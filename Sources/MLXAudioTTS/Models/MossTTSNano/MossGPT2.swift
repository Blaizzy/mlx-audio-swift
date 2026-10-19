import Foundation
@preconcurrency import MLX
import MLXFast
@preconcurrency import MLXAudioCore
import MLXNN

final class MossRotaryEmbedding: @unchecked Sendable {
    private let cosCache: MLXArray
    private let sinCache: MLXArray

    init(headDim: Int, ropeBase: Float, maxPositionEmbeddings: Int) {
        precondition(headDim % 2 == 0, "RoPE head dimension must be even")
        let invFreq = 1.0 / pow(
            MLXArray(ropeBase),
            MLX.arange(0, headDim, step: 2, dtype: .float32) / MLXArray(Float(headDim))
        )
        let positions = MLX.arange(maxPositionEmbeddings, dtype: .float32)
        let angles = MLX.outer(positions, invFreq)
        self.cosCache = cos(angles)
        self.sinCache = sin(angles)
    }

    func apply(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let sequenceLength = x.dim(2)
        let cosValues = cosCache[offset..<(offset + sequenceLength), 0...]
            .reshaped([1, 1, sequenceLength, cosCache.dim(1)])
            .asType(x.dtype)
        let sinValues = sinCache[offset..<(offset + sequenceLength), 0...]
            .reshaped([1, 1, sequenceLength, sinCache.dim(1)])
            .asType(x.dtype)

        let reshaped = x.reshaped(x.shape.dropLast() + [x.shape.last! / 2, 2])
        let xEven = reshaped[0..., 0..., 0..., 0..., 0]
        let xOdd = reshaped[0..., 0..., 0..., 0..., 1]
        let rotated = MLX.stacked([
            xEven * cosValues - xOdd * sinValues,
            xOdd * cosValues + xEven * sinValues,
        ], axis: -1)
        return rotated.reshaped(x.shape)
    }
}

private func mossCausalAdditiveMask(
    queryLength: Int,
    keyLength: Int,
    dtype: DType,
    attentionMask: MLXArray?
) -> MLXArray {
    let offset = max(keyLength - queryLength, 0)
    let queryPositions = (MLXArray(0 ..< queryLength) + MLXArray(offset)).reshaped([queryLength, 1])
    let keyPositions = MLXArray(0 ..< keyLength).reshaped([1, keyLength])
    let visible = keyPositions .<= queryPositions
    let minimum = MLXArray(Float(dtype.finfo?.min ?? -Double.greatestFiniteMagnitude))
    var mask = MLX.where(visible, MLXArray(0.0), minimum).asType(dtype)
        .reshaped([1, 1, queryLength, keyLength])

    if let attentionMask {
        let keyMask = attentionMask.asType(.bool).reshaped([attentionMask.dim(0), 1, 1, attentionMask.dim(1)])
        let keyAdditive = MLX.where(keyMask, MLXArray(0.0), minimum).asType(dtype)
        mask = mask + keyAdditive
    }
    return mask
}

private final class MossGPT2Attention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let rope: MossRotaryEmbedding?

    @ModuleInfo(key: "c_attn") var cAttn: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(config: MossGPT2Config, layerIndex: Int) {
        precondition(config.nEmbd % config.nHead == 0, "n_embd must be divisible by n_head")
        self.embedDim = config.nEmbd
        self.numHeads = config.nHead
        self.headDim = config.nEmbd / config.nHead
        var scale = config.scaleAttnWeights ? pow(Float(headDim), -0.5) : 1.0
        if config.scaleAttnByInverseLayerIdx {
            scale /= Float(layerIndex + 1)
        }
        self.scale = scale
        if config.positionEmbeddingType.lowercased() == "rope" {
            self.rope = MossRotaryEmbedding(
                headDim: headDim,
                ropeBase: config.ropeBase,
                maxPositionEmbeddings: config.nPositions
            )
        } else {
            self.rope = nil
        }

        _cAttn.wrappedValue = Linear(embedDim, 3 * embedDim, bias: true)
        _cProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let queryLength = x.dim(1)
        let qkv = cAttn(x)

        var queries = qkv[0..., 0..., ..<embedDim]
        var keys = qkv[0..., 0..., embedDim..<(2 * embedDim)]
        var values = qkv[0..., 0..., (2 * embedDim)...]

        queries = queries.reshaped(batchSize, queryLength, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(batchSize, queryLength, numHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, queryLength, numHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        if let rope {
            queries = rope.apply(queries, offset: offset)
            keys = rope.apply(keys, offset: offset)
        }

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let keyLength = keys.dim(2)
        let mask = mossCausalAdditiveMask(
            queryLength: queryLength,
            keyLength: keyLength,
            dtype: x.dtype,
            attentionMask: attentionMask
        )
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        return cProj(output.transposed(0, 2, 1, 3).reshaped(batchSize, queryLength, embedDim))
    }
}

private final class MossGPT2MLP: Module {
    let activationFunction: String

    @ModuleInfo(key: "fc_in") var fcIn: Linear
    @ModuleInfo(key: "fc_out") var fcOut: Linear

    init(config: MossGPT2Config) {
        self.activationFunction = config.activationFunction
        _fcIn.wrappedValue = Linear(config.nEmbd, config.intermediateSize, bias: true)
        _fcOut.wrappedValue = Linear(config.intermediateSize, config.nEmbd, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = fcIn(x)
        if activationFunction == "gelu_new" {
            return fcOut(MLXNN.geluApproximate(hidden))
        }
        return fcOut(MLXNN.gelu(hidden))
    }
}

private final class MossGPT2Block: Module {
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "attn") var attention: MossGPT2Attention
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MossGPT2MLP

    init(config: MossGPT2Config, layerIndex: Int) {
        _ln1.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
        _attention.wrappedValue = MossGPT2Attention(config: config, layerIndex: layerIndex)
        _ln2.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
        _mlp.wrappedValue = MossGPT2MLP(config: config)
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        let h = x + attention(ln1(x), attentionMask: attentionMask, cache: cache)
        return h + mlp(ln2(h))
    }
}

final class MossGPT2Model: Module {
    let config: MossGPT2Config
    fileprivate let h: [MossGPT2Block]

    @ModuleInfo(key: "wte") var wte: Embedding?
    @ModuleInfo(key: "wpe") var wpe: Embedding?
    @ModuleInfo(key: "ln_f") var lnF: LayerNorm

    init(config: MossGPT2Config, useTokenEmbedding: Bool = true) {
        self.config = config
        if useTokenEmbedding {
            _wte.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.nEmbd)
        } else {
            _wte.wrappedValue = nil
        }
        if config.positionEmbeddingType.lowercased() == "absolute" {
            _wpe.wrappedValue = Embedding(embeddingCount: config.nPositions, dimensions: config.nEmbd)
        } else {
            _wpe.wrappedValue = nil
        }
        self.h = (0 ..< config.nLayer).map { MossGPT2Block(config: config, layerIndex: $0) }
        _lnF.wrappedValue = LayerNorm(dimensions: config.nEmbd, eps: config.layerNormEpsilon)
    }

    var tokenEmbeddingWeight: MLXArray {
        get throws {
            guard let wte else {
                throw MossTTSNanoError.invalidInput("GPT-2 token embedding is not available.")
            }
            return wte.weight
        }
    }

    func tokenEmbedding(_ inputIDs: MLXArray) throws -> MLXArray {
        guard let wte else {
            throw MossTTSNanoError.invalidInput("GPT-2 token embedding is not available.")
        }
        return wte(inputIDs)
    }

    func makeCache() -> [any KVCache] {
        h.map { _ in KVCacheSimple() }
    }

    func callAsFunction(
        inputIDs: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        cache: [any KVCache]? = nil
    ) throws -> MLXArray {
        var hiddenStates: MLXArray
        if let inputsEmbeds {
            hiddenStates = inputsEmbeds
        } else if let inputIDs {
            hiddenStates = try tokenEmbedding(inputIDs)
        } else {
            throw MossTTSNanoError.invalidInput("inputIDs or inputsEmbeds are required.")
        }

        if let wpe {
            let seqLen = hiddenStates.dim(1)
            let offset = cache?.first?.offset ?? 0
            let positionIDs = MLXArray(Int32(offset)..<Int32(offset + seqLen))
            hiddenStates = hiddenStates + wpe(positionIDs)
        }

        for (index, block) in h.enumerated() {
            hiddenStates = block(
                hiddenStates,
                attentionMask: attentionMask,
                cache: cache?[index]
            )
        }
        return lnF(hiddenStates)
    }
}
