import Foundation
@preconcurrency import MLX
import MLXNN

private func diaActivation(_ name: String, _ x: MLXArray) -> MLXArray {
    switch name {
    case "silu", "swish": return x * MLX.sigmoid(x)
    case "gelu": return MLXNN.geluApproximate(x)
    case "relu": return MLX.maximum(x, 0)
    default: return x
    }
}

/// T5-style projection: a single weight tensor of shape `inShapes + outFeatures`,
/// contracted against `axis` of the input via tensordot.
public final class DiaDenseGeneral: Module {
    public var weight: MLXArray
    private let axis: [Int]

    public init(inShapes: [Int], outFeatures: [Int], axis: [Int] = [-1]) {
        self.axis = axis
        self.weight = MLXArray.zeros(inShapes + outFeatures, dtype: .float32)
    }

    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        let ndim = inputs.ndim
        let normAxis = axis.map { $0 >= 0 ? $0 : ndim + $0 }
        let contract = Array(0..<normAxis.count)
        return tensordot(inputs, weight, axes: (normAxis, contract))
    }
}

/// Rotary embedding with half-split rotation (Dia variant). No parameters.
public final class DiaRotaryEmbedding: Module {
    private let embeddingDims: Int
    private let timescale: [Float]

    public init(embeddingDims: Int, minTimescale: Int, maxTimescale: Int) {
        self.embeddingDims = embeddingDims
        let half = embeddingDims / 2
        self.timescale = (0..<half).map { i in
            let fraction = (2.0 * Float(i)) / Float(embeddingDims)
            return Float(minTimescale) * powf(Float(maxTimescale) / Float(minTimescale), fraction)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, position: MLXArray) -> MLXArray {
        let pos = position.expandedDimensions(axis: -1).expandedDimensions(axis: -1)
        let ts = MLXArray(timescale)
        let sinusoid = pos.asType(.float32) / ts
        let sin = MLX.sin(sinusoid).asType(inputs.dtype)
        let cos = MLX.cos(sinusoid).asType(inputs.dtype)
        let half = embeddingDims / 2
        let first = inputs[.ellipsis, 0 ..< half]
        let second = inputs[.ellipsis, half...]
        let firstPart = first * cos - second * sin
        let secondPart = second * cos + first * sin
        return MLX.concatenated([firstPart, secondPart], axis: -1)
    }
}

/// Concatenation-based KV cache. For cross-attention, `k`/`v` are set once at
/// construction; for decoder self-attention they grow along the sequence axis.
public final class DiaKVCache {
    public var k: MLXArray?
    public var v: MLXArray?

    public init(k: MLXArray? = nil, v: MLXArray? = nil) {
        self.k = k
        self.v = v
    }

    public func prefill(_ nk: MLXArray, _ nv: MLXArray) {
        k = nk
        v = nv
    }

    public func updateAndFetch(_ nk: MLXArray, _ nv: MLXArray) -> (MLXArray, MLXArray) {
        k = k == nil ? nk : MLX.concatenated([k!, nk], axis: 2)
        v = v == nil ? nv : MLX.concatenated([v!, nv], axis: 2)
        return (k!, v!)
    }
}

public final class DiaAttention: Module {
    private let numQueryHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let isCrossAttn: Bool
    private let numGqaGroups: Int

    @ModuleInfo(key: "q_proj") var qProj: DiaDenseGeneral
    @ModuleInfo(key: "k_proj") var kProj: DiaDenseGeneral
    @ModuleInfo(key: "v_proj") var vProj: DiaDenseGeneral
    @ModuleInfo(key: "o_proj") var oProj: DiaDenseGeneral
    @ModuleInfo(key: "rotary_emb") var rotaryEmb: DiaRotaryEmbedding

    public init(
        config: DiaConfig, qEmbedDim: Int, kvEmbedDim: Int,
        numQueryHeads: Int, numKVHeads: Int, headDim: Int,
        isCrossAttn: Bool, outEmbedDim: Int
    ) {
        self.numQueryHeads = numQueryHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.isCrossAttn = isCrossAttn
        self.numGqaGroups = numQueryHeads / numKVHeads
        self._qProj = ModuleInfo(
            wrappedValue: DiaDenseGeneral(inShapes: [qEmbedDim], outFeatures: [numQueryHeads, headDim]), key: "q_proj")
        self._kProj = ModuleInfo(
            wrappedValue: DiaDenseGeneral(inShapes: [kvEmbedDim], outFeatures: [numKVHeads, headDim]), key: "k_proj")
        self._vProj = ModuleInfo(
            wrappedValue: DiaDenseGeneral(inShapes: [kvEmbedDim], outFeatures: [numKVHeads, headDim]), key: "v_proj")
        self._oProj = ModuleInfo(
            wrappedValue: DiaDenseGeneral(inShapes: [numQueryHeads, headDim], outFeatures: [outEmbedDim], axis: [-2, -1]), key: "o_proj")
        self._rotaryEmb = ModuleInfo(
            wrappedValue: DiaRotaryEmbedding(
                embeddingDims: headDim,
                minTimescale: config.model.ropeMinTimescale,
                maxTimescale: config.model.ropeMaxTimescale),
            key: "rotary_emb")
    }

    public func callAsFunction(
        _ xq: MLXArray, _ xkv: MLXArray, qPositions: MLXArray, kvPositions: MLXArray?,
        attnMask: MLXArray?, cache: DiaKVCache?, prefill: Bool = false
    ) -> MLXArray {
        let kvPos = kvPositions ?? qPositions
        let originalDtype = xq.dtype

        var xqBTNH = qProj(xq)
        xqBTNH = rotaryEmb(xqBTNH, position: qPositions)
        let xqBNTH = xqBTNH.transposed(0, 2, 1, 3)

        var attnK: MLXArray
        var attnV: MLXArray

        if isCrossAttn {
            attnK = cache!.k!
            attnV = cache!.v!
        } else {
            var xkBKSH = kProj(xkv)
            var xvBKSH = vProj(xkv)
            xkBKSH = rotaryEmb(xkBKSH, position: kvPos)
            xkBKSH = xkBKSH.transposed(0, 2, 1, 3)
            xvBKSH = xvBKSH.transposed(0, 2, 1, 3)

            if numGqaGroups > 1 {
                xkBKSH = MLX.repeated(xkBKSH, count: numGqaGroups, axis: 1)
                xvBKSH = MLX.repeated(xvBKSH, count: numGqaGroups, axis: 1)
            }

            if cache == nil {
                attnK = xkBKSH
                attnV = xvBKSH
            } else if prefill {
                attnK = xkBKSH
                attnV = xvBKSH
                cache!.prefill(attnK, attnV)
            } else {
                (attnK, attnV) = cache!.updateAndFetch(xkBKSH, xvBKSH)
            }
        }

        var attnScores = MLX.matmul(xqBNTH, attnK.swappedAxes(2, 3))
        if let attnMask {
            attnScores = MLX.where(attnMask, attnScores, MLXArray(Float(-1e9)))
        }
        let attnWeights = MLX.softmax(attnScores, axis: -1)
        var attnOutput = MLX.matmul(attnWeights, attnV)
        attnOutput = attnOutput.transposed(0, 2, 1, 3)
        let output = oProj(attnOutput)
        return output.dtype != originalDtype ? output.asType(originalDtype) : output
    }
}

public final class DiaMlpBlock: Module {
    @ModuleInfo(key: "wi_fused") var wiFused: DiaDenseGeneral
    @ModuleInfo(key: "wo") var wo: DiaDenseGeneral
    private let activations: [String]

    public init(embedDim: Int, intermediateDim: Int, activations: [String]) {
        self.activations = activations
        self._wiFused = ModuleInfo(
            wrappedValue: DiaDenseGeneral(inShapes: [embedDim], outFeatures: [activations.count, intermediateDim]), key: "wi_fused")
        self._wo = ModuleInfo(
            wrappedValue: DiaDenseGeneral(inShapes: [intermediateDim], outFeatures: [embedDim]), key: "wo")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let fused = wiFused(x)
        let gate = diaActivation(activations[0], fused[.ellipsis, 0, 0...])
        let up = diaActivation(activations[1], fused[.ellipsis, 1, 0...])
        return wo(gate * up)
    }
}

public final class DiaEncoderLayer: Module {
    @ModuleInfo(key: "pre_sa_norm") var preSaNorm: RMSNorm
    @ModuleInfo(key: "self_attention") var selfAttention: DiaAttention
    @ModuleInfo(key: "post_sa_norm") var postSaNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: DiaMlpBlock

    public init(config: DiaConfig) {
        let enc = config.model.encoder
        let eps = config.model.normalizationLayerEpsilon
        self._preSaNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: enc.nEmbd, eps: eps), key: "pre_sa_norm")
        self._selfAttention = ModuleInfo(
            wrappedValue: DiaAttention(
                config: config, qEmbedDim: enc.nEmbd, kvEmbedDim: enc.nEmbd,
                numQueryHeads: enc.nHead, numKVHeads: enc.nHead, headDim: enc.headDim,
                isCrossAttn: false, outEmbedDim: enc.nEmbd),
            key: "self_attention")
        self._postSaNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: enc.nEmbd, eps: eps), key: "post_sa_norm")
        self._mlp = ModuleInfo(
            wrappedValue: DiaMlpBlock(embedDim: enc.nEmbd, intermediateDim: enc.nHidden, activations: enc.mlpActivations),
            key: "mlp")
    }

    public func callAsFunction(_ x: MLXArray, srcPositions: MLXArray, attnMask: MLXArray?) -> MLXArray {
        var h = x + selfAttention(preSaNorm(x), preSaNorm(x), qPositions: srcPositions, kvPositions: srcPositions, attnMask: attnMask, cache: nil)
        h = h + mlp(postSaNorm(h))
        return h
    }
}

public final class DiaEncoder: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "layers") var layers: [DiaEncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(config: DiaConfig) {
        let enc = config.model.encoder
        self._embedding = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.model.srcVocabSize, dimensions: enc.nEmbd), key: "embedding")
        self._layers = ModuleInfo(wrappedValue: (0..<enc.nLayer).map { _ in DiaEncoderLayer(config: config) }, key: "layers")
        self._norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: enc.nEmbd, eps: config.model.normalizationLayerEpsilon), key: "norm")
    }

    public func callAsFunction(_ xIds: MLXArray, srcPositions: MLXArray, attnMask: MLXArray?) -> MLXArray {
        var x = embedding(xIds)
        for layer in layers { x = layer(x, srcPositions: srcPositions, attnMask: attnMask) }
        return norm(x)
    }
}

public final class DiaDecoderLayer: Module {
    @ModuleInfo(key: "pre_sa_norm") var preSaNorm: RMSNorm
    @ModuleInfo(key: "pre_ca_norm") var preCaNorm: RMSNorm
    @ModuleInfo(key: "pre_mlp_norm") var preMlpNorm: RMSNorm
    @ModuleInfo(key: "self_attention") var selfAttention: DiaAttention
    @ModuleInfo(key: "cross_attention") var crossAttention: DiaAttention
    @ModuleInfo(key: "mlp") var mlp: DiaMlpBlock

    public init(config: DiaConfig) {
        let dec = config.model.decoder
        let enc = config.model.encoder
        let eps = config.model.normalizationLayerEpsilon
        self._preSaNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dec.nEmbd, eps: eps), key: "pre_sa_norm")
        self._preCaNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dec.nEmbd, eps: eps), key: "pre_ca_norm")
        self._preMlpNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dec.nEmbd, eps: eps), key: "pre_mlp_norm")
        self._selfAttention = ModuleInfo(
            wrappedValue: DiaAttention(
                config: config, qEmbedDim: dec.nEmbd, kvEmbedDim: dec.nEmbd,
                numQueryHeads: dec.gqaQueryHeads, numKVHeads: dec.kvHeads, headDim: dec.gqaHeadDim,
                isCrossAttn: false, outEmbedDim: dec.nEmbd),
            key: "self_attention")
        self._crossAttention = ModuleInfo(
            wrappedValue: DiaAttention(
                config: config, qEmbedDim: dec.nEmbd, kvEmbedDim: enc.nEmbd,
                numQueryHeads: dec.crossQueryHeads, numKVHeads: dec.crossQueryHeads, headDim: dec.crossHeadDim,
                isCrossAttn: true, outEmbedDim: dec.nEmbd),
            key: "cross_attention")
        self._mlp = ModuleInfo(
            wrappedValue: DiaMlpBlock(embedDim: dec.nEmbd, intermediateDim: dec.nHidden, activations: dec.mlpActivations),
            key: "mlp")
    }

    public func callAsFunction(
        _ x: MLXArray, encoderOut: MLXArray, tgtPositions: MLXArray, srcPositions: MLXArray?,
        selfAttnMask: MLXArray?, crossAttnMask: MLXArray?,
        selfAttnCache: DiaKVCache, crossAttnCache: DiaKVCache, prefill: Bool
    ) -> MLXArray {
        var h = x + selfAttention(
            preSaNorm(x), preSaNorm(x), qPositions: tgtPositions, kvPositions: tgtPositions,
            attnMask: selfAttnMask, cache: selfAttnCache, prefill: prefill)
        h = h + crossAttention(
            preCaNorm(h), encoderOut, qPositions: tgtPositions, kvPositions: srcPositions,
            attnMask: crossAttnMask, cache: crossAttnCache)
        h = h + mlp(preMlpNorm(h))
        return h
    }
}

public final class DiaDecoder: Module {
    @ModuleInfo(key: "embeddings") var embeddings: [Embedding]
    @ModuleInfo(key: "layers") var layers: [DiaDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "logits_dense") var logitsDense: DiaDenseGeneral

    public let numChannels: Int
    public let numLayers: Int

    public init(config: DiaConfig) {
        let dec = config.model.decoder
        self.numChannels = config.data.channels
        self.numLayers = dec.nLayer
        self._embeddings = ModuleInfo(
            wrappedValue: (0..<numChannels).map { _ in Embedding(embeddingCount: config.model.tgtVocabSize, dimensions: dec.nEmbd) },
            key: "embeddings")
        self._layers = ModuleInfo(wrappedValue: (0..<numLayers).map { _ in DiaDecoderLayer(config: config) }, key: "layers")
        self._norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dec.nEmbd, eps: config.model.normalizationLayerEpsilon), key: "norm")
        self._logitsDense = ModuleInfo(
            wrappedValue: DiaDenseGeneral(inShapes: [dec.nEmbd], outFeatures: [numChannels, config.model.tgtVocabSize]),
            key: "logits_dense")
    }

    private func embed(_ tgtIds: MLXArray) -> MLXArray {
        var x: MLXArray? = nil
        for i in 0..<numChannels {
            let e = embeddings[i](tgtIds[.ellipsis, i])
            x = x == nil ? e : x! + e
        }
        return x!
    }

    /// Precompute cross-attention K/V for each layer from the encoder output.
    public func precomputeCrossAttentionKV(encoderOut: MLXArray, srcPositions: MLXArray?) -> [DiaKVCache] {
        layers.map { layer in
            let ca = layer.crossAttention
            var kp = ca.kProj(encoderOut)
            let vp = ca.vProj(encoderOut)
            kp = ca.rotaryEmb(kp, position: srcPositions ?? MLXArray(0))
            let k = kp.transposed(0, 2, 1, 3)
            let v = vp.transposed(0, 2, 1, 3)
            return DiaKVCache(k: k, v: v)
        }
    }

    /// Prefill forward over a full target sequence (self-attn caches get filled).
    public func callAsFunction(
        _ tgtIds: MLXArray, encoderOut: MLXArray, tgtPositions: MLXArray, srcPositions: MLXArray?,
        selfAttnMask: MLXArray?, crossAttnMask: MLXArray?,
        selfAttentionCache: [DiaKVCache], crossAttentionCache: [DiaKVCache]
    ) -> MLXArray {
        var x = embed(tgtIds)
        for i in 0..<numLayers {
            x = layers[i](
                x, encoderOut: encoderOut, tgtPositions: tgtPositions, srcPositions: srcPositions,
                selfAttnMask: selfAttnMask, crossAttnMask: crossAttnMask,
                selfAttnCache: selfAttentionCache[i], crossAttnCache: crossAttentionCache[i], prefill: true)
        }
        return logitsDense(norm(x)).asType(.float32)
    }

    /// Single autoregressive decode step.
    public func decodeStep(
        _ tgtIds: MLXArray, tgtPos: MLXArray, encoderOut: MLXArray, crossAttnMask: MLXArray?,
        selfAttentionCache: [DiaKVCache], crossAttentionCache: [DiaKVCache]
    ) -> MLXArray {
        var x = embed(tgtIds)
        for i in 0..<numLayers {
            x = layers[i](
                x, encoderOut: encoderOut, tgtPositions: tgtPos, srcPositions: nil,
                selfAttnMask: nil, crossAttnMask: crossAttnMask,
                selfAttnCache: selfAttentionCache[i], crossAttnCache: crossAttentionCache[i], prefill: false)
        }
        return logitsDense(norm(x)).asType(.float32)
    }
}

public final class DiaModel: Module {
    @ModuleInfo(key: "encoder") var encoder: DiaEncoder
    @ModuleInfo(key: "decoder") var decoder: DiaDecoder

    public init(_ config: DiaConfig) {
        self._encoder = ModuleInfo(wrappedValue: DiaEncoder(config: config), key: "encoder")
        self._decoder = ModuleInfo(wrappedValue: DiaDecoder(config: config), key: "decoder")
    }
}
