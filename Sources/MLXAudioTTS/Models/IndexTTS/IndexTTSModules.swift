import Foundation
@preconcurrency import MLX
import MLXFast
@preconcurrency import MLXLMCommon
import MLXNN

public struct IndexTTSPreparedEmbedding: Sendable {
    public let embeddings: MLXArray
    public let textTokenCount: Int
    public let conditioningTokenCount: Int
}

public struct IndexTTSMelGeneration: Sendable {
    public let tokenIDs: [Int]
    public let latentStates: MLXArray
}

final class IndexTTSLearnedPositionEncoding: Module {
    @ModuleInfo(key: "emb") var emb: Embedding

    init(sequenceLength: Int, modelDim: Int) {
        _emb.wrappedValue = Embedding(embeddingCount: sequenceLength, dimensions: modelDim)
    }

    func callAsFunction(sequenceLength: Int, offset: Int = 0) -> MLXArray {
        let end = offset + sequenceLength
        let positionIDs = MLXArray(Int32(offset)..<Int32(end))
        return emb(positionIDs).expandedDimensions(axis: 0)
    }
}

final class IndexTTSMultiHeadAttention: Module {
    let nHead: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(nHead: Int, nFeat: Int, bias: Bool = true, headDim: Int? = nil) {
        self.nHead = nHead
        self.headDim = headDim ?? (nFeat / nHead)
        self.scale = pow(Float(self.headDim), -0.5)
        _linearQ.wrappedValue = Linear(nFeat, self.headDim * nHead, bias: bias)
        _linearK.wrappedValue = Linear(nFeat, self.headDim * nHead, bias: bias)
        _linearV.wrappedValue = Linear(nFeat, self.headDim * nHead, bias: bias)
        _linearOut.wrappedValue = Linear(self.headDim * nHead, nFeat, bias: bias)
    }

    func callAsFunction(
        q inputQ: MLXArray,
        k inputK: MLXArray,
        v inputV: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        var q = linearQ(inputQ)
        var k = linearK(inputK)
        var v = linearV(inputV)

        let batch = q.dim(0)
        let queryLength = q.dim(1)
        let keyLength = k.dim(1)

        q = q.reshaped(batch, queryLength, nHead, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, keyLength, nHead, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, keyLength, nHead, headDim).transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        return linearOut(out.transposed(0, 2, 1, 3).reshaped(batch, queryLength, -1))
    }
}

final class IndexTTSPerceiverFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear

    init(dim: Int, dFF: Int, useBias: Bool = true) {
        _w1.wrappedValue = Linear(dim, dFF * 2, bias: useBias)
        _w2.wrappedValue = Linear(dFF, dim, bias: useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = w1(x)
        let dFF = projected.dim(-1) / 2
        let hidden = projected[0..., 0..., 0..<dFF]
        let gate = projected[0..., 0..., dFF..<(2 * dFF)]
        return w2(MLXNN.gelu(gate) * hidden)
    }
}

final class IndexTTSPerceiverLayer: Module {
    @ModuleInfo(key: "attention") var attention: IndexTTSMultiHeadAttention
    @ModuleInfo(key: "feed_forward") var feedForward: IndexTTSPerceiverFeedForward

    init(nDim: Int, nHeads: Int, nDimHead: Int, nFFMult: Int) {
        _attention.wrappedValue = IndexTTSMultiHeadAttention(
            nHead: nHeads,
            nFeat: nDim,
            bias: false,
            headDim: nDimHead
        )
        _feedForward.wrappedValue = IndexTTSPerceiverFeedForward(
            dim: nDim,
            dFF: (nDim * nFFMult * 2) / 3
        )
    }

    func callAsFunction(latents: MLXArray, context: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let kv = concatenated([context, latents], axis: -2)
        var hidden = latents + attention(q: latents, k: kv, v: kv, mask: mask)
        hidden = hidden + feedForward(hidden)
        return hidden
    }
}

final class IndexTTSPerceiverResampler: Module {
    let nLatents: Int
    let nDim: Int
    var latents: MLXArray

    @ModuleInfo(key: "proj_context") var projContext: Linear?
    @ModuleInfo(key: "layers") var layers: [IndexTTSPerceiverLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(
        nDim: Int,
        nDepth: Int = 2,
        nDimContext: Int? = nil,
        nLatents: Int = 32,
        nDimHead: Int = 64,
        nHeads: Int = 8,
        nFFMult: Int = 4
    ) {
        self.nLatents = nLatents
        self.nDim = nDim
        let contextDim = nDimContext ?? nDim
        self.latents = MLXArray.zeros([nLatents, nDim])
        if contextDim == nDim {
            _projContext.wrappedValue = nil
        } else {
            _projContext.wrappedValue = Linear(contextDim, nDim)
        }
        _layers.wrappedValue = (0..<nDepth).map { _ in
            IndexTTSPerceiverLayer(nDim: nDim, nHeads: nHeads, nDimHead: nDimHead, nFFMult: nFFMult)
        }
        _norm.wrappedValue = RMSNorm(dimensions: nDim)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let batch = x.dim(0)
        var context = x
        if let projContext {
            context = projContext(context)
        }
        var hidden = MLX.broadcast(latents.expandedDimensions(axis: 0), to: [batch, nLatents, nDim])
        for layer in layers {
            hidden = layer(latents: hidden, context: context, mask: mask)
        }
        return norm(hidden)
    }
}

final class IndexTTSRelPositionalEncoding {
    let dModel: Int
    let scaleInput: Bool
    var maxLen: Int
    var pe: MLXArray

    init(dModel: Int, maxLen: Int = 5000, scaleInput: Bool = true) {
        precondition(dModel % 2 == 0, "IndexTTS relative positional dimension must be even")
        self.dModel = dModel
        self.scaleInput = scaleInput
        self.maxLen = maxLen
        self.pe = Self.makePE(maxLen: maxLen, dModel: dModel)
    }

    private static func makePE(maxLen: Int, dModel: Int) -> MLXArray {
        let positions = MLX.arange(0, maxLen, dtype: .float32).expandedDimensions(axis: 1)
        let divTerm = MLX.exp(
            MLX.arange(0, dModel, step: 2, dtype: .float32)
                * Float(-log(10000.0) / Float(dModel))
        )
        let sin = MLX.sin(positions * divTerm)
        let cos = MLX.cos(positions * divTerm)
        return MLX.stacked([sin, cos], axis: -1).reshaped(maxLen, dModel).expandedDimensions(axis: 0)
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let required = x.dim(1) + offset
        if required > maxLen {
            maxLen = required + 1
            pe = Self.makePE(maxLen: maxLen, dModel: dModel)
        }
        let scaled = scaleInput ? x * sqrt(Float(dModel)) : x
        let posEmb = pe[0..., offset..<(offset + x.dim(1)), 0...].asType(x.dtype)
        return (scaled, posEmb)
    }
}

final class IndexTTSRelPositionMultiHeadAttention: Module {
    let nHead: Int
    let headDim: Int
    let scale: Float
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear

    init(nHead: Int, nFeat: Int, bias: Bool = true, headDim: Int? = nil) {
        self.nHead = nHead
        self.headDim = headDim ?? (nFeat / nHead)
        self.scale = pow(Float(self.headDim), -0.5)
        _posBiasU.wrappedValue = MLXArray.zeros([nHead, self.headDim])
        _posBiasV.wrappedValue = MLXArray.zeros([nHead, self.headDim])
        _linearQ.wrappedValue = Linear(nFeat, self.headDim * nHead, bias: bias)
        _linearK.wrappedValue = Linear(nFeat, self.headDim * nHead, bias: bias)
        _linearV.wrappedValue = Linear(nFeat, self.headDim * nHead, bias: bias)
        _linearOut.wrappedValue = Linear(self.headDim * nHead, nFeat, bias: bias)
        _linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
    }

    func callAsFunction(
        q inputQ: MLXArray,
        k inputK: MLXArray,
        v inputV: MLXArray,
        posEmb: MLXArray?,
        mask: MLXArray? = nil
    ) throws -> MLXArray {
        guard let posEmb else {
            throw IndexTTSError.invalidInput("posEmb is required for IndexTTS relative attention.")
        }

        var q = linearQ(inputQ)
        var k = linearK(inputK)
        var v = linearV(inputV)
        var p = linearPos(posEmb)

        let batch = q.dim(0)
        let queryLength = q.dim(1)
        let keyLength = k.dim(1)
        let posLength = p.dim(1)

        q = q.reshaped(batch, queryLength, nHead, headDim)
        let qU = (q + posBiasU).transposed(0, 2, 1, 3)
        let qV = (q + posBiasV).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, keyLength, nHead, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, keyLength, nHead, headDim).transposed(0, 2, 1, 3)
        p = p.reshaped(batch, posLength, nHead, headDim).transposed(0, 2, 1, 3)

        var relBias = matmul(qV, p.transposed(0, 1, 3, 2)) * scale
        if let mask {
            let expanded = mask.expandedDimensions(axis: 1)
            relBias = MLX.where(expanded .== 0, MLXArray(-Float.infinity), relBias)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: qU,
            keys: k,
            values: v,
            scale: scale,
            mask: relBias
        )
        return linearOut(out.transposed(0, 2, 1, 3).reshaped(batch, queryLength, -1))
    }
}

final class IndexTTSConformerFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear

    init(dim: Int, dFF: Int, useBias: Bool = true) {
        _w1.wrappedValue = Linear(dim, dFF, bias: useBias)
        _w2.wrappedValue = Linear(dFF, dim, bias: useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)))
    }
}

final class IndexTTSConvolutionModule: Module {
    let channels: Int

    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(config: IndexTTSConformerConfig) {
        self.channels = config.outputSize
        _pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: config.outputSize,
            outputChannels: config.outputSize * 2,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: config.useBias
        )
        _depthwiseConv.wrappedValue = Conv1d(
            inputChannels: config.outputSize,
            outputChannels: config.outputSize,
            kernelSize: config.cnnModuleKernel,
            stride: 1,
            padding: (config.cnnModuleKernel - 1) / 2,
            groups: config.outputSize,
            bias: config.useBias
        )
        _norm.wrappedValue = LayerNorm(dimensions: config.outputSize)
        _pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: config.outputSize,
            outputChannels: config.outputSize,
            kernelSize: 1,
            stride: 1,
            padding: 0,
            bias: config.useBias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = pointwiseConv1(x)
        let left = projected[0..., 0..., 0..<channels]
        let right = projected[0..., 0..., channels..<(channels * 2)]
        var hidden = left * sigmoid(right)
        hidden = depthwiseConv(hidden)
        hidden = norm(hidden)
        hidden = silu(hidden)
        return pointwiseConv2(hidden)
    }
}

final class IndexTTSConformerBlock: Module {
    let macaronStyle: Bool
    let ffScale: Float

    @ModuleInfo(key: "norm_ff_macaron") var normFFMacaron: LayerNorm?
    @ModuleInfo(key: "feed_forward_macaron") var feedForwardMacaron: IndexTTSConformerFeedForward?
    @ModuleInfo(key: "norm_mha") var normMHA: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: Module
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv_module") var convModule: IndexTTSConvolutionModule
    @ModuleInfo(key: "norm_ff") var normFF: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: IndexTTSConformerFeedForward
    @ModuleInfo(key: "norm_final") var normFinal: LayerNorm

    init(config: IndexTTSConformerConfig) {
        self.macaronStyle = config.macaronStyle
        self.ffScale = config.macaronStyle ? 0.5 : 1.0
        if config.macaronStyle {
            _normFFMacaron.wrappedValue = LayerNorm(dimensions: config.outputSize)
            _feedForwardMacaron.wrappedValue = IndexTTSConformerFeedForward(
                dim: config.outputSize,
                dFF: config.linearUnits,
                useBias: config.useBias
            )
        } else {
            _normFFMacaron.wrappedValue = nil
            _feedForwardMacaron.wrappedValue = nil
        }
        _normMHA.wrappedValue = LayerNorm(dimensions: config.outputSize)
        if config.posEncLayerType == "rel_pos" {
            _selfAttn.wrappedValue = IndexTTSRelPositionMultiHeadAttention(
                nHead: config.attentionHeads,
                nFeat: config.outputSize,
                bias: config.useBias
            )
        } else {
            _selfAttn.wrappedValue = IndexTTSMultiHeadAttention(
                nHead: config.attentionHeads,
                nFeat: config.outputSize,
                bias: config.useBias
            )
        }
        _normConv.wrappedValue = LayerNorm(dimensions: config.outputSize)
        _convModule.wrappedValue = IndexTTSConvolutionModule(config: config)
        _normFF.wrappedValue = LayerNorm(dimensions: config.outputSize)
        _feedForward.wrappedValue = IndexTTSConformerFeedForward(
            dim: config.outputSize,
            dFF: config.linearUnits,
            useBias: config.useBias
        )
        _normFinal.wrappedValue = LayerNorm(dimensions: config.outputSize)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray? = nil, mask: MLXArray? = nil) throws -> MLXArray {
        var hidden = x
        if let normFFMacaron, let feedForwardMacaron {
            hidden = hidden + ffScale * feedForwardMacaron(normFFMacaron(hidden))
        }

        let attnInput = normMHA(hidden)
        let attnOutput: MLXArray
        if let relAttn = selfAttn as? IndexTTSRelPositionMultiHeadAttention {
            attnOutput = try relAttn(q: attnInput, k: attnInput, v: attnInput, posEmb: posEmb, mask: mask)
        } else if let plainAttn = selfAttn as? IndexTTSMultiHeadAttention {
            attnOutput = plainAttn(q: attnInput, k: attnInput, v: attnInput, mask: mask)
        } else {
            throw IndexTTSError.invalidInput("Unsupported IndexTTS Conformer attention module.")
        }
        hidden = hidden + attnOutput
        hidden = hidden + convModule(normConv(hidden))
        hidden = hidden + ffScale * feedForward(normFF(hidden))
        return normFinal(hidden)
    }
}

final class IndexTTSConv2dSubsampling: Module {
    let convSpecs: [(kernel: Int, stride: Int)]
    let outputSize: Int

    @ModuleInfo(key: "conv") var conv: [Conv2d]
    @ModuleInfo(key: "out") var out: [Linear]

    init(config: IndexTTSConformerConfig) {
        self.outputSize = config.outputSize
        self.convSpecs = Self.specs(for: config.inputLayer)

        var convLayers: [Conv2d] = []
        var inChannels = 1
        var outFreq = config.inputSize
        for spec in convSpecs {
            convLayers.append(Conv2d(
                inputChannels: inChannels,
                outputChannels: config.outputSize,
                kernelSize: IntOrPair(spec.kernel),
                stride: IntOrPair(spec.stride),
                padding: IntOrPair(0)
            ))
            inChannels = config.outputSize
            outFreq = max(1, (outFreq - spec.kernel + spec.stride) / spec.stride)
        }
        _conv.wrappedValue = convLayers
        _out.wrappedValue = [Linear(config.outputSize * outFreq, config.outputSize)]
    }

    private static func specs(for inputLayer: String) -> [(kernel: Int, stride: Int)] {
        switch inputLayer {
        case "conv2d", "conv2d2":
            [(3, 2)]
        case "conv2d3":
            [(5, 3)]
        case "conv2d4":
            [(3, 2), (3, 2)]
        case "conv2d6":
            [(3, 2), (5, 3)]
        case "conv2d8":
            [(3, 2), (3, 2), (3, 2)]
        default:
            [(3, 2)]
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x.expandedDimensions(axis: 3)
        for layer in conv {
            hidden = relu(layer(hidden))
        }
        let batch = hidden.dim(0)
        let time = hidden.dim(1)
        let freq = hidden.dim(2)
        let channels = hidden.dim(3)
        hidden = hidden.transposed(0, 1, 3, 2).reshaped(batch, time, channels * freq)
        return out[0](hidden)
    }
}

final class IndexTTSConformerEncoder: Module {
    let config: IndexTTSConformerConfig

    let posEnc: IndexTTSRelPositionalEncoding?
    @ModuleInfo(key: "embed") var embed: IndexTTSConv2dSubsampling
    @ModuleInfo(key: "encoders") var encoders: [IndexTTSConformerBlock]
    @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm

    init(config: IndexTTSConformerConfig) {
        self.config = config
        if config.posEncLayerType == "rel_pos" {
            self.posEnc = IndexTTSRelPositionalEncoding(
                dModel: config.outputSize,
                maxLen: config.posEmbMaxLen,
                scaleInput: config.xscaling
            )
        } else {
            self.posEnc = nil
        }
        _embed.wrappedValue = IndexTTSConv2dSubsampling(config: config)
        _encoders.wrappedValue = (0..<config.numBlocks).map { _ in IndexTTSConformerBlock(config: config) }
        _afterNorm.wrappedValue = LayerNorm(dimensions: config.outputSize, eps: 1e-5)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) throws -> MLXArray {
        var hidden = embed(x)
        var posEmb: MLXArray?
        if let posEnc {
            (hidden, posEmb) = posEnc(hidden)
        }
        for layer in encoders {
            hidden = try layer(hidden, posEmb: posEmb, mask: mask)
        }
        return afterNorm(hidden)
    }
}

final class IndexTTSGPT2Attention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "c_attn") var cAttn: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(config: IndexTTSGPTConfig) {
        self.embedDim = config.modelDim
        self.numHeads = config.heads
        self.headDim = config.modelDim / config.heads
        self.scale = pow(Float(headDim), -0.5)
        _cAttn.wrappedValue = Linear(embedDim, embedDim * 3, bias: true)
        _cProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let qkv = cAttn(x)

        var q = qkv[0..., 0..., 0..<embedDim]
        var k = qkv[0..., 0..., embedDim..<(2 * embedDim)]
        var v = qkv[0..., 0..., (2 * embedDim)..<(3 * embedDim)]

        q = q.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )
        return cProj(out.transposed(0, 2, 1, 3).reshaped(batch, length, embedDim))
    }
}

final class IndexTTSGPT2MLP: Module {
    @ModuleInfo(key: "c_fc") var cFc: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(config: IndexTTSGPTConfig) {
        _cFc.wrappedValue = Linear(config.modelDim, config.modelDim * 4, bias: true)
        _cProj.wrappedValue = Linear(config.modelDim * 4, config.modelDim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        cProj(MLXNN.geluApproximate(cFc(x)))
    }
}

final class IndexTTSGPT2Block: Module {
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "attn") var attention: IndexTTSGPT2Attention
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: IndexTTSGPT2MLP

    init(config: IndexTTSGPTConfig) {
        _ln1.wrappedValue = LayerNorm(dimensions: config.modelDim, eps: 1e-5)
        _attention.wrappedValue = IndexTTSGPT2Attention(config: config)
        _ln2.wrappedValue = LayerNorm(dimensions: config.modelDim, eps: 1e-5)
        _mlp.wrappedValue = IndexTTSGPT2MLP(config: config)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        let hidden = x + attention(ln1(x), mask: mask, cache: cache)
        return hidden + mlp(ln2(hidden))
    }
}

final class IndexTTSGPT2Model: Module {
    @ModuleInfo(key: "h") var h: [IndexTTSGPT2Block]
    @ModuleInfo(key: "ln_f") var lnF: LayerNorm

    init(config: IndexTTSGPTConfig) {
        _h.wrappedValue = (0..<config.layers).map { _ in IndexTTSGPT2Block(config: config) }
        _lnF.wrappedValue = LayerNorm(dimensions: config.modelDim, eps: 1e-5)
    }

    func makeCache() -> [any KVCache] {
        h.map { _ in KVCacheSimple() }
    }

    func callAsFunction(inputsEmbeds: MLXArray, cache: [any KVCache]? = nil) -> MLXArray {
        var hidden = inputsEmbeds
        let mask = createAttentionMask(h: hidden, cache: cache?.first)
        for (index, block) in h.enumerated() {
            hidden = block(hidden, mask: mask, cache: cache?[index])
        }
        return lnF(hidden)
    }
}

public final class IndexTTSCore: Module {
    public let config: IndexTTSConfig

    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding
    @ModuleInfo(key: "mel_embedding") var melEmbedding: Embedding
    @ModuleInfo(key: "mel_pos_embedding") var melPositionEmbedding: IndexTTSLearnedPositionEncoding
    @ModuleInfo(key: "text_pos_embedding") var textPositionEmbedding: IndexTTSLearnedPositionEncoding
    @ModuleInfo(key: "text_head") var textHead: Linear
    @ModuleInfo(key: "mel_head") var melHead: Linear
    @ModuleInfo(key: "conditioning_encoder") var conditioningEncoder: IndexTTSConformerEncoder
    @ModuleInfo(key: "perceiver_encoder") var perceiverEncoder: IndexTTSPerceiverResampler
    @ModuleInfo(key: "gpt") var gpt: IndexTTSGPT2Model
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    public init(config: IndexTTSConfig) {
        self.config = config
        let gptConfig = config.gpt
        _textEmbedding.wrappedValue = Embedding(
            embeddingCount: gptConfig.numberTextTokens + 1,
            dimensions: gptConfig.modelDim
        )
        _melEmbedding.wrappedValue = Embedding(
            embeddingCount: gptConfig.numberMelCodes,
            dimensions: gptConfig.modelDim
        )
        _melPositionEmbedding.wrappedValue = IndexTTSLearnedPositionEncoding(
            sequenceLength: gptConfig.maxMelTokens + 2 + gptConfig.maxConditioningInputs,
            modelDim: gptConfig.modelDim
        )
        _textPositionEmbedding.wrappedValue = IndexTTSLearnedPositionEncoding(
            sequenceLength: gptConfig.maxTextTokens + 2,
            modelDim: gptConfig.modelDim
        )
        _textHead.wrappedValue = Linear(gptConfig.modelDim, gptConfig.numberTextTokens + 1, bias: true)
        _melHead.wrappedValue = Linear(gptConfig.modelDim, gptConfig.numberMelCodes, bias: true)
        _conditioningEncoder.wrappedValue = IndexTTSConformerEncoder(config: gptConfig.conditionModule)
        _perceiverEncoder.wrappedValue = IndexTTSPerceiverResampler(
            nDim: gptConfig.modelDim,
            nDimContext: gptConfig.conditionModule.outputSize,
            nLatents: gptConfig.conditionNumLatent,
            nHeads: gptConfig.conditionModule.attentionHeads,
            nFFMult: gptConfig.conditionModule.perceiverMult
        )
        _gpt.wrappedValue = IndexTTSGPT2Model(config: gptConfig)
        _finalNorm.wrappedValue = LayerNorm(dimensions: gptConfig.modelDim, eps: 1e-5)
    }

    public func getConditioning(referenceFeatures: MLXArray) throws -> MLXArray {
        perceiverEncoder(try conditioningEncoder(referenceFeatures))
    }

    public func prepareInputEmbedding(
        textTokenIDs rawTextTokenIDs: [Int],
        conditioningLatents: MLXArray
    ) throws -> IndexTTSPreparedEmbedding {
        let gptConfig = config.gpt
        guard conditioningLatents.ndim == 3, conditioningLatents.dim(2) == gptConfig.modelDim else {
            throw IndexTTSError.invalidInput(
                "conditioningLatents must have shape [batch, tokens, \(gptConfig.modelDim)]"
            )
        }
        guard conditioningLatents.dim(0) == 1 else {
            throw IndexTTSError.invalidInput("IndexTTS Swift currently prepares one prompt at a time.")
        }

        let textTokenIDs = rawTextTokenIDs.map { min(max($0, 0), gptConfig.numberTextTokens) }
        let tokens = [gptConfig.startTextToken] + textTokenIDs + [gptConfig.stopTextToken, gptConfig.startMelToken]
        guard tokens.count <= gptConfig.maxTextTokens + 2 else {
            throw IndexTTSError.invalidInput("Text token count \(tokens.count) exceeds max_text_tokens + 2.")
        }
        let tokenArray = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
        let textEmbeds = textEmbedding(tokenArray) + textPositionEmbedding(sequenceLength: tokens.count)
        let embeddings = concatenated([conditioningLatents, textEmbeds], axis: 1)
        return IndexTTSPreparedEmbedding(
            embeddings: embeddings,
            textTokenCount: tokens.count,
            conditioningTokenCount: conditioningLatents.dim(1)
        )
    }

    public func prepareInputEmbedding(
        textTokenIDs: [Int],
        referenceFeatures: MLXArray
    ) throws -> IndexTTSPreparedEmbedding {
        try prepareInputEmbedding(
            textTokenIDs: textTokenIDs,
            conditioningLatents: getConditioning(referenceFeatures: referenceFeatures)
        )
    }

    public func logits(inputEmbeddings: MLXArray) -> MLXArray {
        melHead(finalNorm(gpt(inputsEmbeds: inputEmbeddings)))
    }

    public func generateMelTokens(
        textTokenIDs: [Int],
        conditioningLatents: MLXArray,
        maxTokens: Int,
        temperature: Float = 0,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0
    ) throws -> IndexTTSMelGeneration {
        guard maxTokens > 0 else {
            return IndexTTSMelGeneration(
                tokenIDs: [],
                latentStates: MLXArray.zeros([1, 0, config.gpt.modelDim])
            )
        }

        let prepared = try prepareInputEmbedding(textTokenIDs: textTokenIDs, conditioningLatents: conditioningLatents)
        let cache = gpt.makeCache()
        var input = prepared.embeddings
        var tokenIDs: [Int] = []
        var latents: [MLXArray] = []
        let sampler = temperature > 0
            ? TopPSampler(temperature: temperature, topP: topP, topK: topK, minP: minP)
            : nil

        for position in 0..<maxTokens {
            let hidden = finalNorm(gpt(inputsEmbeds: input, cache: cache))
            let last = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...]
            latents.append(last)
            let logits = melHead(last)
            let next = (sampler?.sample(logits: logits.squeezed(axis: 1)) ?? argMax(logits, axis: -1)).asType(.int32)
            eval(next)
            let nextID = next.item(Int.self)
            if nextID == config.gpt.stopMelToken {
                break
            }
            tokenIDs.append(nextID)
            let nextArray = MLXArray([Int32(nextID)]).reshaped(1, 1)
            input = melEmbedding(nextArray) + melPositionEmbedding(
                sequenceLength: 1,
                offset: prepared.embeddings.dim(1) + position
            )
        }

        let latentStates = latents.isEmpty
            ? MLXArray.zeros([1, 0, config.gpt.modelDim])
            : concatenated(latents, axis: 1)
        return IndexTTSMelGeneration(tokenIDs: tokenIDs, latentStates: latentStates)
    }
}
