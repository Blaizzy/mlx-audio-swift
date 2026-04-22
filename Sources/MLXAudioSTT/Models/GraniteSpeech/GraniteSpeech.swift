//
//  GraniteSpeech.swift
//  MLXAudioSTT
//

import Foundation
import MLX
import MLXAudioCore
import MLXNN
import MLXLMCommon
import HuggingFace
import Tokenizers

// MARK: - Language Codes

private let languageCodes: [String: String] = [
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
]

// MARK: - BatchNorm1d (Inference-only)

class GraniteSpeechBatchNorm1d: Module {
    var weight: MLXArray
    var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    let eps: Float

    init(numFeatures: Int, eps: Float = 1e-5) {
        self.weight = MLXArray.ones([numFeatures])
        self.bias = MLXArray.zeros([numFeatures])
        self._runningMean.wrappedValue = MLXArray.zeros([numFeatures])
        self._runningVar.wrappedValue = MLXArray.ones([numFeatures])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - runningMean) / MLX.sqrt(runningVar + eps) * weight + bias
    }
}

// MARK: - Conformer Feed Forward

class GraniteSpeechConformerFeedForward: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GraniteSpeechEncoderConfig) {
        let ffDim = config.hiddenDim * config.feedforwardMult
        self._preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenDim)
        self._upProj.wrappedValue = Linear(config.hiddenDim, ffDim)
        self._downProj.wrappedValue = Linear(ffDim, config.hiddenDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(upProj(preNorm(x))))
    }
}

// MARK: - Conformer Attention (Block-wise with Relative Position)

class GraniteSpeechConformerAttention: Module {
    let maxPosEmb: Int
    let contextSize: Int
    let numHeads: Int
    let dimHead: Int
    let scale: Float

    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_kv") var toKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "rel_pos_emb") var relPosEmb: Embedding

    init(_ config: GraniteSpeechEncoderConfig) {
        let innerDim = config.dimHead * config.numHeads
        self.maxPosEmb = config.maxPosEmb
        self.contextSize = config.contextSize
        self.numHeads = config.numHeads
        self.dimHead = config.dimHead
        self.scale = pow(Float(config.dimHead), -0.5)

        self._preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenDim)
        self._toQ.wrappedValue = Linear(config.hiddenDim, innerDim, bias: false)
        self._toKV.wrappedValue = Linear(config.hiddenDim, innerDim * 2, bias: false)
        self._toOut.wrappedValue = Linear(innerDim, config.hiddenDim)
        self._relPosEmb.wrappedValue = Embedding(
            embeddingCount: 2 * config.maxPosEmb + 1, dimensions: config.dimHead
        )
    }

    func callAsFunction(_ x: MLXArray, attentionDists: MLXArray) -> MLXArray {
        var x = preNorm(x)
        let B = x.dim(0)
        let N = x.dim(1)

        let numBlocks = (N + contextSize - 1) / contextSize
        let remainder = N % contextSize

        if remainder > 0 {
            let padLen = contextSize - remainder
            x = MLX.padded(x, widths: [.init((0, 0)), .init((0, padLen)), .init((0, 0))])
        }

        var q = toQ(x)
        let kv = toKV(x)
        let split = MLX.split(kv, parts: 2, axis: -1)
        var k = split[0]
        var v = split[1]

        q = q.reshaped(B, numBlocks, contextSize, numHeads, -1)
        k = k.reshaped(B, numBlocks, contextSize, numHeads, -1)
        v = v.reshaped(B, numBlocks, contextSize, numHeads, -1)

        q = q.transposed(0, 1, 3, 2, 4)
        k = k.transposed(0, 1, 3, 2, 4)
        v = v.transposed(0, 1, 3, 2, 4)

        let relEmb = relPosEmb(attentionDists)

        let qExpanded = q.expandedDimensions(axis: 4)
        let relEmbExpanded = relEmb.expandedDimensions(axes: [0, 1, 2])
        var posAttn = (qExpanded * relEmbExpanded).sum(axis: -1) * scale

        if remainder > 0 {
            let C = contextSize
            let rowIndices = MLXArray(Int32(0)..<Int32(C)).reshaped(C, 1)
            let colIndices = MLXArray(Int32(0)..<Int32(C)).reshaped(1, C)
            let rowValid = rowIndices .< Int32(remainder)
            let colValid = colIndices .< Int32(remainder)
            let mask = .!(rowValid .&& colValid)
            let maskValue = MLXArray(Float(-1e9))

            let lastBlock = posAttn[0..., (-1)..., 0..., 0..., 0...]
            let maskedLast = MLX.where(
                mask.expandedDimensions(axes: [0, 1, 2]),
                maskValue,
                lastBlock
            )
            posAttn = MLX.concatenated(
                [posAttn[0..., ..<(-1), 0..., 0..., 0...], maskedLast],
                axis: 1
            )
        }

        var attnWeights = (q.matmul(k.transposed(0, 1, 2, 4, 3))) * scale + posAttn
        attnWeights = softmax(attnWeights, axis: -1)

        var out = attnWeights.matmul(v)
        out = out.transposed(0, 1, 3, 2, 4)
        out = out.reshaped(B, -1, numHeads * dimHead)
        out = out[0..., ..<N, 0...]
        return toOut(out)
    }
}

// MARK: - Depthwise Conv1d

class GraniteSpeechDepthWiseConv1d: Module {
    let paddingLeft: Int
    let paddingRight: Int
    let conv: Conv1d

    init(chanIn: Int, chanOut: Int, kernelSize: Int) {
        let pad = kernelSize / 2
        let padOffset = (kernelSize + 1) % 2
        self.paddingLeft = pad
        self.paddingRight = pad - padOffset
        self.conv = Conv1d(inputChannels: chanIn, outputChannels: chanOut, kernelSize: kernelSize, groups: chanIn, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(
            x,
            widths: [.init((0, 0)), .init((paddingLeft, paddingRight)), .init((0, 0))]
        )
        return conv(padded)
    }
}

// MARK: - Conformer Conv Module

class GraniteSpeechConformerConvModule: Module {
    let norm: LayerNorm
    @ModuleInfo(key: "up_conv") var upConv: Conv1d
    @ModuleInfo(key: "depth_conv") var depthConv: GraniteSpeechDepthWiseConv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: GraniteSpeechBatchNorm1d
    @ModuleInfo(key: "down_conv") var downConv: Conv1d

    init(_ config: GraniteSpeechEncoderConfig) {
        let innerDim = config.hiddenDim * config.convExpansionFactor
        self.norm = LayerNorm(dimensions: config.hiddenDim)
        self._upConv.wrappedValue = Conv1d(inputChannels: config.hiddenDim, outputChannels: innerDim * 2, kernelSize: 1)
        self._depthConv.wrappedValue = GraniteSpeechDepthWiseConv1d(
            chanIn: innerDim, chanOut: innerDim, kernelSize: config.convKernelSize
        )
        self._batchNorm.wrappedValue = GraniteSpeechBatchNorm1d(numFeatures: innerDim)
        self._downConv.wrappedValue = Conv1d(inputChannels: innerDim, outputChannels: config.hiddenDim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = norm(x)
        x = upConv(x)
        let parts = MLX.split(x, parts: 2, axis: -1)
        x = parts[0] * sigmoid(parts[1])
        x = depthConv(x)
        x = silu(batchNorm(x))
        x = downConv(x)
        return x
    }
}

// MARK: - Conformer Block

class GraniteSpeechConformerBlock: Module {
    let ff1: GraniteSpeechConformerFeedForward
    let attn: GraniteSpeechConformerAttention
    let conv: GraniteSpeechConformerConvModule
    let ff2: GraniteSpeechConformerFeedForward
    @ModuleInfo(key: "post_norm") var postNorm: LayerNorm

    init(_ config: GraniteSpeechEncoderConfig) {
        self.ff1 = GraniteSpeechConformerFeedForward(config)
        self.attn = GraniteSpeechConformerAttention(config)
        self.conv = GraniteSpeechConformerConvModule(config)
        self.ff2 = GraniteSpeechConformerFeedForward(config)
        self._postNorm.wrappedValue = LayerNorm(dimensions: config.hiddenDim)
    }

    func callAsFunction(_ x: MLXArray, attentionDists: MLXArray) -> MLXArray {
        var x = 0.5 * ff1(x) + x
        x = attn(x, attentionDists: attentionDists) + x
        x = conv(x) + x
        x = 0.5 * ff2(x) + x
        x = postNorm(x)
        return x
    }
}

// MARK: - CTC Encoder

class GraniteSpeechCTCEncoder: Module {
    let config: GraniteSpeechEncoderConfig
    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    let layers: [GraniteSpeechConformerBlock]
    let out: Linear
    @ModuleInfo(key: "out_mid") var outMid: Linear
    let numLayers: Int

    /// Relative position distance matrix — computed at init, not a trained parameter.
    private var _attentionDists: MLXArray

    init(_ config: GraniteSpeechEncoderConfig) {
        self.config = config
        self._inputLinear.wrappedValue = Linear(config.inputDim, config.hiddenDim)
        self.layers = (0..<config.numLayers).map { _ in GraniteSpeechConformerBlock(config) }
        self.out = Linear(config.hiddenDim, config.outputDim)
        self._outMid.wrappedValue = Linear(config.outputDim, config.hiddenDim)
        self.numLayers = config.numLayers

        let seq = MLXArray(Int32(0)..<Int32(config.contextSize))
        let relposDist = seq.expandedDimensions(axis: 1) - seq.expandedDimensions(axis: 0)
        self._attentionDists = MLX.clip(relposDist, min: -config.contextSize, max: config.contextSize) + config.maxPosEmb
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = inputLinear(x)
        for (idx, layer) in layers.enumerated() {
            x = layer(x, attentionDists: _attentionDists)
            if idx + 1 == numLayers / 2 {
                let xMid = out(x)
                x = x + outMid(softmax(xMid, axis: -1))
            }
        }
        return x
    }
}

// MARK: - QFormer Components

class GraniteSpeechQFormerMultiHeadAttention: Module {
    let numHeads: Int
    let headDim: Int

    let query: Linear
    let key: Linear
    let value: Linear

    init(hiddenSize: Int, numHeads: Int, kvHiddenSize: Int? = nil) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        let kvDim = kvHiddenSize ?? hiddenSize

        self.query = Linear(hiddenSize, hiddenSize)
        self.key = Linear(kvDim, hiddenSize)
        self.value = Linear(kvDim, hiddenSize)
    }

    func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray? = nil) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)

        let q = query(hiddenStates)
        let kvInput = encoderHiddenStates ?? hiddenStates
        let k = key(kvInput)
        let v = value(kvInput)

        let qr = q.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let kr = k.reshaped(B, -1, numHeads, headDim).transposed(0, 2, 1, 3)
        let vr = v.reshaped(B, -1, numHeads, headDim).transposed(0, 2, 1, 3)

        let scale = pow(Float(headDim), -0.5)
        var attn = (qr * scale).matmul(kr.transposed(0, 1, 3, 2))
        attn = softmax(attn, axis: -1)
        let out = attn.matmul(vr).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return out
    }
}

class GraniteSpeechQFormerSelfOutput: Module {
    let dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(hiddenSize: Int, eps: Float = 1e-12) {
        self.dense = Linear(hiddenSize, hiddenSize)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: eps)
    }

    func callAsFunction(_ hiddenStates: MLXArray, inputTensor: MLXArray) -> MLXArray {
        layerNorm(dense(hiddenStates) + inputTensor)
    }
}

class GraniteSpeechQFormerAttention: Module {
    let attention: GraniteSpeechQFormerMultiHeadAttention
    let output: GraniteSpeechQFormerSelfOutput

    init(hiddenSize: Int, numHeads: Int, kvHiddenSize: Int? = nil, eps: Float = 1e-12) {
        self.attention = GraniteSpeechQFormerMultiHeadAttention(
            hiddenSize: hiddenSize, numHeads: numHeads, kvHiddenSize: kvHiddenSize
        )
        self.output = GraniteSpeechQFormerSelfOutput(hiddenSize: hiddenSize, eps: eps)
    }

    func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray? = nil) -> MLXArray {
        let attnOut = attention(hiddenStates, encoderHiddenStates: encoderHiddenStates)
        return output(attnOut, inputTensor: hiddenStates)
    }
}

class GraniteSpeechQFormerIntermediate: Module {
    let dense: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self.dense = Linear(hiddenSize, intermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gelu(dense(x))
    }
}

class GraniteSpeechQFormerOutput: Module {
    let dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(intermediateSize: Int, hiddenSize: Int, eps: Float = 1e-12) {
        self.dense = Linear(intermediateSize, hiddenSize)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: eps)
    }

    func callAsFunction(_ hiddenStates: MLXArray, inputTensor: MLXArray) -> MLXArray {
        layerNorm(dense(hiddenStates) + inputTensor)
    }
}

// MARK: - QFormer Layer

class GraniteSpeechQFormerLayer: Module {
    let attention: GraniteSpeechQFormerAttention
    let crossattention: GraniteSpeechQFormerAttention
    @ModuleInfo(key: "intermediate_query") var intermediateQuery: GraniteSpeechQFormerIntermediate
    @ModuleInfo(key: "output_query") var outputQuery: GraniteSpeechQFormerOutput

    init(_ config: GraniteSpeechProjectorConfig) {
        self.attention = GraniteSpeechQFormerAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            eps: config.layerNormEps
        )
        self.crossattention = GraniteSpeechQFormerAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            kvHiddenSize: config.encoderHiddenSize,
            eps: config.layerNormEps
        )
        self._intermediateQuery.wrappedValue = GraniteSpeechQFormerIntermediate(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize
        )
        self._outputQuery.wrappedValue = GraniteSpeechQFormerOutput(
            intermediateSize: config.intermediateSize,
            hiddenSize: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        var h = attention(hiddenStates)
        h = crossattention(h, encoderHiddenStates: encoderHiddenStates)
        let intermediate = intermediateQuery(h)
        return outputQuery(intermediate, inputTensor: h)
    }
}

// MARK: - QFormer Encoder & Model

class GraniteSpeechQFormerEncoder: Module {
    let layer: [GraniteSpeechQFormerLayer]

    init(_ config: GraniteSpeechProjectorConfig) {
        self.layer = (0..<config.numHiddenLayers).map { _ in GraniteSpeechQFormerLayer(config) }
    }

    func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates
        for l in layer {
            h = l(h, encoderHiddenStates: encoderHiddenStates)
        }
        return h
    }
}

class GraniteSpeechQFormerModel: Module {
    let layernorm: LayerNorm
    let encoder: GraniteSpeechQFormerEncoder

    init(_ config: GraniteSpeechProjectorConfig) {
        self.layernorm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.encoder = GraniteSpeechQFormerEncoder(config)
    }

    func callAsFunction(_ queryEmbeds: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
        encoder(layernorm(queryEmbeds), encoderHiddenStates: encoderHiddenStates)
    }
}

// MARK: - Encoder Projector

class GraniteSpeechEncoderProjector: Module {
    let hiddenSize: Int
    let downsampleRate: Int
    let windowSize: Int
    let numQueries: Int

    var query: MLXArray
    let qformer: GraniteSpeechQFormerModel
    let linear: Linear

    init(_ config: GraniteSpeechModelConfig) {
        self.hiddenSize = config.projectorConfig.hiddenSize
        self.downsampleRate = config.downsampleRate
        self.windowSize = config.windowSize
        self.numQueries = config.windowSize / config.downsampleRate

        self.query = MLXArray.zeros([1, config.windowSize / config.downsampleRate, config.projectorConfig.hiddenSize])
        self.qformer = GraniteSpeechQFormerModel(config.projectorConfig)
        self.linear = Linear(config.projectorConfig.hiddenSize, config.textConfig.hiddenSize)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let B = hiddenStates.dim(0)
        let L = hiddenStates.dim(1)
        let D = hiddenStates.dim(2)

        let nblocks = (L + windowSize - 1) / windowSize
        let pad = nblocks * windowSize - L
        var h = hiddenStates
        if pad > 0 {
            h = MLX.padded(h, widths: [.init((0, 0)), .init((0, pad)), .init((0, 0))])
        }

        h = h.reshaped(B * nblocks, windowSize, D)
        let q = MLX.broadcast(query, to: [B * nblocks, numQueries, hiddenSize])
        let queryOutput = qformer(q, encoderHiddenStates: h)
        let reshaped = queryOutput.reshaped(B, nblocks * numQueries, -1)
        return linear(reshaped)
    }
}

// MARK: - Granite LM Components

class GraniteSpeechLMAttention: Module {
    let scale: Float
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ config: GraniteSpeechTextConfig) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = dim / numHeads
        self.scale = config.attentionMultiplier

        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: config.attentionBias)
        self._wk.wrappedValue = Linear(dim, numKVHeads * headDim, bias: config.attentionBias)
        self._wv.wrappedValue = Linear(dim, numKVHeads * headDim, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: config.attentionBias)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)
        var values = wv(x).reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return wo(output)
    }
}

class GraniteSpeechLMMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: GraniteSpeechTextConfig) {
        self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class GraniteSpeechLMBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: GraniteSpeechLMAttention
    let mlp: GraniteSpeechLMMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    let residualMultiplier: Float

    init(_ config: GraniteSpeechTextConfig) {
        self._attention.wrappedValue = GraniteSpeechLMAttention(config)
        self.mlp = GraniteSpeechLMMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.residualMultiplier = config.residualMultiplier
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r * residualMultiplier
        r = mlp(postAttentionLayerNorm(h))
        return h + r * residualMultiplier
    }
}

class GraniteSpeechLMModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [GraniteSpeechLMBlock]
    let norm: RMSNorm
    let embeddingMultiplier: Float

    init(_ config: GraniteSpeechTextConfig) {
        precondition(config.vocabSize > 0)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize
        )
        self.layers = (0..<config.numHiddenLayers).map { _ in GraniteSpeechLMBlock(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.embeddingMultiplier = config.embeddingMultiplier
    }

    func callAsFunction(
        _ inputs: MLXArray? = nil, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbeddings {
            h = inputEmbeddings
        } else if let inputs {
            h = embedTokens(inputs)
        } else {
            fatalError("Either inputs or inputEmbeddings must be provided")
        }

        h = h * embeddingMultiplier

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

class GraniteSpeechLanguageModel: Module, KVCacheDimensionProvider {
    let config: GraniteSpeechTextConfig

    @ModuleInfo(key: "model") var model: GraniteSpeechLMModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let logitsScaling: Float

    public var kvHeads: [Int] {
        (0..<config.numHiddenLayers).map { _ in config.numKeyValueHeads }
    }

    init(_ config: GraniteSpeechTextConfig) {
        self.config = config
        self._model.wrappedValue = GraniteSpeechLMModelInner(config)
        self.logitsScaling = config.logitsScaling

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(
        inputs: MLXArray? = nil, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var out = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)

        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        return out / logitsScaling
    }

    var embedTokens: Embedding {
        model.embedTokens
    }
}

// MARK: - Generation Context

private struct GenerationContext {
    let tokenizer: Tokenizers.Tokenizer
    let cache: [KVCache]
    let eosTokenId: Int
    var logits: MLXArray

    func sampleNextToken(temperature: Float) -> Int {
        var lastLogits = logits[0..., -1, 0...]
        if temperature > 0 {
            lastLogits = lastLogits / temperature
        }
        return lastLogits.argMax(axis: -1).item(Int.self)
    }

    func isEOS(_ token: Int) -> Bool {
        token == eosTokenId
    }

    func decode(_ token: Int) -> String {
        tokenizer.decode(tokens: [token])
    }

    func decode(_ tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }
}

// MARK: - Granite Speech Model

public class GraniteSpeechModel: Module {
    public let config: GraniteSpeechModelConfig

    let encoder: GraniteSpeechCTCEncoder
    let projector: GraniteSpeechEncoderProjector
    @ModuleInfo(key: "language_model") var languageModel: GraniteSpeechLanguageModel

    let audioTokenId: Int
    public var tokenizer: Tokenizers.Tokenizer?

    public init(_ config: GraniteSpeechModelConfig) {
        self.config = config
        self.encoder = GraniteSpeechCTCEncoder(config.encoderConfig)
        self.projector = GraniteSpeechEncoderProjector(config)
        self._languageModel.wrappedValue = GraniteSpeechLanguageModel(config.textConfig)
        self.audioTokenId = config.audioTokenIndex
    }

    // MARK: - Forward Pass

    public func callAsFunction(
        inputIds: MLXArray,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        languageModel(
            inputs: inputEmbeddings != nil ? nil : inputIds,
            cache: cache,
            inputEmbeddings: inputEmbeddings
        )
    }

    // MARK: - Audio Processing

    func getAudioFeatures(_ inputFeatures: MLXArray) -> MLXArray {
        let encoderOutput = encoder(inputFeatures)
        return projector(encoderOutput)
    }

    func extractFeatures(_ audio: MLXArray) -> (MLXArray, Int) {
        let nFft = 512
        let winLength = 400
        let hopLength = 160
        let nMels = 80
        let sampleRate = 16000

        let audio1d = audio.reshaped(-1)

        // Periodic Hanning window of size winLength, padded to nFft
        var winValues = [Float](repeating: 0, count: winLength)
        for n in 0..<winLength {
            winValues[n] = 0.5 * (1 - cos(2.0 * Float.pi * Float(n) / Float(winLength)))
        }
        let win = MLXArray(winValues)
        let padLeft = (nFft - winLength) / 2
        let padRight = nFft - winLength - padLeft
        let winPadded = MLX.concatenated([
            MLXArray.zeros([padLeft]), win, MLXArray.zeros([padRight]),
        ])

        let spec = stft(audio: audio1d, window: winPadded, nFft: nFft, hopLength: hopLength, padMode: .reflect)
        let power = MLX.abs(spec).square()
        let melFb = melFilters(sampleRate: sampleRate, nFft: nFft, nMels: nMels, norm: nil, melScale: .htk)
        let melSpec = power.matmul(melFb)

        var logmel = MLX.log10(MLX.clip(melSpec, min: 1e-10))
        let maxVal = logmel.max()
        logmel = MLX.maximum(logmel, maxVal - 8.0) / 4.0 + 1.0

        var numFrames = logmel.dim(0)
        if numFrames % 2 == 1 {
            logmel = logmel[..<(numFrames - 1)]
            numFrames -= 1
        }
        let encoderInput = logmel.reshaped(-1, 2 * nMels)

        let encoderLength = encoderInput.dim(0)
        let nblocks = (encoderLength + config.windowSize - 1) / config.windowSize
        let numAudioTokens = nblocks * (config.windowSize / config.downsampleRate)

        let inputFeatures = encoderInput.expandedDimensions(axis: 0)
        return (inputFeatures, numAudioTokens)
    }

    // MARK: - Prompt Building

    func buildPrompt(numAudioTokens: Int, userPrompt: String?) -> MLXArray {
        guard let tokenizer else { fatalError("Tokenizer not loaded") }

        let prompt = userPrompt ?? "can you transcribe the speech into a written format?"
        let audioPlaceholder = String(repeating: "<|audio|>", count: numAudioTokens)
        let content = "\(audioPlaceholder)\(prompt)"

        let promptIds: [Int]
        let messages: [Tokenizers.Message] = [["role": "user", "content": content]]
        if let tokenIds = try? tokenizer.applyChatTemplate(messages: messages) {
            promptIds = tokenIds
        } else {
            let promptStr = "USER: \(content)\nASSISTANT:"
            promptIds = tokenizer.encode(text: promptStr)
        }

        return MLXArray(promptIds.map { Int32($0) })
    }

    func buildInputEmbeds(_ inputIds: MLXArray, audioFeatures: MLXArray) -> MLXArray {
        let isAudio = inputIds .== Int32(audioTokenId)
        let llmIds = MLX.where(isAudio, MLXArray(Int32(0)), inputIds)

        var inputsEmbeds = languageModel.embedTokens(llmIds.expandedDimensions(axis: 0))

        let isAudioFlat = isAudio.asType(.int32)
        let numTokens = inputIds.dim(0)
        let numAudio = min(audioFeatures.dim(1), isAudioFlat.sum().item(Int.self))

        if numAudio > 0 {
            var audioIdx = 0
            var embedSlices: [MLXArray] = []
            var lastEnd = 0

            for i in 0..<numTokens {
                if isAudioFlat[i].item(Int.self) == 1 && audioIdx < numAudio {
                    if lastEnd < i {
                        embedSlices.append(inputsEmbeds[0..., lastEnd..<i, 0...])
                    }
                    embedSlices.append(audioFeatures[0..., audioIdx..<(audioIdx + 1), 0...])
                    audioIdx += 1
                    lastEnd = i + 1
                }
            }

            if lastEnd < numTokens {
                embedSlices.append(inputsEmbeds[0..., lastEnd..<numTokens, 0...])
            }

            if !embedSlices.isEmpty {
                inputsEmbeds = MLX.concatenated(embedSlices, axis: 1)
            }
        }

        return inputsEmbeds
    }

    // MARK: - Generation

    public func generate(
        audio: MLXArray,
        maxTokens: Int = 4096,
        temperature: Float = 0.0,
        prompt: String? = nil,
        language: String? = nil,
        verbose: Bool = false
    ) -> STTOutput {
        guard let tokenizer else { fatalError("Tokenizer not loaded") }

        var userPrompt = prompt
        if userPrompt == nil, let language {
            let langName = languageCodes[language.lowercased()] ?? language
            userPrompt = "Translate the speech to \(langName)."
        }

        let startTime = Date()

        let (inputFeatures, numAudioTokens) = extractFeatures(audio)

        if verbose { print("Encoding audio...") }
        let audioFeatures = getAudioFeatures(inputFeatures)
        eval(audioFeatures)

        let promptIds = buildPrompt(numAudioTokens: numAudioTokens, userPrompt: userPrompt)
        let inputsEmbeds = buildInputEmbeds(promptIds, audioFeatures: audioFeatures)
        eval(inputsEmbeds)

        let promptTokenCount = promptIds.dim(0)

        let cache = makeCache()
        let logits = languageModel(cache: cache, inputEmbeddings: inputsEmbeds)
        eval(logits)

        var ctx = GenerationContext(
            tokenizer: tokenizer,
            cache: cache,
            eosTokenId: tokenizer.eosTokenId ?? 0,
            logits: logits
        )

        var generatedTokens: [Int] = []

        for _ in 0..<maxTokens {
            let nextToken = ctx.sampleNextToken(temperature: temperature)
            if ctx.isEOS(nextToken) { break }
            generatedTokens.append(nextToken)

            if verbose { print(ctx.decode(nextToken), terminator: "") }

            let nextTokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            let newLogits = languageModel(inputs: nextTokenArray, cache: cache)
            eval(newLogits)
            ctx = GenerationContext(
                tokenizer: tokenizer, cache: cache,
                eosTokenId: ctx.eosTokenId, logits: newLogits
            )
        }

        if verbose { print() }

        Memory.clearCache()

        let text = ctx.decode(generatedTokens)
        let totalTime = Date().timeIntervalSince(startTime)

        return STTOutput(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            promptTokens: promptTokenCount,
            generationTokens: generatedTokens.count,
            totalTokens: promptTokenCount + generatedTokens.count,
            promptTps: Double(promptTokenCount) / totalTime,
            generationTps: Double(generatedTokens.count) / totalTime,
            totalTime: totalTime,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9
        )
    }

    public func generateStream(
        audio: MLXArray,
        maxTokens: Int = 4096,
        temperature: Float = 0.0,
        prompt: String? = nil,
        language: String? = nil
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        AsyncThrowingStream { continuation in
            do {
                guard let tokenizer = self.tokenizer else {
                    throw STTError.modelNotInitialized("Tokenizer not loaded")
                }

                var userPrompt = prompt
                if userPrompt == nil, let language {
                    let langName = languageCodes[language.lowercased()] ?? language
                    userPrompt = "Translate the speech to \(langName)."
                }

                let startTime = Date()

                let (inputFeatures, numAudioTokens) = self.extractFeatures(audio)
                let audioFeatures = self.getAudioFeatures(inputFeatures)
                eval(audioFeatures)

                let promptIds = self.buildPrompt(numAudioTokens: numAudioTokens, userPrompt: userPrompt)
                let inputsEmbeds = self.buildInputEmbeds(promptIds, audioFeatures: audioFeatures)
                eval(inputsEmbeds)

                let promptTokenCount = promptIds.dim(0)
                let prefillEndTime = Date()
                let prefillTime = prefillEndTime.timeIntervalSince(startTime)

                let cache = self.makeCache()
                let logits = self.languageModel(cache: cache, inputEmbeddings: inputsEmbeds)
                eval(logits)

                let eosTokenId = tokenizer.eosTokenId ?? 0
                var ctx = GenerationContext(
                    tokenizer: tokenizer, cache: cache,
                    eosTokenId: eosTokenId, logits: logits
                )

                let generateStartTime = Date()
                var generatedTokens: [Int] = []

                for _ in 0..<maxTokens {
                    let nextToken = ctx.sampleNextToken(temperature: temperature)
                    if ctx.isEOS(nextToken) { break }
                    generatedTokens.append(nextToken)

                    continuation.yield(.token(ctx.decode(nextToken)))

                    let nextTokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
                    let newLogits = self.languageModel(inputs: nextTokenArray, cache: cache)
                    eval(newLogits)
                    ctx = GenerationContext(
                        tokenizer: tokenizer, cache: cache,
                        eosTokenId: eosTokenId, logits: newLogits
                    )
                }

                let endTime = Date()
                let generateTime = endTime.timeIntervalSince(generateStartTime)
                let totalTime = endTime.timeIntervalSince(startTime)

                Memory.clearCache()

                let tokensPerSecond = generateTime > 0
                    ? Double(generatedTokens.count) / generateTime : 0
                let peakMemory = Double(Memory.peakMemory) / 1e9

                continuation.yield(.info(STTGenerationInfo(
                    promptTokenCount: promptTokenCount,
                    generationTokenCount: generatedTokens.count,
                    prefillTime: prefillTime,
                    generateTime: generateTime,
                    tokensPerSecond: tokensPerSecond,
                    peakMemoryUsage: peakMemory
                )))

                let text = ctx.decode(generatedTokens)
                continuation.yield(.result(STTOutput(
                    text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                    promptTokens: promptTokenCount,
                    generationTokens: generatedTokens.count,
                    totalTokens: promptTokenCount + generatedTokens.count,
                    promptTps: Double(promptTokenCount) / prefillTime,
                    generationTps: tokensPerSecond,
                    totalTime: totalTime,
                    peakMemoryUsage: peakMemory
                )))

                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    // MARK: - Cache

    public func makeCache() -> [KVCache] {
        (0..<config.textConfig.numHiddenLayers).map { _ in KVCacheSimple() }
    }

    // MARK: - Weight Sanitization

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let alreadyConverted = weights.keys.contains { $0.contains("scales") }

        var sanitized: [String: MLXArray] = [:]
        for (k, v) in weights {
            if k.contains("num_batches_tracked") { continue }

            var value = v
            if !alreadyConverted
                && ["up_conv", "down_conv", "depth_conv"].contains(where: { k.contains($0) })
                && k.contains("weight")
                && v.ndim == 3
            {
                value = v.transposed(0, 2, 1)
            }

            sanitized[k] = value
        }
        return sanitized
    }

    // MARK: - Load from Pretrained

    public static func fromPretrained(
        _ modelPath: String,
        cache: HubCache = .default
    ) async throws -> GraniteSpeechModel {
        guard let repoID = Repo.ID(rawValue: modelPath) else {
            throw NSError(
                domain: "GraniteSpeechModel", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid repository ID: \(modelPath)"]
            )
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            cache: cache
        )

        return try await fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) async throws -> GraniteSpeechModel {
        let configPath = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configPath)
        let config = try JSONDecoder().decode(GraniteSpeechModelConfig.self, from: configData)

        let model = GraniteSpeechModel(config)

        model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)

        var weights: [String: MLXArray] = [:]
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let safetensorFiles = files.filter { $0.pathExtension == "safetensors" }

        for file in safetensorFiles {
            let fileWeights = try MLX.loadArrays(url: file)
            weights.merge(fileWeights) { _, new in new }
        }

        let sanitizedWeights = GraniteSpeechModel.sanitize(weights: weights)

        if let perLayerQuantization = config.perLayerQuantization {
            quantize(model: model) { path, module in
                if sanitizedWeights["\(path).scales"] != nil {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                }
                return nil
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(sanitizedWeights),
            verify: .all
        )

        eval(model)

        return model
    }
}

// MARK: - STTGenerationModel Conformance

extension GraniteSpeechModel: STTGenerationModel {
    public var defaultGenerationParameters: STTGenerateParameters {
        STTGenerateParameters(
            maxTokens: 4096,
            temperature: 0.0,
            topP: 1.0,
            topK: 0,
            verbose: false,
            language: "English"
        )
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        generate(
            audio: audio,
            maxTokens: generationParameters.maxTokens,
            temperature: generationParameters.temperature,
            language: generationParameters.language,
            verbose: generationParameters.verbose
        )
    }

    public func generateStream(
        audio: MLXArray,
        generationParameters: STTGenerateParameters
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        generateStream(
            audio: audio,
            maxTokens: generationParameters.maxTokens,
            temperature: generationParameters.temperature,
            language: generationParameters.language
        )
    }
}
