@preconcurrency import MLX
import MLXNN
import Foundation

func breezeTextAttentionMask(
    length: Int,
    layerType: String,
    slidingWindow: Int,
    attentionMask: MLXArray? = nil
) -> MLXArray? {
    precondition(length >= 0, "Attention length must be non-negative")
    precondition(
        layerType == "full_attention" || layerType == "sliding_attention",
        "Unsupported T5Gemma2 attention layer"
    )

    var allowed: MLXArray?
    if layerType == "sliding_attention" {
        precondition(slidingWindow > 0, "Sliding attention needs a positive window")
        let positions = MLX.arange(length)
        let distance = positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)
        let left = (slidingWindow + 1) / 2
        let right = slidingWindow / 2 + 1
        let local = logicalOr(
            logicalAnd(distance .>= 0, distance .< left),
            logicalAnd(distance .< 0, -distance .< right)
        )
        allowed = local.reshaped([1, 1, length, length])
    }

    if let attentionMask {
        let valid = attentionMask.asType(.bool).reshaped([attentionMask.dim(0), 1, 1, length])
        if let current = allowed {
            allowed = logicalAnd(current, valid)
        } else {
            allowed = valid
        }
    }

    return allowed
}

final class BreezeTTSTextEmbedding: Module {
    let dimensions: Int
    let eoiTokenIndex: Int

    @ParameterInfo var weight: MLXArray
    @ParameterInfo(key: "eoi_embedding") var eoiEmbedding: MLXArray

    init(vocabSize: Int, dimensions: Int, eoiTokenIndex: Int) {
        self.dimensions = dimensions
        self.eoiTokenIndex = eoiTokenIndex
        _weight.wrappedValue = MLXArray.zeros([vocabSize, dimensions])
        _eoiEmbedding.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ inputIDs: MLXArray) -> MLXArray {
        let embeddings = weight[inputIDs] * Foundation.sqrt(Float(dimensions))
        return MLX.where(
            (inputIDs .== eoiTokenIndex).expandedDimensions(axis: -1),
            eoiEmbedding,
            embeddings
        )
    }
}

final class BreezeT5GemmaRMSNorm: Module {
    let eps: Float
    @ParameterInfo var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}

final class BreezeTTSTextAttention: Module {
    let config: BreezeTextEncoderConfig
    let layerType: String
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: BreezeT5GemmaRMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: BreezeT5GemmaRMSNorm

    init(config: BreezeTextEncoderConfig, layerIndex: Int) {
        self.config = config
        self.layerType = config.layerTypes[layerIndex]
        self.scale = Foundation.pow(config.queryPreAttentionScalar, -0.5)
        let ropeBase: Float = layerType == "sliding_attention" ? 10_000 : 1_000_000
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: ropeBase, scale: 1)
        _query.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: false)
        _key.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        _value.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        _output.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: false)
        _queryNorm.wrappedValue = BreezeT5GemmaRMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = BreezeT5GemmaRMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let queries = rope(
            queryNorm(query(x).reshaped([batch, length, config.numAttentionHeads, config.headDim]))
                .transposed(0, 2, 1, 3)
        )
        let keys = rope(
            keyNorm(key(x).reshaped([batch, length, config.numKeyValueHeads, config.headDim]))
                .transposed(0, 2, 1, 3)
        )
        let values = value(x)
            .reshaped([batch, length, config.numKeyValueHeads, config.headDim])
            .transposed(0, 2, 1, 3)
        let mask = breezeTextAttentionMask(
            length: length,
            layerType: layerType,
            slidingWindow: config.slidingWindow,
            attentionMask: attentionMask
        )
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask.map(MLXFast.ScaledDotProductAttentionMaskMode.array) ?? .none
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped([batch, length, -1]))
    }
}

final class BreezeTTSTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(config: BreezeTextEncoderConfig) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(MLXNN.geluApproximate(gate(x)) * up(x))
    }
}

final class BreezeTTSTextLayer: Module {
    let layerType: String
    @ModuleInfo(key: "self_attn") var attention: BreezeTTSTextAttention
    @ModuleInfo(key: "pre_self_attn_layernorm") var preAttentionNorm: BreezeT5GemmaRMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postAttentionNorm: BreezeT5GemmaRMSNorm
    @ModuleInfo var mlp: BreezeTTSTextMLP
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: BreezeT5GemmaRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: BreezeT5GemmaRMSNorm

    init(config: BreezeTextEncoderConfig, layerIndex: Int) {
        layerType = config.layerTypes[layerIndex]
        _attention.wrappedValue = BreezeTTSTextAttention(config: config, layerIndex: layerIndex)
        _preAttentionNorm.wrappedValue = BreezeT5GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = BreezeT5GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlp.wrappedValue = BreezeTTSTextMLP(config: config)
        _preFeedForwardNorm.wrappedValue = BreezeT5GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedForwardNorm.wrappedValue = BreezeT5GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let attended = x + postAttentionNorm(attention(preAttentionNorm(x), attentionMask: attentionMask))
        return attended + postFeedForwardNorm(mlp(preFeedForwardNorm(attended)))
    }
}

final class BreezeTTSTextEncoder: Module {
    let config: BreezeTextEncoderConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: BreezeTTSTextEmbedding
    @ModuleInfo var layers: [BreezeTTSTextLayer]
    @ModuleInfo var norm: BreezeT5GemmaRMSNorm

    init(config: BreezeTextEncoderConfig) {
        self.config = config
        _embedTokens.wrappedValue = BreezeTTSTextEmbedding(
            vocabSize: config.vocabSize,
            dimensions: config.hiddenSize,
            eoiTokenIndex: config.eoiTokenIndex
        )
        _layers.wrappedValue = (0..<config.numHiddenLayers).map {
            BreezeTTSTextLayer(config: config, layerIndex: $0)
        }
        _norm.wrappedValue = BreezeT5GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputIDs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var hidden = embedTokens(inputIDs)
        for layer in layers {
            hidden = layer(hidden, attentionMask: attentionMask)
        }
        return norm(hidden)
    }
}
