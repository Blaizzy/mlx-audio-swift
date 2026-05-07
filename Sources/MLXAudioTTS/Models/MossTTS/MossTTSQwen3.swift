@preconcurrency import MLX
import MLXAudioCore
import MLXFast
@preconcurrency import MLXLMCommon
import MLXNN

final class MossQwen3Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ config: MossQwen3Config) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = Float(1.0 / Double(config.headDim).squareRoot())

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let length = hiddenStates.dim(1)

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = qNorm(queries.reshaped(batch, length, numHeads, headDim))
            .transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(batch, length, numKVHeads, headDim))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(batch, length, numKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class MossQwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: MossQwen3Config) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class MossQwen3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MossQwen3Attention
    @ModuleInfo(key: "mlp") var mlp: MossQwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MossQwen3Config) {
        _selfAttention.wrappedValue = MossQwen3Attention(config)
        _mlp.wrappedValue = MossQwen3MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var residual = hiddenStates
        var h = selfAttention(inputLayerNorm(hiddenStates), mask: mask, cache: cache)
        h = residual + h
        residual = h
        h = mlp(postAttentionLayerNorm(h))
        return residual + h
    }
}

final class MossQwen3Model: Module {
    let config: MossQwen3Config

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [MossQwen3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: MossQwen3Config) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            MossQwen3DecoderLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func makeCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    func callAsFunction(
        inputIDs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) throws -> MLXArray {
        var hiddenStates: MLXArray
        if let inputEmbeddings {
            hiddenStates = inputEmbeddings
        } else if let inputIDs {
            hiddenStates = embedTokens(inputIDs)
        } else {
            throw AudioGenerationError.invalidInput("inputIDs or inputEmbeddings are required")
        }

        let mask = createAttentionMask(h: hiddenStates, cache: cache?.first)
        let caches = cache ?? [KVCache?](repeating: nil, count: layers.count)
        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: caches[index])
        }
        return norm(hiddenStates)
    }
}

final class MossTTSMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(inputSize: Int, ffnHiddenSize: Int, outputSize: Int) {
        _gateProj.wrappedValue = Linear(inputSize, ffnHiddenSize, bias: false)
        _upProj.wrappedValue = Linear(inputSize, ffnHiddenSize, bias: false)
        _downProj.wrappedValue = Linear(ffnHiddenSize, outputSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class MossTTSLocalAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: MossQwen3Config) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = Float(1.0 / Double(config.headDim).squareRoot())
        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)
        queries = qNorm(queries.reshaped(batch, length, numHeads, headDim)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(batch, length, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        values = values.reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class MossTTSLocalTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MossTTSLocalAttention
    @ModuleInfo(key: "mlp") var mlp: MossTTSMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MossQwen3Config) {
        _selfAttention.wrappedValue = MossTTSLocalAttention(config)
        _mlp.wrappedValue = MossTTSMLP(
            inputSize: config.hiddenSize,
            ffnHiddenSize: config.intermediateSize,
            outputSize: config.hiddenSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let h = x + selfAttention(inputLayerNorm(x), mask: mask)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

final class MossTTSLocalTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [MossTTSLocalTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: MossQwen3Config) {
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            MossTTSLocalTransformerBlock(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputEmbeddings: MLXArray) -> MLXArray {
        var hiddenStates = inputEmbeddings
        let mask = createAttentionMask(h: hiddenStates, cache: nil as KVCache?)
        for layer in layers {
            hiddenStates = layer(hiddenStates, mask: mask)
        }
        return norm(hiddenStates)
    }
}
