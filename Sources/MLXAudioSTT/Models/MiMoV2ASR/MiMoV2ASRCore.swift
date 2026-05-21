import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct MiMoTransformerConfiguration: Sendable {
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let kvHeads: Int
    let vocabularySize: Int?
    let rmsNormEps: Float
    let ropeTheta: Float
    let attentionBias: Bool
    let useCausalMask: Bool
}

private final class MiMoAttention: Module {
    let config: MiMoTransformerConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ config: MiMoTransformerConfiguration) {
        self.config = config

        let headDim = config.hiddenSize / config.attentionHeads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeads * headDim,
            bias: config.attentionBias
        )
        _wk.wrappedValue = Linear(
            config.hiddenSize,
            config.kvHeads * headDim,
            bias: config.attentionBias
        )
        _wv.wrappedValue = Linear(
            config.hiddenSize,
            config.kvHeads * headDim,
            bias: config.attentionBias
        )
        _wo.wrappedValue = Linear(config.attentionHeads * headDim, config.hiddenSize, bias: false)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries
            .reshaped(batch, sequence, config.attentionHeads, -1)
            .transposed(0, 2, 1, 3)
        keys = keys
            .reshaped(batch, sequence, config.kvHeads, -1)
            .transposed(0, 2, 1, 3)
        values = values
            .reshaped(batch, sequence, config.kvHeads, -1)
            .transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, sequence, -1)

        return wo(output)
    }
}

private final class MiMoMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(config: MiMoTransformerConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private final class MiMoTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: MiMoAttention
    let mlp: MiMoMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MiMoTransformerConfiguration) {
        _attention.wrappedValue = MiMoAttention(config)
        self.mlp = MiMoMLP(config: config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var residual = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let hidden = x + residual
        residual = mlp(postAttentionLayerNorm(hidden))
        return hidden + residual
    }
}

final class MiMoTransformerBackbone: Module, KVCacheDimensionProvider {
    let config: MiMoTransformerConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    fileprivate let layers: [MiMoTransformerBlock]
    let norm: RMSNorm

    var kvHeads: [Int] {
        (0..<config.hiddenLayers).map { _ in config.kvHeads }
    }

    init(_ config: MiMoTransformerConfiguration, includeEmbeddings: Bool) {
        self.config = config
        if includeEmbeddings, let vocabularySize = config.vocabularySize {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: vocabularySize,
                dimensions: config.hiddenSize
            )
        } else {
            _embedTokens.wrappedValue = nil
        }
        self.layers = (0..<config.hiddenLayers).map { _ in MiMoTransformerBlock(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputIDs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        let hidden: MLXArray
        if let inputEmbeddings {
            hidden = inputEmbeddings
        } else if let inputIDs, let embedTokens {
            hidden = embedTokens(inputIDs)
        } else {
            fatalError("MiMoTransformerBackbone requires input IDs with embeddings or explicit input embeddings.")
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if config.useCausalMask {
            mask = createAttentionMask(h: hidden, cache: cache?.first)
        } else {
            mask = .none
        }

        var output = hidden
        for (index, layer) in layers.enumerated() {
            output = layer(output, mask: mask, cache: cache?[index])
        }
        return norm(output)
    }
}

final class MiMoV2ASRCore: Module, BaseLanguageModel {
    let config: MiMoV2ASRConfig

    @ModuleInfo(key: "model") var model: MiMoTransformerBackbone
    @ModuleInfo(key: "local_transformer") var localTransformer: MiMoTransformerBackbone
    @ModuleInfo(key: "input_local_transformer") var inputLocalTransformer: MiMoTransformerBackbone
    @ModuleInfo(key: "local_transformer_lm_heads") var localTransformerLMHeads: [Linear]
    @ModuleInfo(key: "speech_embeddings") var speechEmbeddings: [Embedding]
    @ModuleInfo(key: "speech_group_downcast") var speechGroupDowncast: Linear
    @ModuleInfo(key: "hidden_states_downcast") var hiddenStatesDowncast: Linear
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(config: MiMoV2ASRConfig) {
        self.config = config

        let globalConfig = MiMoTransformerConfiguration(
            hiddenSize: config.hiddenSize,
            hiddenLayers: config.numHiddenLayers,
            intermediateSize: config.intermediateSize,
            attentionHeads: config.numAttentionHeads,
            kvHeads: config.numKeyValueHeads,
            vocabularySize: config.vocabSize,
            rmsNormEps: config.rmsNormEps,
            ropeTheta: config.ropeTheta,
            attentionBias: config.attentionBias,
            useCausalMask: true
        )
        _model.wrappedValue = MiMoTransformerBackbone(globalConfig, includeEmbeddings: true)

        let localConfig = MiMoTransformerConfiguration(
            hiddenSize: config.localDim,
            hiddenLayers: config.localLayers,
            intermediateSize: config.localFFNDim,
            attentionHeads: config.localAttentionHeads,
            kvHeads: config.localAttentionHeads,
            vocabularySize: nil,
            rmsNormEps: config.rmsNormEps,
            ropeTheta: config.ropeTheta,
            attentionBias: config.attentionBias,
            useCausalMask: true
        )
        _localTransformer.wrappedValue = MiMoTransformerBackbone(localConfig, includeEmbeddings: false)

        let inputLocalConfig = MiMoTransformerConfiguration(
            hiddenSize: config.inputLocalDim,
            hiddenLayers: config.inputLocalLayers,
            intermediateSize: config.inputLocalDim * 4,
            attentionHeads: config.localAttentionHeads,
            kvHeads: config.localAttentionHeads,
            vocabularySize: nil,
            rmsNormEps: config.rmsNormEps,
            ropeTheta: config.ropeTheta,
            attentionBias: config.attentionBias,
            useCausalMask: !config.inputFullAttention
        )
        _inputLocalTransformer.wrappedValue = MiMoTransformerBackbone(inputLocalConfig, includeEmbeddings: false)

        _localTransformerLMHeads.wrappedValue = config.activeSpeechCodebookSizes.map {
            Linear(config.localDim, $0, bias: false)
        }
        _speechEmbeddings.wrappedValue = config.activeSpeechCodebookSizes.enumerated().map { index, size in
            Embedding(
                embeddingCount: size,
                dimensions: config.inputLocalDim
            )
        }
        _speechGroupDowncast.wrappedValue = Linear(
            config.inputLocalDim * config.groupSize,
            config.hiddenSize,
            bias: false
        )
        _hiddenStatesDowncast.wrappedValue = Linear(config.hiddenSize, config.localDim, bias: false)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    func makeGlobalCache() -> [KVCache] {
        (0..<model.kvHeads.count).map { _ in KVCacheSimple() }
    }

    func makeLocalCache() -> [KVCache] {
        (0..<localTransformer.kvHeads.count).map { _ in KVCacheSimple() }
    }

    func textEmbedding(_ inputIDs: MLXArray) -> MLXArray {
        guard let embedTokens = model.embedTokens else {
            fatalError("MiMoV2ASRCore global embed_tokens are not initialized.")
        }
        return embedTokens(inputIDs)
    }

    func textLogits(_ hiddenStates: MLXArray) -> MLXArray {
        lmHead(hiddenStates)
    }

    func localLogits(channel: Int, hiddenStates: MLXArray) -> MLXArray {
        localTransformerLMHeads[channel](hiddenStates)
    }

    static func load(
        modelDirectory: URL,
        config: MiMoV2ASRConfig
    ) throws -> MiMoV2ASRCore {
        let model = MiMoV2ASRCore(config: config)
        try loadWeights(
            modelDirectory: modelDirectory,
            model: model,
            quantization: config.quantization,
            perLayerQuantization: config.perLayerQuantization
        )
        return model
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
