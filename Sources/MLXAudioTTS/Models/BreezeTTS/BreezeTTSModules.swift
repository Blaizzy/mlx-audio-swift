@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN
import Foundation

final class BreezeAudioEmbedding: Module {
    let numCodebooks: Int
    let vocabSize: Int

    @ModuleInfo(key: "embed_audio_tokens") var embedAudioTokens: Embedding
    @ModuleInfo(key: "audio_embeds_projector") var audioEmbedsProjector: Linear?

    init(numCodebooks: Int, vocabSize: Int, audioEmbedSize: Int, hiddenSize: Int) {
        self.numCodebooks = numCodebooks
        self.vocabSize = vocabSize
        _embedAudioTokens.wrappedValue = Embedding(
            embeddingCount: numCodebooks * vocabSize,
            dimensions: audioEmbedSize
        )
        _audioEmbedsProjector.wrappedValue = audioEmbedSize == hiddenSize
            ? nil
            : Linear(audioEmbedSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ codebooks: MLXArray) -> MLXArray {
        precondition(
            codebooks.ndim == 3 && codebooks.dim(-1) == numCodebooks,
            "Breeze audio codebooks must have shape [batch, time, numCodebooks]"
        )
        let offsets = MLX.arange(numCodebooks).reshaped([1, 1, numCodebooks]) * vocabSize
        var hidden = embedAudioTokens(codebooks + offsets)
        if let audioEmbedsProjector {
            hidden = audioEmbedsProjector(hidden)
        }
        return hidden.sum(axis: -2)
    }
}

final class BreezeBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: BreezeAudioEmbedding
    @ModuleInfo var layers: [Qwen3TransformerBlock]
    @ModuleInfo var norm: RMSNorm

    init(config: BreezeTTSConfig) {
        let backbone = config.backboneConfig
        let qwen = Qwen3Configuration(
            hiddenSize: backbone.hiddenSize,
            hiddenLayers: backbone.numHiddenLayers,
            intermediateSize: backbone.intermediateSize,
            attentionHeads: backbone.numAttentionHeads,
            kvHeads: backbone.numKeyValueHeads,
            headDim: backbone.headDim,
            vocabularySize: backbone.vocabSize,
            rmsNormEps: backbone.rmsNormEps,
            ropeTheta: backbone.ropeTheta,
            ropeScaling: backbone.ropeScaling,
            maxPositionEmbeddings: backbone.maxPositionEmbeddings
        )
        _embedTokens.wrappedValue = BreezeAudioEmbedding(
            numCodebooks: config.numCodebooks,
            vocabSize: config.audioVocabSize,
            audioEmbedSize: config.audioEmbedSize,
            hiddenSize: backbone.hiddenSize
        )
        _layers.wrappedValue = (0..<backbone.numHiddenLayers).map { _ in Qwen3TransformerBlock(qwen) }
        _norm.wrappedValue = RMSNorm(dimensions: backbone.hiddenSize, eps: backbone.rmsNormEps)
    }

    func callAsFunction(
        inputIDs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        precondition((inputIDs == nil) != (inputEmbeddings == nil), "Pass input IDs or embeddings")
        var hidden = inputEmbeddings ?? embedTokens(inputIDs!)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }
        return norm(hidden)
    }

    func makeCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }
}

final class BreezeDepthAttention: Module {
    let config: BreezeDepthDecoderConfig
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear

    init(config: BreezeDepthDecoderConfig) {
        self.config = config
        self.scale = Foundation.pow(Float(config.headDim), -0.5)
        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta, scale: 1)
        _query.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: false)
        _key.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        _value.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        _output.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let queries = rope(
            query(x).reshaped([batch, length, config.numAttentionHeads, config.headDim])
                .transposed(0, 2, 1, 3)
        )
        let keys = rope(
            key(x).reshaped([batch, length, config.numKeyValueHeads, config.headDim])
                .transposed(0, 2, 1, 3)
        )
        let values = value(x)
            .reshaped([batch, length, config.numKeyValueHeads, config.headDim])
            .transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .causal
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped([batch, length, -1]))
    }
}

final class BreezeDepthMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(config: BreezeDepthDecoderConfig) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class BreezeDepthLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: BreezeDepthAttention
    @ModuleInfo var mlp: BreezeDepthMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

    init(config: BreezeDepthDecoderConfig) {
        _attention.wrappedValue = BreezeDepthAttention(config: config)
        _mlp.wrappedValue = BreezeDepthMLP(config: config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attended = x + attention(inputNorm(x))
        return attended + mlp(postAttentionNorm(attended))
    }
}

final class BreezeDepthModel: Module {
    let config: BreezeDepthDecoderConfig
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "backbone_hidden_state_projector") var backboneProjector: Linear?
    @ModuleInfo(key: "inputs_embeds_projector") var inputProjector: Linear
    @ModuleInfo var layers: [BreezeDepthLayer]
    @ModuleInfo var norm: RMSNorm

    init(config: BreezeDepthDecoderConfig) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.numCodebooks * config.vocabSize,
            dimensions: config.audioEmbedSize
        )
        _backboneProjector.wrappedValue = config.backboneHiddenSize == config.audioEmbedSize
            ? nil
            : Linear(config.backboneHiddenSize, config.audioEmbedSize, bias: false)
        _inputProjector.wrappedValue = Linear(config.audioEmbedSize, config.hiddenSize, bias: false)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in BreezeDepthLayer(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ tokenIDs: MLXArray, backboneHiddenState: MLXArray) -> MLXArray {
        precondition(tokenIDs.ndim == 2, "Depth token IDs must have shape [batch, time]")
        precondition(backboneHiddenState.ndim == 2, "Backbone hidden state must have shape [batch, hidden]")
        precondition(tokenIDs.dim(0) == backboneHiddenState.dim(0), "Depth batch sizes must match")
        let length = tokenIDs.dim(1)
        let positions = MLX.maximum(MLX.arange(length) - 1, MLXArray(0))
        var embeddings = embedTokens(tokenIDs + positions.reshaped([1, length]) * config.vocabSize)
        var backbone = backboneHiddenState
        if let backboneProjector {
            backbone = backboneProjector(backbone)
        }
        embeddings = concatenated([
            backbone.expandedDimensions(axis: 1),
            embeddings[0..., 1..., 0...],
        ], axis: 1)
        var hidden = inputProjector(embeddings)
        for layer in layers {
            hidden = layer(hidden)
        }
        return norm(hidden)
    }
}

final class BreezeCodebooksHead: Module {
    @ParameterInfo var weight: MLXArray

    init(headCount: Int, hiddenSize: Int, vocabSize: Int) {
        _weight.wrappedValue = MLXArray.zeros([headCount, hiddenSize, vocabSize])
    }
}

final class BreezeDepthDecoder: Module {
    @ModuleInfo var model: BreezeDepthModel
    @ModuleInfo(key: "codebooks_head") var codebooksHead: BreezeCodebooksHead

    init(config: BreezeDepthDecoderConfig) {
        _model.wrappedValue = BreezeDepthModel(config: config)
        _codebooksHead.wrappedValue = BreezeCodebooksHead(
            headCount: config.numCodebooks - 1,
            hiddenSize: config.hiddenSize,
            vocabSize: config.vocabSize
        )
    }

    func nextLogits(tokenIDs: MLXArray, backboneHiddenState: MLXArray) -> MLXArray {
        let headIndex = tokenIDs.dim(1) - 2
        precondition(headIndex >= 0 && headIndex < codebooksHead.weight.dim(0), "Invalid depth head")
        let hidden = model(tokenIDs, backboneHiddenState: backboneHiddenState)[0..., -1, 0...]
        return matmul(hidden, codebooksHead.weight[headIndex])
    }
}
