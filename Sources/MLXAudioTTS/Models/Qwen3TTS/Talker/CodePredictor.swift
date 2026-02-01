//
//  CodePredictor.swift
//  Qwen3TTS
//
//  Code Predictor for multi-codebook token prediction in Qwen3-TTS.
//  Ported from mlx_audio/tts/models/qwen3_tts/talker.py
//

import Foundation
import MLX
import MLXNN

// MARK: - CodePredictorAttention

/// Attention for the code predictor with standard RoPE.
///
/// Similar to TalkerAttention but uses standard (non-multimodal) RoPE.
public class CodePredictorAttention: Module {
    public let hiddenSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let numKVGroups: Int
    public let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    public init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.numKVGroups = numHeads / numKVHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(
            hiddenSize,
            numHeads * headDim,
            bias: config.attentionBias
        )
        self._kProj.wrappedValue = Linear(
            hiddenSize,
            numKVHeads * headDim,
            bias: config.attentionBias
        )
        self._vProj.wrappedValue = Linear(
            hiddenSize,
            numKVHeads * headDim,
            bias: config.attentionBias
        )
        self._oProj.wrappedValue = Linear(
            numHeads * headDim,
            hiddenSize,
            bias: config.attentionBias
        )

        self._qNorm.wrappedValue = RMSNorm(dims: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dims: headDim, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil,
        cache: TalkerKVCache? = nil
    ) -> MLXArray {
        let batch = x.shape[0]
        let seqLen = x.shape[1]

        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        q = q.reshaped([batch, seqLen, numHeads, headDim])
        k = k.reshaped([batch, seqLen, numKVHeads, headDim])
        v = v.reshaped([batch, seqLen, numKVHeads, headDim])

        q = qNorm(q)
        k = kNorm(k)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // Apply standard RoPE (not multimodal)
        let (cos, sin) = positionEmbeddings
        (q, k) = applyRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)

        if let cache = cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        output = output.transposed(0, 2, 1, 3)
        output = output.reshaped([batch, seqLen, -1])

        return oProj(output)
    }
}

// MARK: - CodePredictorMLP

/// MLP for code predictor with SwiGLU activation.
public class CodePredictorMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(config: Qwen3TTSTalkerCodePredictorConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - CodePredictorDecoderLayer

/// Decoder layer for code predictor.
public class CodePredictorDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: CodePredictorAttention
    @ModuleInfo(key: "mlp") var mlp: CodePredictorMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    public init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = CodePredictorAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = CodePredictorMLP(config: config)
        self._inputLayernorm.wrappedValue = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil,
        cache: TalkerKVCache? = nil
    ) -> MLXArray {
        var residual = x
        var hidden = inputLayernorm(x)
        hidden = selfAttn(hidden, positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
        hidden = residual + hidden

        residual = hidden
        hidden = postAttentionLayernorm(hidden)
        hidden = mlp(hidden)
        hidden = residual + hidden

        return hidden
    }
}

// MARK: - CodePredictorModel

/// Inner model for code predictor (matches PyTorch weight structure).
public class CodePredictorModel: Module {
    public let config: Qwen3TTSTalkerCodePredictorConfig

    // Embeddings for each code group (except first)
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]

    // Transformer layers
    @ModuleInfo(key: "layers") var layers: [CodePredictorDecoderLayer]

    @ModuleInfo(key: "norm") var norm: RMSNorm

    // Rotary embeddings (standard, not multimodal)
    public let rotaryEmb: RotaryEmbedding

    public init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config

        // Embeddings for each code group (except first)
        var embeddings: [Embedding] = []
        for _ in 0..<(config.numCodeGroups - 1) {
            embeddings.append(Embedding(embeddingCount: config.vocabSize, dimensions: talkerHiddenSize))
        }
        self._codecEmbedding.wrappedValue = embeddings

        // Transformer layers
        var decoderLayers: [CodePredictorDecoderLayer] = []
        for i in 0..<config.numHiddenLayers {
            decoderLayers.append(CodePredictorDecoderLayer(config: config, layerIdx: i))
        }
        self._layers.wrappedValue = decoderLayers

        self._norm.wrappedValue = RMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)

        // Standard rotary embeddings
        self.rotaryEmb = RotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )
    }

    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [TalkerKVCache]? = nil
    ) -> MLXArray {
        let batch = inputsEmbeds.shape[0]
        let seqLen = inputsEmbeds.shape[1]

        // Get offset from cache for position calculation
        let offset: Int
        if let cache = cache, !cache.isEmpty, let firstCache = cache[0] as? TalkerSimpleKVCache {
            offset = firstCache.sequenceLength
        } else {
            offset = 0
        }

        // Position ids
        let posIds: MLXArray
        if let providedPosIds = positionIds {
            posIds = providedPosIds
        } else {
            let positions = MLXArray(Array(offset..<(offset + seqLen)).map { Int32($0) })
            posIds = broadcast(positions.reshaped([1, seqLen]), to: [batch, seqLen])
        }

        let positionEmbeddings = rotaryEmb(inputsEmbeds, positionIds: posIds)

        // Create causal mask if needed
        let attentionMask: MLXArray?
        if mask == nil && seqLen > 1 {
            attentionMask = createCausalMask(seqLen: seqLen, dtype: inputsEmbeds.dtype)
        } else {
            attentionMask = mask
        }

        var x = inputsEmbeds

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            x = layer(x, positionEmbeddings: positionEmbeddings, mask: attentionMask, cache: layerCache)
        }

        x = norm(x)
        return x
    }

    /// Create KV cache for all layers.
    public func makeCache() -> [TalkerKVCache] {
        return layers.map { _ in TalkerSimpleKVCache() }
    }
}

// MARK: - Qwen3TTSTalkerCodePredictor

/// Code predictor sub-model for multi-codebook token prediction.
///
/// Predicts tokens for code groups 1 to N-1 (group 0 is predicted by main talker).
public class Qwen3TTSTalkerCodePredictor: Module {
    public let config: Qwen3TTSTalkerCodePredictorConfig
    public let numCodeGroups: Int
    public let talkerHiddenSize: Int

    // Optional projection (when talker and code predictor have different hidden sizes)
    @ModuleInfo(key: "small_to_mtp_projection") var smallToMtpProjection: Linear?

    // Inner model
    @ModuleInfo(key: "model") var model: CodePredictorModel

    // LM heads for each code group (except first)
    @ModuleInfo(key: "lm_head") var lmHead: [Linear]

    public init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self.numCodeGroups = config.numCodeGroups
        self.talkerHiddenSize = talkerHiddenSize

        // Projection from talker hidden size to code predictor hidden size
        if config.hiddenSize != talkerHiddenSize {
            self._smallToMtpProjection.wrappedValue = Linear(
                talkerHiddenSize,
                config.hiddenSize,
                bias: true
            )
        } else {
            self._smallToMtpProjection.wrappedValue = nil
        }

        // Inner model
        self._model.wrappedValue = CodePredictorModel(config: config, talkerHiddenSize: talkerHiddenSize)

        // LM heads for each code group (except first)
        var heads: [Linear] = []
        for _ in 0..<(config.numCodeGroups - 1) {
            heads.append(Linear(config.hiddenSize, config.vocabSize, bias: false))
        }
        self._lmHead.wrappedValue = heads
    }

    /// Access codec embeddings from inner model.
    public var codecEmbedding: [Embedding] {
        return model.codecEmbedding
    }

    /// Forward pass for code prediction.
    ///
    /// - Parameters:
    ///   - inputsEmbeds: Input embeddings [batch, seq_len, hidden_size]
    ///   - positionIds: Optional position IDs
    ///   - mask: Optional attention mask
    ///   - cache: Optional KV cache
    ///   - generationStep: Which code group to predict (0 to numCodeGroups-2)
    /// - Returns: (logits, cache, next_generation_step)
    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [TalkerKVCache]? = nil,
        generationStep: Int = 0
    ) -> (logits: MLXArray, cache: [TalkerKVCache]?, nextStep: Int) {
        var embeds = inputsEmbeds

        // Apply projection if needed
        if let projection = smallToMtpProjection {
            embeds = projection(embeds)
        }

        // Forward through inner model
        var x = model(embeds, positionIds: positionIds, mask: mask, cache: cache)

        // Get logits from appropriate head
        let logits = lmHead[generationStep](x)

        return (logits, cache, generationStep + 1)
    }

    /// Create KV cache for all layers.
    public func makeCache() -> [TalkerKVCache] {
        return model.makeCache()
    }
}
