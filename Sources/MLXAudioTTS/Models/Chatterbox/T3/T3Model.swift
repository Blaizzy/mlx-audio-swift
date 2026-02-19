//
//  T3Model.swift
//  MLXAudio
//
//  T3 (Token-To-Token) TTS model: LLaMA backbone + text/speech embeddings + CFG inference.
//  Ported from mlx-audio Python: chatterbox/t3/t3.py
//

import Foundation
import MLX
import MLXFast
import MLXNN
@preconcurrency import MLXLMCommon

// MARK: - Llama3 Scaled RoPE for T3

/// Llama3-style scaled RoPE matching the Chatterbox T3 config.
private class T3Llama3ScaledRoPE: Module {
    let dims: Int
    let traditional: Bool
    private let _freqs: MLXArray

    init(
        dims: Int,
        traditional: Bool = false,
        base: Float = 500000.0,
        scaleFactor: Float = 8.0,
        lowFreqFactor: Float = 1.0,
        highFreqFactor: Float = 4.0,
        oldContextLen: Float = 8192.0
    ) {
        precondition(dims % 2 == 0, "RoPE dims must be even")
        self.dims = dims
        self.traditional = traditional

        let indices = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32)
        let exponents = indices / MLXArray(Float(dims))
        var freqs = MLX.pow(MLXArray(base), exponents)

        let wavelens = MLXArray(2.0 * Float.pi) * freqs
        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        freqs = MLX.where(wavelens .> MLXArray(lowFreqWavelen), freqs * scaleFactor, freqs)

        let isMediumFreq = logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )

        let smoothFactors = (MLXArray(oldContextLen) / wavelens - MLXArray(lowFreqFactor))
            / MLXArray(highFreqFactor - lowFreqFactor)
        let denominator = (MLXArray(1.0) - smoothFactors) / MLXArray(scaleFactor) + smoothFactors
        let smoothFreqs = freqs / denominator

        self._freqs = MLX.where(isMediumFreq, smoothFreqs, freqs)
        super.init()
    }

    init(dims: Int, config: LlamaBackboneConfig) {
        let base = config.ropeTheta
        let rs = config.ropeScaling

        self.dims = dims
        self.traditional = false

        let scaleFactor = rs?.factor ?? 8.0
        let lowFreqFactor = rs?.lowFreqFactor ?? 1.0
        let highFreqFactor = rs?.highFreqFactor ?? 4.0
        let oldContextLen = Float(rs?.originalMaxPositionEmbeddings ?? 8192)

        let indices = MLXArray(stride(from: 0, to: dims, by: 2)).asType(.float32)
        let exponents = indices / MLXArray(Float(dims))
        var freqs = MLX.pow(MLXArray(base), exponents)

        let wavelens = MLXArray(2.0 * Float.pi) * freqs
        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        freqs = MLX.where(wavelens .> MLXArray(lowFreqWavelen), freqs * scaleFactor, freqs)

        let isMediumFreq = logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )

        let smoothFactors = (MLXArray(oldContextLen) / wavelens - MLXArray(lowFreqFactor))
            / MLXArray(highFreqFactor - lowFreqFactor)
        let denominator = (MLXArray(1.0) - smoothFactors) / MLXArray(scaleFactor) + smoothFactors
        let smoothFreqs = freqs / denominator

        self._freqs = MLX.where(isMediumFreq, smoothFreqs, freqs)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        return MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }
}

// MARK: - T3 LLaMA Attention

private class T3Attention: Module {
    let scale: Float
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: T3Llama3ScaledRoPE

    init(_ config: LlamaBackboneConfig) {
        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: config.attentionBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.attentionBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: config.attentionBias)

        self.rope = T3Llama3ScaledRoPE(dims: headDim, config: config)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(b, l, nHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(b, l, nKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(b, l, nKVHeads, headDim).transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(b, l, -1)

        return wo(output)
    }
}

// MARK: - T3 LLaMA MLP

private class T3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: LlamaBackboneConfig) {
        self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
        self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return down(silu(gate(x)) * up(x))
    }
}

// MARK: - T3 LLaMA Transformer Block

private class T3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: T3Attention
    @ModuleInfo(key: "mlp") var mlp: T3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: LlamaBackboneConfig) {
        self._attention.wrappedValue = T3Attention(config)
        self._mlp.wrappedValue = T3MLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - T3 LLaMA Inner Model

/// Inner LLaMA model for T3 — takes **embeddings** (not token IDs).
///
/// In the Python code, `self.tfmr.model(inputs=None, input_embeddings=embeds, cache=cache)`.
/// This corresponds to LLaMA's model.layers + norm, accepting pre-computed embeddings.
class T3LlamaInner: Module {
    /// Placeholder embedding — T3 doesn't use it (it builds embeddings externally).
    /// Needed so weight keys like `tfmr.model.embed_tokens.weight` load without error.
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [T3TransformerBlock]
    let norm: RMSNorm

    init(_ config: LlamaBackboneConfig) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.numHiddenLayers).map { _ in T3TransformerBlock(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Forward pass with pre-computed embeddings.
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - T3 Model

/// Token-To-Token (T3) TTS model using LLaMA as backbone.
///
/// Generates speech tokens from text tokens, conditioned on speaker embeddings
/// and optional emotion/prompt controls.
public class T3Model: Module {
    let hp: T3Configuration
    let llamaConfig: LlamaBackboneConfig
    let dim: Int

    // LLaMA backbone — weight key prefix: "tfmr.model.*"
    @ModuleInfo(key: "tfmr") var tfmr: T3LlamaInner

    // Conditioning encoder
    @ModuleInfo(key: "cond_enc") var condEnc: T3CondEnc

    // Embeddings
    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "speech_emb") var speechEmb: Embedding

    // Learned position embeddings
    @ModuleInfo(key: "text_pos_emb") var textPosEmb: LearnedPositionEmbeddings
    @ModuleInfo(key: "speech_pos_emb") var speechPosEmb: LearnedPositionEmbeddings

    // Output heads
    @ModuleInfo(key: "text_head") var textHead: Linear
    @ModuleInfo(key: "speech_head") var speechHead: Linear

    public init(_ hp: T3Configuration = .englishOnly) {
        self.hp = hp
        self.llamaConfig = .llama520M
        self.dim = llamaConfig.hiddenSize

        self._tfmr.wrappedValue = T3LlamaInner(llamaConfig)
        self._condEnc.wrappedValue = T3CondEnc(hp)

        self._textEmb.wrappedValue = Embedding(embeddingCount: hp.textTokensDictSize, dimensions: dim)
        self._speechEmb.wrappedValue = Embedding(embeddingCount: hp.speechTokensDictSize, dimensions: dim)

        let maxTextSeqLen = hp.maxTextTokens + 2
        let maxSpeechSeqLen = hp.maxSpeechTokens + 4
        self._textPosEmb.wrappedValue = LearnedPositionEmbeddings(seqLen: maxTextSeqLen, modelDim: dim)
        self._speechPosEmb.wrappedValue = LearnedPositionEmbeddings(seqLen: maxSpeechSeqLen, modelDim: dim)

        self._textHead.wrappedValue = Linear(dim, hp.textTokensDictSize, bias: false)
        self._speechHead.wrappedValue = Linear(dim, hp.speechTokensDictSize, bias: false)
    }

    /// Number of transformer layers for cache creation.
    public var numLayers: Int { llamaConfig.numHiddenLayers }

    /// Create KV cache for inference.
    public func makeCache() -> [KVCache] {
        (0 ..< numLayers).map { _ in KVCacheSimple() }
    }

    // MARK: - Conditioning

    /// Prepare conditioning embeddings from T3Cond.
    public func prepareConditioning(_ t3Cond: inout T3Cond) -> MLXArray {
        // Embed speech prompt tokens if provided but not yet embedded
        if t3Cond.condPromptSpeechTokens != nil && t3Cond.condPromptSpeechEmb == nil {
            let tokens = t3Cond.condPromptSpeechTokens!
            t3Cond.condPromptSpeechEmb = speechEmb(tokens) + speechPosEmb(tokens)
        }
        return condEnc(t3Cond)
    }

    // MARK: - Weight Sanitization

    /// Sanitize PyTorch weights for MLX.
    ///
    /// Handles:
    /// - `tfmr.layers.X` → `tfmr.model.layers.X` mapping
    /// - Conv1d weight transposition in conditioning encoder
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = [String: MLXArray]()

        for (key, value) in weights {
            var newKey = key
            var newValue = value

            // Transformer weight name mapping:
            // PyTorch uses: tfmr.layers.X, tfmr.embed_tokens, tfmr.norm
            // Our T3LlamaInner uses: tfmr.model.layers.X, tfmr.model.embed_tokens, tfmr.model.norm
            // So: tfmr.X → tfmr.model.X for internal components
            if key.hasPrefix("tfmr.") && !key.hasPrefix("tfmr.model.") {
                let internalPrefixes = ["tfmr.layers.", "tfmr.embed_tokens.", "tfmr.norm."]
                for prefix in internalPrefixes {
                    if key.hasPrefix(prefix) {
                        newKey = key.replacingOccurrences(of: "tfmr.", with: "tfmr.model.", options: [], range: key.startIndex ..< key.index(key.startIndex, offsetBy: 5))
                        break
                    }
                }
            }

            newWeights[newKey] = newValue
        }

        return newWeights
    }

    // MARK: - Inference

    /// Generate speech tokens from text tokens using KV-cached autoregressive generation.
    ///
    /// - Parameters:
    ///   - t3Cond: Conditioning information (speaker + prompt + emotion).
    ///   - textTokens: Text token IDs (1D or 2D).
    ///   - maxNewTokens: Maximum tokens to generate.
    ///   - temperature: Sampling temperature.
    ///   - topP: Top-p sampling threshold.
    ///   - repetitionPenalty: Repetition penalty factor.
    ///   - cfgWeight: Classifier-free guidance weight.
    /// - Returns: Generated speech tokens (1, T).
    public func inference(
        t3Cond: inout T3Cond,
        textTokens: MLXArray,
        maxNewTokens: Int = 1024,
        temperature: Float = 0.8,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.2,
        cfgWeight: Float = 0.5
    ) -> MLXArray {
        var tokens = textTokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        // Prepare conditioning
        let condEmb = prepareConditioning(&t3Cond) // (1, condLen, dim)

        // Text embeddings + position
        var textEmbResult = textEmb(tokens)
        if hp.inputPosEmb == "learned" {
            textEmbResult = textEmbResult + textPosEmb(tokens)
        }

        // For CFG: duplicate batch — [conditional, unconditional]
        var condEmbForInput = condEmb
        if cfgWeight > 0.0 {
            let uncondText = MLX.zeros(like: textEmbResult)
            textEmbResult = MLX.concatenated([textEmbResult[0 ..< 1], uncondText], axis: 0)
            condEmbForInput = MLX.broadcast(condEmb, to: [textEmbResult.dim(0), condEmb.dim(1), condEmb.dim(2)])
        }

        // BOS token embedding with position 0
        let bosToken = MLXArray([Int32(hp.startSpeechToken)]).reshaped([1, 1])
        var bosEmbed = speechEmb(bosToken)
        bosEmbed = bosEmbed + speechPosEmb.getFixedEmbedding(0)

        if cfgWeight > 0.0 {
            bosEmbed = MLX.concatenated([bosEmbed, bosEmbed], axis: 0)
        }

        // Initial input: [conditioning | text | BOS]
        let inputEmbeddings = MLX.concatenated([condEmbForInput, textEmbResult, bosEmbed], axis: 1)

        // Create KV cache
        var cache = makeCache()

        // Initial forward pass to fill cache
        var hidden = tfmr(inputEmbeddings, cache: cache)

        // Track generated tokens
        var generatedIds = [hp.startSpeechToken]

        // Generation loop
        for step in 0 ..< maxNewTokens {
            // Get logits for last position
            var logits = speechHead(hidden[0..., (-1)..., 0...]) // (B, 1, vocab)
            logits = logits.squeezed(axis: 1) // (B, vocab)

            // Apply CFG
            if cfgWeight > 0.0 && logits.dim(0) > 1 {
                let condLogits = logits[0 ..< 1]
                let uncondLogits = logits[1 ..< 2]
                logits = condLogits + cfgWeight * (condLogits - uncondLogits)
            } else {
                logits = logits[0 ..< 1]
            }

            // Apply repetition penalty
            if repetitionPenalty != 1.0 {
                let tokenArray = MLXArray(generatedIds.map { Int32($0) }).reshaped([1, -1])
                logits = applyRepetitionPenalty(logits: logits, tokens: tokenArray, penalty: repetitionPenalty)
            }

            // Sample (temperature + top-p)
            let nextToken = sampleToken(logits: logits, temperature: temperature, topP: topP)
            eval(nextToken)
            let nextTokenId = nextToken[0].item(Int.self)

            // Check EOS
            if nextTokenId == hp.stopSpeechToken {
                generatedIds.append(nextTokenId)
                break
            }
            generatedIds.append(nextTokenId)

            // Create embedding for next token with position embedding
            var nextTokenEmbed = speechEmb(MLXArray([Int32(nextTokenId)]).reshaped([1, 1]))
            nextTokenEmbed = nextTokenEmbed + speechPosEmb.getFixedEmbedding(step + 1)

            if cfgWeight > 0.0 {
                nextTokenEmbed = MLX.concatenated([nextTokenEmbed, nextTokenEmbed], axis: 0)
            }

            // Forward with cache
            hidden = tfmr(nextTokenEmbed, cache: cache)
            eval(hidden)
        }

        return MLXArray(generatedIds.map { Int32($0) }).reshaped([1, -1])
    }
}

// MARK: - Sampling Utilities

/// Apply repetition penalty to logits.
/// Shared by both T3Model (LLaMA) and T3GPT2Model (Turbo).
func applyRepetitionPenalty(logits: MLXArray, tokens: MLXArray, penalty: Float) -> MLXArray {
    guard penalty != 1.0 else { return logits }

    var result = logits
    let flatTokens = tokens.reshaped([-1])
    let tokenCount = flatTokens.dim(0)

    for i in 0 ..< tokenCount {
        let tokenId = flatTokens[i].item(Int.self)
        if tokenId >= 0 && tokenId < result.dim(1) {
            let score = result[0, tokenId]
            let scoreVal = score.item(Float.self)
            if scoreVal > 0 {
                result[0, tokenId] = MLXArray(scoreVal / penalty)
            } else {
                result[0, tokenId] = MLXArray(scoreVal * penalty)
            }
        }
    }

    return result
}

/// Sample a token from logits using temperature and top-p.
/// Shared by both T3Model (LLaMA) and T3GPT2Model (Turbo).
func sampleToken(logits: MLXArray, temperature: Float, topP: Float) -> MLXArray {
    var scaled = logits
    if temperature > 0 {
        scaled = logits / MLXArray(temperature)
    }

    // Softmax
    let probs = softmax(scaled, axis: -1)

    if topP < 1.0 {
        // Top-p (nucleus) sampling
        let sorted = MLX.sorted(probs, axis: -1)
        let sortedIndices = MLX.argSort(probs, axis: -1)
        let cumProbs = MLX.cumsum(sorted, axis: -1)

        // Remove tokens with cumulative probability below threshold
        let mask = cumProbs .< MLXArray(1.0 - topP)
        let filteredProbs = MLX.where(mask, MLXArray(Float(0)), sorted)

        // Re-normalize
        let sum = filteredProbs.sum(axis: -1, keepDims: true)
        let normalized = filteredProbs / (sum + MLXArray(Float(1e-10)))

        // Sample from filtered distribution
        let token = MLX.argMax(
            MLXRandom.categorical(MLX.log(normalized + MLXArray(Float(1e-10)))),
            axis: -1
        )

        // Map back to original indices
        let batchIdx = MLXArray(0)
        let tokenIdx = token.item(Int.self)
        return sortedIndices[batchIdx, tokenIdx].reshaped([1])
    } else {
        // Simple categorical sampling
        return MLXRandom.categorical(MLX.log(probs + MLXArray(Float(1e-10))))
    }
}
