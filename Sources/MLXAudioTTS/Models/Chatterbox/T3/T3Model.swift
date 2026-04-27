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
    /// Needed so weight key `tfmr.embed_tokens.weight` loads without error.
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

    // LLaMA backbone — weight keys: "tfmr.layers.*", "tfmr.embed_tokens.*", "tfmr.norm.*"
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

    /// Sanitize weights for MLX.
    ///
    /// Handles:
    /// - `tfmr.model.layers.X` → `tfmr.layers.X` (strip intermediate `model.`)
    ///
    /// The HuggingFace MLX weights use the Python LLaMA structure:
    ///   Model (outer, at `tfmr`) → LlamaModel (inner, at `tfmr.model`)
    ///
    /// Our Swift T3LlamaInner sits directly at `tfmr` with `layers`, `embed_tokens`,
    /// `norm` as direct children — no intermediate `model` level. So we strip `model.`
    /// from `tfmr.model.*` keys to match the flat Swift structure.
    ///
    /// Also handles raw PyTorch keys (`tfmr.layers.*` without `model.`) which pass
    /// through unchanged since they already match our structure.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = [String: MLXArray]()

        for (key, value) in weights {
            var newKey = key

            // Strip the intermediate `model.` from `tfmr.model.*` keys.
            // Python has: Model (at tfmr) → LlamaModel (at tfmr.model)
            // Swift has: T3LlamaInner (at tfmr) → layers/embed_tokens/norm directly
            if key.hasPrefix("tfmr.model.") {
                newKey = "tfmr." + key.dropFirst("tfmr.model.".count)
            }

            // Drop lm_head weights (Python's Model wrapper has one, we don't use it)
            if newKey.hasPrefix("tfmr.lm_head.") {
                continue
            }

            newWeights[newKey] = value
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
        minP: Float = 0.05,
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
        let cache = makeCache()

        // Initial forward pass to fill cache
        var hidden = tfmr(inputEmbeddings, cache: cache)
        eval(hidden)

        // Track generated tokens
        var generatedIds = [hp.startSpeechToken]

        print("[T3-LLaMA] Starting generation (maxNewTokens=\(maxNewTokens), temp=\(temperature), topP=\(topP), minP=\(minP), cfg=\(cfgWeight))")
        let genStart = CFAbsoluteTimeGetCurrent()

        // Generation loop
        for step in 0 ..< maxNewTokens {
            if Task.isCancelled { break }
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
                logits = applyRepetitionPenalty(logits: logits, generatedIds: generatedIds, penalty: repetitionPenalty, vocabSize: hp.speechTokensDictSize)
            }

            // Sample (temperature + min-p + top-p)
            let nextToken = sampleToken(logits: logits, temperature: temperature, topP: topP, minP: minP)
            eval(nextToken)
            let nextTokenId = nextToken[0].item(Int.self)

            // Check EOS
            if nextTokenId == hp.stopSpeechToken {
                let elapsed = CFAbsoluteTimeGetCurrent() - genStart
                print("[T3-LLaMA] EOS at step \(step)/\(maxNewTokens) (\(generatedIds.count) tokens, \(String(format: "%.2f", elapsed))s)")
                generatedIds.append(nextTokenId)
                break
            }
            generatedIds.append(nextTokenId)

            if step % 100 == 0 && step > 0 {
                let elapsed = CFAbsoluteTimeGetCurrent() - genStart
                print("[T3-LLaMA] Step \(step)/\(maxNewTokens) (\(generatedIds.count) tokens, \(String(format: "%.2f", elapsed))s)")
            }

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

        let totalElapsed = CFAbsoluteTimeGetCurrent() - genStart
        let uniqueCount = Set(generatedIds).count
        let first10 = Array(generatedIds.prefix(10))
        let last10 = Array(generatedIds.suffix(10))
        print("[T3-LLaMA] Generation complete: \(generatedIds.count) tokens (\(uniqueCount) unique) in \(String(format: "%.2f", totalElapsed))s")
        print("[T3-LLaMA]   first10=\(first10), last10=\(last10)")
        print("[T3-LLaMA]   stopToken=\(hp.stopSpeechToken), startToken=\(hp.startSpeechToken)")
        return MLXArray(generatedIds.map { Int32($0) }).reshaped([1, -1])
    }
}

// MARK: - Sampling Utilities

/// Apply repetition penalty to logits using vectorized MLX operations.
///
/// Matches Python Chatterbox Turbo's `_apply_repetition_penalty`: extracts unique tokens
/// from the generated history, then applies penalty once per unique token ID.
/// Positive logits are divided by penalty, negative logits are multiplied.
///
/// Uses unique token IDs (from a Swift `[Int]` array) to avoid undefined behavior
/// with duplicate indices in `putAlong`. The caller provides the generated IDs as
/// a Swift array so we avoid GPU→CPU sync for each token.
///
/// Shared by both T3Model (LLaMA) and T3GPT2Model (Turbo).
func applyRepetitionPenalty(logits: MLXArray, generatedIds: [Int], penalty: Float, vocabSize: Int) -> MLXArray {
    guard penalty != 1.0 else { return logits }
    guard !generatedIds.isEmpty else { return logits }

    // Extract unique token IDs on CPU (matches Python's np.unique approach).
    // This avoids undefined behavior from duplicate indices in putAlong.
    let unique = Array(Set(generatedIds)).filter { $0 >= 0 && $0 < vocabSize }
    guard !unique.isEmpty else { return logits }

    let tokenIds = MLXArray(unique.map { Int32($0) }).reshaped([1, -1])

    // Vectorized gather → penalize → scatter (all on GPU)
    let selected = takeAlong(logits, tokenIds, axis: -1)
    let penalized = which(
        selected .< 0,
        selected * MLXArray(penalty),
        selected / MLXArray(penalty)
    )
    return putAlong(logits, tokenIds, values: penalized, axis: -1)
}

/// Sample a token from logits using temperature, top-k, min-p, and top-p.
///
/// Uses fully vectorized MLX operations (no element-by-element loops).
/// Matches the Python implementation: operates on logits throughout,
/// then uses categorical (which applies softmax internally) to sample.
/// Shared by both T3Model (LLaMA) and T3GPT2Model (Turbo).
func sampleToken(logits: MLXArray, temperature: Float, topK: Int = 0, topP: Float = 1.0, minP: Float = 0.0) -> MLXArray {
    var filtered = logits

    // 1. Temperature scaling (on logits)
    if temperature > 0 && temperature != 1.0 {
        filtered = filtered / MLXArray(temperature)
    }

    // 2. Top-k filtering using argPartition (vectorized, no loops)
    let vocabSize = filtered.dim(filtered.ndim - 1)
    if topK > 0 {
        let k = min(topK, vocabSize)
        if k < vocabSize {
            // argPartition: indices beyond position k are the ones NOT in top-k
            let kth = min(k - 1, max(vocabSize - 1, 0))
            let maskIdx = argPartition(-filtered, kth: kth, axis: -1)[0..., k...]
            let negInf = MLXArray.full(maskIdx.shape, values: MLXArray(-Float.infinity), dtype: filtered.dtype)
            filtered = putAlong(filtered, maskIdx, values: negInf, axis: -1)
        }
    }

    // 3. Min-p filtering — matches Python mlx_lm apply_min_p.
    // Keeps all tokens whose probability is at least minP × max_token_probability.
    // More adaptive than top-k: aggressive when the model is confident, permissive when uncertain.
    if minP > 0.0 {
        let probs = softmax(filtered, axis: -1)
        let topProb = MLX.max(probs, axis: -1, keepDims: true)
        let threshold = topProb * MLXArray(minP)
        let mask = probs .< threshold
        filtered = MLX.where(mask, MLXArray(-Float.infinity), filtered)
    }

    // 4. Top-p (nucleus) filtering using vectorized takeAlong/putAlong
    if topP < 1.0 {
        // Sort in descending order
        let sortedIndices = argSort(-filtered, axis: -1)
        let sortedLogits = takeAlong(filtered, sortedIndices, axis: -1)

        // Compute cumulative probabilities from sorted logits
        let sortedProbs = softmax(sortedLogits, axis: -1)
        let cumProbs = MLX.cumsum(sortedProbs, axis: -1)

        // Mark tokens to remove: cumulative prob > topP (shift right to keep first token)
        let toRemoveRaw = cumProbs .> MLXArray(topP)
        let keepFirst = MLXArray.zeros([1, 1]).asType(.bool)
        let toRemove = MLX.concatenated([keepFirst, toRemoveRaw[0..., ..<(vocabSize - 1)]], axis: -1)

        // Set removed tokens to -inf in sorted space
        let maskedSorted = MLX.where(toRemove, MLXArray(-Float.infinity), sortedLogits)

        // Scatter back to original order using inverse indices
        let arangeIndices = MLXArray(0 ..< vocabSize).reshaped([1, -1]).asType(.int32)
        let inverseIndices = putAlong(
            MLXArray.zeros(sortedIndices.shape, type: Int32.self),
            sortedIndices.asType(.int32),
            values: arangeIndices,
            axis: -1
        )
        filtered = takeAlong(maskedSorted, inverseIndices, axis: -1)
    }

    // 5. Sample using categorical (applies softmax internally on logits)
    return MLXRandom.categorical(filtered)
}
