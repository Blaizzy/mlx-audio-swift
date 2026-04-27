//
//  T3GPT2Model.swift
//  MLXAudio
//
//  GPT-2 backbone for Chatterbox Turbo T3 model.
//  The Turbo variant uses GPT-2 Medium (24 layers, LayerNorm, GELU, fused c_attn).
//  Ported from original Chatterbox: resemble-ai/chatterbox tts_turbo.py + t3.py
//

import Foundation
import MLX
import MLXFast
import MLXNN
@preconcurrency import MLXLMCommon

// MARK: - GPT-2 Attention

/// GPT-2 style multi-head attention with fused QKV projection.
///
/// Weight keys: `c_attn.weight` (3*dim, dim), `c_attn.bias` (3*dim),
///              `c_proj.weight` (dim, dim), `c_proj.bias` (dim)
private class T3GPT2Attention: Module {
    let nHead: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "c_attn") var cAttn: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(_ config: GPT2BackboneConfig) {
        let dim = config.hiddenSize
        self.nHead = config.nHead
        self.headDim = config.headDim
        self.scale = pow(Float(headDim), -0.5)

        self._cAttn.wrappedValue = Linear(dim, 3 * dim)
        self._cProj.wrappedValue = Linear(dim, dim)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Fused QKV projection
        let qkv = cAttn(x)
        let dim = nHead * headDim

        // Split into Q, K, V
        var queries = qkv[0..., 0..., ..<dim]
        var keys = qkv[0..., 0..., dim ..< (2 * dim)]
        var values = qkv[0..., 0..., (2 * dim)...]

        // Reshape to multi-head format: (B, L, D) → (B, nHead, L, headDim)
        queries = queries.reshaped(b, l, nHead, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(b, l, nHead, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(b, l, nHead, headDim).transposed(0, 2, 1, 3)

        // KV cache update (no RoPE for GPT-2)
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Scaled dot-product attention
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(b, l, -1)

        return cProj(output)
    }
}

// MARK: - GPT-2 MLP

/// GPT-2 style feed-forward network: Linear → GELU → Linear.
///
/// Weight keys: `c_fc.weight` (4*dim, dim), `c_fc.bias` (4*dim),
///              `c_proj.weight` (dim, 4*dim), `c_proj.bias` (dim)
private class T3GPT2MLP: Module {
    @ModuleInfo(key: "c_fc") var cFc: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(_ config: GPT2BackboneConfig) {
        let dim = config.hiddenSize
        let intermediate = config.intermediateSize
        self._cFc.wrappedValue = Linear(dim, intermediate)
        self._cProj.wrappedValue = Linear(intermediate, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return cProj(geluApproximate(cFc(x)))
    }
}

// MARK: - GPT-2 Transformer Block

/// GPT-2 style pre-norm transformer block.
///
/// Weight keys: `ln_1.weight/bias`, `attn.c_attn.*`, `attn.c_proj.*`,
///              `ln_2.weight/bias`, `mlp.c_fc.*`, `mlp.c_proj.*`
private class T3GPT2Block: Module {
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "attn") var attention: T3GPT2Attention
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: T3GPT2MLP

    init(_ config: GPT2BackboneConfig) {
        self._ln1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        self._attention.wrappedValue = T3GPT2Attention(config)
        self._ln2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        self._mlp.wrappedValue = T3GPT2MLP(config)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Pre-norm: ln → attn → residual, ln → mlp → residual
        let r = attention(ln1(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(ln2(h))
    }
}

// MARK: - GPT-2 Inner Model

/// Inner GPT-2 model for T3 Turbo — takes **embeddings** (not token IDs).
///
/// Weight keys: `wte.weight` (placeholder), `wpe.weight`, `h.{N}.*`, `ln_f.weight/bias`
class T3GPT2Inner: Module {
    /// Placeholder token embedding — T3 doesn't use it but the weight key exists.
    @ModuleInfo(key: "wte") var wte: Embedding

    /// Learned positional embeddings (GPT-2 style).
    /// Added to hidden states on every forward pass — matches Python `self.wpe(position_ids)`.
    @ModuleInfo(key: "wpe") var wpe: Embedding

    fileprivate let h: [T3GPT2Block]
    @ModuleInfo(key: "ln_f") var lnF: LayerNorm

    init(_ config: GPT2BackboneConfig) {
        self._wte.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize
        )
        self._wpe.wrappedValue = Embedding(
            embeddingCount: config.nCtx, dimensions: config.hiddenSize
        )
        self.h = (0 ..< config.nLayer).map { _ in T3GPT2Block(config) }
        self._lnF.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
    }

    /// Forward pass with pre-computed embeddings.
    /// Adds learned positional embeddings (wpe) matching Python GPT2Model.
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embeddings
        let seqLen = hidden.dim(1)

        // Compute past sequence length from KV cache offset (matches Python: cache[0].offset)
        let pastLength: Int
        if let firstCache = cache?.first {
            pastLength = firstCache.offset
        } else {
            pastLength = 0
        }

        // Add learned positional embeddings (GPT-2 wpe)
        let positionIds = MLXArray(Int32(pastLength) ..< Int32(pastLength + seqLen))
        let positionEmbeds = wpe(positionIds)
        hidden = hidden + positionEmbeds

        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for (i, layer) in h.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[i])
        }
        return lnF(hidden)
    }
}

// MARK: - T3 GPT-2 Model (Turbo)

/// Token-To-Token (T3) TTS model using GPT-2 as backbone (Chatterbox Turbo).
///
/// Generates speech tokens from text tokens, conditioned on speaker embeddings.
/// Unlike the LLaMA variant, Turbo uses:
/// - No CFG (classifier-free guidance)
/// - No learned text position embeddings
/// - speech_head has bias
/// - Simpler inference loop
public class T3GPT2Model: Module {
    let hp: T3Configuration
    let gpt2Config: GPT2BackboneConfig
    let dim: Int

    // GPT-2 backbone — weight key prefix: "tfmr.*"
    @ModuleInfo(key: "tfmr") var tfmr: T3GPT2Inner

    // Conditioning encoder
    @ModuleInfo(key: "cond_enc") var condEnc: T3CondEnc

    // Embeddings
    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "speech_emb") var speechEmb: Embedding

    // Learned position embeddings — speech only for Turbo (text pos emb is nil)
    @ModuleInfo(key: "text_pos_emb") var textPosEmb: LearnedPositionEmbeddings?
    @ModuleInfo(key: "speech_pos_emb") var speechPosEmb: LearnedPositionEmbeddings

    // Output heads — speech_head has bias for GPT-2 variant
    @ModuleInfo(key: "text_head") var textHead: Linear
    @ModuleInfo(key: "speech_head") var speechHead: Linear

    public init(_ hp: T3Configuration = .turbo, gpt2Config: GPT2BackboneConfig = .medium) {
        self.hp = hp
        self.gpt2Config = gpt2Config
        self.dim = gpt2Config.hiddenSize

        self._tfmr.wrappedValue = T3GPT2Inner(gpt2Config)
        self._condEnc.wrappedValue = T3CondEnc(hp)

        self._textEmb.wrappedValue = Embedding(embeddingCount: hp.textTokensDictSize, dimensions: dim)
        self._speechEmb.wrappedValue = Embedding(embeddingCount: hp.speechTokensDictSize, dimensions: dim)

        // Position embeddings: text only if configured, speech always
        if hp.inputPosEmb == "learned" {
            let maxTextSeqLen = hp.maxTextTokens + 2
            self._textPosEmb.wrappedValue = LearnedPositionEmbeddings(seqLen: maxTextSeqLen, modelDim: dim)
        } else {
            self._textPosEmb.wrappedValue = nil
        }
        let maxSpeechSeqLen = hp.maxSpeechTokens + 4
        self._speechPosEmb.wrappedValue = LearnedPositionEmbeddings(seqLen: maxSpeechSeqLen, modelDim: dim)

        self._textHead.wrappedValue = Linear(dim, hp.textTokensDictSize, bias: false)
        self._speechHead.wrappedValue = Linear(dim, hp.speechTokensDictSize, bias: true)
    }

    /// Number of transformer layers for cache creation.
    public var numLayers: Int { gpt2Config.nLayer }

    /// Create KV cache for inference.
    public func makeCache() -> [KVCache] {
        (0 ..< numLayers).map { _ in KVCacheSimple() }
    }

    // MARK: - Conditioning

    /// Prepare conditioning embeddings from T3Cond.
    public func prepareConditioning(_ t3Cond: inout T3Cond) -> MLXArray {
        // For GPT-2 Turbo: embed speech prompt tokens without position embeddings
        if t3Cond.condPromptSpeechTokens != nil && t3Cond.condPromptSpeechEmb == nil {
            let tokens = t3Cond.condPromptSpeechTokens!
            t3Cond.condPromptSpeechEmb = speechEmb(tokens)
            // Note: Turbo does NOT add position embeddings to conditioning speech tokens
        }
        return condEnc(t3Cond)
    }

    // MARK: - Weight Sanitization

    /// Sanitize weights for GPT-2 Turbo T3.
    ///
    /// GPT-2 weight keys map directly — no prefix remapping needed.
    /// Keys like `tfmr.h.0.attn.c_attn.weight` match the module structure.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // GPT-2 keys map naturally to our module hierarchy — pass through
        return weights
    }

    // MARK: - Inference (Turbo)

    /// Generate speech tokens using Turbo inference (no CFG).
    ///
    /// Matches Python's `inference_turbo` method.
    public func inference(
        t3Cond: inout T3Cond,
        textTokens: MLXArray,
        maxNewTokens: Int = 1000,
        temperature: Float = 0.8,
        topK: Int = 1000,
        topP: Float = 0.95,
        repetitionPenalty: Float = 1.2
    ) -> MLXArray {
        var tokens = textTokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        // Prepare conditioning
        let condEmb = prepareConditioning(&t3Cond) // (1, condLen, dim)

        // Text embeddings (no position embeddings for Turbo)
        var textEmbResult = textEmb(tokens)
        if hp.inputPosEmb == "learned", let textPosEmb {
            textEmbResult = textEmbResult + textPosEmb(tokens)
        }

        // Speech start token
        let speechStartToken = MLXArray([Int32(hp.startSpeechToken)]).reshaped([1, 1])
        let speechStartEmbed = speechEmb(speechStartToken)

        // Build initial input: [conditioning | text | speech_start]
        let inputEmbeddings = MLX.concatenated([condEmb, textEmbResult, speechStartEmbed], axis: 1)

        // Create KV cache
        let cache = makeCache()

        // Initial forward pass to fill cache
        var hidden = tfmr(inputEmbeddings, cache: cache)

        // Get first speech logits
        var speechLogits = speechHead(hidden[0..., (-1)..., 0...])

        var generatedIds = [Int]()

        print("[T3-Turbo] Starting generation (maxNewTokens=\(maxNewTokens), temp=\(temperature), topK=\(topK), topP=\(topP))")
        let genStart = CFAbsoluteTimeGetCurrent()

        // Generation loop
        for step in 0 ..< maxNewTokens {
            if Task.isCancelled { break }
            var logits = speechLogits.squeezed(axis: 1) // (1, vocab)

            // Apply repetition penalty
            if repetitionPenalty != 1.0 && !generatedIds.isEmpty {
                logits = applyRepetitionPenalty(logits: logits, generatedIds: generatedIds, penalty: repetitionPenalty, vocabSize: hp.speechTokensDictSize)
            }

            // Sample
            let nextToken = sampleToken(logits: logits, temperature: temperature, topK: topK, topP: topP)
            eval(nextToken)
            let nextTokenId = nextToken[0].item(Int.self)

            // Check EOS
            if nextTokenId == hp.stopSpeechToken {
                let elapsed = CFAbsoluteTimeGetCurrent() - genStart
                print("[T3-Turbo] EOS at step \(step)/\(maxNewTokens) (\(generatedIds.count) tokens, \(String(format: "%.2f", elapsed))s)")
                break
            }
            generatedIds.append(nextTokenId)

            if step % 100 == 0 && step > 0 {
                let elapsed = CFAbsoluteTimeGetCurrent() - genStart
                print("[T3-Turbo] Step \(step)/\(maxNewTokens) (\(generatedIds.count) tokens, \(String(format: "%.2f", elapsed))s)")
            }

            // Embed next token for next step
            let nextTokenEmbed = speechEmb(MLXArray([Int32(nextTokenId)]).reshaped([1, 1]))

            // Forward with cache
            hidden = tfmr(nextTokenEmbed, cache: cache)
            speechLogits = speechHead(hidden[0..., (-1)..., 0...])
            eval(speechLogits)
        }

        let totalElapsed = CFAbsoluteTimeGetCurrent() - genStart
        print("[T3-Turbo] Generation complete: \(generatedIds.count) tokens in \(String(format: "%.2f", totalElapsed))s")
        return MLXArray(generatedIds.map { Int32($0) }).reshaped([1, -1])
    }
}

// MARK: - GELU Approximate

/// GPT-2 uses GELU with tanh approximation ("gelu_new").
/// Must use geluApproximate (tanh version), NOT gelu (erf version).
/// Python: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
private func geluApproximate(_ x: MLXArray) -> MLXArray {
    return MLXNN.geluApproximate(x)
}
