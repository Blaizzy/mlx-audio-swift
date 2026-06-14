import Foundation
import MLX
import MLXAudioCore
import MLXNN

// MARK: - Irodori DiT (MLX Swift port of mlx_audio/tts/models/irodori_tts/model.py)
//
// Reuses Echo TTS primitives where the math and weight keys are identical:
//   EchoRMSNorm == RMSNorm, EchoLowRankAdaLN == LowRankAdaLN, EchoMLP == SwiGLU,
//   EchoSelfAttention == SelfAttention (non-causal), EchoEncoderTransformerBlock == TextBlock,
//   echoTtsPrecomputeFreqsCis / echoTtsApplyRotaryEmb / echoTtsTimestepEmbedding.

typealias IrodoriKVCache = (keys: MLXArray, values: MLXArray)

// MARK: - Helpers

/// Convert boolean mask (B, Sq, Sk) to additive float mask (B, 1, Sq, Sk).
func irodoriBoolToAdditiveMask(_ mask: MLXArray) -> MLXArray {
    let zero = MLXArray.zeros(mask.shape, dtype: .float32)
    let negInf = MLXArray.full(mask.shape, values: MLXArray(-1e9), dtype: .float32)
    return MLX.where(mask, zero, negInf).expandedDimensions(axis: 1)
}

/// Patch along the sequence axis.
///   seq  : (B, S, D)   -> (B, S/patch, D*patch)
///   mask : (B, S) bool -> (B, S/patch) bool (true iff all tokens in patch are valid)
func irodoriPatchSequenceWithMask(
    seq: MLXArray,
    mask: MLXArray,
    patchSize: Int
) -> (MLXArray, MLXArray) {
    if patchSize <= 1 {
        return (seq, mask)
    }
    let bsz = seq.shape[0]
    let seqLen = seq.shape[1]
    let dim = seq.shape[2]
    let usable = (seqLen / patchSize) * patchSize
    let patchedSeq = seq[0..., 0..<usable, 0...].reshaped([bsz, usable / patchSize, dim * patchSize])
    let patchedMask = mask[0..., 0..<usable].reshaped([bsz, usable / patchSize, patchSize])
    return (patchedSeq, patchedMask.all(axis: -1))
}

/// Ensure at least one position is valid per batch for attention pooling.
func irodoriSafeAttentionMask(_ x: MLXArray, _ mask: MLXArray) throws -> (MLXArray, MLXArray) {
    guard mask.ndim == 2, mask.shape[0] == x.shape[0], mask.shape[1] == x.shape[1] else {
        throw IrodoriTTSError.generation(
            "mask must have shape (B, S) matching x, got x=\(x.shape) mask=\(mask.shape)"
        )
    }
    let maskBool = mask.asType(.bool)
    let hasAny = maskBool.any(axis: 1)
    if MLX.all(hasAny).item(Bool.self) {
        return (x, maskBool)
    }
    guard x.shape[1] > 0 else {
        throw IrodoriTTSError.generation("Cannot attention-pool an empty sequence.")
    }
    let zeroedX = MLX.where(
        hasAny.expandedDimensions(axes: [1, 2]),
        x,
        MLXArray.zeros(x.shape, dtype: x.dtype)
    )
    // Set first position valid for batches with no valid positions
    let fallback = .!hasAny
    let firstValid = MLX.concatenated(
        [MLXArray.ones([x.shape[0], 1], dtype: .bool), maskBool[0..., 1...]],
        axis: 1
    )
    let fixedMask = MLX.where(fallback.expandedDimensions(axis: 1), firstValid, maskBool)
    return (zeroedX, fixedMask)
}

// MARK: - Joint attention

/// Joint attention over latent self-tokens, text context, and speaker/caption context.
/// Half-RoPE: RoPE applied to the first half of attention heads.
/// speakerCtxDim and/or captionCtxDim may be provided independently;
/// dual mode (v3 VoiceDesign) sets both.
final class IrodoriJointAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let hasSpeakerCondition: Bool
    let hasCaptionCondition: Bool

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wk_text") var wkText: Linear
    @ModuleInfo(key: "wv_text") var wvText: Linear
    @ModuleInfo(key: "wk_speaker") var wkSpeaker: Linear?
    @ModuleInfo(key: "wv_speaker") var wvSpeaker: Linear?
    @ModuleInfo(key: "wk_caption") var wkCaption: Linear?
    @ModuleInfo(key: "wv_caption") var wvCaption: Linear?
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: EchoRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EchoRMSNorm

    init(
        dim: Int,
        heads: Int,
        textCtxDim: Int,
        speakerCtxDim: Int?,
        normEps: Float,
        captionCtxDim: Int?
    ) throws {
        precondition(dim % heads == 0, "dim must be divisible by heads")
        guard speakerCtxDim != nil || captionCtxDim != nil else {
            throw IrodoriTTSError.generation(
                "At least one of speakerCtxDim or captionCtxDim must be set"
            )
        }
        self.numHeads = heads
        self.headDim = dim / heads
        self.scale = 1 / sqrt(Float(self.headDim))
        self.hasSpeakerCondition = speakerCtxDim != nil
        self.hasCaptionCondition = captionCtxDim != nil

        self._wq = ModuleInfo(wrappedValue: Linear(dim, dim, bias: false))
        self._wk = ModuleInfo(wrappedValue: Linear(dim, dim, bias: false))
        self._wv = ModuleInfo(wrappedValue: Linear(dim, dim, bias: false))
        self._wkText = ModuleInfo(wrappedValue: Linear(textCtxDim, dim, bias: false))
        self._wvText = ModuleInfo(wrappedValue: Linear(textCtxDim, dim, bias: false))
        self._wkSpeaker = ModuleInfo(wrappedValue: speakerCtxDim.map { Linear($0, dim, bias: false) })
        self._wvSpeaker = ModuleInfo(wrappedValue: speakerCtxDim.map { Linear($0, dim, bias: false) })
        self._wkCaption = ModuleInfo(wrappedValue: captionCtxDim.map { Linear($0, dim, bias: false) })
        self._wvCaption = ModuleInfo(wrappedValue: captionCtxDim.map { Linear($0, dim, bias: false) })
        self._gate = ModuleInfo(wrappedValue: Linear(dim, dim, bias: false))
        self._wo = ModuleInfo(wrappedValue: Linear(dim, dim, bias: false))
        self._qNorm = ModuleInfo(wrappedValue: EchoRMSNorm(shape: [heads, self.headDim], eps: normEps))
        self._kNorm = ModuleInfo(wrappedValue: EchoRMSNorm(shape: [heads, self.headDim], eps: normEps))
    }

    /// Apply RoPE to the first half of attention heads only.
    private func applyRotaryHalf(_ y: MLXArray, freqsCis: EchoTTSRotaryCache) -> MLXArray {
        let half = y.shape[2] / 2
        let rotated = echoTtsApplyRotaryEmb(y[0..., 0..., 0..<half, 0...], freqsCis: freqsCis)
        return MLX.concatenated([rotated, y[0..., 0..., half..., 0...]], axis: 2)
    }

    func getKVCacheText(_ textState: MLXArray) -> IrodoriKVCache {
        let bsz = textState.shape[0]
        let length = textState.shape[1]
        let keys = kNorm(wkText(textState).reshaped([bsz, length, numHeads, headDim]))
        let values = wvText(textState).reshaped([bsz, length, numHeads, headDim])
        return (keys, values)
    }

    func getKVCacheSpeaker(_ speakerState: MLXArray) throws -> IrodoriKVCache {
        guard let wkSpeaker, let wvSpeaker else {
            throw IrodoriTTSError.generation("Speaker condition modules are missing")
        }
        let bsz = speakerState.shape[0]
        let length = speakerState.shape[1]
        let keys = kNorm(wkSpeaker(speakerState).reshaped([bsz, length, numHeads, headDim]))
        let values = wvSpeaker(speakerState).reshaped([bsz, length, numHeads, headDim])
        return (keys, values)
    }

    func getKVCacheCaption(_ captionState: MLXArray) throws -> IrodoriKVCache {
        guard let wkCaption, let wvCaption else {
            throw IrodoriTTSError.generation("Caption condition modules are missing")
        }
        let bsz = captionState.shape[0]
        let length = captionState.shape[1]
        let keys = kNorm(wkCaption(captionState).reshaped([bsz, length, numHeads, headDim]))
        let values = wvCaption(captionState).reshaped([bsz, length, numHeads, headDim])
        return (keys, values)
    }

    func callAsFunction(
        _ x: MLXArray,
        textMask: MLXArray,
        freqsCis: EchoTTSRotaryCache,
        kvCacheText: IrodoriKVCache,
        kvCacheSpeaker: IrodoriKVCache?,
        speakerMask: MLXArray?,
        kvCacheCaption: IrodoriKVCache?,
        captionMask: MLXArray?,
        startPos: Int
    ) -> MLXArray {
        let bsz = x.shape[0]
        let seqLen = x.shape[1]

        var q = wq(x).reshaped([bsz, seqLen, numHeads, headDim])
        var kSelf = wk(x).reshaped([bsz, seqLen, numHeads, headDim])
        let vSelf = wv(x).reshaped([bsz, seqLen, numHeads, headDim])
        let gateValues = gate(x)

        q = qNorm(q)
        kSelf = kNorm(kSelf)

        let qFreqs = (
            cos: freqsCis.cos[startPos ..< (startPos + seqLen)],
            sin: freqsCis.sin[startPos ..< (startPos + seqLen)]
        )
        q = applyRotaryHalf(q, freqsCis: qFreqs)
        kSelf = applyRotaryHalf(kSelf, freqsCis: qFreqs)

        let selfMask = MLXArray.ones([bsz, seqLen], dtype: .bool)
        var kParts = [kSelf, kvCacheText.keys]
        var vParts = [vSelf, kvCacheText.values]
        var maskParts = [selfMask, textMask]

        if let kvCacheSpeaker, let speakerMask {
            kParts.append(kvCacheSpeaker.keys)
            vParts.append(kvCacheSpeaker.values)
            maskParts.append(speakerMask)
        }
        if let kvCacheCaption, let captionMask {
            kParts.append(kvCacheCaption.keys)
            vParts.append(kvCacheCaption.values)
            maskParts.append(captionMask)
        }

        let keys = MLX.concatenated(kParts, axis: 1)
        let values = MLX.concatenated(vParts, axis: 1)
        let fullMask = MLX.concatenated(maskParts, axis: 1)
        let attentionMask = irodoriBoolToAdditiveMask(
            MLX.broadcast(fullMask.expandedDimensions(axis: 1), to: [bsz, seqLen, fullMask.shape[1]])
        )

        let output = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3),
            keys: keys.transposed(0, 2, 1, 3),
            values: values.transposed(0, 2, 1, 3),
            scale: scale,
            mask: attentionMask
        )
        let merged = output.transposed(0, 2, 1, 3).reshaped([bsz, seqLen, numHeads * headDim])
        return wo(merged * sigmoid(gateValues))
    }
}

// MARK: - Encoders

/// Text encoder: embedding + non-causal Transformer blocks.
/// Applies mask zeroing after each block so fully-masked positions stay zero.
/// Used for both text and caption encoding.
final class IrodoriTextEncoder: Module {
    let headDim: Int

    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding
    @ModuleInfo(key: "blocks") var blocks: [EchoEncoderTransformerBlock]

    init(
        vocabSize: Int,
        dim: Int,
        heads: Int,
        numLayers: Int,
        mlpRatio: Float,
        normEps: Float
    ) {
        self.headDim = dim / heads
        self._textEmbedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: vocabSize, dimensions: dim))
        let mlpHidden = Int(Float(dim) * mlpRatio)
        self._blocks = ModuleInfo(wrappedValue: (0 ..< numLayers).map { _ in
            EchoEncoderTransformerBlock(
                modelSize: dim,
                numHeads: heads,
                intermediateSize: mlpHidden,
                isCausal: false,
                normEps: normEps
            )
        })
    }

    func callAsFunction(_ inputIDs: MLXArray, mask: MLXArray?) -> MLXArray {
        var x = textEmbedding(inputIDs)
        let freqs = echoTtsPrecomputeFreqsCis(dim: headDim, end: inputIDs.shape[1])
        if let mask {
            let maskF = mask.expandedDimensions(axis: -1).asType(x.dtype)
            x = x * maskF
            for block in blocks {
                x = block(x, mask: mask, freqsCis: freqs)
                x = x * maskF
            }
            return x * maskF
        } else {
            for block in blocks {
                x = block(x, mask: nil, freqsCis: freqs)
            }
            return x
        }
    }
}

/// Encoder for reference (speaker) audio latents.
/// Receives already-patched DACVAE latents: (B, S, latentDim * speakerPatchSize).
/// Uses non-causal attention (unlike Echo TTS which uses causal).
final class IrodoriReferenceLatentEncoder: Module {
    let headDim: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "blocks") var blocks: [EchoEncoderTransformerBlock]

    init(
        inDim: Int,
        dim: Int,
        heads: Int,
        numLayers: Int,
        mlpRatio: Float,
        normEps: Float
    ) {
        self.headDim = dim / heads
        self._inProj = ModuleInfo(wrappedValue: Linear(inDim, dim, bias: true))
        let mlpHidden = Int(Float(dim) * mlpRatio)
        self._blocks = ModuleInfo(wrappedValue: (0 ..< numLayers).map { _ in
            EchoEncoderTransformerBlock(
                modelSize: dim,
                numHeads: heads,
                intermediateSize: mlpHidden,
                isCausal: false,
                normEps: normEps
            )
        })
    }

    func callAsFunction(_ latent: MLXArray, mask: MLXArray?) -> MLXArray {
        var x = inProj(latent) / 6
        let freqs = echoTtsPrecomputeFreqsCis(dim: headDim, end: x.shape[1])
        if let mask {
            let maskF = mask.expandedDimensions(axis: -1).asType(x.dtype)
            x = x * maskF
            for block in blocks {
                x = block(x, mask: mask, freqsCis: freqs)
                x = x * maskF
            }
            return x * maskF
        } else {
            for block in blocks {
                x = block(x, mask: nil, freqsCis: freqs)
            }
            return x
        }
    }
}

// MARK: - Diffusion block

/// Single DiT block: JointAttention + SwiGLU, both conditioned via LowRankAdaLN.
final class IrodoriDiffusionBlock: Module {
    @ModuleInfo(key: "attention") var attention: IrodoriJointAttention
    @ModuleInfo(key: "mlp") var mlp: EchoMLP
    @ModuleInfo(key: "attention_adaln") var attentionAdaLN: EchoLowRankAdaLN
    @ModuleInfo(key: "mlp_adaln") var mlpAdaLN: EchoLowRankAdaLN

    init(
        dim: Int,
        heads: Int,
        mlpHiddenDim: Int,
        textCtxDim: Int,
        speakerCtxDim: Int?,
        adalnRank: Int,
        normEps: Float,
        captionCtxDim: Int?
    ) throws {
        self._attention = ModuleInfo(wrappedValue: try IrodoriJointAttention(
            dim: dim,
            heads: heads,
            textCtxDim: textCtxDim,
            speakerCtxDim: speakerCtxDim,
            normEps: normEps,
            captionCtxDim: captionCtxDim
        ))
        self._mlp = ModuleInfo(wrappedValue: EchoMLP(modelSize: dim, intermediateSize: mlpHiddenDim))
        self._attentionAdaLN = ModuleInfo(wrappedValue: EchoLowRankAdaLN(modelSize: dim, rank: adalnRank, eps: normEps))
        self._mlpAdaLN = ModuleInfo(wrappedValue: EchoLowRankAdaLN(modelSize: dim, rank: adalnRank, eps: normEps))
    }

    func callAsFunction(
        _ x: MLXArray,
        condEmbed: MLXArray,
        textMask: MLXArray,
        freqsCis: EchoTTSRotaryCache,
        kvCacheText: IrodoriKVCache,
        kvCacheSpeaker: IrodoriKVCache?,
        speakerMask: MLXArray?,
        kvCacheCaption: IrodoriKVCache?,
        captionMask: MLXArray?,
        startPos: Int
    ) -> MLXArray {
        let (attentionInput, attentionGate) = attentionAdaLN(x, condEmbed: condEmbed)
        let attended = attention(
            attentionInput,
            textMask: textMask,
            freqsCis: freqsCis,
            kvCacheText: kvCacheText,
            kvCacheSpeaker: kvCacheSpeaker,
            speakerMask: speakerMask,
            kvCacheCaption: kvCacheCaption,
            captionMask: captionMask,
            startPos: startPos
        )
        let withAttention = x + attentionGate * attended

        let (mlpInput, mlpGate) = mlpAdaLN(withAttention, condEmbed: condEmbed)
        return withAttention + mlpGate * mlp(mlpInput)
    }
}

// MARK: - Duration predictor (v3, token-sum architectures)

/// SwiGLU block with optional AdaRN-Zero modulation for the duration predictor.
/// Supports dual modulation (speaker + caption) added together, matching
/// the token_sum_dual_adarn_zero_no_aux architecture.
final class IrodoriDurationSwiGLUBlock: Module {
    @ModuleInfo(key: "norm") var norm: EchoRMSNorm
    @ModuleInfo(key: "mlp") var mlp: EchoMLP
    @ModuleInfo(key: "modulation") var modulation: Linear?
    @ModuleInfo(key: "caption_modulation") var captionModulation: Linear?

    init(
        dim: Int,
        hiddenDim: Int,
        normEps: Float,
        condDim: Int?,
        captionCondDim: Int?
    ) {
        self._norm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: dim, eps: normEps))
        self._mlp = ModuleInfo(wrappedValue: EchoMLP(modelSize: dim, intermediateSize: hiddenDim))
        self._modulation = ModuleInfo(wrappedValue: condDim.map { Linear($0, dim * 3, bias: true) })
        self._captionModulation = ModuleInfo(wrappedValue: captionCondDim.map { Linear($0, dim * 3, bias: true) })
    }

    func callAsFunction(
        _ x: MLXArray,
        cond: MLXArray?,
        captionCond: MLXArray?
    ) throws -> MLXArray {
        var h = norm(x)
        guard modulation != nil || captionModulation != nil else {
            return x + mlp(h)
        }

        var shift = MLXArray.zeros(h.shape, dtype: h.dtype)
        var scale = MLXArray.zeros(h.shape, dtype: h.dtype)
        var gateAccum = MLXArray.zeros(h.shape, dtype: h.dtype)

        if let modulation {
            guard let cond else {
                throw IrodoriTTSError.generation("cond is required for AdaRN-Zero duration blocks.")
            }
            var parts = modulation(silu(cond)).split(parts: 3, axis: -1)
            if h.ndim == 3, parts[0].ndim == 2 {
                parts = parts.map { $0.expandedDimensions(axis: 1) }
            }
            shift = shift + parts[0]
            scale = scale + parts[1]
            gateAccum = gateAccum + parts[2]
        }
        if let captionModulation {
            guard let captionCond else {
                throw IrodoriTTSError.generation("captionCond is required for caption AdaRN-Zero blocks.")
            }
            var parts = captionModulation(silu(captionCond)).split(parts: 3, axis: -1)
            if h.ndim == 3, parts[0].ndim == 2 {
                parts = parts.map { $0.expandedDimensions(axis: 1) }
            }
            shift = shift + parts[0]
            scale = scale + parts[1]
            gateAccum = gateAccum + parts[2]
        }

        h = h * (scale + 1) + shift
        return x + tanh(gateAccum) * mlp(h)
    }
}

/// Duration predictor that regresses log1p(num_frames) from text state.
///
/// Only the token-sum architectures are ported (the v3 defaults):
///   - token_sum_adarn_zero_no_aux (speaker-only)
///   - token_sum_dual_adarn_zero_no_aux (speaker + caption, v3 VoiceDesign)
/// The legacy pooled architectures throw at init.
final class IrodoriDurationPredictor: Module {
    let textDim: Int
    let auxDim: Int
    let speakerDim: Int?
    let captionDim: Int?
    let architecture: String

    @ParameterInfo(key: "null_speaker") var nullSpeaker: MLXArray?
    @ParameterInfo(key: "null_caption") var nullCaption: MLXArray?
    @ModuleInfo(key: "token_input_proj") var tokenInputProj: Linear
    @ModuleInfo(key: "token_blocks") var tokenBlocks: [IrodoriDurationSwiGLUBlock]
    @ModuleInfo(key: "token_out_norm") var tokenOutNorm: EchoRMSNorm
    @ModuleInfo(key: "token_out_proj") var tokenOutProj: Linear

    init(
        textDim: Int,
        auxDim: Int,
        hiddenDim: Int,
        layers: Int,
        normEps: Float,
        speakerDim: Int?,
        captionDim: Int?,
        architecture: String,
        tokenInitFrames: Float
    ) throws {
        guard architecture == "token_sum_adarn_zero_no_aux"
            || architecture == "token_sum_dual_adarn_zero_no_aux"
        else {
            throw IrodoriTTSError.generation(
                "Unsupported duration architecture: \(architecture). "
                    + "Only token-sum architectures are ported."
            )
        }
        let isDual = architecture == "token_sum_dual_adarn_zero_no_aux"

        self.textDim = textDim
        self.auxDim = auxDim
        self.speakerDim = speakerDim
        self.captionDim = captionDim
        self.architecture = architecture

        self._nullSpeaker = ParameterInfo(wrappedValue: speakerDim.map { MLXArray.zeros([$0]) })
        self._nullCaption = ParameterInfo(wrappedValue: captionDim.map { MLXArray.zeros([$0]) })

        self._tokenInputProj = ModuleInfo(wrappedValue: Linear(textDim, hiddenDim, bias: true))
        self._tokenBlocks = ModuleInfo(wrappedValue: (0 ..< layers).map { _ in
            IrodoriDurationSwiGLUBlock(
                dim: hiddenDim,
                hiddenDim: hiddenDim,
                normEps: normEps,
                condDim: speakerDim,
                captionCondDim: isDual ? captionDim : nil
            )
        })
        self._tokenOutNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: hiddenDim, eps: normEps))
        self._tokenOutProj = ModuleInfo(wrappedValue: Linear(hiddenDim, 1, bias: true))
    }

    private func speakerVec(
        batchSize: Int,
        dtype: DType,
        speakerState: MLXArray?,
        hasSpeaker: MLXArray
    ) throws -> MLXArray {
        guard let nullSpeaker, let speakerDim else {
            throw IrodoriTTSError.generation("Duration speaker modules are missing.")
        }
        let nullVec = MLX.broadcast(
            nullSpeaker.asType(dtype).expandedDimensions(axis: 0),
            to: [batchSize, speakerDim]
        )
        guard let speakerState else { return nullVec }
        let vec = speakerState[0..., 0, 0...].asType(dtype)
        return MLX.where(hasSpeaker.expandedDimensions(axis: 1), vec, nullVec)
    }

    private func captionVec(
        batchSize: Int,
        dtype: DType,
        captionState: MLXArray?,
        captionMask: MLXArray?,
        hasCaption: MLXArray
    ) throws -> MLXArray {
        guard let nullCaption, let captionDim else {
            throw IrodoriTTSError.generation("Duration caption modules are missing.")
        }
        let nullVec = MLX.broadcast(
            nullCaption.asType(dtype).expandedDimensions(axis: 0),
            to: [batchSize, captionDim]
        )
        guard let captionState else { return nullVec }
        let state = captionState.asType(dtype)
        let vec: MLXArray
        if let captionMask {
            let maskF = captionMask.expandedDimensions(axis: -1).asType(dtype)
            let denom = MLX.maximum(maskF.sum(axis: 1), MLXArray(1.0).asType(dtype))
            vec = (state * maskF).sum(axis: 1) / denom
        } else {
            vec = state.mean(axis: 1)
        }
        return MLX.where(hasCaption.expandedDimensions(axis: 1), vec, nullVec)
    }

    func callAsFunction(
        textState rawTextState: MLXArray,
        textMask rawTextMask: MLXArray,
        auxFeatures: MLXArray,
        speakerState: MLXArray?,
        hasSpeaker: MLXArray?,
        captionState: MLXArray?,
        captionMask: MLXArray?,
        hasCaption: MLXArray?
    ) throws -> MLXArray {
        guard rawTextState.ndim == 3, rawTextState.shape[2] == textDim else {
            throw IrodoriTTSError.generation(
                "textState must have shape (B, S, \(textDim)), got \(rawTextState.shape)"
            )
        }
        guard auxFeatures.ndim == 2, auxFeatures.shape[1] == auxDim else {
            throw IrodoriTTSError.generation(
                "auxFeatures must have shape (B, \(auxDim)), got \(auxFeatures.shape)"
            )
        }
        let (textState, textMask) = try irodoriSafeAttentionMask(rawTextState, rawTextMask)
        let isDual = architecture == "token_sum_dual_adarn_zero_no_aux"

        guard speakerDim != nil else {
            throw IrodoriTTSError.generation("Token-sum duration architecture requires speaker modules.")
        }
        guard let hasSpeaker else {
            throw IrodoriTTSError.generation(
                "hasSpeaker is required for speaker-conditioned duration prediction."
            )
        }
        let speakerVector = try speakerVec(
            batchSize: textState.shape[0],
            dtype: textState.dtype,
            speakerState: speakerState,
            hasSpeaker: hasSpeaker.asType(.bool)
        )

        var captionVector: MLXArray?
        if isDual {
            guard captionDim != nil else {
                throw IrodoriTTSError.generation(
                    "Dual token-sum architecture requires both speaker and caption modules."
                )
            }
            guard let hasCaption else {
                throw IrodoriTTSError.generation("hasCaption is required for dual duration prediction.")
            }
            captionVector = try captionVec(
                batchSize: textState.shape[0],
                dtype: textState.dtype,
                captionState: captionState,
                captionMask: captionMask,
                hasCaption: hasCaption.asType(.bool)
            )
        }

        var h = tokenInputProj(textState)
        for block in tokenBlocks {
            h = try block(h, cond: speakerVector, captionCond: captionVector)
        }
        let tokenLogits = tokenOutProj(tokenOutNorm(h)).squeezed(axis: -1)
        // NOTE: softplus, computed as log(1 + exp(x)) in float32 (matches Python port)
        let tokenFrames = MLX.log(1 + MLX.exp(tokenLogits.asType(.float32)))
        let totalFrames = (tokenFrames * textMask.asType(tokenFrames.dtype)).sum(axis: 1)
        return MLX.log1p(MLX.maximum(totalFrames, MLXArray(Float(0))))
    }
}

// MARK: - Main DiT model

/// Irodori-TTS DiT model (Swift port of TextToLatentRFDiT).
///
/// Supports speaker-only, caption-only, and dual (v3 VoiceDesign) conditioning.
/// Input x_t : (B, S, latentDim * latentPatchSize); output v_t : same shape
/// (velocity prediction for the Rectified Flow ODE).
public final class IrodoriDiT: Module {
    let cfg: IrodoriDiTConfig
    let headDim: Int

    @ModuleInfo(key: "text_encoder") var textEncoder: IrodoriTextEncoder
    @ModuleInfo(key: "text_norm") var textNorm: EchoRMSNorm
    @ModuleInfo(key: "speaker_encoder") var speakerEncoder: IrodoriReferenceLatentEncoder?
    @ModuleInfo(key: "speaker_norm") var speakerNorm: EchoRMSNorm?
    @ModuleInfo(key: "caption_encoder") var captionEncoder: IrodoriTextEncoder?
    @ModuleInfo(key: "caption_norm") var captionNorm: EchoRMSNorm?
    @ModuleInfo(key: "duration_predictor") var durationPredictor: IrodoriDurationPredictor?
    @ModuleInfo(key: "cond_module") var condModule: EchoSequential
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "blocks") var blocks: [IrodoriDiffusionBlock]
    @ModuleInfo(key: "out_norm") var outNorm: EchoRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(cfg: IrodoriDiTConfig) throws {
        self.cfg = cfg
        self.headDim = cfg.modelDim / cfg.numHeads

        self._textEncoder = ModuleInfo(wrappedValue: IrodoriTextEncoder(
            vocabSize: cfg.textVocabSize,
            dim: cfg.textDim,
            heads: cfg.textHeads,
            numLayers: cfg.textLayers,
            mlpRatio: cfg.textMlpRatioResolved,
            normEps: cfg.normEps
        ))
        self._textNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: cfg.textDim, eps: cfg.normEps))

        var speakerCtxDim: Int?
        var captionCtxDim: Int?

        if cfg.useSpeakerConditionResolved {
            self._speakerEncoder = ModuleInfo(wrappedValue: IrodoriReferenceLatentEncoder(
                inDim: cfg.speakerPatchedLatentDim,
                dim: cfg.speakerDim,
                heads: cfg.speakerHeads,
                numLayers: cfg.speakerLayers,
                mlpRatio: cfg.speakerMlpRatioResolved,
                normEps: cfg.normEps
            ))
            self._speakerNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: cfg.speakerDim, eps: cfg.normEps))
            speakerCtxDim = cfg.speakerDim
        } else {
            self._speakerEncoder = ModuleInfo(wrappedValue: nil)
            self._speakerNorm = ModuleInfo(wrappedValue: nil)
        }

        if cfg.useCaptionCondition {
            self._captionEncoder = ModuleInfo(wrappedValue: IrodoriTextEncoder(
                vocabSize: cfg.captionVocabSizeResolved,
                dim: cfg.captionDimResolved,
                heads: cfg.captionHeadsResolved,
                numLayers: cfg.captionLayersResolved,
                mlpRatio: cfg.captionMlpRatioResolved,
                normEps: cfg.normEps
            ))
            self._captionNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: cfg.captionDimResolved, eps: cfg.normEps))
            captionCtxDim = cfg.captionDimResolved
        } else {
            self._captionEncoder = ModuleInfo(wrappedValue: nil)
            self._captionNorm = ModuleInfo(wrappedValue: nil)
        }

        if cfg.useDurationPredictor {
            self._durationPredictor = ModuleInfo(wrappedValue: try IrodoriDurationPredictor(
                textDim: cfg.textDim,
                auxDim: cfg.durationAuxDim,
                hiddenDim: cfg.durationHiddenDim,
                layers: cfg.durationLayers,
                normEps: cfg.normEps,
                speakerDim: cfg.useSpeakerConditionResolved ? cfg.speakerDim : nil,
                captionDim: cfg.useCaptionCondition ? cfg.captionDimResolved : nil,
                architecture: cfg.durationArchitecture,
                tokenInitFrames: cfg.durationTokenInitFrames
            ))
        } else {
            self._durationPredictor = ModuleInfo(wrappedValue: nil)
        }

        self._condModule = ModuleInfo(wrappedValue: EchoSequential([
            Linear(cfg.timestepEmbedDim, cfg.modelDim, bias: false),
            SiLU(),
            Linear(cfg.modelDim, cfg.modelDim, bias: false),
            SiLU(),
            Linear(cfg.modelDim, cfg.modelDim * 3, bias: false),
        ]))

        self._inProj = ModuleInfo(wrappedValue: Linear(cfg.patchedLatentDim, cfg.modelDim, bias: true))
        let mlpHidden = Int(Float(cfg.modelDim) * cfg.mlpRatio)
        self._blocks = ModuleInfo(wrappedValue: try (0 ..< cfg.numLayers).map { _ in
            try IrodoriDiffusionBlock(
                dim: cfg.modelDim,
                heads: cfg.numHeads,
                mlpHiddenDim: mlpHidden,
                textCtxDim: cfg.textDim,
                speakerCtxDim: speakerCtxDim,
                adalnRank: cfg.adalnRank,
                normEps: cfg.normEps,
                captionCtxDim: captionCtxDim
            )
        })
        self._outNorm = ModuleInfo(wrappedValue: EchoRMSNorm(dimensions: cfg.modelDim, eps: cfg.normEps))
        self._outProj = ModuleInfo(wrappedValue: Linear(cfg.modelDim, cfg.patchedLatentDim, bias: true))
    }

    // MARK: Condition encoding (cached across sampling steps)

    /// Encode all conditions including both speaker and caption states.
    /// Returns (textState, textMask, speakerState, speakerMask, captionState, captionMask).
    func encodeConditionsFull(
        textInputIDs: MLXArray,
        textMask: MLXArray,
        refLatent: MLXArray?,
        refMask: MLXArray?,
        captionInputIDs: MLXArray?,
        captionMask: MLXArray?
    ) -> (
        textState: MLXArray,
        textMask: MLXArray,
        speakerState: MLXArray?,
        speakerMask: MLXArray?,
        captionState: MLXArray?,
        captionMask: MLXArray?
    ) {
        let textState = textNorm(textEncoder(textInputIDs, mask: textMask))

        var speakerState: MLXArray?
        var speakerMaskOut: MLXArray?
        if cfg.useSpeakerConditionResolved, let speakerEncoder, let speakerNorm {
            if let refLatent, let refMask {
                let (refLatentP, refMaskP) = irodoriPatchSequenceWithMask(
                    seq: refLatent,
                    mask: refMask,
                    patchSize: cfg.speakerPatchSize
                )
                speakerState = speakerNorm(speakerEncoder(refLatentP, mask: refMaskP))
                speakerMaskOut = refMaskP
            } else {
                // Zero speaker state for duration prediction with no reference
                speakerState = MLXArray.zeros(
                    [textInputIDs.shape[0], 1, cfg.speakerDim],
                    dtype: textState.dtype
                )
                speakerMaskOut = MLXArray.zeros([textInputIDs.shape[0], 1], dtype: .bool)
            }
        }

        var captionState: MLXArray?
        var captionMaskOut: MLXArray?
        if cfg.useCaptionCondition, let captionEncoder, let captionNorm {
            if let captionInputIDs, let captionMask {
                captionState = captionNorm(captionEncoder(captionInputIDs, mask: captionMask))
                captionMaskOut = captionMask
            }
        }

        return (textState, textMask, speakerState, speakerMaskOut, captionState, captionMaskOut)
    }

    /// Pre-compute per-layer text/speaker/caption KV projections for fast sampling.
    func buildKVCache(
        textState: MLXArray,
        speakerState: MLXArray?,
        captionState: MLXArray?
    ) throws -> (
        kvText: [IrodoriKVCache],
        kvSpeaker: [IrodoriKVCache]?,
        kvCaption: [IrodoriKVCache]?
    ) {
        let kvText = blocks.map { $0.attention.getKVCacheText(textState) }
        var kvSpeaker: [IrodoriKVCache]?
        if let speakerState, cfg.useSpeakerConditionResolved {
            kvSpeaker = try blocks.map { try $0.attention.getKVCacheSpeaker(speakerState) }
        }
        var kvCaption: [IrodoriKVCache]?
        if let captionState, cfg.useCaptionCondition {
            kvCaption = try blocks.map { try $0.attention.getKVCacheCaption(captionState) }
        }
        return (kvText, kvSpeaker, kvCaption)
    }

    /// Predict log1p(num_frames) from text state and duration features. Returns (B,) float32.
    func predictDurationLogFrames(
        textState: MLXArray,
        textMask: MLXArray,
        speakerState: MLXArray?,
        durationFeatures: MLXArray,
        hasSpeaker: MLXArray?,
        captionState: MLXArray?,
        captionMask: MLXArray?,
        hasCaption: MLXArray?
    ) throws -> MLXArray {
        guard let durationPredictor else {
            throw IrodoriTTSError.generation("Duration predictor is disabled for this model.")
        }
        guard durationFeatures.ndim == 2 else {
            throw IrodoriTTSError.generation(
                "durationFeatures must have shape (B, D), got \(durationFeatures.shape)"
            )
        }
        guard durationFeatures.shape[1] == cfg.durationAuxDim else {
            throw IrodoriTTSError.generation(
                "durationFeatures dim mismatch: expected \(cfg.durationAuxDim), got \(durationFeatures.shape[1])"
            )
        }
        let pred = try durationPredictor(
            textState: textState,
            textMask: textMask,
            auxFeatures: durationFeatures,
            speakerState: speakerState,
            hasSpeaker: hasSpeaker,
            captionState: captionState,
            captionMask: captionMask,
            hasCaption: hasCaption
        )
        return pred.asType(.float32)
    }

    // MARK: Forward (with pre-encoded conditions)

    /// Forward pass with pre-encoded conditions.
    ///
    /// For speaker-only models: speakerState/Mask carry the speaker context.
    /// For caption-only models: speakerState/Mask carry the caption context
    /// (backward-compat; internally routed to the caption branch).
    /// For dual models (v3 VoiceDesign): speakerState=speaker, captionState=caption.
    func forwardWithConditions(
        xT: MLXArray,
        t: MLXArray,
        textState: MLXArray,
        textMask: MLXArray,
        speakerState: MLXArray?,
        speakerMask: MLXArray?,
        kvText: [IrodoriKVCache]? = nil,
        kvSpeaker: [IrodoriKVCache]? = nil,
        startPos: Int = 0,
        captionState: MLXArray? = nil,
        captionMask: MLXArray? = nil,
        kvCaption: [IrodoriKVCache]? = nil
    ) throws -> MLXArray {
        let tEmbed = echoTtsTimestepEmbedding(t, embedSize: cfg.timestepEmbedDim).asType(xT.dtype)
        let condEmbed = condModule(tEmbed).expandedDimensions(axis: 1) // (B, 1, 3*modelDim)

        var x = inProj(xT)
        let freqs = echoTtsPrecomputeFreqsCis(dim: headDim, end: startPos + x.shape[1])

        let useSpk = cfg.useSpeakerConditionResolved
        let useCap = cfg.useCaptionCondition

        // For caption-only models: speakerState arg carries caption context (backward compat)
        let actualSpeakerState: MLXArray?
        let actualSpeakerMask: MLXArray?
        let actualKVSpeaker: [IrodoriKVCache]?
        let actualCaptionState: MLXArray?
        let actualCaptionMask: MLXArray?
        let actualKVCaption: [IrodoriKVCache]?
        if !useSpk && useCap {
            actualCaptionState = captionState ?? speakerState
            actualCaptionMask = captionMask ?? speakerMask
            actualKVCaption = kvCaption ?? kvSpeaker
            actualSpeakerState = nil
            actualSpeakerMask = nil
            actualKVSpeaker = nil
        } else {
            actualSpeakerState = speakerState
            actualSpeakerMask = speakerMask
            actualKVSpeaker = kvSpeaker
            actualCaptionState = captionState
            actualCaptionMask = captionMask
            actualKVCaption = kvCaption
        }

        for (index, block) in blocks.enumerated() {
            let kvT = kvText?[index] ?? block.attention.getKVCacheText(textState)
            var kvS: IrodoriKVCache?
            if useSpk, let actualSpeakerState {
                kvS = try actualKVSpeaker?[index]
                    ?? block.attention.getKVCacheSpeaker(actualSpeakerState)
            }
            var kvC: IrodoriKVCache?
            if useCap, let actualCaptionState {
                kvC = try actualKVCaption?[index]
                    ?? block.attention.getKVCacheCaption(actualCaptionState)
            }
            x = block(
                x,
                condEmbed: condEmbed,
                textMask: textMask,
                freqsCis: freqs,
                kvCacheText: kvT,
                kvCacheSpeaker: kvS,
                speakerMask: actualSpeakerMask,
                kvCacheCaption: kvC,
                captionMask: actualCaptionMask,
                startPos: startPos
            )
        }

        x = outNorm(x)
        return outProj(x).asType(.float32)
    }
}
