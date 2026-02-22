//
//  T3CondEnc.swift
//  MLXAudio
//
//  T3 conditioning encoder: speaker + emotion + prompt speech tokens.
//  Ported from mlx-audio Python: chatterbox/t3/cond_enc.py
//

import Foundation
import MLX
import MLXNN

// MARK: - T3 Conditioning Data

/// Container for T3 conditioning information.
public struct T3Cond {
    /// Speaker embedding from voice encoder (B, speakerDim).
    public var speakerEmb: MLXArray

    /// Optional CLAP embedding for semantic conditioning.
    public var clapEmb: MLXArray?

    /// Optional speech token prompt (B, T).
    public var condPromptSpeechTokens: MLXArray?

    /// Optional embedded speech prompt (B, T, D).
    public var condPromptSpeechEmb: MLXArray?

    /// Emotion exaggeration factor, typically 0.3–0.7.
    public var emotionAdv: MLXArray

    public init(
        speakerEmb: MLXArray,
        clapEmb: MLXArray? = nil,
        condPromptSpeechTokens: MLXArray? = nil,
        condPromptSpeechEmb: MLXArray? = nil,
        emotionAdv: MLXArray? = nil
    ) {
        self.speakerEmb = speakerEmb
        self.clapEmb = clapEmb
        self.condPromptSpeechTokens = condPromptSpeechTokens
        self.condPromptSpeechEmb = condPromptSpeechEmb
        self.emotionAdv = emotionAdv ?? MLXArray(Float(0.5))
    }
}

// MARK: - T3 Conditioning Encoder

/// Conditioning encoder for T3 model.
/// Handles speaker embeddings, emotion control, and prompt speech tokens.
public class T3CondEnc: Module {
    let hp: T3Configuration

    @ModuleInfo(key: "spkr_enc") var spkrEnc: Linear
    @ModuleInfo(key: "emotion_adv_fc") var emotionAdvFc: Linear?
    @ModuleInfo var perceiver: Perceiver?

    public init(_ hp: T3Configuration) {
        self.hp = hp

        // Speaker embedding projection
        self._spkrEnc.wrappedValue = Linear(hp.speakerEmbedSize, hp.nChannels)

        // Emotion control
        if hp.emotionAdv {
            self._emotionAdvFc.wrappedValue = Linear(1, hp.nChannels, bias: false)
        } else {
            self._emotionAdvFc.wrappedValue = nil
        }

        // Perceiver resampler for prompt speech tokens
        if hp.usePerceiverResampler {
            self._perceiver.wrappedValue = Perceiver()
        } else {
            self._perceiver.wrappedValue = nil
        }
    }

    /// Process conditioning inputs into a single conditioning tensor.
    ///
    /// - Parameter cond: T3Cond with conditioning information.
    /// - Returns: Conditioning embeddings (B, condLen, D).
    public func callAsFunction(_ cond: T3Cond) -> MLXArray {
        let b = cond.speakerEmb.dim(0)

        // Speaker embedding projection (B, speakerDim) → (B, 1, D)
        let speakerReshaped = cond.speakerEmb.reshaped([b, hp.speakerEmbedSize])
        var condSpkr = spkrEnc(speakerReshaped)
        condSpkr = condSpkr.expandedDimensions(axis: 1) // (B, 1, D)

        // Empty placeholder for concatenation — (B, 0, D)
        let empty = condSpkr[0..., ..<0, 0...]

        // CLAP embedding (not implemented)
        let condClap = empty

        // Conditional prompt speech embeddings
        var condPromptSpeechEmb: MLXArray
        if let speechEmb = cond.condPromptSpeechEmb {
            if hp.usePerceiverResampler, let perceiver = perceiver {
                condPromptSpeechEmb = perceiver(speechEmb)
            } else {
                condPromptSpeechEmb = speechEmb
            }
        } else {
            condPromptSpeechEmb = empty
        }

        // Emotion exaggeration
        var condEmotionAdv = empty
        if hp.emotionAdv, let emotionFc = emotionAdvFc {
            var emotionVal = cond.emotionAdv
            if emotionVal.ndim == 0 {
                emotionVal = emotionVal.reshaped([1, 1, 1])
            } else if emotionVal.ndim == 1 {
                emotionVal = emotionVal.reshaped([-1, 1, 1])
            } else if emotionVal.ndim == 2 {
                emotionVal = emotionVal.expandedDimensions(axis: -1)
            }
            condEmotionAdv = emotionFc(emotionVal)
        }

        // Concatenate all conditioning signals along sequence dimension
        let condEmbeds = MLX.concatenated(
            [condSpkr, condClap, condPromptSpeechEmb, condEmotionAdv],
            axis: 1
        )

        return condEmbeds
    }
}
