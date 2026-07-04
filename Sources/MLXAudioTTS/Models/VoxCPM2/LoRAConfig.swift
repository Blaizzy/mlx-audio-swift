// LoRA configuration for VoxCPM2 fine-tuning.
//
// Matches the official Python API's LoRA parameter schema:
//   lora:
//     enable_lm:   true   — attention layers in base_lm + residual_lm
//     enable_dit:  true   — attention layers in DiT decoder (critical for voice quality)
//     enable_proj: false  — projection heads
//     r:     8            — rank (8 for speaker adaptation, 32–64 for new languages)
//     alpha: 16
//     dropout: 0.0

import Foundation

/// Configuration for LoRA (Low-Rank Adaptation) fine-tuning of a VoxCPM2 model.
///
/// LoRA freezes the base model and trains only small rank‑delta matrices,
/// making it the recommended default starting point for voice adaptation.
///
/// Usage:
/// ```swift
/// let loraConfig = LoRAConfig(r: 8, alpha: 16)
/// let model = try await VoxCPM2Model.fromModelDirectory(
///     modelDir,
///     loraConfig: loraConfig
/// )
/// try model.loadLoRA(weightsPath: "path/to/lora_weights.safetensors")
/// model.setLoRAEnabled(true)
/// ```
public struct LoRAConfig: Codable, Sendable {
    /// Apply LoRA to the attention layers of `base_lm` and `residual_lm`.
    ///
    /// Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj` in every decoder layer
    /// of both language models.
    public var enableLM: Bool = true

    /// Apply LoRA to the attention layers of the DiT decoder.
    ///
    /// Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj` in the
    /// `feat_decoder.estimator.decoder` (the diffusion‑backbone transformer).
    ///
    /// > Important: This is **critical for voice quality** — do not disable
    /// > when fine‑tuning for voice cloning or timbre adaptation.
    public var enableDiT: Bool = true

    /// Apply LoRA to the linear projection heads.
    ///
    /// Targets: `enc_to_lm_proj`, `lm_to_dit_proj`, `res_to_dit_proj`,
    /// `fusion_concat_proj`, `stop_proj`, `stop_head`.
    ///
    /// Usually left disabled — the projection heads are small and full‑rank
    /// fine‑tuning suffices.
    public var enableProj: Bool = false

    /// LoRA rank. Lower = fewer parameters, faster training.
    ///
    /// - `r = 8`:  sufficient for speaker/voice adaptation
    /// - `r = 32–64`: recommended for new languages or large domain shifts
    public var r: Int = 8

    /// LoRA alpha scaling factor. The effective scaling is `alpha / r`.
    ///
    /// A common starting point is `alpha = 16` for `r = 8` (ratio 2.0).
    public var alpha: Int = 16

    /// Dropout rate applied to the LoRA branch during training (0.0–1.0).
    public var dropout: Float = 0.0

    public init(
        enableLM: Bool = true,
        enableDiT: Bool = true,
        enableProj: Bool = false,
        r: Int = 8,
        alpha: Int = 16,
        dropout: Float = 0.0
    ) {
        self.enableLM = enableLM
        self.enableDiT = enableDiT
        self.enableProj = enableProj
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
    }

    enum CodingKeys: String, CodingKey {
        case enableLM = "enable_lm"
        case enableDiT = "enable_dit"
        case enableProj = "enable_proj"
        case r
        case alpha
        case dropout
    }
}
