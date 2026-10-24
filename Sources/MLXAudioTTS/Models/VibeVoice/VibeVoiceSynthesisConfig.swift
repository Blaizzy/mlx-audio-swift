// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLXAudioCore

public struct VibeVoiceAcousticConfig: Codable, Sendable {
    public var channels = 1
    public var causal = true
    public var vaeDim = 64
    public var encoderNFilters = 32
    public var encoderRatios = [8, 5, 5, 4, 2, 2]
    public var encoderDepths = [3, 3, 3, 3, 3, 3, 8]
    public var decoderNFilters = 32
    public var decoderRatios: [Int]?
    public var decoderDepths: [Int]?
    public var convBias = true
    public var disableLastNorm = true
    public var layernormEps: Float = 1e-5
    public var layerScaleInitValue: Float = 1e-6
    public var fixStd: Float = 0.5
    public var hopLength: Int { (decoderRatios ?? encoderRatios).reduce(1, *) }
    public init() {}
    enum CodingKeys: String, CodingKey {
        case channels, causal
        case vaeDim = "vae_dim", encoderNFilters = "encoder_n_filters"
        case encoderRatios = "encoder_ratios", encoderDepths = "encoder_depths"
        case decoderNFilters = "decoder_n_filters", decoderRatios = "decoder_ratios"
        case decoderDepths = "decoder_depths", convBias = "conv_bias"
        case disableLastNorm = "disable_last_norm", layernormEps = "layernorm_eps"
        case layerScaleInitValue = "layer_scale_init_value", fixStd = "fix_std"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? channels
        causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? causal
        vaeDim = try c.decodeIfPresent(Int.self, forKey: .vaeDim) ?? vaeDim
        encoderNFilters = try c.decodeIfPresent(Int.self, forKey: .encoderNFilters) ?? encoderNFilters
        encoderRatios = try c.decodeIfPresent([Int].self, forKey: .encoderRatios) ?? encoderRatios
        func depths(_ key: CodingKeys) throws -> [Int]? {
            if let text = try? c.decode(String.self, forKey: key) { return text.split(separator: "-").compactMap { Int($0) } }
            return try c.decodeIfPresent([Int].self, forKey: key)
        }
        encoderDepths = try depths(.encoderDepths) ?? encoderDepths
        decoderDepths = try depths(.decoderDepths)
        decoderNFilters = try c.decodeIfPresent(Int.self, forKey: .decoderNFilters) ?? encoderNFilters
        decoderRatios = try c.decodeIfPresent([Int].self, forKey: .decoderRatios)
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? convBias
        disableLastNorm = try c.decodeIfPresent(Bool.self, forKey: .disableLastNorm) ?? disableLastNorm
        layernormEps = try c.decodeIfPresent(Float.self, forKey: .layernormEps) ?? layernormEps
        layerScaleInitValue = try c.decodeIfPresent(Float.self, forKey: .layerScaleInitValue) ?? layerScaleInitValue
        fixStd = try c.decodeIfPresent(Float.self, forKey: .fixStd) ?? fixStd
    }
}
public struct VibeVoiceDiffusionConfig: Codable, Sendable {
    public var hiddenSize = 896
    public var headLayers = 4
    public var headFFNRatio: Float = 3
    public var rmsNormEps: Float = 1e-5
    public var latentSize = 64
    public var predictionType = "v_prediction"
    public var ddpmNumSteps = 1000
    public var ddpmNumInferenceSteps = 20
    public var ddpmBetaSchedule = "cosine"
    public init() {}
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", headLayers = "head_layers", headFFNRatio = "head_ffn_ratio"
        case rmsNormEps = "rms_norm_eps", latentSize = "latent_size", predictionType = "prediction_type"
        case ddpmNumSteps = "ddpm_num_steps", ddpmNumInferenceSteps = "ddpm_num_inference_steps"
        case ddpmBetaSchedule = "ddpm_beta_schedule"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        headLayers = try c.decodeIfPresent(Int.self, forKey: .headLayers) ?? headLayers
        headFFNRatio = try c.decodeIfPresent(Float.self, forKey: .headFFNRatio) ?? headFFNRatio
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
        latentSize = try c.decodeIfPresent(Int.self, forKey: .latentSize) ?? latentSize
        predictionType = try c.decodeIfPresent(String.self, forKey: .predictionType) ?? predictionType
        ddpmNumSteps = try c.decodeIfPresent(Int.self, forKey: .ddpmNumSteps) ?? ddpmNumSteps
        ddpmNumInferenceSteps = try c.decodeIfPresent(Int.self, forKey: .ddpmNumInferenceSteps) ?? ddpmNumInferenceSteps
        ddpmBetaSchedule = try c.decodeIfPresent(String.self, forKey: .ddpmBetaSchedule) ?? ddpmBetaSchedule
    }
}
