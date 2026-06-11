//
//  VoxCPM2Config.swift
//  MLXAudio
//
//  VoxCPM2 TTS configuration.
//  Ported from mlx-audio Python: voxcpm2/config.py
//

import Foundation
import MLXLMCommon

// MARK: - LM Configuration

public struct VoxCPM2LMConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var intermediateSize: Int
    public var vocabSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var ropeScalingType: String
    public var ropeLongFactor: [Float]
    public var ropeShortFactor: [Float]
    public var scaleEmb: Int
    public var dimModelBase: Int
    public var scaleDepth: Float
    public var originalMaxPositionEmbeddings: Int
    public var maxPositionEmbeddings: Int
    public var bosTokenId: Int
    public var eosTokenId: Int
    public var useMup: Bool
    public var kvChannels: Int?
    public var noRope: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeScalingType = "rope_scaling_type"
        case ropeLongFactor = "rope_long_factor"
        case ropeShortFactor = "rope_short_factor"
        case scaleEmb = "scale_emb"
        case dimModelBase = "dim_model_base"
        case scaleDepth = "scale_depth"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case useMup = "use_mup"
        case kvChannels = "kv_channels"
        case noRope = "no_rope"
    }

    public init(
        hiddenSize: Int = 1024,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 2,
        intermediateSize: Int = 4096,
        vocabSize: Int = 73448,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10000.0,
        ropeScalingType: String = "longrope",
        ropeLongFactor: [Float] = [],
        ropeShortFactor: [Float] = [],
        scaleEmb: Int = 12,
        dimModelBase: Int = 256,
        scaleDepth: Float = 1.4,
        originalMaxPositionEmbeddings: Int = 32768,
        maxPositionEmbeddings: Int = 32768,
        bosTokenId: Int = 1,
        eosTokenId: Int = 2,
        useMup: Bool = true,
        kvChannels: Int? = nil,
        noRope: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScalingType = ropeScalingType
        self.ropeLongFactor = ropeLongFactor
        self.ropeShortFactor = ropeShortFactor
        self.scaleEmb = scaleEmb
        self.dimModelBase = dimModelBase
        self.scaleDepth = scaleDepth
        self.originalMaxPositionEmbeddings = originalMaxPositionEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.useMup = useMup
        self.kvChannels = kvChannels
        self.noRope = noRope
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        self.numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 73448
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        self.scaleEmb = try container.decodeIfPresent(Int.self, forKey: .scaleEmb) ?? 12
        self.dimModelBase = try container.decodeIfPresent(Int.self, forKey: .dimModelBase) ?? 256
        self.scaleDepth = try container.decodeIfPresent(Float.self, forKey: .scaleDepth) ?? 1.4
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        self.eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        self.useMup = try container.decodeIfPresent(Bool.self, forKey: .useMup) ?? true
        self.kvChannels = try container.decodeIfPresent(Int.self, forKey: .kvChannels)
        self.noRope = try container.decodeIfPresent(Bool.self, forKey: .noRope) ?? false

        // rope_scaling can be a nested dict or flat fields
        if let ropeScaling = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling) {
            self.ropeScalingType = ropeScaling.type
            self.ropeLongFactor = ropeScaling.longFactor
            self.ropeShortFactor = ropeScaling.shortFactor
            self.originalMaxPositionEmbeddings = ropeScaling.originalMaxPositionEmbeddings ?? 32768
        } else {
            self.ropeScalingType = try container.decodeIfPresent(String.self, forKey: .ropeScalingType) ?? "longrope"
            self.ropeLongFactor = try container.decodeIfPresent([Float].self, forKey: .ropeLongFactor) ?? []
            self.ropeShortFactor = try container.decodeIfPresent([Float].self, forKey: .ropeShortFactor) ?? []
            self.originalMaxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .originalMaxPositionEmbeddings) ?? 32768
        }
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try container.encode(numKeyValueHeads, forKey: .numKeyValueHeads)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(ropeScalingType, forKey: .ropeScalingType)
        try container.encode(ropeLongFactor, forKey: .ropeLongFactor)
        try container.encode(ropeShortFactor, forKey: .ropeShortFactor)
        try container.encode(scaleEmb, forKey: .scaleEmb)
        try container.encode(dimModelBase, forKey: .dimModelBase)
        try container.encode(scaleDepth, forKey: .scaleDepth)
        try container.encode(originalMaxPositionEmbeddings, forKey: .originalMaxPositionEmbeddings)
        try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(bosTokenId, forKey: .bosTokenId)
        try container.encode(eosTokenId, forKey: .eosTokenId)
        try container.encode(useMup, forKey: .useMup)
        try container.encodeIfPresent(kvChannels, forKey: .kvChannels)
        try container.encode(noRope, forKey: .noRope)
    }

    /// Head dimension for attention.
    public var headDim: Int {
        kvChannels ?? (hiddenSize / numAttentionHeads)
    }
}

private struct RopeScaling: Codable {
    var type: String
    var longFactor: [Float]
    var shortFactor: [Float]
    var originalMaxPositionEmbeddings: Int?

    enum CodingKeys: String, CodingKey {
        case type
        case longFactor = "long_factor"
        case shortFactor = "short_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }
}

// MARK: - Encoder Configuration

public struct VoxCPM2EncoderConfig: Codable, Sendable {
    public var hiddenDim: Int
    public var ffnDim: Int
    public var numHeads: Int
    public var numLayers: Int
    public var kvChannels: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenDim = "hidden_dim"
        case ffnDim = "ffn_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case kvChannels = "kv_channels"
    }

    public init(
        hiddenDim: Int = 1024,
        ffnDim: Int = 4096,
        numHeads: Int = 16,
        numLayers: Int = 4,
        kvChannels: Int? = nil
    ) {
        self.hiddenDim = hiddenDim
        self.ffnDim = ffnDim
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.kvChannels = kvChannels
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenDim = try container.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 1024
        self.ffnDim = try container.decodeIfPresent(Int.self, forKey: .ffnDim) ?? 4096
        self.numHeads = try container.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
        self.numLayers = try container.decodeIfPresent(Int.self, forKey: .numLayers) ?? 4
        self.kvChannels = try container.decodeIfPresent(Int.self, forKey: .kvChannels)
    }
}

// MARK: - CFM Configuration

public struct VoxCPM2CFMConfig: Codable, Sendable {
    public var sigmaMin: Float
    public var solver: String
    public var tScheduler: String
    public var inferenceCfgRate: Float

    enum CodingKeys: String, CodingKey {
        case sigmaMin = "sigma_min"
        case solver
        case tScheduler = "t_scheduler"
        case inferenceCfgRate = "inference_cfg_rate"
    }

    public init(
        sigmaMin: Float = 1e-6,
        solver: String = "euler",
        tScheduler: String = "log-norm",
        inferenceCfgRate: Float = 2.0
    ) {
        self.sigmaMin = sigmaMin
        self.solver = solver
        self.tScheduler = tScheduler
        self.inferenceCfgRate = inferenceCfgRate
    }
}

// MARK: - DiT Configuration

public struct VoxCPM2DiTConfig: Codable, Sendable {
    public var hiddenDim: Int
    public var ffnDim: Int
    public var numHeads: Int
    public var numLayers: Int
    public var kvChannels: Int?
    public var ditMeanMode: Bool
    public var cfmConfig: VoxCPM2CFMConfig

    enum CodingKeys: String, CodingKey {
        case hiddenDim = "hidden_dim"
        case ffnDim = "ffn_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case kvChannels = "kv_channels"
        case ditMeanMode = "dit_mean_mode"
        case meanMode = "mean_mode"
        case cfmConfig = "cfm_config"
    }

    public init(
        hiddenDim: Int = 1024,
        ffnDim: Int = 4096,
        numHeads: Int = 16,
        numLayers: Int = 4,
        kvChannels: Int? = nil,
        ditMeanMode: Bool = false,
        cfmConfig: VoxCPM2CFMConfig = VoxCPM2CFMConfig()
    ) {
        self.hiddenDim = hiddenDim
        self.ffnDim = ffnDim
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.kvChannels = kvChannels
        self.ditMeanMode = ditMeanMode
        self.cfmConfig = cfmConfig
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenDim = try container.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 1024
        self.ffnDim = try container.decodeIfPresent(Int.self, forKey: .ffnDim) ?? 4096
        self.numHeads = try container.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
        self.numLayers = try container.decodeIfPresent(Int.self, forKey: .numLayers) ?? 4
        self.kvChannels = try container.decodeIfPresent(Int.self, forKey: .kvChannels)
        self.cfmConfig = try container.decodeIfPresent(VoxCPM2CFMConfig.self, forKey: .cfmConfig) ?? VoxCPM2CFMConfig()

        // Handle "mean_mode" → "dit_mean_mode" naming
        if let ditMM = try container.decodeIfPresent(Bool.self, forKey: .ditMeanMode) {
            self.ditMeanMode = ditMM
        } else {
            self.ditMeanMode = try container.decodeIfPresent(Bool.self, forKey: .meanMode) ?? false
        }
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hiddenDim, forKey: .hiddenDim)
        try container.encode(ffnDim, forKey: .ffnDim)
        try container.encode(numHeads, forKey: .numHeads)
        try container.encode(numLayers, forKey: .numLayers)
        try container.encodeIfPresent(kvChannels, forKey: .kvChannels)
        try container.encode(ditMeanMode, forKey: .ditMeanMode)
        try container.encode(cfmConfig, forKey: .cfmConfig)
    }
}

// MARK: - AudioVAE Configuration

public struct VoxCPM2AudioVAEConfig: Codable, Sendable {
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var depthwise: Bool
    public var sampleRate: Int
    public var outSampleRate: Int
    public var useNoiseBlock: Bool
    public var srBinBoundaries: [Int]?
    public var condType: String
    public var condDim: Int
    public var condOutLayer: Bool

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim"
        case encoderRates = "encoder_rates"
        case latentDim = "latent_dim"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case depthwise
        case sampleRate = "sample_rate"
        case outSampleRate = "out_sample_rate"
        case useNoiseBlock = "use_noise_block"
        case srBinBoundaries = "sr_bin_boundaries"
        case condType = "cond_type"
        case condDim = "cond_dim"
        case condOutLayer = "cond_out_layer"
    }

    public init(
        encoderDim: Int = 128,
        encoderRates: [Int] = [2, 5, 8, 8],
        latentDim: Int = 64,
        decoderDim: Int = 2048,
        decoderRates: [Int] = [8, 6, 5, 2, 2, 2],
        depthwise: Bool = true,
        sampleRate: Int = 16000,
        outSampleRate: Int = 48000,
        useNoiseBlock: Bool = false,
        srBinBoundaries: [Int]? = [20000, 30000, 40000],
        condType: String = "scale_bias",
        condDim: Int = 128,
        condOutLayer: Bool = false
    ) {
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.latentDim = latentDim
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.depthwise = depthwise
        self.sampleRate = sampleRate
        self.outSampleRate = outSampleRate
        self.useNoiseBlock = useNoiseBlock
        self.srBinBoundaries = srBinBoundaries
        self.condType = condType
        self.condDim = condDim
        self.condOutLayer = condOutLayer
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.encoderDim = try container.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 128
        self.encoderRates = try container.decodeIfPresent([Int].self, forKey: .encoderRates) ?? [2, 5, 8, 8]
        self.latentDim = try container.decodeIfPresent(Int.self, forKey: .latentDim) ?? 64
        self.decoderDim = try container.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 2048
        self.decoderRates = try container.decodeIfPresent([Int].self, forKey: .decoderRates) ?? [8, 6, 5, 2, 2, 2]
        self.depthwise = try container.decodeIfPresent(Bool.self, forKey: .depthwise) ?? true
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16000
        self.outSampleRate = try container.decodeIfPresent(Int.self, forKey: .outSampleRate) ?? 48000
        self.useNoiseBlock = try container.decodeIfPresent(Bool.self, forKey: .useNoiseBlock) ?? false
        self.srBinBoundaries = try container.decodeIfPresent([Int].self, forKey: .srBinBoundaries)
        self.condType = try container.decodeIfPresent(String.self, forKey: .condType) ?? "scale_bias"
        self.condDim = try container.decodeIfPresent(Int.self, forKey: .condDim) ?? 128
        self.condOutLayer = try container.decodeIfPresent(Bool.self, forKey: .condOutLayer) ?? false
    }
}

// MARK: - Top-Level Model Configuration

public struct VoxCPM2Configuration: Codable, Sendable {
    public var modelType: String
    public var lmConfig: VoxCPM2LMConfig
    public var encoderConfig: VoxCPM2EncoderConfig
    public var ditConfig: VoxCPM2DiTConfig
    public var audioVaeConfig: VoxCPM2AudioVAEConfig
    public var patchSize: Int
    public var featDim: Int
    public var scalarQuantizationLatentDim: Int
    public var scalarQuantizationScale: Int
    public var residualLmNumLayers: Int
    public var residualLmNoRope: Bool
    public var maxLength: Int
    public var inferenceTimesteps: Int
    public var cfgScale: Float

    public var quantization: BaseConfiguration.Quantization?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case lmConfig = "lm_config"
        case encoderConfig = "encoder_config"
        case ditConfig = "dit_config"
        case audioVaeConfig = "audio_vae_config"
        case patchSize = "patch_size"
        case featDim = "feat_dim"
        case scalarQuantizationLatentDim = "scalar_quantization_latent_dim"
        case scalarQuantizationScale = "scalar_quantization_scale"
        case residualLmNumLayers = "residual_lm_num_layers"
        case residualLmNoRope = "residual_lm_no_rope"
        case maxLength = "max_length"
        case inferenceTimesteps = "inference_timesteps"
        case cfgScale = "cfg_scale"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(
        modelType: String = "voxcpm2",
        lmConfig: VoxCPM2LMConfig = VoxCPM2LMConfig(),
        encoderConfig: VoxCPM2EncoderConfig = VoxCPM2EncoderConfig(),
        ditConfig: VoxCPM2DiTConfig = VoxCPM2DiTConfig(),
        audioVaeConfig: VoxCPM2AudioVAEConfig = VoxCPM2AudioVAEConfig(),
        patchSize: Int = 4,
        featDim: Int = 64,
        scalarQuantizationLatentDim: Int = 512,
        scalarQuantizationScale: Int = 9,
        residualLmNumLayers: Int = 8,
        residualLmNoRope: Bool = false,
        maxLength: Int = 8192,
        inferenceTimesteps: Int = 10,
        cfgScale: Float = 2.0,
        quantization: BaseConfiguration.Quantization? = nil,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.modelType = modelType
        self.lmConfig = lmConfig
        self.encoderConfig = encoderConfig
        self.ditConfig = ditConfig
        self.audioVaeConfig = audioVaeConfig
        self.patchSize = patchSize
        self.featDim = featDim
        self.scalarQuantizationLatentDim = scalarQuantizationLatentDim
        self.scalarQuantizationScale = scalarQuantizationScale
        self.residualLmNumLayers = residualLmNumLayers
        self.residualLmNoRope = residualLmNoRope
        self.maxLength = maxLength
        self.inferenceTimesteps = inferenceTimesteps
        self.cfgScale = cfgScale
        self.quantization = quantization
        self.perLayerQuantization = perLayerQuantization
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "voxcpm2"
        self.lmConfig = try container.decodeIfPresent(VoxCPM2LMConfig.self, forKey: .lmConfig) ?? VoxCPM2LMConfig()
        self.encoderConfig = try container.decodeIfPresent(VoxCPM2EncoderConfig.self, forKey: .encoderConfig) ?? VoxCPM2EncoderConfig()
        self.ditConfig = try container.decodeIfPresent(VoxCPM2DiTConfig.self, forKey: .ditConfig) ?? VoxCPM2DiTConfig()
        self.audioVaeConfig = try container.decodeIfPresent(VoxCPM2AudioVAEConfig.self, forKey: .audioVaeConfig) ?? VoxCPM2AudioVAEConfig()
        self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 4
        self.featDim = try container.decodeIfPresent(Int.self, forKey: .featDim) ?? 64
        self.scalarQuantizationLatentDim = try container.decodeIfPresent(Int.self, forKey: .scalarQuantizationLatentDim) ?? 512
        self.scalarQuantizationScale = try container.decodeIfPresent(Int.self, forKey: .scalarQuantizationScale) ?? 9
        self.residualLmNumLayers = try container.decodeIfPresent(Int.self, forKey: .residualLmNumLayers) ?? 8
        self.residualLmNoRope = try container.decodeIfPresent(Bool.self, forKey: .residualLmNoRope) ?? false
        self.maxLength = try container.decodeIfPresent(Int.self, forKey: .maxLength) ?? 8192
        self.inferenceTimesteps = try container.decodeIfPresent(Int.self, forKey: .inferenceTimesteps) ?? 10
        self.cfgScale = try container.decodeIfPresent(Float.self, forKey: .cfgScale) ?? 2.0

        let baseConfig = try? BaseConfiguration(from: decoder)
        let globalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        let altGlobalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
        self.quantization = globalQuant ?? altGlobalQuant ?? baseConfig?.perLayerQuantization?.quantization
        self.perLayerQuantization = baseConfig?.perLayerQuantization
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(lmConfig, forKey: .lmConfig)
        try container.encode(encoderConfig, forKey: .encoderConfig)
        try container.encode(ditConfig, forKey: .ditConfig)
        try container.encode(audioVaeConfig, forKey: .audioVaeConfig)
        try container.encode(patchSize, forKey: .patchSize)
        try container.encode(featDim, forKey: .featDim)
        try container.encode(scalarQuantizationLatentDim, forKey: .scalarQuantizationLatentDim)
        try container.encode(scalarQuantizationScale, forKey: .scalarQuantizationScale)
        try container.encode(residualLmNumLayers, forKey: .residualLmNumLayers)
        try container.encode(residualLmNoRope, forKey: .residualLmNoRope)
        try container.encode(maxLength, forKey: .maxLength)
        try container.encode(inferenceTimesteps, forKey: .inferenceTimesteps)
        try container.encode(cfgScale, forKey: .cfgScale)
        try container.encodeIfPresent(quantization, forKey: .quantization)
    }
}
