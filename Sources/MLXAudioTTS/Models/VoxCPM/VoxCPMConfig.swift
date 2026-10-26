import Foundation
import MLXAudioCore

/// Configuration for the VoxCPM 1 and 1.5 architecture.
public struct VoxCPMConfiguration: Codable, Sendable {
    public var lmConfig: VoxCPM2LMConfig
    public var encoderConfig: VoxCPM2EncoderConfig
    public var ditConfig: VoxCPM2DiTConfig
    public var audioVaeConfig: VoxCPMAudioVAEConfig
    public var patchSize: Int
    public var featDim: Int
    public var scalarQuantizationLatentDim: Int
    public var scalarQuantizationScale: Int
    public var residualLmNumLayers: Int
    public var maxLength: Int
    public var quantization: BaseConfiguration.Quantization?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case architecture
        case lmConfig = "lm_config", encoderConfig = "encoder_config", ditConfig = "dit_config"
        case audioVaeConfig = "audio_vae_config", patchSize = "patch_size", featDim = "feat_dim"
        case scalarQuantizationLatentDim = "scalar_quantization_latent_dim"
        case scalarQuantizationScale = "scalar_quantization_scale"
        case residualLmNumLayers = "residual_lm_num_layers", maxLength = "max_length"
        case quantization
        case quantizationConfig = "quantization_config"
    }
    private enum NestedKeys: String, CodingKey {
        case useMup = "use_mup", numLayers = "num_layers"
    }

    public init(
        lmConfig: VoxCPM2LMConfig = VoxCPM2LMConfig(useMup: false),
        encoderConfig: VoxCPM2EncoderConfig = VoxCPM2EncoderConfig(numLayers: 8),
        ditConfig: VoxCPM2DiTConfig = VoxCPM2DiTConfig(numLayers: 8),
        audioVaeConfig: VoxCPMAudioVAEConfig = VoxCPMAudioVAEConfig(),
        patchSize: Int = 4, featDim: Int = 64,
        scalarQuantizationLatentDim: Int = 256, scalarQuantizationScale: Int = 9,
        residualLmNumLayers: Int = 8, maxLength: Int = 8192,
        quantization: BaseConfiguration.Quantization? = nil,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.lmConfig = lmConfig
        self.encoderConfig = encoderConfig
        self.ditConfig = ditConfig
        self.audioVaeConfig = audioVaeConfig
        self.patchSize = patchSize
        self.featDim = featDim
        self.scalarQuantizationLatentDim = scalarQuantizationLatentDim
        self.scalarQuantizationScale = scalarQuantizationScale
        self.residualLmNumLayers = residualLmNumLayers
        self.maxLength = maxLength
        self.quantization = quantization
        self.perLayerQuantization = perLayerQuantization
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init()
        lmConfig = try c.decodeIfPresent(VoxCPM2LMConfig.self, forKey: .lmConfig) ?? lmConfig
        if c.contains(.lmConfig) {
            let nested = try c.nestedContainer(keyedBy: NestedKeys.self, forKey: .lmConfig)
            lmConfig.useMup = try nested.decodeIfPresent(Bool.self, forKey: .useMup) ?? false
        }
        encoderConfig = try c.decodeIfPresent(VoxCPM2EncoderConfig.self, forKey: .encoderConfig) ?? encoderConfig
        if c.contains(.encoderConfig) {
            let nested = try c.nestedContainer(keyedBy: NestedKeys.self, forKey: .encoderConfig)
            encoderConfig.numLayers = try nested.decodeIfPresent(Int.self, forKey: .numLayers) ?? 8
        }
        ditConfig = try c.decodeIfPresent(VoxCPM2DiTConfig.self, forKey: .ditConfig) ?? ditConfig
        if c.contains(.ditConfig) {
            let nested = try c.nestedContainer(keyedBy: NestedKeys.self, forKey: .ditConfig)
            ditConfig.numLayers = try nested.decodeIfPresent(Int.self, forKey: .numLayers) ?? 8
        }
        audioVaeConfig = try c.decodeIfPresent(VoxCPMAudioVAEConfig.self, forKey: .audioVaeConfig) ?? audioVaeConfig
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
        featDim = try c.decodeIfPresent(Int.self, forKey: .featDim) ?? featDim
        scalarQuantizationLatentDim = try c.decodeIfPresent(Int.self, forKey: .scalarQuantizationLatentDim) ?? scalarQuantizationLatentDim
        scalarQuantizationScale = try c.decodeIfPresent(Int.self, forKey: .scalarQuantizationScale) ?? scalarQuantizationScale
        residualLmNumLayers = try c.decodeIfPresent(Int.self, forKey: .residualLmNumLayers) ?? residualLmNumLayers
        maxLength = try c.decodeIfPresent(Int.self, forKey: .maxLength) ?? maxLength
        let base = try? BaseConfiguration(from: decoder)
        perLayerQuantization = base?.perLayerQuantization
        let global = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        let alternative = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
        quantization = global ?? alternative ?? perLayerQuantization?.quantization
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode("voxcpm", forKey: .architecture)
        try c.encode(lmConfig, forKey: .lmConfig)
        try c.encode(encoderConfig, forKey: .encoderConfig)
        try c.encode(ditConfig, forKey: .ditConfig)
        try c.encode(audioVaeConfig, forKey: .audioVaeConfig)
        try c.encode(patchSize, forKey: .patchSize)
        try c.encode(featDim, forKey: .featDim)
        try c.encode(scalarQuantizationLatentDim, forKey: .scalarQuantizationLatentDim)
        try c.encode(scalarQuantizationScale, forKey: .scalarQuantizationScale)
        try c.encode(residualLmNumLayers, forKey: .residualLmNumLayers)
        try c.encode(maxLength, forKey: .maxLength)
        try c.encodeIfPresent(quantization, forKey: .quantization)
    }
}

public struct VoxCPMAudioVAEConfig: Codable, Sendable {
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var sampleRate: Int

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim", encoderRates = "encoder_rates", latentDim = "latent_dim"
        case decoderDim = "decoder_dim", decoderRates = "decoder_rates", sampleRate = "sample_rate"
    }

    public init(encoderDim: Int = 64, encoderRates: [Int] = [2, 3, 6, 7, 7],
                latentDim: Int = 64, decoderDim: Int = 2048,
                decoderRates: [Int] = [7, 7, 6, 3, 2], sampleRate: Int = 44100) {
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.latentDim = latentDim
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.sampleRate = sampleRate
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init()
        encoderDim = try c.decodeIfPresent(Int.self, forKey: .encoderDim) ?? encoderDim
        encoderRates = try c.decodeIfPresent([Int].self, forKey: .encoderRates) ?? encoderRates
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? latentDim
        decoderDim = try c.decodeIfPresent(Int.self, forKey: .decoderDim) ?? decoderDim
        decoderRates = try c.decodeIfPresent([Int].self, forKey: .decoderRates) ?? decoderRates
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? sampleRate
    }

    var sharedConfiguration: VoxCPM2AudioVAEConfig {
        VoxCPM2AudioVAEConfig(encoderDim: encoderDim, encoderRates: encoderRates,
            latentDim: latentDim, decoderDim: decoderDim, decoderRates: decoderRates,
            depthwise: true, sampleRate: sampleRate, outSampleRate: sampleRate,
            srBinBoundaries: nil)
    }
}
