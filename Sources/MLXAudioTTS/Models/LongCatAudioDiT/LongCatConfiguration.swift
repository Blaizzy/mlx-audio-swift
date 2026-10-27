import Foundation
import MLXAudioCore

public struct LongCatVAEConfiguration: Codable, Sendable {
    public var inChannels: Int
    public var channels: Int
    public var cMults: [Int]
    public var strides: [Int]
    public var latentDim: Int
    public var encoderLatentDim: Int
    public var useSnake: Bool
    public var downsampleShortcut: String
    public var upsampleShortcut: String
    public var outShortcut: String
    public var inShortcut: String
    public var finalTanh: Bool
    public var downsamplingRatio: Int
    public var sampleRate: Int
    public var scale: Float

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case channels = "channels"
        case cMults = "c_mults"
        case strides = "strides"
        case latentDim = "latent_dim"
        case encoderLatentDim = "encoder_latent_dim"
        case useSnake = "use_snake"
        case downsampleShortcut = "downsample_shortcut"
        case upsampleShortcut = "upsample_shortcut"
        case outShortcut = "out_shortcut"
        case inShortcut = "in_shortcut"
        case finalTanh = "final_tanh"
        case downsamplingRatio = "downsampling_ratio"
        case sampleRate = "sample_rate"
        case scale = "scale"
    }

    public init(
        inChannels: Int = 1,
        channels: Int = 128,
        cMults: [Int] = [1, 2, 4, 8, 16],
        strides: [Int] = [2, 4, 4, 8, 8],
        latentDim: Int = 64,
        encoderLatentDim: Int = 128,
        useSnake: Bool = true,
        downsampleShortcut: String = "averaging",
        upsampleShortcut: String = "duplicating",
        outShortcut: String = "averaging",
        inShortcut: String = "duplicating",
        finalTanh: Bool = false,
        downsamplingRatio: Int = 2048,
        sampleRate: Int = 24000,
        scale: Float = 0.71
    ) {
        self.inChannels = inChannels
        self.channels = channels
        self.cMults = cMults
        self.strides = strides
        self.latentDim = latentDim
        self.encoderLatentDim = encoderLatentDim
        self.useSnake = useSnake
        self.downsampleShortcut = downsampleShortcut
        self.upsampleShortcut = upsampleShortcut
        self.outShortcut = outShortcut
        self.inShortcut = inShortcut
        self.finalTanh = finalTanh
        self.downsamplingRatio = downsamplingRatio
        self.sampleRate = sampleRate
        self.scale = scale
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init()
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? inChannels
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? channels
        cMults = try c.decodeIfPresent([Int].self, forKey: .cMults) ?? cMults
        strides = try c.decodeIfPresent([Int].self, forKey: .strides) ?? strides
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? latentDim
        encoderLatentDim = try c.decodeIfPresent(Int.self, forKey: .encoderLatentDim) ?? encoderLatentDim
        useSnake = try c.decodeIfPresent(Bool.self, forKey: .useSnake) ?? useSnake
        downsampleShortcut = try c.decodeIfPresent(String.self, forKey: .downsampleShortcut) ?? downsampleShortcut
        upsampleShortcut = try c.decodeIfPresent(String.self, forKey: .upsampleShortcut) ?? upsampleShortcut
        outShortcut = try c.decodeIfPresent(String.self, forKey: .outShortcut) ?? outShortcut
        inShortcut = try c.decodeIfPresent(String.self, forKey: .inShortcut) ?? inShortcut
        finalTanh = try c.decodeIfPresent(Bool.self, forKey: .finalTanh) ?? finalTanh
        downsamplingRatio = try c.decodeIfPresent(Int.self, forKey: .downsamplingRatio) ?? downsamplingRatio
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? sampleRate
        scale = try c.decodeIfPresent(Float.self, forKey: .scale) ?? scale
    }
}

public struct LongCatTextEncoderConfiguration: Codable, Sendable {
    public var vocabSize: Int
    public var dModel: Int
    public var dKv: Int
    public var dFf: Int
    public var numLayers: Int
    public var numHeads: Int
    public var relativeAttentionNumBuckets: Int
    public var relativeAttentionMaxDistance: Int
    public var dropoutRate: Float
    public var layerNormEpsilon: Float
    public var isGatedAct: Bool
    public var denseActFn: String

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case dModel = "d_model"
        case dKv = "d_kv"
        case dFf = "d_ff"
        case numLayers = "num_layers"
        case numHeads = "num_heads"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case relativeAttentionMaxDistance = "relative_attention_max_distance"
        case dropoutRate = "dropout_rate"
        case layerNormEpsilon = "layer_norm_epsilon"
        case isGatedAct = "is_gated_act"
        case denseActFn = "dense_act_fn"
    }

    public init(
        vocabSize: Int = 256384,
        dModel: Int = 768,
        dKv: Int = 64,
        dFf: Int = 2048,
        numLayers: Int = 12,
        numHeads: Int = 12,
        relativeAttentionNumBuckets: Int = 32,
        relativeAttentionMaxDistance: Int = 128,
        dropoutRate: Float = 0.1,
        layerNormEpsilon: Float = 1e-06,
        isGatedAct: Bool = true,
        denseActFn: String = "gelu_new"
    ) {
        self.vocabSize = vocabSize
        self.dModel = dModel
        self.dKv = dKv
        self.dFf = dFf
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.relativeAttentionNumBuckets = relativeAttentionNumBuckets
        self.relativeAttentionMaxDistance = relativeAttentionMaxDistance
        self.dropoutRate = dropoutRate
        self.layerNormEpsilon = layerNormEpsilon
        self.isGatedAct = isGatedAct
        self.denseActFn = denseActFn
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init()
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? vocabSize
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? dModel
        dKv = try c.decodeIfPresent(Int.self, forKey: .dKv) ?? dKv
        dFf = try c.decodeIfPresent(Int.self, forKey: .dFf) ?? dFf
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? numLayers
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? numHeads
        relativeAttentionNumBuckets = try c.decodeIfPresent(Int.self, forKey: .relativeAttentionNumBuckets) ?? relativeAttentionNumBuckets
        relativeAttentionMaxDistance = try c.decodeIfPresent(Int.self, forKey: .relativeAttentionMaxDistance) ?? relativeAttentionMaxDistance
        dropoutRate = try c.decodeIfPresent(Float.self, forKey: .dropoutRate) ?? dropoutRate
        layerNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? layerNormEpsilon
        isGatedAct = try c.decodeIfPresent(Bool.self, forKey: .isGatedAct) ?? isGatedAct
        denseActFn = try c.decodeIfPresent(String.self, forKey: .denseActFn) ?? denseActFn
    }
}

public struct LongCatConfiguration: Codable, Sendable {
    public var modelType = "longcat_audiodit"
    public var ditDim: Int
    public var ditDepth: Int
    public var ditHeads: Int
    public var ditFfMult: Float
    public var ditTextDim: Int
    public var ditDropout: Float
    public var ditBias: Bool
    public var ditCrossAttn: Bool
    public var ditAdalnType: String
    public var ditAdalnUseTextCond: Bool
    public var ditLongSkip: Bool
    public var ditTextConv: Bool
    public var ditQkNorm: Bool
    public var ditCrossAttnNorm: Bool
    public var ditEps: Float
    public var ditUseLatentCondition: Bool
    public var repaDitLayer: Int
    public var latentDim: Int
    public var sigma: Float
    public var samplingRate: Int
    public var latentHop: Int
    public var maxWavDuration: Float
    public var textEncoderModel: String
    public var textAddEmbed: Bool
    public var textNormFeat: Bool
    public var vaeConfig: LongCatVAEConfiguration
    public var textEncoderConfig: LongCatTextEncoderConfiguration
    public var quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case ditDim = "dit_dim"
        case ditDepth = "dit_depth"
        case ditHeads = "dit_heads"
        case ditFfMult = "dit_ff_mult"
        case ditTextDim = "dit_text_dim"
        case ditDropout = "dit_dropout"
        case ditBias = "dit_bias"
        case ditCrossAttn = "dit_cross_attn"
        case ditAdalnType = "dit_adaln_type"
        case ditAdalnUseTextCond = "dit_adaln_use_text_cond"
        case ditLongSkip = "dit_long_skip"
        case ditTextConv = "dit_text_conv"
        case ditQkNorm = "dit_qk_norm"
        case ditCrossAttnNorm = "dit_cross_attn_norm"
        case ditEps = "dit_eps"
        case ditUseLatentCondition = "dit_use_latent_condition"
        case repaDitLayer = "repa_dit_layer"
        case latentDim = "latent_dim"
        case sigma = "sigma"
        case samplingRate = "sampling_rate"
        case latentHop = "latent_hop"
        case maxWavDuration = "max_wav_duration"
        case textEncoderModel = "text_encoder_model"
        case textAddEmbed = "text_add_embed"
        case textNormFeat = "text_norm_feat"
        case vaeConfig = "vae_config"
        case textEncoderConfig = "text_encoder_config"
        case quantization = "quantization"
    }

    public init(
        ditDim: Int = 1536,
        ditDepth: Int = 24,
        ditHeads: Int = 24,
        ditFfMult: Float = 4.0,
        ditTextDim: Int = 768,
        ditDropout: Float = 0.0,
        ditBias: Bool = true,
        ditCrossAttn: Bool = true,
        ditAdalnType: String = "global",
        ditAdalnUseTextCond: Bool = true,
        ditLongSkip: Bool = true,
        ditTextConv: Bool = true,
        ditQkNorm: Bool = true,
        ditCrossAttnNorm: Bool = false,
        ditEps: Float = 1e-06,
        ditUseLatentCondition: Bool = true,
        repaDitLayer: Int = 8,
        latentDim: Int = 64,
        sigma: Float = 0.0,
        samplingRate: Int = 24000,
        latentHop: Int = 2048,
        maxWavDuration: Float = 30.0,
        textEncoderModel: String = "google/umt5-base",
        textAddEmbed: Bool = true,
        textNormFeat: Bool = true,
        vaeConfig: LongCatVAEConfiguration = LongCatVAEConfiguration(),
        textEncoderConfig: LongCatTextEncoderConfiguration = LongCatTextEncoderConfiguration(),
        quantization: BaseConfiguration.Quantization? = nil
    ) {
        self.ditDim = ditDim
        self.ditDepth = ditDepth
        self.ditHeads = ditHeads
        self.ditFfMult = ditFfMult
        self.ditTextDim = ditTextDim
        self.ditDropout = ditDropout
        self.ditBias = ditBias
        self.ditCrossAttn = ditCrossAttn
        self.ditAdalnType = ditAdalnType
        self.ditAdalnUseTextCond = ditAdalnUseTextCond
        self.ditLongSkip = ditLongSkip
        self.ditTextConv = ditTextConv
        self.ditQkNorm = ditQkNorm
        self.ditCrossAttnNorm = ditCrossAttnNorm
        self.ditEps = ditEps
        self.ditUseLatentCondition = ditUseLatentCondition
        self.repaDitLayer = repaDitLayer
        self.latentDim = latentDim
        self.sigma = sigma
        self.samplingRate = samplingRate
        self.latentHop = latentHop
        self.maxWavDuration = maxWavDuration
        self.textEncoderModel = textEncoderModel
        self.textAddEmbed = textAddEmbed
        self.textNormFeat = textNormFeat
        self.vaeConfig = vaeConfig
        self.textEncoderConfig = textEncoderConfig
        self.quantization = quantization
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init()
        ditDim = try c.decodeIfPresent(Int.self, forKey: .ditDim) ?? ditDim
        ditDepth = try c.decodeIfPresent(Int.self, forKey: .ditDepth) ?? ditDepth
        ditHeads = try c.decodeIfPresent(Int.self, forKey: .ditHeads) ?? ditHeads
        ditFfMult = try c.decodeIfPresent(Float.self, forKey: .ditFfMult) ?? ditFfMult
        ditTextDim = try c.decodeIfPresent(Int.self, forKey: .ditTextDim) ?? ditTextDim
        ditDropout = try c.decodeIfPresent(Float.self, forKey: .ditDropout) ?? ditDropout
        ditBias = try c.decodeIfPresent(Bool.self, forKey: .ditBias) ?? ditBias
        ditCrossAttn = try c.decodeIfPresent(Bool.self, forKey: .ditCrossAttn) ?? ditCrossAttn
        ditAdalnType = try c.decodeIfPresent(String.self, forKey: .ditAdalnType) ?? ditAdalnType
        ditAdalnUseTextCond = try c.decodeIfPresent(Bool.self, forKey: .ditAdalnUseTextCond) ?? ditAdalnUseTextCond
        ditLongSkip = try c.decodeIfPresent(Bool.self, forKey: .ditLongSkip) ?? ditLongSkip
        ditTextConv = try c.decodeIfPresent(Bool.self, forKey: .ditTextConv) ?? ditTextConv
        ditQkNorm = try c.decodeIfPresent(Bool.self, forKey: .ditQkNorm) ?? ditQkNorm
        ditCrossAttnNorm = try c.decodeIfPresent(Bool.self, forKey: .ditCrossAttnNorm) ?? ditCrossAttnNorm
        ditEps = try c.decodeIfPresent(Float.self, forKey: .ditEps) ?? ditEps
        ditUseLatentCondition = try c.decodeIfPresent(Bool.self, forKey: .ditUseLatentCondition) ?? ditUseLatentCondition
        repaDitLayer = try c.decodeIfPresent(Int.self, forKey: .repaDitLayer) ?? repaDitLayer
        latentDim = try c.decodeIfPresent(Int.self, forKey: .latentDim) ?? latentDim
        sigma = try c.decodeIfPresent(Float.self, forKey: .sigma) ?? sigma
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? samplingRate
        latentHop = try c.decodeIfPresent(Int.self, forKey: .latentHop) ?? latentHop
        maxWavDuration = try c.decodeIfPresent(Float.self, forKey: .maxWavDuration) ?? maxWavDuration
        textEncoderModel = try c.decodeIfPresent(String.self, forKey: .textEncoderModel) ?? textEncoderModel
        textAddEmbed = try c.decodeIfPresent(Bool.self, forKey: .textAddEmbed) ?? textAddEmbed
        textNormFeat = try c.decodeIfPresent(Bool.self, forKey: .textNormFeat) ?? textNormFeat
        vaeConfig = try c.decodeIfPresent(LongCatVAEConfiguration.self, forKey: .vaeConfig) ?? vaeConfig
        textEncoderConfig = try c.decodeIfPresent(LongCatTextEncoderConfiguration.self, forKey: .textEncoderConfig) ?? textEncoderConfig
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization) ?? quantization
    }
}
