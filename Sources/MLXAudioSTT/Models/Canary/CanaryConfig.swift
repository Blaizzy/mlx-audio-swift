import Foundation
import MLXLMCommon

public struct CanaryPreprocessConfig: Decodable, Sendable {
    public let sampleRate: Int
    public let normalize: String
    public let features: Int
    public let nFft: Int
    public let windowSize: Float
    public let windowStride: Float
    public let window: String
    public let dither: Float
    public let padTo: Int
    public let padValue: Float
    public let preemph: Float

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case normalize
        case features
        case nFft = "n_fft"
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case window
        case dither
        case padTo = "pad_to"
        case padValue = "pad_value"
        case preemph
    }

    public init(
        sampleRate: Int = 16_000,
        normalize: String = "per_feature",
        features: Int = 128,
        nFft: Int = 512,
        windowSize: Float = 0.025,
        windowStride: Float = 0.01,
        window: String = "hann",
        dither: Float = 0.0,
        padTo: Int = 0,
        padValue: Float = 0.0,
        preemph: Float = 0.97
    ) {
        self.sampleRate = sampleRate
        self.normalize = normalize
        self.features = features
        self.nFft = nFft
        self.windowSize = windowSize
        self.windowStride = windowStride
        self.window = window
        self.dither = dither
        self.padTo = padTo
        self.padValue = padValue
        self.preemph = preemph
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16_000
        normalize = try c.decodeIfPresent(String.self, forKey: .normalize) ?? "per_feature"
        features = try c.decodeIfPresent(Int.self, forKey: .features) ?? 128
        nFft = try c.decodeIfPresent(Int.self, forKey: .nFft) ?? 512
        windowSize = try c.decodeIfPresent(Float.self, forKey: .windowSize) ?? 0.025
        windowStride = try c.decodeIfPresent(Float.self, forKey: .windowStride) ?? 0.01
        window = try c.decodeIfPresent(String.self, forKey: .window) ?? "hann"
        dither = try c.decodeIfPresent(Float.self, forKey: .dither) ?? 0.0
        padTo = try c.decodeIfPresent(Int.self, forKey: .padTo) ?? 0
        padValue = try c.decodeIfPresent(Float.self, forKey: .padValue) ?? 0.0
        preemph = try c.decodeIfPresent(Float.self, forKey: .preemph) ?? 0.97
    }

    var parakeetConfig: ParakeetPreprocessConfig {
        ParakeetPreprocessConfig(
            sampleRate: sampleRate,
            normalize: normalize,
            windowSize: windowSize,
            windowStride: windowStride,
            window: window,
            features: features,
            nFft: nFft,
            dither: dither,
            padTo: padTo,
            padValue: padValue,
            preemph: preemph
        )
    }
}

public struct CanaryEncoderConfig: Decodable, Sendable {
    public let featIn: Int
    public let nLayers: Int
    public let dModel: Int
    public let nHeads: Int
    public let ffExpansionFactor: Int
    public let subsamplingFactor: Int
    public let selfAttentionModel: String
    public let subsampling: String
    public let convKernelSize: Int
    public let subsamplingConvChannels: Int
    public let posEmbMaxLen: Int
    public let causalDownsampling: Bool
    public let useBias: Bool
    public let xscaling: Bool
    public let subsamplingConvChunkingFactor: Int

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case nLayers = "n_layers"
        case dModel = "d_model"
        case nHeads = "n_heads"
        case ffExpansionFactor = "ff_expansion_factor"
        case subsamplingFactor = "subsampling_factor"
        case selfAttentionModel = "self_attention_model"
        case subsampling
        case convKernelSize = "conv_kernel_size"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case posEmbMaxLen = "pos_emb_max_len"
        case causalDownsampling = "causal_downsampling"
        case useBias = "use_bias"
        case xscaling
        case subsamplingConvChunkingFactor = "subsampling_conv_chunking_factor"
    }

    public init(
        featIn: Int = 128,
        nLayers: Int = 32,
        dModel: Int = 1024,
        nHeads: Int = 8,
        ffExpansionFactor: Int = 4,
        subsamplingFactor: Int = 8,
        selfAttentionModel: String = "rel_pos",
        subsampling: String = "dw_striding",
        convKernelSize: Int = 9,
        subsamplingConvChannels: Int = 256,
        posEmbMaxLen: Int = 5000,
        causalDownsampling: Bool = false,
        useBias: Bool = true,
        xscaling: Bool = true,
        subsamplingConvChunkingFactor: Int = 1
    ) {
        self.featIn = featIn
        self.nLayers = nLayers
        self.dModel = dModel
        self.nHeads = nHeads
        self.ffExpansionFactor = ffExpansionFactor
        self.subsamplingFactor = subsamplingFactor
        self.selfAttentionModel = selfAttentionModel
        self.subsampling = subsampling
        self.convKernelSize = convKernelSize
        self.subsamplingConvChannels = subsamplingConvChannels
        self.posEmbMaxLen = posEmbMaxLen
        self.causalDownsampling = causalDownsampling
        self.useBias = useBias
        self.xscaling = xscaling
        self.subsamplingConvChunkingFactor = subsamplingConvChunkingFactor
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try c.decodeIfPresent(Int.self, forKey: .featIn) ?? 128
        nLayers = try c.decodeIfPresent(Int.self, forKey: .nLayers) ?? 32
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 1024
        nHeads = try c.decodeIfPresent(Int.self, forKey: .nHeads) ?? 8
        ffExpansionFactor = try c.decodeIfPresent(Int.self, forKey: .ffExpansionFactor) ?? 4
        subsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        selfAttentionModel = try c.decodeIfPresent(String.self, forKey: .selfAttentionModel) ?? "rel_pos"
        subsampling = try c.decodeIfPresent(String.self, forKey: .subsampling) ?? "dw_striding"
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 9
        subsamplingConvChannels = try c.decodeIfPresent(Int.self, forKey: .subsamplingConvChannels) ?? 256
        posEmbMaxLen = try c.decodeIfPresent(Int.self, forKey: .posEmbMaxLen) ?? 5000
        causalDownsampling = try c.decodeIfPresent(Bool.self, forKey: .causalDownsampling) ?? false
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? true
        xscaling = try c.decodeIfPresent(Bool.self, forKey: .xscaling) ?? true
        subsamplingConvChunkingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingConvChunkingFactor) ?? 1
    }

    var parakeetConfig: ParakeetConformerConfig {
        ParakeetConformerConfig(
            featIn: featIn,
            nLayers: nLayers,
            dModel: dModel,
            nHeads: nHeads,
            ffExpansionFactor: ffExpansionFactor,
            subsamplingFactor: subsamplingFactor,
            selfAttentionModel: selfAttentionModel,
            subsampling: subsampling,
            convKernelSize: convKernelSize,
            subsamplingConvChannels: subsamplingConvChannels,
            posEmbMaxLen: posEmbMaxLen,
            causalDownsampling: causalDownsampling,
            useBias: useBias,
            xscaling: xscaling,
            subsamplingConvChunkingFactor: subsamplingConvChunkingFactor
        )
    }
}

public struct CanaryDecoderConfig: Decodable, Sendable {
    public let numLayers: Int
    public let hiddenSize: Int
    public let numAttentionHeads: Int
    public let innerSize: Int
    public let ffnDropout: Float
    public let attnScoreDropout: Float
    public let attnLayerDropout: Float

    enum CodingKeys: String, CodingKey {
        case decoder
        case numLayers = "num_layers"
        case hiddenSize = "hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case innerSize = "inner_size"
        case ffnDropout = "ffn_dropout"
        case attnScoreDropout = "attn_score_dropout"
        case attnLayerDropout = "attn_layer_dropout"
    }

    public init(
        numLayers: Int = 8,
        hiddenSize: Int = 1024,
        numAttentionHeads: Int = 16,
        innerSize: Int = 4096,
        ffnDropout: Float = 0,
        attnScoreDropout: Float = 0,
        attnLayerDropout: Float = 0
    ) {
        self.numLayers = numLayers
        self.hiddenSize = hiddenSize
        self.numAttentionHeads = numAttentionHeads
        self.innerSize = innerSize
        self.ffnDropout = ffnDropout
        self.attnScoreDropout = attnScoreDropout
        self.attnLayerDropout = attnLayerDropout
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let source: KeyedDecodingContainer<CodingKeys>
        if c.contains(.decoder) {
            source = try c.nestedContainer(keyedBy: CodingKeys.self, forKey: .decoder)
        } else {
            source = c
        }
        numLayers = try source.decodeIfPresent(Int.self, forKey: .numLayers) ?? 8
        hiddenSize = try source.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        numAttentionHeads = try source.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        innerSize = try source.decodeIfPresent(Int.self, forKey: .innerSize) ?? 4096
        ffnDropout = try source.decodeIfPresent(Float.self, forKey: .ffnDropout) ?? 0
        attnScoreDropout = try source.decodeIfPresent(Float.self, forKey: .attnScoreDropout) ?? 0
        attnLayerDropout = try source.decodeIfPresent(Float.self, forKey: .attnLayerDropout) ?? 0
    }
}

public struct CanaryConfig: Decodable, Sendable {
    public let modelType: String
    public let preprocessor: CanaryPreprocessConfig
    public let encoder: CanaryEncoderConfig
    public let decoder: CanaryDecoderConfig
    public let vocabSize: Int
    public let encoderOutputDim: Int
    public let startOfContextId: Int
    public let startOfTranscriptId: Int
    public let emotionUndefinedId: Int
    public let endOfTextId: Int
    public let supportedLanguages: [String]
    public let tokenizerModelBase64: String?
    public let quantization: BaseConfiguration.Quantization?
    public let perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case preprocessor
        case encoder
        case decoder
        case transfDecoder = "transf_decoder"
        case vocabSize = "vocab_size"
        case encoderOutputDim = "enc_output_dim"
        case startOfContextId = "startofcontext_id"
        case startOfTranscriptId = "startoftranscript_id"
        case emotionUndefinedId = "emo_undefined_id"
        case endOfTextId = "endoftext_id"
        case supportedLanguages = "supported_languages"
        case tokenizer
        case quantization
        case quantizationConfig = "quantization_config"
    }

    enum TokenizerCodingKeys: String, CodingKey {
        case modelBase64 = "model_base64"
    }

    public static let defaultSupportedLanguages = [
        "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
        "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
    ]

    public init(
        modelType: String = "canary",
        preprocessor: CanaryPreprocessConfig = CanaryPreprocessConfig(),
        encoder: CanaryEncoderConfig = CanaryEncoderConfig(),
        decoder: CanaryDecoderConfig = CanaryDecoderConfig(),
        vocabSize: Int = 16_384,
        encoderOutputDim: Int = 1024,
        startOfContextId: Int = 0,
        startOfTranscriptId: Int = 1,
        emotionUndefinedId: Int = 2,
        endOfTextId: Int = 3,
        supportedLanguages: [String] = CanaryConfig.defaultSupportedLanguages,
        tokenizerModelBase64: String? = nil,
        quantization: BaseConfiguration.Quantization? = nil,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.modelType = modelType
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.vocabSize = vocabSize
        self.encoderOutputDim = encoderOutputDim
        self.startOfContextId = startOfContextId
        self.startOfTranscriptId = startOfTranscriptId
        self.emotionUndefinedId = emotionUndefinedId
        self.endOfTextId = endOfTextId
        self.supportedLanguages = supportedLanguages
        self.tokenizerModelBase64 = tokenizerModelBase64
        self.quantization = quantization
        self.perLayerQuantization = perLayerQuantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let baseConfig = try? BaseConfiguration(from: decoder)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "canary"
        preprocessor = try c.decodeIfPresent(CanaryPreprocessConfig.self, forKey: .preprocessor) ?? CanaryPreprocessConfig()
        encoder = try c.decodeIfPresent(CanaryEncoderConfig.self, forKey: .encoder) ?? CanaryEncoderConfig()
        if let transfDecoder = try c.decodeIfPresent(CanaryDecoderConfig.self, forKey: .transfDecoder) {
            self.decoder = transfDecoder
        } else {
            self.decoder = try c.decodeIfPresent(CanaryDecoderConfig.self, forKey: .decoder) ?? CanaryDecoderConfig()
        }
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 16_384
        encoderOutputDim = try c.decodeIfPresent(Int.self, forKey: .encoderOutputDim) ?? 1024
        startOfContextId = try c.decodeIfPresent(Int.self, forKey: .startOfContextId) ?? 0
        startOfTranscriptId = try c.decodeIfPresent(Int.self, forKey: .startOfTranscriptId) ?? 1
        emotionUndefinedId = try c.decodeIfPresent(Int.self, forKey: .emotionUndefinedId) ?? 2
        endOfTextId = try c.decodeIfPresent(Int.self, forKey: .endOfTextId) ?? 3
        supportedLanguages = try c.decodeIfPresent([String].self, forKey: .supportedLanguages) ?? CanaryConfig.defaultSupportedLanguages
        if let tokenizerContainer = try? c.nestedContainer(keyedBy: TokenizerCodingKeys.self, forKey: .tokenizer) {
            tokenizerModelBase64 = try tokenizerContainer.decodeIfPresent(String.self, forKey: .modelBase64)
        } else {
            tokenizerModelBase64 = nil
        }

        let globalQuant = try? c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        let altGlobalQuant = try? c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
        let fallbackQuant = try? c.decodeIfPresent(CanaryFlatQuantization.self, forKey: .quantization)?.baseQuantization
        quantization = globalQuant ?? altGlobalQuant ?? fallbackQuant
        perLayerQuantization = baseConfig?.perLayerQuantization
    }
}

private struct CanaryFlatQuantization: Decodable {
    let bits: Int
    let groupSize: Int?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }

    var baseQuantization: BaseConfiguration.Quantization {
        BaseConfiguration.Quantization(groupSize: groupSize ?? 64, bits: bits)
    }
}
