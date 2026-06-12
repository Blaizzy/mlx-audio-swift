import Foundation

public struct NemotronASRPreprocessConfig: Codable, Sendable {
    public let sampleRate: Int
    public let features: Int
    public let nFft: Int
    public let windowSize: Float
    public let windowStride: Float
    public let window: String
    public let preemph: Float
    public let dither: Float
    public let normalize: String
    public let logZeroGuardValue: Float
    public let padTo: Int
    public let padValue: Float

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case features
        case nFft = "n_fft"
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case window
        case preemph
        case dither
        case normalize
        case logZeroGuardValue = "log_zero_guard_value"
        case padTo = "pad_to"
        case padValue = "pad_value"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16_000
        features = try container.decodeIfPresent(Int.self, forKey: .features) ?? 128
        nFft = try container.decodeIfPresent(Int.self, forKey: .nFft) ?? 512
        windowSize = try container.decodeIfPresent(Float.self, forKey: .windowSize) ?? 0.025
        windowStride = try container.decodeIfPresent(Float.self, forKey: .windowStride) ?? 0.01
        window = try container.decodeIfPresent(String.self, forKey: .window) ?? "hann"
        preemph = try container.decodeIfPresent(Float.self, forKey: .preemph) ?? 0.97
        dither = try container.decodeIfPresent(Float.self, forKey: .dither) ?? 1e-5
        normalize = try container.decodeIfPresent(String.self, forKey: .normalize) ?? "NA"
        logZeroGuardValue = try container.decodeIfPresent(Float.self, forKey: .logZeroGuardValue) ?? pow(2, -24)
        padTo = try container.decodeIfPresent(Int.self, forKey: .padTo) ?? 0
        padValue = try container.decodeIfPresent(Float.self, forKey: .padValue) ?? 0
    }

    public init(
        sampleRate: Int = 16_000,
        features: Int = 128,
        nFft: Int = 512,
        windowSize: Float = 0.025,
        windowStride: Float = 0.01,
        window: String = "hann",
        preemph: Float = 0.97,
        dither: Float = 1e-5,
        normalize: String = "NA",
        logZeroGuardValue: Float = pow(2, -24),
        padTo: Int = 0,
        padValue: Float = 0
    ) {
        self.sampleRate = sampleRate
        self.features = features
        self.nFft = nFft
        self.windowSize = windowSize
        self.windowStride = windowStride
        self.window = window
        self.preemph = preemph
        self.dither = dither
        self.normalize = normalize
        self.logZeroGuardValue = logZeroGuardValue
        self.padTo = padTo
        self.padValue = padValue
    }

    public var winLength: Int {
        Int(windowSize * Float(sampleRate))
    }

    public var hopLength: Int {
        Int(windowStride * Float(sampleRate))
    }
}

public enum NemotronASRConvContextSize: Codable, Sendable, Equatable {
    case causal
    case explicit(left: Int, right: Int)

    public init(from decoder: Decoder) throws {
        let single = try decoder.singleValueContainer()
        if let string = try? single.decode(String.self), string.lowercased() == "causal" {
            self = .causal
            return
        }
        let values = try single.decode([Int].self)
        self = .explicit(left: values.first ?? 0, right: values.dropFirst().first ?? 0)
    }

    public func encode(to encoder: Encoder) throws {
        var single = encoder.singleValueContainer()
        switch self {
        case .causal:
            try single.encode("causal")
        case .explicit(let left, let right):
            try single.encode([left, right])
        }
    }
}

public struct NemotronASRConformerConfig: Codable, Sendable {
    public let featIn: Int
    public let nLayers: Int
    public let dModel: Int
    public let nHeads: Int
    public let ffExpansionFactor: Int
    public let subsamplingFactor: Int
    public let subsamplingConvChannels: Int
    public let convKernelSize: Int
    public let causalDownsampling: Bool
    public let convContextSize: NemotronASRConvContextSize
    public let convNormType: String
    public let selfAttentionModel: String
    public let attContextStyle: String
    public let attContextSize: [[Int]]
    public let posEmbMaxLen: Int
    public let useBias: Bool
    public let xscaling: Bool

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case nLayers = "n_layers"
        case dModel = "d_model"
        case nHeads = "n_heads"
        case ffExpansionFactor = "ff_expansion_factor"
        case subsamplingFactor = "subsampling_factor"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case causalDownsampling = "causal_downsampling"
        case convContextSize = "conv_context_size"
        case convNormType = "conv_norm_type"
        case selfAttentionModel = "self_attention_model"
        case attContextStyle = "att_context_style"
        case attContextSize = "att_context_size"
        case posEmbMaxLen = "pos_emb_max_len"
        case useBias = "use_bias"
        case xscaling
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try container.decodeIfPresent(Int.self, forKey: .featIn) ?? 128
        nLayers = try container.decodeIfPresent(Int.self, forKey: .nLayers) ?? 24
        dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 1024
        nHeads = try container.decodeIfPresent(Int.self, forKey: .nHeads) ?? 8
        ffExpansionFactor = try container.decodeIfPresent(Int.self, forKey: .ffExpansionFactor) ?? 4
        subsamplingFactor = try container.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        subsamplingConvChannels = try container.decodeIfPresent(Int.self, forKey: .subsamplingConvChannels) ?? 256
        convKernelSize = try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 9
        causalDownsampling = try container.decodeIfPresent(Bool.self, forKey: .causalDownsampling) ?? true
        convContextSize = try container.decodeIfPresent(NemotronASRConvContextSize.self, forKey: .convContextSize) ?? .causal
        convNormType = try container.decodeIfPresent(String.self, forKey: .convNormType) ?? "layer_norm"
        selfAttentionModel = try container.decodeIfPresent(String.self, forKey: .selfAttentionModel) ?? "rel_pos"
        attContextStyle = try container.decodeIfPresent(String.self, forKey: .attContextStyle) ?? "chunked_limited"
        attContextSize = try container.decodeIfPresent([[Int]].self, forKey: .attContextSize) ?? [[56, 13]]
        posEmbMaxLen = try container.decodeIfPresent(Int.self, forKey: .posEmbMaxLen) ?? 5000
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        xscaling = try container.decodeIfPresent(Bool.self, forKey: .xscaling) ?? false
    }

    public init(
        featIn: Int = 128,
        nLayers: Int = 24,
        dModel: Int = 1024,
        nHeads: Int = 8,
        ffExpansionFactor: Int = 4,
        subsamplingFactor: Int = 8,
        subsamplingConvChannels: Int = 256,
        convKernelSize: Int = 9,
        causalDownsampling: Bool = true,
        convContextSize: NemotronASRConvContextSize = .causal,
        convNormType: String = "layer_norm",
        selfAttentionModel: String = "rel_pos",
        attContextStyle: String = "chunked_limited",
        attContextSize: [[Int]] = [[56, 13]],
        posEmbMaxLen: Int = 5000,
        useBias: Bool = false,
        xscaling: Bool = false
    ) {
        self.featIn = featIn
        self.nLayers = nLayers
        self.dModel = dModel
        self.nHeads = nHeads
        self.ffExpansionFactor = ffExpansionFactor
        self.subsamplingFactor = subsamplingFactor
        self.subsamplingConvChannels = subsamplingConvChannels
        self.convKernelSize = convKernelSize
        self.causalDownsampling = causalDownsampling
        self.convContextSize = convContextSize
        self.convNormType = convNormType
        self.selfAttentionModel = selfAttentionModel
        self.attContextStyle = attContextStyle
        self.attContextSize = attContextSize
        self.posEmbMaxLen = posEmbMaxLen
        self.useBias = useBias
        self.xscaling = xscaling
    }
}

public struct NemotronASRPromptConfig: Codable, Sendable {
    public let numPrompts: Int
    public let promptHidden: Int
    public let promptDictionary: [String: Int]

    enum CodingKeys: String, CodingKey {
        case numPrompts = "num_prompts"
        case promptHidden = "prompt_hidden"
        case promptDictionary = "prompt_dictionary"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numPrompts = try container.decodeIfPresent(Int.self, forKey: .numPrompts) ?? 128
        promptHidden = try container.decodeIfPresent(Int.self, forKey: .promptHidden) ?? 2048
        promptDictionary = try container.decodeIfPresent([String: Int].self, forKey: .promptDictionary) ?? [:]
    }

    public init(numPrompts: Int = 128, promptHidden: Int = 2048, promptDictionary: [String: Int] = [:]) {
        self.numPrompts = numPrompts
        self.promptHidden = promptHidden
        self.promptDictionary = promptDictionary
    }
}

public struct NemotronASRPredictConfig: Codable, Sendable {
    public let predHidden: Int
    public let predRnnLayers: Int
    public let vocabSize: Int
    public let blankAsPad: Bool

    enum CodingKeys: String, CodingKey {
        case predHidden = "pred_hidden"
        case predRnnLayers = "pred_rnn_layers"
        case vocabSize = "vocab_size"
        case blankAsPad = "blank_as_pad"
    }
}

public struct NemotronASRJointConfig: Codable, Sendable {
    public let jointHidden: Int
    public let activation: String
    public let encoderHidden: Int
    public let predHidden: Int
    public let numClasses: Int

    enum CodingKeys: String, CodingKey {
        case jointHidden = "joint_hidden"
        case activation
        case encoderHidden = "encoder_hidden"
        case predHidden = "pred_hidden"
        case numClasses = "num_classes"
    }
}

public struct NemotronASRConfig: Codable, Sendable {
    public let modelType: String
    public let target: String
    public let preprocessor: NemotronASRPreprocessConfig
    public let encoder: NemotronASRConformerConfig
    public let prompt: NemotronASRPromptConfig
    public let decoder: NemotronASRPredictConfig
    public let joint: NemotronASRJointConfig
    public let vocabulary: [String]
    public let defaultLanguage: String
    public let defaultAttContextSize: [Int]
    public let maxSymbols: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case target
        case preprocessor
        case encoder
        case prompt
        case decoder
        case joint
        case vocabulary
        case defaultLanguage = "default_language"
        case defaultAttContextSize = "default_att_context_size"
        case maxSymbols = "max_symbols"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_asr"
        self.target = try container.decodeIfPresent(String.self, forKey: .target)
            ?? "nemo.collections.asr.models.rnnt_bpe_models_prompt.EncDecRNNTBPEModelWithPrompt"
        self.preprocessor = try container.decodeIfPresent(NemotronASRPreprocessConfig.self, forKey: .preprocessor)
            ?? NemotronASRPreprocessConfig()
        self.encoder = try container.decodeIfPresent(NemotronASRConformerConfig.self, forKey: .encoder)
            ?? NemotronASRConformerConfig()
        self.prompt = try container.decodeIfPresent(NemotronASRPromptConfig.self, forKey: .prompt)
            ?? NemotronASRPromptConfig()
        self.decoder = try container.decode(NemotronASRPredictConfig.self, forKey: .decoder)
        self.joint = try container.decode(NemotronASRJointConfig.self, forKey: .joint)
        self.vocabulary = try container.decodeIfPresent([String].self, forKey: .vocabulary) ?? []
        self.defaultLanguage = try container.decodeIfPresent(String.self, forKey: .defaultLanguage) ?? "auto"
        self.defaultAttContextSize = try container.decodeIfPresent([Int].self, forKey: .defaultAttContextSize) ?? [56, 13]
        self.maxSymbols = try container.decodeIfPresent(Int.self, forKey: .maxSymbols)
    }
}
