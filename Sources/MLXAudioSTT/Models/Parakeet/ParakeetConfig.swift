import Foundation

public struct ParakeetPreprocessConfig: Codable, Sendable {
    public let sampleRate: Int
    public let normalize: String
    public let windowSize: Float
    public let windowStride: Float
    public let window: String
    public let features: Int
    public let nFft: Int
    public let dither: Float
    public let padTo: Int
    public let padValue: Float
    public let preemph: Float

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case normalize
        case windowSize = "window_size"
        case windowStride = "window_stride"
        case window
        case features
        case nFft = "n_fft"
        case dither
        case padTo = "pad_to"
        case padValue = "pad_value"
        case preemph
    }

    public init(
        sampleRate: Int,
        normalize: String,
        windowSize: Float,
        windowStride: Float,
        window: String,
        features: Int,
        nFft: Int,
        dither: Float,
        padTo: Int = 0,
        padValue: Float = 0,
        preemph: Float = 0.97
    ) {
        self.sampleRate = sampleRate
        self.normalize = normalize
        self.windowSize = windowSize
        self.windowStride = windowStride
        self.window = window
        self.features = features
        self.nFft = nFft
        self.dither = dither
        self.padTo = padTo
        self.padValue = padValue
        self.preemph = preemph
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        sampleRate = try container.decode(Int.self, forKey: .sampleRate)
        normalize = try container.decode(String.self, forKey: .normalize)
        windowSize = try container.decode(Float.self, forKey: .windowSize)
        windowStride = try container.decode(Float.self, forKey: .windowStride)
        window = try container.decode(String.self, forKey: .window)
        features = try container.decode(Int.self, forKey: .features)
        nFft = try container.decode(Int.self, forKey: .nFft)
        dither = try container.decode(Float.self, forKey: .dither)
        padTo = try container.decodeIfPresent(Int.self, forKey: .padTo) ?? 0
        padValue = try container.decodeIfPresent(Float.self, forKey: .padValue) ?? 0
        preemph = try container.decodeIfPresent(Float.self, forKey: .preemph) ?? 0.97
    }

    public var winLength: Int {
        Int(windowSize * Float(sampleRate))
    }

    public var hopLength: Int {
        Int(windowStride * Float(sampleRate))
    }
}

public struct ParakeetConformerConfig: Codable, Sendable {
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

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try container.decode(Int.self, forKey: .featIn)
        nLayers = try container.decode(Int.self, forKey: .nLayers)
        dModel = try container.decode(Int.self, forKey: .dModel)
        nHeads = try container.decode(Int.self, forKey: .nHeads)
        ffExpansionFactor = try container.decode(Int.self, forKey: .ffExpansionFactor)
        subsamplingFactor = try container.decode(Int.self, forKey: .subsamplingFactor)
        selfAttentionModel = try container.decode(String.self, forKey: .selfAttentionModel)
        subsampling = try container.decode(String.self, forKey: .subsampling)
        convKernelSize = try container.decode(Int.self, forKey: .convKernelSize)
        subsamplingConvChannels = try container.decode(Int.self, forKey: .subsamplingConvChannels)
        posEmbMaxLen = try container.decode(Int.self, forKey: .posEmbMaxLen)
        causalDownsampling = try container.decodeIfPresent(Bool.self, forKey: .causalDownsampling) ?? false
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? true
        xscaling = try container.decodeIfPresent(Bool.self, forKey: .xscaling) ?? false
        subsamplingConvChunkingFactor = try container.decodeIfPresent(Int.self, forKey: .subsamplingConvChunkingFactor) ?? 1
    }
}

public struct ParakeetPredictNetworkConfig: Codable, Sendable {
    public let predHidden: Int
    public let predRnnLayers: Int
    public let rnnHiddenSize: Int?

    enum CodingKeys: String, CodingKey {
        case predHidden = "pred_hidden"
        case predRnnLayers = "pred_rnn_layers"
        case rnnHiddenSize = "rnn_hidden_size"
    }
}

public struct ParakeetJointNetworkConfig: Codable, Sendable {
    public let jointHidden: Int
    public let activation: String
    public let encoderHidden: Int
    public let predHidden: Int

    enum CodingKeys: String, CodingKey {
        case jointHidden = "joint_hidden"
        case activation
        case encoderHidden = "encoder_hidden"
        case predHidden = "pred_hidden"
    }
}

public struct ParakeetPredictConfig: Codable, Sendable {
    public let blankAsPad: Bool
    public let vocabSize: Int
    public let prednet: ParakeetPredictNetworkConfig

    enum CodingKeys: String, CodingKey {
        case blankAsPad = "blank_as_pad"
        case vocabSize = "vocab_size"
        case prednet
    }
}

public struct ParakeetJointConfig: Codable, Sendable {
    public let numClasses: Int
    public let vocabulary: [String]
    public let jointnet: ParakeetJointNetworkConfig
    public let numExtraOutputs: Int

    enum CodingKeys: String, CodingKey {
        case numClasses = "num_classes"
        case vocabulary
        case jointnet
        case numExtraOutputs = "num_extra_outputs"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numClasses = try container.decode(Int.self, forKey: .numClasses)
        vocabulary = try container.decode([String].self, forKey: .vocabulary)
        jointnet = try container.decode(ParakeetJointNetworkConfig.self, forKey: .jointnet)
        numExtraOutputs = try container.decodeIfPresent(Int.self, forKey: .numExtraOutputs) ?? 0
    }
}

public struct ParakeetGreedyConfig: Codable, Sendable {
    public let maxSymbols: Int?

    enum CodingKeys: String, CodingKey {
        case maxSymbols = "max_symbols"
    }
}

public struct ParakeetTDTDecodingConfig: Codable, Sendable {
    public let modelType: String
    public let durations: [Int]
    public let greedy: ParakeetGreedyConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case durations
        case greedy
    }
}

public struct ParakeetRNNTDecodingConfig: Codable, Sendable {
    public let greedy: ParakeetGreedyConfig?
}

public struct ParakeetCTCDecodingConfig: Codable, Sendable {
    public let greedy: ParakeetGreedyConfig?
}

public struct ParakeetConvASRDecoderConfig: Codable, Sendable {
    public let featIn: Int?
    public let numClasses: Int
    public let vocabulary: [String]

    enum CodingKeys: String, CodingKey {
        case featIn = "feat_in"
        case numClasses = "num_classes"
        case vocabulary
    }

    public init(featIn: Int?, numClasses: Int, vocabulary: [String]) {
        self.featIn = featIn
        self.numClasses = numClasses
        self.vocabulary = vocabulary
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        featIn = try container.decodeIfPresent(Int.self, forKey: .featIn)
        numClasses = try container.decode(Int.self, forKey: .numClasses)
        vocabulary = try container.decode([String].self, forKey: .vocabulary)
    }
}

public struct ParakeetAuxCTCConfig: Codable, Sendable {
    public let decoder: ParakeetConvASRDecoderConfig

    public init(decoder: ParakeetConvASRDecoderConfig) {
        self.decoder = decoder
    }
}

public struct ParakeetModelDefaults: Codable, Sendable {
    public let tdtDurations: [Int]?

    enum CodingKeys: String, CodingKey {
        case tdtDurations = "tdt_durations"
    }
}

public struct ParakeetRawConfig: Codable, Sendable {
    public let target: String?
    public let modelDefaults: ParakeetModelDefaults?
    public let preprocessor: ParakeetPreprocessConfig
    public let encoder: ParakeetConformerConfig
    public let decoder: CodableJSONValue
    public let joint: ParakeetJointConfig?
    public let decoding: CodableJSONValue
    public let auxCtc: CodableJSONValue?

    enum CodingKeys: String, CodingKey {
        case target
        case modelDefaults = "model_defaults"
        case preprocessor
        case encoder
        case decoder
        case joint
        case decoding
        case auxCtc = "aux_ctc"
    }
}

public struct ParakeetTDTConfig: Sendable {
    public let preprocessor: ParakeetPreprocessConfig
    public let encoder: ParakeetConformerConfig
    public let decoder: ParakeetPredictConfig
    public let joint: ParakeetJointConfig
    public let decoding: ParakeetTDTDecodingConfig
}

public struct ParakeetRNNTConfig: Sendable {
    public let preprocessor: ParakeetPreprocessConfig
    public let encoder: ParakeetConformerConfig
    public let decoder: ParakeetPredictConfig
    public let joint: ParakeetJointConfig
    public let decoding: ParakeetRNNTDecodingConfig
}

public struct ParakeetCTCConfig: Sendable {
    public let preprocessor: ParakeetPreprocessConfig
    public let encoder: ParakeetConformerConfig
    public let decoder: ParakeetConvASRDecoderConfig
    public let decoding: ParakeetCTCDecodingConfig
}

public struct ParakeetTDTCTCConfig: Sendable {
    public let preprocessor: ParakeetPreprocessConfig
    public let encoder: ParakeetConformerConfig
    public let decoder: ParakeetPredictConfig
    public let joint: ParakeetJointConfig
    public let decoding: ParakeetTDTDecodingConfig
    public let auxCTC: ParakeetAuxCTCConfig
}

public enum ParakeetVariant: Sendable {
    case tdt
    case tdtCtc
    case rnnt
    case ctc
}

public enum ParakeetVariantResolver {
    public static func resolve(_ config: ParakeetRawConfig) throws -> ParakeetVariant {
        let target = config.target ?? ""
        let hasTdt = config.modelDefaults?.tdtDurations != nil

        if target == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel" && hasTdt {
            return .tdt
        }
        if target == "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models.EncDecHybridRNNTCTCBPEModel" && hasTdt {
            return .tdtCtc
        }
        if target == "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel" && !hasTdt {
            return .rnnt
        }
        if target == "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE" {
            return .ctc
        }

        throw ParakeetConfigError.unsupportedModelTarget(target)
    }
}

public enum ParakeetConfigParser {
    private static func resolveDecoderFeatIn(
        _ decoder: ParakeetConvASRDecoderConfig,
        fallback: Int
    ) -> ParakeetConvASRDecoderConfig {
        if decoder.featIn != nil {
            return decoder
        }
        return ParakeetConvASRDecoderConfig(
            featIn: fallback,
            numClasses: decoder.numClasses,
            vocabulary: decoder.vocabulary
        )
    }

    public static func parseTDT(_ config: ParakeetRawConfig) throws -> ParakeetTDTConfig {
        guard let joint = config.joint else {
            throw ParakeetConfigError.missingField("joint")
        }
        return ParakeetTDTConfig(
            preprocessor: config.preprocessor,
            encoder: config.encoder,
            decoder: try config.decoder.decode(as: ParakeetPredictConfig.self),
            joint: joint,
            decoding: try config.decoding.decode(as: ParakeetTDTDecodingConfig.self)
        )
    }

    public static func parseRNNT(_ config: ParakeetRawConfig) throws -> ParakeetRNNTConfig {
        guard let joint = config.joint else {
            throw ParakeetConfigError.missingField("joint")
        }
        return ParakeetRNNTConfig(
            preprocessor: config.preprocessor,
            encoder: config.encoder,
            decoder: try config.decoder.decode(as: ParakeetPredictConfig.self),
            joint: joint,
            decoding: try config.decoding.decode(as: ParakeetRNNTDecodingConfig.self)
        )
    }

    public static func parseCTC(_ config: ParakeetRawConfig) throws -> ParakeetCTCConfig {
        let rawDecoder = try config.decoder.decode(as: ParakeetConvASRDecoderConfig.self)
        return ParakeetCTCConfig(
            preprocessor: config.preprocessor,
            encoder: config.encoder,
            decoder: resolveDecoderFeatIn(rawDecoder, fallback: config.encoder.dModel),
            decoding: try config.decoding.decode(as: ParakeetCTCDecodingConfig.self)
        )
    }

    public static func parseTDTCTC(_ config: ParakeetRawConfig) throws -> ParakeetTDTCTCConfig {
        guard let joint = config.joint else {
            throw ParakeetConfigError.missingField("joint")
        }
        guard let auxCTCRaw = config.auxCtc else {
            throw ParakeetConfigError.missingField("aux_ctc")
        }
        let auxCTC = try auxCTCRaw.decode(as: ParakeetAuxCTCConfig.self)
        let resolvedAuxCTC = ParakeetAuxCTCConfig(
            decoder: resolveDecoderFeatIn(auxCTC.decoder, fallback: config.encoder.dModel)
        )
        return ParakeetTDTCTCConfig(
            preprocessor: config.preprocessor,
            encoder: config.encoder,
            decoder: try config.decoder.decode(as: ParakeetPredictConfig.self),
            joint: joint,
            decoding: try config.decoding.decode(as: ParakeetTDTDecodingConfig.self),
            auxCTC: resolvedAuxCTC
        )
    }
}

public enum ParakeetConfigError: Error, LocalizedError {
    case unsupportedModelTarget(String)
    case missingField(String)
    case decodeFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelTarget(let target):
            return "Unsupported Parakeet model target: \(target)"
        case .missingField(let field):
            return "Missing required Parakeet config field: \(field)"
        case .decodeFailed(let typeName):
            return "Failed to decode Parakeet config object as \(typeName)"
        }
    }
}

// Simple JSON wrapper so we can parse variant-dependent decoder/decoding payloads after target resolution.
public enum CodableJSONValue: Codable, Sendable {
    case object([String: CodableJSONValue])
    case array([CodableJSONValue])
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else if let value = try? container.decode(Double.self) {
            self = .double(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: CodableJSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([CodableJSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .object(let value):
            try container.encode(value)
        case .array(let value):
            try container.encode(value)
        case .string(let value):
            try container.encode(value)
        case .int(let value):
            try container.encode(value)
        case .double(let value):
            try container.encode(value)
        case .bool(let value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }

    func decode<T: Decodable>(as type: T.Type) throws -> T {
        let data = try JSONSerialization.data(withJSONObject: toAny())
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw ParakeetConfigError.decodeFailed(String(describing: T.self))
        }
    }

    private func toAny() -> Any {
        switch self {
        case .object(let value):
            var object: [String: Any] = [:]
            object.reserveCapacity(value.count)
            for (key, item) in value {
                object[key] = item.toAny()
            }
            return object
        case .array(let value):
            return value.map { $0.toAny() }
        case .string(let value):
            return value
        case .int(let value):
            return value
        case .double(let value):
            return value
        case .bool(let value):
            return value
        case .null:
            return NSNull()
        }
    }
}
