import Foundation

public struct PocketTTSFlowConfig: Codable, Sendable {
    public let dim: Int
    public let depth: Int
}

public struct PocketTTSFlowLMTransformerConfig: Codable, Sendable {
    public let hiddenScale: Double
    public let maxPeriod: Double
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let gelu: String

    enum CodingKeys: String, CodingKey {
        case hiddenScale = "hidden_scale"
        case maxPeriod = "max_period"
        case dModel = "d_model"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case gelu
    }

    init(hiddenScale: Double, maxPeriod: Double, dModel: Int, numHeads: Int, numLayers: Int, gelu: String = "erf") {
        self.hiddenScale = hiddenScale
        self.maxPeriod = maxPeriod
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.gelu = gelu
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenScale = try c.decode(Double.self, forKey: .hiddenScale)
        maxPeriod = try c.decode(Double.self, forKey: .maxPeriod)
        dModel = try c.decode(Int.self, forKey: .dModel)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        numLayers = try c.decode(Int.self, forKey: .numLayers)
        gelu = try c.decodeIfPresent(String.self, forKey: .gelu) ?? "erf"
    }
}

public struct PocketTTSLookupTableConfig: Codable, Sendable {
    public let dim: Int
    public let nBins: Int
    public let tokenizer: String
    public let tokenizerPath: String

    enum CodingKeys: String, CodingKey {
        case dim
        case nBins = "n_bins"
        case tokenizer
        case tokenizerPath = "tokenizer_path"
    }
}

public struct PocketTTSFlowLMConfig: Codable, Sendable {
    public let dtype: String?
    public let flow: PocketTTSFlowConfig
    public let transformer: PocketTTSFlowLMTransformerConfig
    public let lookupTable: PocketTTSLookupTableConfig
    public let weightsPath: String?
    public let ldim: Int?
    public let timeRmsNorm: String

    enum CodingKeys: String, CodingKey {
        case dtype
        case flow
        case transformer
        case lookupTable = "lookup_table"
        case weightsPath = "weights_path"
        case ldim
        case timeRmsNorm = "time_rms_norm"
    }

    private enum FlatKeys: String, CodingKey {
        case dModel = "d_model", numHeads = "num_heads", numLayers = "num_layers"
        case dimFeedforward = "dim_feedforward", maxPeriod = "max_period"
        case nBins = "n_bins", lutDim = "lut_dim", flowDim = "flow_dim", flowDepth = "flow_depth"
        case gelu
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype)
        weightsPath = try c.decodeIfPresent(String.self, forKey: .weightsPath)
        ldim = try c.decodeIfPresent(Int.self, forKey: .ldim)
        timeRmsNorm = try c.decodeIfPresent(String.self, forKey: .timeRmsNorm) ?? "rms_var"
        let flat = try decoder.container(keyedBy: FlatKeys.self)
        if flat.contains(.dModel) {
            let dim = try flat.decode(Int.self, forKey: .dModel)
            flow = PocketTTSFlowConfig(
                dim: try flat.decode(Int.self, forKey: .flowDim),
                depth: try flat.decode(Int.self, forKey: .flowDepth))
            transformer = PocketTTSFlowLMTransformerConfig(
                hiddenScale: Double(try flat.decode(Int.self, forKey: .dimFeedforward)) / Double(dim),
                maxPeriod: try flat.decode(Double.self, forKey: .maxPeriod),
                dModel: dim,
                numHeads: try flat.decode(Int.self, forKey: .numHeads),
                numLayers: try flat.decode(Int.self, forKey: .numLayers),
                gelu: try flat.decodeIfPresent(String.self, forKey: .gelu) ?? "erf")
            lookupTable = PocketTTSLookupTableConfig(
                dim: try flat.decode(Int.self, forKey: .lutDim),
                nBins: try flat.decode(Int.self, forKey: .nBins),
                tokenizer: "huggingface", tokenizerPath: "tokenizer.json")
        } else {
            flow = try c.decode(PocketTTSFlowConfig.self, forKey: .flow)
            transformer = try c.decode(PocketTTSFlowLMTransformerConfig.self, forKey: .transformer)
            lookupTable = try c.decode(PocketTTSLookupTableConfig.self, forKey: .lookupTable)
        }
    }
}

public struct PocketTTSSeanetConfig: Codable, Sendable {
    public let dimension: Int
    public let channels: Int
    public let nFilters: Int
    public let nResidualLayers: Int
    public let ratios: [Int]
    public let kernelSize: Int
    public let residualKernelSize: Int
    public let lastKernelSize: Int
    public let dilationBase: Int
    public let padMode: String
    public let compress: Int

    enum CodingKeys: String, CodingKey {
        case dimension
        case channels
        case nFilters = "n_filters"
        case nResidualLayers = "n_residual_layers"
        case ratios
        case kernelSize = "kernel_size"
        case residualKernelSize = "residual_kernel_size"
        case lastKernelSize = "last_kernel_size"
        case dilationBase = "dilation_base"
        case padMode = "pad_mode"
        case compress
    }
}

public struct PocketTTSMimiTransformerConfig: Codable, Sendable {
    public let dModel: Int
    public let inputDimension: Int
    public let outputDimensions: [Int]
    public let numHeads: Int
    public let numLayers: Int
    public let layerScale: Double
    public let context: Int
    public let dimFeedforward: Int
    public let maxPeriod: Double

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case inputDimension = "input_dimension"
        case outputDimensions = "output_dimensions"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case layerScale = "layer_scale"
        case context
        case dimFeedforward = "dim_feedforward"
        case maxPeriod = "max_period"
    }
}

public struct PocketTTSQuantizerConfig: Codable, Sendable {
    public let dimension: Int
    public let outputDimension: Int

    enum CodingKeys: String, CodingKey {
        case dimension
        case outputDimension = "output_dimension"
    }
}

public struct PocketTTSMimiConfig: Codable, Sendable {
    public let dtype: String?
    public let sampleRate: Int
    public let channels: Int
    public let frameRate: Double
    public let seanet: PocketTTSSeanetConfig
    public let transformer: PocketTTSMimiTransformerConfig
    public let quantizer: PocketTTSQuantizerConfig
    public let weightsPath: String?
    public let encoderEnabled: Bool
    public let gelu: String

    enum CodingKeys: String, CodingKey {
        case dtype
        case sampleRate = "sample_rate"
        case channels
        case frameRate = "frame_rate"
        case seanet
        case transformer
        case quantizer
        case weightsPath = "weights_path"
        case encoderEnabled = "encoder_enabled"
        case gelu
    }

    private enum FlatKeys: String, CodingKey {
        case dimension, channels, ratios, compress
        case nFilters = "n_filters", nResidualLayers = "n_residual_layers"
        case kernelSize = "kernel_size", residualKernelSize = "residual_kernel_size"
        case lastKernelSize = "last_kernel_size", dilationBase = "dilation_base", padMode = "pad_mode"
        case dModel = "transformer_d_model", numHeads = "transformer_num_heads"
        case numLayers = "transformer_num_layers", layerScale = "transformer_layer_scale"
        case context = "transformer_context", dimFeedforward = "transformer_dim_feedforward"
        case maxPeriod = "transformer_max_period"
        case quantizerDimension = "quantizer_dimension", quantizerOutputDimension = "quantizer_output_dimension"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype)
        sampleRate = try c.decode(Int.self, forKey: .sampleRate)
        channels = try c.decode(Int.self, forKey: .channels)
        frameRate = try c.decode(Double.self, forKey: .frameRate)
        weightsPath = try c.decodeIfPresent(String.self, forKey: .weightsPath)
        encoderEnabled = try c.decodeIfPresent(Bool.self, forKey: .encoderEnabled) ?? true
        gelu = try c.decodeIfPresent(String.self, forKey: .gelu) ?? "tanh"
        let flat = try decoder.container(keyedBy: FlatKeys.self)
        if flat.contains(.dimension) {
            let dimension = try flat.decode(Int.self, forKey: .dimension)
            seanet = PocketTTSSeanetConfig(
                dimension: dimension, channels: channels,
                nFilters: try flat.decode(Int.self, forKey: .nFilters),
                nResidualLayers: try flat.decode(Int.self, forKey: .nResidualLayers),
                ratios: try flat.decode([Int].self, forKey: .ratios),
                kernelSize: try flat.decode(Int.self, forKey: .kernelSize),
                residualKernelSize: try flat.decode(Int.self, forKey: .residualKernelSize),
                lastKernelSize: try flat.decode(Int.self, forKey: .lastKernelSize),
                dilationBase: try flat.decode(Int.self, forKey: .dilationBase),
                padMode: try flat.decodeIfPresent(String.self, forKey: .padMode) ?? "constant",
                compress: try flat.decode(Int.self, forKey: .compress))
            transformer = PocketTTSMimiTransformerConfig(
                dModel: try flat.decode(Int.self, forKey: .dModel),
                inputDimension: dimension, outputDimensions: [dimension],
                numHeads: try flat.decode(Int.self, forKey: .numHeads),
                numLayers: try flat.decode(Int.self, forKey: .numLayers),
                layerScale: try flat.decode(Double.self, forKey: .layerScale),
                context: try flat.decode(Int.self, forKey: .context),
                dimFeedforward: try flat.decode(Int.self, forKey: .dimFeedforward),
                maxPeriod: try flat.decodeIfPresent(Double.self, forKey: .maxPeriod) ?? 10000)
            quantizer = PocketTTSQuantizerConfig(
                dimension: try flat.decode(Int.self, forKey: .quantizerDimension),
                outputDimension: try flat.decode(Int.self, forKey: .quantizerOutputDimension))
        } else {
            seanet = try c.decode(PocketTTSSeanetConfig.self, forKey: .seanet)
            transformer = try c.decode(PocketTTSMimiTransformerConfig.self, forKey: .transformer)
            quantizer = try c.decode(PocketTTSQuantizerConfig.self, forKey: .quantizer)
        }
    }
}

public struct PocketTTSConditionerConfig: Codable, Sendable {
    public let name: String
    public let type: String
    public let lut: LookupTable?
    public let continuous: Continuous?

    public struct LookupTable: Codable, Sendable {
        public let nBins: Int
        public let dim: Int

        enum CodingKeys: String, CodingKey {
            case nBins = "n_bins", dim
        }
    }

    public struct Continuous: Codable, Sendable {
        public let dim: Int
        public let scaleFactor: Float
        public let maxPeriod: Float

        enum CodingKeys: String, CodingKey {
            case dim, scaleFactor = "scale_factor", maxPeriod = "max_period"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            dim = try c.decode(Int.self, forKey: .dim)
            scaleFactor = try c.decode(Float.self, forKey: .scaleFactor)
            maxPeriod = try c.decodeIfPresent(Float.self, forKey: .maxPeriod) ?? 10000
        }
    }
}

public struct PocketTTSFuserConfig: Codable, Sendable {
    public var sum: [String] = []
    public var cross: [String] = []
    public var streamingSum: [String] = []

    enum CodingKeys: String, CodingKey {
        case sum, cross, streamingSum = "streaming_sum"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sum = try c.decodeIfPresent([String].self, forKey: .sum) ?? []
        cross = try c.decodeIfPresent([String].self, forKey: .cross) ?? []
        streamingSum = try c.decodeIfPresent([String].self, forKey: .streamingSum) ?? []
    }
}

public struct PocketTTSModelConfig: Codable, Sendable {
    public let modelType: String
    public let flowLM: PocketTTSFlowLMConfig
    public let mimi: PocketTTSMimiConfig
    public let weightsPath: String?
    public let weightsPathWithoutVoiceCloning: String?
    public let modelPath: String?
    public let temp: Float
    public let lsdDecodeSteps: Int
    public let eosThreshold: Float
    public let noiseClamp: Float?
    public let audioPromptMinDuration: Double
    public let audioPromptMaxDuration: Double?
    public let conditioners: [PocketTTSConditionerConfig]
    public let fuser: PocketTTSFuserConfig
    public let weightsName: String?
    public let tokenizerName: String?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case flowLM = "flow_lm"
        case mimi
        case weightsPath = "weights_path"
        case weightsPathWithoutVoiceCloning = "weights_path_without_voice_cloning"
        case modelPath = "model_path"
        case temp, lsdDecodeSteps = "lsd_decode_steps", eosThreshold = "eos_threshold"
        case noiseClamp = "noise_clamp"
        case audioPromptMinDuration = "audio_prompt_min_duration", audioPromptMaxDuration = "audio_prompt_max_duration"
        case conditioners, fuser, weightsName = "weights_name", tokenizerName = "tokenizer_name"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "pocket_tts"
        flowLM = try c.decode(PocketTTSFlowLMConfig.self, forKey: .flowLM)
        mimi = try c.decode(PocketTTSMimiConfig.self, forKey: .mimi)
        weightsPath = try c.decodeIfPresent(String.self, forKey: .weightsPath)
        weightsPathWithoutVoiceCloning = try c.decodeIfPresent(String.self, forKey: .weightsPathWithoutVoiceCloning)
        modelPath = try c.decodeIfPresent(String.self, forKey: .modelPath)
        temp = try c.decodeIfPresent(Float.self, forKey: .temp) ?? 0.7
        lsdDecodeSteps = try c.decodeIfPresent(Int.self, forKey: .lsdDecodeSteps) ?? 1
        eosThreshold = try c.decodeIfPresent(Float.self, forKey: .eosThreshold) ?? -4
        noiseClamp = try c.decodeIfPresent(Float.self, forKey: .noiseClamp)
        audioPromptMinDuration = try c.decodeIfPresent(Double.self, forKey: .audioPromptMinDuration) ?? 0
        audioPromptMaxDuration = try c.decodeIfPresent(Double.self, forKey: .audioPromptMaxDuration)
        conditioners = try c.decodeIfPresent([PocketTTSConditionerConfig].self, forKey: .conditioners) ?? []
        fuser = try c.decodeIfPresent(PocketTTSFuserConfig.self, forKey: .fuser) ?? PocketTTSFuserConfig()
        weightsName = try c.decodeIfPresent(String.self, forKey: .weightsName)
        tokenizerName = try c.decodeIfPresent(String.self, forKey: .tokenizerName)
        guard audioPromptMinDuration.isFinite, audioPromptMinDuration >= 0,
              audioPromptMaxDuration.map({ $0.isFinite && $0 >= audioPromptMinDuration }) ?? true else {
            throw DecodingError.dataCorruptedError(
                forKey: .audioPromptMinDuration, in: c, debugDescription: "Invalid audio prompt duration limits")
        }
    }

    public static func load(from url: URL) throws -> PocketTTSModelConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(PocketTTSModelConfig.self, from: data)
    }
}
