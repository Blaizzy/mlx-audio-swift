import Foundation

public struct PocketTTSFlowConfig: Codable {
    public let dim: Int
    public let depth: Int
}

public struct PocketTTSFlowLMTransformerConfig: Codable {
    public let hiddenScale: Int
    public let maxPeriod: Double
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int

    enum CodingKeys: String, CodingKey {
        case hiddenScale = "hidden_scale"
        case maxPeriod = "max_period"
        case dModel = "d_model"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
    }
}

public struct PocketTTSLookupTableConfig: Codable {
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

public struct PocketTTSFlowLMConfig: Codable {
    public let dtype: String?
    public let flow: PocketTTSFlowConfig
    public let transformer: PocketTTSFlowLMTransformerConfig
    public let lookupTable: PocketTTSLookupTableConfig
    public let weightsPath: String?

    enum CodingKeys: String, CodingKey {
        case dtype
        case flow
        case transformer
        case lookupTable = "lookup_table"
        case weightsPath = "weights_path"
    }
}

public struct PocketTTSSeanetConfig: Codable {
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

public struct PocketTTSMimiTransformerConfig: Codable {
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

public struct PocketTTSQuantizerConfig: Codable {
    public let dimension: Int
    public let outputDimension: Int

    enum CodingKeys: String, CodingKey {
        case dimension
        case outputDimension = "output_dimension"
    }
}

public struct PocketTTSMimiConfig: Codable {
    public let dtype: String?
    public let sampleRate: Int
    public let channels: Int
    public let frameRate: Double
    public let seanet: PocketTTSSeanetConfig
    public let transformer: PocketTTSMimiTransformerConfig
    public let quantizer: PocketTTSQuantizerConfig
    public let weightsPath: String?

    enum CodingKeys: String, CodingKey {
        case dtype
        case sampleRate = "sample_rate"
        case channels
        case frameRate = "frame_rate"
        case seanet
        case transformer
        case quantizer
        case weightsPath = "weights_path"
    }
}

public struct PocketTTSModelConfig: Codable {
    public let modelType: String
    public let flowLM: PocketTTSFlowLMConfig
    public let mimi: PocketTTSMimiConfig
    public let weightsPath: String?
    public let weightsPathWithoutVoiceCloning: String?
    public let modelPath: String?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case flowLM = "flow_lm"
        case mimi
        case weightsPath = "weights_path"
        case weightsPathWithoutVoiceCloning = "weights_path_without_voice_cloning"
        case modelPath = "model_path"
    }

    public static func load(from url: URL) throws -> PocketTTSModelConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(PocketTTSModelConfig.self, from: data)
    }
}
