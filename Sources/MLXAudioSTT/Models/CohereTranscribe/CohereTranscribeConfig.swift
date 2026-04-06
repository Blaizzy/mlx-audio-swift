import Foundation
import MLX
import MLXLMCommon

public struct CohereTranscribeAudioEncoderConfig: Codable, Sendable {
    public let dModel: Int
    public let ffExpansionFactor: Int
    public let nHeads: Int
    public let convKernelSize: Int
    public let nLayers: Int
    public let posEmbMaxLen: Int
    public let subsamplingConvChannels: Int
    public let subsamplingFactor: Int
    public let featIn: Int

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case ffExpansionFactor = "ff_expansion_factor"
        case nHeads = "n_heads"
        case convKernelSize = "conv_kernel_size"
        case nLayers = "n_layers"
        case posEmbMaxLen = "pos_emb_max_len"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case subsamplingFactor = "subsampling_factor"
        case featIn = "feat_in"
    }
}

public struct CohereTranscribeTextDecoderConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let innerSize: Int
    public let numAttentionHeads: Int
    public let numLayers: Int
    public let maxSequenceLength: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case innerSize = "inner_size"
        case numAttentionHeads = "num_attention_heads"
        case numLayers = "num_layers"
        case maxSequenceLength = "max_sequence_length"
    }
}

public struct CohereTranscribeConfig: Decodable, Sendable {
    public let modelType: String
    public let vocabSize: Int
    public let sampleRate: Int
    public let maxAudioClipS: Int
    public var quantization: BaseConfiguration.Quantization?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    
    public let encoder: CohereTranscribeAudioEncoderConfig
    
    private let transfDecoder: TransfDecoderWrapper
    public var decoder: CohereTranscribeTextDecoderConfig {
        transfDecoder.configDict
    }
    
    struct TransfDecoderWrapper: Codable, Sendable {
        let configDict: CohereTranscribeTextDecoderConfig
        
        enum CodingKeys: String, CodingKey {
            case configDict = "config_dict"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case sampleRate = "sample_rate"
        case maxAudioClipS = "max_audio_clip_s"
        case encoder
        case transfDecoder = "transf_decoder"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        sampleRate = try container.decode(Int.self, forKey: .sampleRate)
        maxAudioClipS = try container.decode(Int.self, forKey: .maxAudioClipS)
        encoder = try container.decode(CohereTranscribeAudioEncoderConfig.self, forKey: .encoder)
        transfDecoder = try container.decode(TransfDecoderWrapper.self, forKey: .transfDecoder)

        let baseConfig = try? BaseConfiguration(from: decoder)
        let globalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        let altGlobalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
        quantization = globalQuant ?? altGlobalQuant
        perLayerQuantization = baseConfig?.perLayerQuantization
    }
}
