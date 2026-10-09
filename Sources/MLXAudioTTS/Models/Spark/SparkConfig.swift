import Foundation

/// BiCodec codec configuration, decoded from the checkpoint's `BiCodec/config.yaml`.
public struct BiCodecConfiguration: Codable, Sendable {
    public struct MelParams: Codable, Sendable {
        public var sampleRate: Int

        enum CodingKeys: String, CodingKey {
            case sampleRate = "sample_rate"
        }
    }

    /// Vocos-backbone config for the conditioned prenet.
    public struct VocosBackbone: Codable, Sendable {
        public var inputChannels: Int
        public var vocosDim: Int
        public var vocosIntermediateDim: Int
        public var vocosNumLayers: Int
        public var outChannels: Int
        public var conditionDim: Int?
        public var sampleRatios: [Int]?
        public var useTanhAtFinal: Bool?

        enum CodingKeys: String, CodingKey {
            case inputChannels = "input_channels"
            case vocosDim = "vocos_dim"
            case vocosIntermediateDim = "vocos_intermediate_dim"
            case vocosNumLayers = "vocos_num_layers"
            case outChannels = "out_channels"
            case conditionDim = "condition_dim"
            case sampleRatios = "sample_ratios"
            case useTanhAtFinal = "use_tanh_at_final"
        }
    }

    /// HiFiGAN-style wave generator (decoder).
    public struct WaveGenerator: Codable, Sendable {
        public var inputChannel: Int
        public var channels: Int
        public var rates: [Int]
        public var kernelSizes: [Int]

        enum CodingKeys: String, CodingKey {
            case inputChannel = "input_channel"
            case channels
            case rates
            case kernelSizes = "kernel_sizes"
        }
    }

    /// Factorized vector quantizer holding the semantic-token codebook.
    public struct Quantizer: Codable, Sendable {
        public var inputDim: Int
        public var codebookSize: Int
        public var codebookDim: Int

        enum CodingKeys: String, CodingKey {
            case inputDim = "input_dim"
            case codebookSize = "codebook_size"
            case codebookDim = "codebook_dim"
        }
    }

    /// Finite-scalar-quantized speaker encoder decoding global tokens.
    public struct SpeakerEncoder: Codable, Sendable {
        public var inputDim: Int?
        public var outDim: Int
        public var latentDim: Int
        public var tokenNum: Int
        public var fsqLevels: [Int]

        enum CodingKeys: String, CodingKey {
            case inputDim = "input_dim"
            case outDim = "out_dim"
            case latentDim = "latent_dim"
            case tokenNum = "token_num"
            case fsqLevels = "fsq_levels"
        }
    }

    public var melParams: MelParams
    public var encoder: VocosBackbone?
    public var decoder: WaveGenerator
    public var quantizer: Quantizer
    public var speakerEncoder: SpeakerEncoder
    public var prenet: VocosBackbone

    enum CodingKeys: String, CodingKey {
        case melParams = "mel_params"
        case encoder, decoder, quantizer
        case speakerEncoder = "speaker_encoder"
        case prenet
    }
}
