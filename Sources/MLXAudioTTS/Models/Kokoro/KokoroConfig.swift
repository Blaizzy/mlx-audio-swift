import Foundation

/// Configuration for the Kokoro TTS model, loaded from HuggingFace config.json.
public struct KokoroConfig: Codable {

    /// iSTFT decoder network configuration.
    public struct ISTFTNetConfig: Codable {
        public let upsampleKernelSizes: [Int]
        public let upsampleRates: [Int]
        public let genIstftHopSize: Int
        public let genIstftNFft: Int
        public let resblockDilationSizes: [[Int]]
        public let resblockKernelSizes: [Int]
        public let upsampleInitialChannel: Int

        enum CodingKeys: String, CodingKey {
            case upsampleKernelSizes = "upsample_kernel_sizes"
            case upsampleRates = "upsample_rates"
            case genIstftHopSize = "gen_istft_hop_size"
            case genIstftNFft = "gen_istft_n_fft"
            case resblockDilationSizes = "resblock_dilation_sizes"
            case resblockKernelSizes = "resblock_kernel_sizes"
            case upsampleInitialChannel = "upsample_initial_channel"
        }
    }

    /// PLBERT encoder configuration.
    public struct PLBERTConfig: Codable {
        public let hiddenSize: Int
        public let numAttentionHeads: Int
        public let intermediateSize: Int
        public let maxPositionEmbeddings: Int
        public let numHiddenLayers: Int
        public let dropout: Double

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numAttentionHeads = "num_attention_heads"
            case intermediateSize = "intermediate_size"
            case maxPositionEmbeddings = "max_position_embeddings"
            case numHiddenLayers = "num_hidden_layers"
            case dropout
        }
    }

    public let istftNet: ISTFTNetConfig
    public let dimIn: Int
    public let dropout: Double
    public let hiddenDim: Int
    public let maxConvDim: Int
    public let maxDur: Int
    public let multispeaker: Bool
    public let nLayer: Int
    public let nMels: Int
    public let nToken: Int
    public let styleDim: Int
    public let textEncoderKernelSize: Int
    public let plbert: PLBERTConfig
    public let vocab: [String: Int]

    enum CodingKeys: String, CodingKey {
        case istftNet = "istftnet"
        case dimIn = "dim_in"
        case dropout
        case hiddenDim = "hidden_dim"
        case maxConvDim = "max_conv_dim"
        case maxDur = "max_dur"
        case multispeaker
        case nLayer = "n_layer"
        case nMels = "n_mels"
        case nToken = "n_token"
        case styleDim = "style_dim"
        case textEncoderKernelSize = "text_encoder_kernel_size"
        case plbert
        case vocab
    }

    /// Loads config from a JSON file at the given path.
    public static func load(from url: URL) throws -> KokoroConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(KokoroConfig.self, from: data)
    }
}
