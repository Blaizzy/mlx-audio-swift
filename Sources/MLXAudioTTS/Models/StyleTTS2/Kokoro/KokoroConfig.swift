import Foundation
@preconcurrency import MLXLMCommon

public struct KokoroConfig: Decodable {
    let modelType: String
    let dimIn: Int
    let dropout: Float
    let hiddenDim: Int
    let maxConvDim: Int
    let maxDur: Int
    let multispeaker: Bool
    let nLayer: Int
    let nMels: Int
    let nToken: Int
    let styleDim: Int
    let textEncoderKernelSize: Int
    let vocab: [String: Int]

    let plbert: PLBertConfig
    let istftnet: ISTFTNetConfig

    let sampleRate: Int
    let asrResDim: Int
    let voicesPath: String?
    let speedPriors: [String: Float]?
    let voiceAliases: [String: String]?

    let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
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
        case vocab
        case plbert
        case istftnet
        case sampleRate = "sample_rate"
        case asrResDim = "asr_res_dim"
        case voicesPath = "voices_path"
        case speedPriors = "speed_priors"
        case voiceAliases = "voice_aliases"
        case quantization
        case quantizationAlt = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "kokoro"
        dimIn = try container.decodeIfPresent(Int.self, forKey: .dimIn) ?? 64
        dropout = try container.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.2
        hiddenDim = try container.decode(Int.self, forKey: .hiddenDim)
        maxConvDim = try container.decodeIfPresent(Int.self, forKey: .maxConvDim) ?? 512
        maxDur = try container.decodeIfPresent(Int.self, forKey: .maxDur) ?? 50
        multispeaker = try container.decodeIfPresent(Bool.self, forKey: .multispeaker) ?? false
        nLayer = try container.decodeIfPresent(Int.self, forKey: .nLayer) ?? 3
        nMels = try container.decodeIfPresent(Int.self, forKey: .nMels) ?? 80
        nToken = try container.decode(Int.self, forKey: .nToken)
        styleDim = try container.decodeIfPresent(Int.self, forKey: .styleDim) ?? 128
        textEncoderKernelSize = try container.decodeIfPresent(Int.self, forKey: .textEncoderKernelSize) ?? 5
        vocab = try container.decode([String: Int].self, forKey: .vocab)
        plbert = try container.decode(PLBertConfig.self, forKey: .plbert)
        istftnet = try container.decode(ISTFTNetConfig.self, forKey: .istftnet)
        sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        asrResDim = try container.decodeIfPresent(Int.self, forKey: .asrResDim) ?? 64
        voicesPath = try container.decodeIfPresent(String.self, forKey: .voicesPath)
        speedPriors = try container.decodeIfPresent([String: Float].self, forKey: .speedPriors)
        voiceAliases = try container.decodeIfPresent([String: String].self, forKey: .voiceAliases)
        quantization =
            try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
            ?? container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantizationAlt)
    }
}
