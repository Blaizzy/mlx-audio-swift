import Foundation
import MLXLMCommon

public struct MiMoAudioTokenizerConfig: Decodable, Sendable {
    public let maxAudioSeconds: Int
    public let strideSize: Int
    public let avgPooler: Int
    public let dModel: Int
    public let scaleEmbedding: Bool
    public let kernelSize: Int
    public let activationFunction: String
    public let encoderLayers: Int
    public let encoderSkipLayerID: Int?
    public let encoderAttentionHeads: Int
    public let encoderFFNDim: Int
    public let encoderCausal: Bool
    public let encoderAttentionWindowSize: [Int]
    public let decoderLayers: Int
    public let decoderAttentionHeads: Int
    public let decoderFFNDim: Int
    public let decoderKernelSize: Int
    public let decoderStrideSize: Int
    public let decoderCausal: Bool
    public let decoderAttentionWindowSize: [Int]
    public let nfft: Int
    public let vocoderDim: Int
    public let vocoderIntermediateDim: Int
    public let vocoderNumLayers: Int
    public let nMels: Int
    public let samplingRate: Int
    public let hopLength: Int
    public let windowSize: Int
    public let vocoderPadding: String
    public let fmin: Int
    public let fmax: Int?
    public let numQuantizers: Int
    public let codebookSize: [Int]
    public let thresholdEMADeadCode: Int
    public let positionEmbeddingType: String
    public let ropeTheta: Int
    public let ropeType: String
    public let layerNormType: String
    public let vocoderAttentionHeads: Int
    public let vocoderAttentionWindowSize: [Int]
    public let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case maxAudioSeconds = "max_audio_seconds"
        case strideSize = "stride_size"
        case avgPooler = "avg_pooler"
        case dModel = "d_model"
        case scaleEmbedding = "scale_embedding"
        case kernelSize = "kernel_size"
        case activationFunction = "activation_function"
        case encoderLayers = "encoder_layers"
        case encoderSkipLayerID = "encoder_skip_layer_id"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFFNDim = "encoder_ffn_dim"
        case encoderCausal = "encoder_causal"
        case encoderAttentionWindowSize = "encoder_attn_window_size"
        case decoderLayers = "decoder_layers"
        case decoderAttentionHeads = "decoder_attention_heads"
        case decoderFFNDim = "decoder_ffn_dim"
        case decoderKernelSize = "decoder_kernel_size"
        case decoderStrideSize = "decoder_stride_size"
        case decoderCausal = "decoder_causal"
        case decoderAttentionWindowSize = "decoder_attn_window_size"
        case nfft
        case vocoderDim = "vocoder_dim"
        case vocoderIntermediateDim = "vocoder_intermediate_dim"
        case vocoderNumLayers = "vocoder_num_layers"
        case nMels = "n_mels"
        case samplingRate = "sampling_rate"
        case hopLength = "hop_length"
        case windowSize = "window_size"
        case vocoderPadding = "vocoder_padding"
        case fmin
        case fmax
        case numQuantizers = "num_quantizers"
        case codebookSize = "codebook_size"
        case thresholdEMADeadCode = "threshold_ema_dead_code"
        case positionEmbeddingType = "position_embedding_type"
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case layerNormType = "ln_type"
        case vocoderAttentionHeads = "vocoder_attention_heads"
        case vocoderAttentionWindowSize = "vocoder_attn_window_size"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        maxAudioSeconds = try container.decode(Int.self, forKey: .maxAudioSeconds)
        strideSize = try container.decode(Int.self, forKey: .strideSize)
        avgPooler = try container.decode(Int.self, forKey: .avgPooler)
        dModel = try container.decode(Int.self, forKey: .dModel)
        scaleEmbedding = try container.decode(Bool.self, forKey: .scaleEmbedding)
        kernelSize = try container.decode(Int.self, forKey: .kernelSize)
        activationFunction = try container.decode(String.self, forKey: .activationFunction)
        encoderLayers = try container.decode(Int.self, forKey: .encoderLayers)
        encoderSkipLayerID = try container.decodeIfPresent(Int.self, forKey: .encoderSkipLayerID)
        encoderAttentionHeads = try container.decode(Int.self, forKey: .encoderAttentionHeads)
        encoderFFNDim = try container.decode(Int.self, forKey: .encoderFFNDim)
        encoderCausal = try container.decode(Bool.self, forKey: .encoderCausal)
        encoderAttentionWindowSize = try container.decode([Int].self, forKey: .encoderAttentionWindowSize)
        decoderLayers = try container.decode(Int.self, forKey: .decoderLayers)
        decoderAttentionHeads = try container.decode(Int.self, forKey: .decoderAttentionHeads)
        decoderFFNDim = try container.decode(Int.self, forKey: .decoderFFNDim)
        decoderKernelSize = try container.decode(Int.self, forKey: .decoderKernelSize)
        decoderStrideSize = try container.decode(Int.self, forKey: .decoderStrideSize)
        decoderCausal = try container.decode(Bool.self, forKey: .decoderCausal)
        decoderAttentionWindowSize = try container.decode([Int].self, forKey: .decoderAttentionWindowSize)
        nfft = try container.decode(Int.self, forKey: .nfft)
        vocoderDim = try container.decode(Int.self, forKey: .vocoderDim)
        vocoderIntermediateDim = try container.decode(Int.self, forKey: .vocoderIntermediateDim)
        vocoderNumLayers = try container.decode(Int.self, forKey: .vocoderNumLayers)
        nMels = try container.decode(Int.self, forKey: .nMels)
        samplingRate = try container.decode(Int.self, forKey: .samplingRate)
        hopLength = try container.decode(Int.self, forKey: .hopLength)
        windowSize = try container.decode(Int.self, forKey: .windowSize)
        vocoderPadding = try container.decode(String.self, forKey: .vocoderPadding)
        fmin = try container.decode(Int.self, forKey: .fmin)
        fmax = try container.decodeIfPresent(Int.self, forKey: .fmax)
        numQuantizers = try container.decode(Int.self, forKey: .numQuantizers)
        codebookSize = try container.decode([Int].self, forKey: .codebookSize)
        thresholdEMADeadCode = try container.decode(Int.self, forKey: .thresholdEMADeadCode)
        positionEmbeddingType = try container.decode(String.self, forKey: .positionEmbeddingType)
        ropeTheta = try container.decode(Int.self, forKey: .ropeTheta)
        ropeType = try container.decode(String.self, forKey: .ropeType)
        layerNormType = try container.decode(String.self, forKey: .layerNormType)
        vocoderAttentionHeads = try container.decode(Int.self, forKey: .vocoderAttentionHeads)
        vocoderAttentionWindowSize = try container.decode([Int].self, forKey: .vocoderAttentionWindowSize)

        let globalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        let altGlobalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
        quantization = globalQuant ?? altGlobalQuant
    }

    public func activeCodebookSizes(prefixCount: Int? = nil) -> [Int] {
        guard let prefixCount else { return codebookSize }
        return Array(codebookSize.prefix(max(0, prefixCount)))
    }
}
