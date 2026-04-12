import Foundation
import MLXLMCommon

// MARK: - LLM Config (nested within OmniVoice config)

public struct OmniVoiceLLMConfig: Codable, Sendable {
    var architectures: [String]
    var attentionBias: Bool
    var attentionDropout: Float
    var bosTokenId: Int
    var chunkSizeFeedForward: Int
    var dtype: String
    var eosTokenId: Int
    var headDim: Int
    var hiddenAct: String
    var hiddenSize: Int
    var initializerRange: Float
    var intermediateSize: Int
    var isEncoderDecoder: Bool
    var layerTypes: [String]
    var maxPositionEmbeddings: Int
    var maxWindowLayers: Int
    var modelType: String
    var numAttentionHeads: Int
    var numHiddenLayers: Int
    var numKeyValueHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeType: String
    var tieWordEmbeddings: Bool
    var useCache: Bool
    var vocabSize: Int

    enum CodingKeys: String, CodingKey {
        case architectures
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case bosTokenId = "bos_token_id"
        case chunkSizeFeedForward = "chunk_size_feed_forward"
        case dtype
        case eosTokenId = "eos_token_id"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case hiddenSize = "hidden_size"
        case initializerRange = "initializer_range"
        case intermediateSize = "intermediate_size"
        case isEncoderDecoder = "is_encoder_decoder"
        case layerTypes = "layer_types"
        case maxPositionEmbeddings = "max_position_embeddings"
        case maxWindowLayers = "max_window_layers"
        case modelType = "model_type"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useCache = "use_cache"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? ["Qwen3ForCausalLM"]
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 151643
        chunkSizeFeedForward = try c.decodeIfPresent(Int.self, forKey: .chunkSizeFeedForward) ?? 0
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype) ?? "float32"
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 151645
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        isEncoderDecoder = try c.decodeIfPresent(Bool.self, forKey: .isEncoderDecoder) ?? false
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 40960
        maxWindowLayers = try c.decodeIfPresent(Int.self, forKey: .maxWindowLayers) ?? 28
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeType = try c.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        useCache = try c.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151676
    }
}

// MARK: - Audio Tokenizer Config

public struct OmniVoiceAudioTokenizerConfig: Codable, Sendable {
    // Acoustic model (DAC) config
    var codebookSize: Int
    var codebookDim: Int
    var nCodebooks: Int
    var hopLength: Int
    var samplingRate: Int
    var downsamplingRatios: [Int]
    var upsamplingRatios: [Int]
    var encoderHiddenSize: Int
    var decoderHiddenSize: Int
    var kernelSize: Int

    // Hubert semantic config
    var hiddenSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var intermediateSize: Int
    var convDim: [Int]
    var convKernel: [Int]
    var convStride: [Int]

    // General
    var sampleRate: Int
    var semanticSampleRate: Int
    var downsampleFactor: Int

    enum CodingKeys: String, CodingKey {
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case nCodebooks = "n_codebooks"
        case hopLength = "hop_length"
        case samplingRate = "sampling_rate"
        case downsamplingRatios = "downsampling_ratios"
        case upsamplingRatios = "upsampling_ratios"
        case encoderHiddenSize = "encoder_hidden_size"
        case decoderHiddenSize = "decoder_hidden_size"
        case kernelSize = "kernel_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case convDim = "conv_dim"
        case convKernel = "conv_kernel"
        case convStride = "conv_stride"
        case sampleRate = "sample_rate"
        case semanticSampleRate = "semantic_sample_rate"
        case downsampleFactor = "downsample_factor"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 1024
        codebookDim = try c.decodeIfPresent(Int.self, forKey: .codebookDim) ?? 64
        nCodebooks = try c.decodeIfPresent(Int.self, forKey: .nCodebooks) ?? 9
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 960
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 16000
        downsamplingRatios = try c.decodeIfPresent([Int].self, forKey: .downsamplingRatios) ?? [8, 5, 4, 2, 3]
        upsamplingRatios = try c.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [8, 5, 4, 2, 3]
        encoderHiddenSize = try c.decodeIfPresent(Int.self, forKey: .encoderHiddenSize) ?? 64
        decoderHiddenSize = try c.decodeIfPresent(Int.self, forKey: .decoderHiddenSize) ?? 1024
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 3
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        convDim = try c.decodeIfPresent([Int].self, forKey: .convDim) ?? [512, 512, 512, 512, 512, 512, 512]
        convKernel = try c.decodeIfPresent([Int].self, forKey: .convKernel) ?? [10, 3, 3, 3, 3, 2, 2]
        convStride = try c.decodeIfPresent([Int].self, forKey: .convStride) ?? [5, 2, 2, 2, 2, 2, 2]
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        semanticSampleRate = try c.decodeIfPresent(Int.self, forKey: .semanticSampleRate) ?? 16000
        downsampleFactor = try c.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 320
    }
}

// MARK: - Top-level OmniVoice Config

public struct OmniVoiceConfig: Codable, Sendable {
    var modelType: String
    var architectures: [String]
    var dtype: String

    // LLM backbone config
    var llmConfig: OmniVoiceLLMConfig

    // Audio codebook settings
    var audioCodebookWeights: [Int]
    var audioMaskId: Int
    var audioVocabSize: Int

    // Token IDs
    var bosTokenId: Int?
    var eosTokenId: Int
    var padTokenId: Int

    // Audio codebook
    var numAudioCodebook: Int

    // Quantization
    var quantization: BaseConfiguration.Quantization?
    var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case dtype
        case llmConfig = "llm_config"
        case audioCodebookWeights = "audio_codebook_weights"
        case audioMaskId = "audio_mask_id"
        case audioVocabSize = "audio_vocab_size"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case numAudioCodebook = "num_audio_codebook"
        case quantization
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "omnivoice"
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? ["OmniVoice"]
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype) ?? "bfloat16"
        llmConfig = try c.decode(OmniVoiceLLMConfig.self, forKey: .llmConfig)
        audioCodebookWeights = try c.decodeIfPresent([Int].self, forKey: .audioCodebookWeights) ?? [8, 8, 6, 6, 4, 4, 2, 2]
        audioMaskId = try c.decodeIfPresent(Int.self, forKey: .audioMaskId) ?? 1024
        audioVocabSize = try c.decodeIfPresent(Int.self, forKey: .audioVocabSize) ?? 1025
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId)
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 151645
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 151643
        numAudioCodebook = try c.decodeIfPresent(Int.self, forKey: .numAudioCodebook) ?? 8

        // Try global quantization
        let baseConfig = try? BaseConfiguration(from: decoder)
        quantization = try c.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        if quantization == nil {
            quantization = try? BaseConfiguration.Quantization(from: decoder)
        }
        perLayerQuantization = baseConfig?.perLayerQuantization
    }

    /// The expected sample rate for output audio.
    public var sampleRate: Int {
        24000
    }
}
