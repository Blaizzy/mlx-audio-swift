import Foundation

// MARK: - FastConformer Encoder Config

public struct FCEncoderConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var intermediateSize: Int
    public var numMelBins: Int
    public var convKernelSize: Int
    public var subsamplingFactor: Int
    public var subsamplingConvChannels: Int
    public var subsamplingConvKernelSize: Int
    public var subsamplingConvStride: Int
    public var maxPositionEmbeddings: Int
    public var attentionBias: Bool
    public var scaleInput: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case numMelBins = "num_mel_bins"
        case convKernelSize = "conv_kernel_size"
        case subsamplingFactor = "subsampling_factor"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case subsamplingConvKernelSize = "subsampling_conv_kernel_size"
        case subsamplingConvStride = "subsampling_conv_stride"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case scaleInput = "scale_input"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 18
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2048
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 80
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 9
        subsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        subsamplingConvChannels = try c.decodeIfPresent(Int.self, forKey: .subsamplingConvChannels) ?? 256
        subsamplingConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .subsamplingConvKernelSize) ?? 3
        subsamplingConvStride = try c.decodeIfPresent(Int.self, forKey: .subsamplingConvStride) ?? 2
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 5000
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        scaleInput = try c.decodeIfPresent(Bool.self, forKey: .scaleInput) ?? true
    }
}

// MARK: - Transformer Encoder Config

public struct TFEncoderConfig: Codable, Sendable {
    public var dModel: Int
    public var encoderLayers: Int
    public var encoderAttentionHeads: Int
    public var encoderFfnDim: Int
    public var layerNormEps: Float
    public var maxSourcePositions: Int
    public var kProjBias: Bool

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
        case layerNormEps = "layer_norm_eps"
        case maxSourcePositions = "max_source_positions"
        case kProjBias = "k_proj_bias"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 192
        encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 18
        encoderAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 8
        encoderFfnDim = try c.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? 768
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        kProjBias = try c.decodeIfPresent(Bool.self, forKey: .kProjBias) ?? false
    }
}

// MARK: - Sortformer Modules Config

public struct ModulesConfig: Codable, Sendable {
    public var numSpeakers: Int
    public var fcDModel: Int
    public var tfDModel: Int
    public var subsamplingFactor: Int
    public var chunkLen: Int
    public var fifoLen: Int
    public var spkcacheLen: Int
    public var spkcacheUpdatePeriod: Int
    public var chunkLeftContext: Int
    public var chunkRightContext: Int
    public var spkcacheSilFramesPerSpk: Int
    public var predScoreThreshold: Float
    public var maxIndex: Int
    public var scoresBoostLatest: Float
    public var silThreshold: Float
    public var strongBoostRate: Float
    public var weakBoostRate: Float
    public var minPosScoresRate: Float
    public var useAosc: Bool

    enum CodingKeys: String, CodingKey {
        case numSpeakers = "num_speakers"
        case fcDModel = "fc_d_model"
        case tfDModel = "tf_d_model"
        case subsamplingFactor = "subsampling_factor"
        case chunkLen = "chunk_len"
        case fifoLen = "fifo_len"
        case spkcacheLen = "spkcache_len"
        case spkcacheUpdatePeriod = "spkcache_update_period"
        case chunkLeftContext = "chunk_left_context"
        case chunkRightContext = "chunk_right_context"
        case spkcacheSilFramesPerSpk = "spkcache_sil_frames_per_spk"
        case predScoreThreshold = "pred_score_threshold"
        case maxIndex = "max_index"
        case scoresBoostLatest = "scores_boost_latest"
        case silThreshold = "sil_threshold"
        case strongBoostRate = "strong_boost_rate"
        case weakBoostRate = "weak_boost_rate"
        case minPosScoresRate = "min_pos_scores_rate"
        case useAosc = "use_aosc"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numSpeakers = try c.decodeIfPresent(Int.self, forKey: .numSpeakers) ?? 4
        fcDModel = try c.decodeIfPresent(Int.self, forKey: .fcDModel) ?? 512
        tfDModel = try c.decodeIfPresent(Int.self, forKey: .tfDModel) ?? 192
        subsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        chunkLen = try c.decodeIfPresent(Int.self, forKey: .chunkLen) ?? 188
        fifoLen = try c.decodeIfPresent(Int.self, forKey: .fifoLen) ?? 0
        spkcacheLen = try c.decodeIfPresent(Int.self, forKey: .spkcacheLen) ?? 188
        spkcacheUpdatePeriod = try c.decodeIfPresent(Int.self, forKey: .spkcacheUpdatePeriod) ?? 188
        chunkLeftContext = try c.decodeIfPresent(Int.self, forKey: .chunkLeftContext) ?? 1
        chunkRightContext = try c.decodeIfPresent(Int.self, forKey: .chunkRightContext) ?? 1
        spkcacheSilFramesPerSpk = try c.decodeIfPresent(Int.self, forKey: .spkcacheSilFramesPerSpk) ?? 5
        predScoreThreshold = try c.decodeIfPresent(Float.self, forKey: .predScoreThreshold) ?? 1e-6
        maxIndex = try c.decodeIfPresent(Int.self, forKey: .maxIndex) ?? 10000
        scoresBoostLatest = try c.decodeIfPresent(Float.self, forKey: .scoresBoostLatest) ?? 0.5
        silThreshold = try c.decodeIfPresent(Float.self, forKey: .silThreshold) ?? 0.1
        strongBoostRate = try c.decodeIfPresent(Float.self, forKey: .strongBoostRate) ?? 0.3
        weakBoostRate = try c.decodeIfPresent(Float.self, forKey: .weakBoostRate) ?? 0.7
        minPosScoresRate = try c.decodeIfPresent(Float.self, forKey: .minPosScoresRate) ?? 0.5
        useAosc = try c.decodeIfPresent(Bool.self, forKey: .useAosc) ?? false
    }
}

// MARK: - Processor Config

public struct ProcessorConfig: Codable, Sendable {
    public var featureSize: Int
    public var samplingRate: Int
    public var hopLength: Int
    public var nFft: Int
    public var winLength: Int
    public var preemphasis: Float

    enum CodingKeys: String, CodingKey {
        case featureSize = "feature_size"
        case samplingRate = "sampling_rate"
        case hopLength = "hop_length"
        case nFft = "n_fft"
        case winLength = "win_length"
        case preemphasis
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featureSize = try c.decodeIfPresent(Int.self, forKey: .featureSize) ?? 80
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 16000
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 160
        nFft = try c.decodeIfPresent(Int.self, forKey: .nFft) ?? 512
        winLength = try c.decodeIfPresent(Int.self, forKey: .winLength) ?? 400
        preemphasis = try c.decodeIfPresent(Float.self, forKey: .preemphasis) ?? 0.97
    }
}

// MARK: - Top-Level Model Config

public struct SortformerConfig: Codable, Sendable {
    public var modelType: String
    public var numSpeakers: Int
    public var fcEncoderConfig: FCEncoderConfig
    public var tfEncoderConfig: TFEncoderConfig
    public var modulesConfig: ModulesConfig
    public var processorConfig: ProcessorConfig

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case numSpeakers = "num_speakers"
        case fcEncoderConfig = "fc_encoder_config"
        case tfEncoderConfig = "tf_encoder_config"
        case modulesConfig = "modules_config"
        case processorConfig = "processor_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "sortformer"
        numSpeakers = try c.decodeIfPresent(Int.self, forKey: .numSpeakers) ?? 4
        fcEncoderConfig = try c.decodeIfPresent(FCEncoderConfig.self, forKey: .fcEncoderConfig) ?? FCEncoderConfig(from: decoder)
        tfEncoderConfig = try c.decodeIfPresent(TFEncoderConfig.self, forKey: .tfEncoderConfig) ?? TFEncoderConfig(from: decoder)
        modulesConfig = try c.decodeIfPresent(ModulesConfig.self, forKey: .modulesConfig) ?? ModulesConfig(from: decoder)
        processorConfig = try c.decodeIfPresent(ProcessorConfig.self, forKey: .processorConfig) ?? ProcessorConfig(from: decoder)
    }
}
