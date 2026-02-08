//
//  Qwen3ASRConfig.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 06/02/2026.
//

import Foundation
import MLXLMCommon

// MARK: - Audio Encoder Config

public struct Qwen3AudioEncoderConfig: Codable {
    public var numMelBins: Int
    public var encoderLayers: Int
    public var encoderAttentionHeads: Int
    public var encoderFfnDim: Int
    public var dModel: Int
    public var dropout: Float
    public var attentionDropout: Float
    public var activationFunction: String
    public var activationDropout: Float
    public var scaleEmbedding: Bool
    public var maxSourcePositions: Int
    public var nWindow: Int
    public var outputDim: Int
    public var nWindowInfer: Int
    public var convChunksize: Int
    public var downsampleHiddenSize: Int

    enum CodingKeys: String, CodingKey {
        case numMelBins = "num_mel_bins"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
        case dModel = "d_model"
        case dropout
        case attentionDropout = "attention_dropout"
        case activationFunction = "activation_function"
        case activationDropout = "activation_dropout"
        case scaleEmbedding = "scale_embedding"
        case maxSourcePositions = "max_source_positions"
        case nWindow = "n_window"
        case outputDim = "output_dim"
        case nWindowInfer = "n_window_infer"
        case convChunksize = "conv_chunksize"
        case downsampleHiddenSize = "downsample_hidden_size"
    }

    public init(
        numMelBins: Int = 128,
        encoderLayers: Int = 24,
        encoderAttentionHeads: Int = 16,
        encoderFfnDim: Int = 4096,
        dModel: Int = 1024,
        dropout: Float = 0.0,
        attentionDropout: Float = 0.0,
        activationFunction: String = "gelu",
        activationDropout: Float = 0.0,
        scaleEmbedding: Bool = false,
        maxSourcePositions: Int = 1500,
        nWindow: Int = 50,
        outputDim: Int = 2048,
        nWindowInfer: Int = 800,
        convChunksize: Int = 500,
        downsampleHiddenSize: Int = 480
    ) {
        self.numMelBins = numMelBins
        self.encoderLayers = encoderLayers
        self.encoderAttentionHeads = encoderAttentionHeads
        self.encoderFfnDim = encoderFfnDim
        self.dModel = dModel
        self.dropout = dropout
        self.attentionDropout = attentionDropout
        self.activationFunction = activationFunction
        self.activationDropout = activationDropout
        self.scaleEmbedding = scaleEmbedding
        self.maxSourcePositions = maxSourcePositions
        self.nWindow = nWindow
        self.outputDim = outputDim
        self.nWindowInfer = nWindowInfer
        self.convChunksize = convChunksize
        self.downsampleHiddenSize = downsampleHiddenSize
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numMelBins = try container.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        encoderLayers = try container.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 24
        encoderAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 16
        encoderFfnDim = try container.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? 4096
        dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 1024
        dropout = try container.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.0
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        activationFunction = try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
        activationDropout = try container.decodeIfPresent(Float.self, forKey: .activationDropout) ?? 0.0
        scaleEmbedding = try container.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? false
        maxSourcePositions = try container.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        nWindow = try container.decodeIfPresent(Int.self, forKey: .nWindow) ?? 50
        outputDim = try container.decodeIfPresent(Int.self, forKey: .outputDim) ?? 2048
        nWindowInfer = try container.decodeIfPresent(Int.self, forKey: .nWindowInfer) ?? 800
        convChunksize = try container.decodeIfPresent(Int.self, forKey: .convChunksize) ?? 500
        downsampleHiddenSize = try container.decodeIfPresent(Int.self, forKey: .downsampleHiddenSize) ?? 480
    }
}

// MARK: - Text Config

public struct Qwen3TextConfig: Codable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var hiddenAct: String
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var useCache: Bool
    public var tieWordEmbeddings: Bool
    public var ropeTheta: Float
    public var ropeScaling: [String: StringAnyCodable]?
    public var attentionBias: Bool
    public var attentionDropout: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case useCache = "use_cache"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
    }

    public init(
        modelType: String = "qwen3",
        vocabSize: Int = 151936,
        hiddenSize: Int = 1024,
        intermediateSize: Int = 3072,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 65536,
        rmsNormEps: Float = 1e-6,
        useCache: Bool = true,
        tieWordEmbeddings: Bool = true,
        ropeTheta: Float = 1000000.0,
        ropeScaling: [String: StringAnyCodable]? = nil,
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.useCache = useCache
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151936
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 65536
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        useCache = try container.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1000000.0
        ropeScaling = try container.decodeIfPresent([String: StringAnyCodable].self, forKey: .ropeScaling)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
    }
}

// MARK: - Model Config

public struct Qwen3ASRConfig: Codable {
    public var audioConfig: Qwen3AudioEncoderConfig
    public var textConfig: Qwen3TextConfig
    public var modelType: String
    public var modelRepo: String?
    public var audioTokenId: Int
    public var audioStartTokenId: Int
    public var audioEndTokenId: Int
    public var supportLanguages: [String]

    // Quantization
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    // Forced aligner specific fields
    public var timestampTokenId: Int?
    public var timestampSegmentTime: Float?
    public var classifyNum: Int?

    public var isForcedAligner: Bool {
        return modelType == "qwen3_forced_aligner" || classifyNum != nil
    }

    enum CodingKeys: String, CodingKey {
        case audioConfig = "audio_config"
        case textConfig = "text_config"
        case modelType = "model_type"
        case modelRepo = "model_repo"
        case audioTokenId = "audio_token_id"
        case audioStartTokenId = "audio_start_token_id"
        case audioEndTokenId = "audio_end_token_id"
        case supportLanguages = "support_languages"
        case thinkerConfig = "thinker_config"
        case timestampTokenId = "timestamp_token_id"
        case timestampSegmentTime = "timestamp_segment_time"
        case classifyNum = "classify_num"
    }

    public init(
        audioConfig: Qwen3AudioEncoderConfig = Qwen3AudioEncoderConfig(),
        textConfig: Qwen3TextConfig = Qwen3TextConfig(),
        modelType: String = "qwen3_asr",
        modelRepo: String? = nil,
        audioTokenId: Int = 151676,
        audioStartTokenId: Int = 151669,
        audioEndTokenId: Int = 151670,
        supportLanguages: [String] = [],
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
        timestampTokenId: Int? = nil,
        timestampSegmentTime: Float? = nil,
        classifyNum: Int? = nil
    ) {
        self.audioConfig = audioConfig
        self.textConfig = textConfig
        self.modelType = modelType
        self.modelRepo = modelRepo
        self.audioTokenId = audioTokenId
        self.audioStartTokenId = audioStartTokenId
        self.audioEndTokenId = audioEndTokenId
        self.supportLanguages = supportLanguages
        self.perLayerQuantization = perLayerQuantization
        self.timestampTokenId = timestampTokenId
        self.timestampSegmentTime = timestampSegmentTime
        self.classifyNum = classifyNum
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_asr"
        modelRepo = try container.decodeIfPresent(String.self, forKey: .modelRepo)
        supportLanguages = try container.decodeIfPresent([String].self, forKey: .supportLanguages) ?? []

        let topTimestampTokenId = try container.decodeIfPresent(Int.self, forKey: .timestampTokenId)
        let topTimestampSegmentTime = try container.decodeIfPresent(Float.self, forKey: .timestampSegmentTime)

        // Try to decode from thinker_config first (HF format)
        if let thinkerContainer = try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .thinkerConfig) {
            audioConfig = try thinkerContainer.decodeIfPresent(Qwen3AudioEncoderConfig.self, forKey: .audioConfig) ?? Qwen3AudioEncoderConfig()
            textConfig = try thinkerContainer.decodeIfPresent(Qwen3TextConfig.self, forKey: .textConfig) ?? Qwen3TextConfig()
            audioTokenId = try thinkerContainer.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151676
            audioStartTokenId = try thinkerContainer.decodeIfPresent(Int.self, forKey: .audioStartTokenId) ?? 151669
            audioEndTokenId = try thinkerContainer.decodeIfPresent(Int.self, forKey: .audioEndTokenId) ?? 151670
            classifyNum = try thinkerContainer.decodeIfPresent(Int.self, forKey: .classifyNum)

            if let thinkerModelType = try? thinkerContainer.decodeIfPresent(String.self, forKey: .modelType),
               thinkerModelType == "qwen3_forced_aligner" {
                self.modelType = "qwen3_forced_aligner"
            }

            timestampTokenId = try thinkerContainer.decodeIfPresent(Int.self, forKey: .timestampTokenId) ?? topTimestampTokenId
            timestampSegmentTime = try thinkerContainer.decodeIfPresent(Float.self, forKey: .timestampSegmentTime) ?? topTimestampSegmentTime
        } else {
            audioConfig = try container.decodeIfPresent(Qwen3AudioEncoderConfig.self, forKey: .audioConfig) ?? Qwen3AudioEncoderConfig()
            textConfig = try container.decodeIfPresent(Qwen3TextConfig.self, forKey: .textConfig) ?? Qwen3TextConfig()
            audioTokenId = try container.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151676
            audioStartTokenId = try container.decodeIfPresent(Int.self, forKey: .audioStartTokenId) ?? 151669
            audioEndTokenId = try container.decodeIfPresent(Int.self, forKey: .audioEndTokenId) ?? 151670
            classifyNum = try container.decodeIfPresent(Int.self, forKey: .classifyNum)
            timestampTokenId = topTimestampTokenId
            timestampSegmentTime = topTimestampSegmentTime
        }

        let baseConfig = try? BaseConfiguration(from: decoder)
        perLayerQuantization = baseConfig?.perLayerQuantization
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(audioConfig, forKey: .audioConfig)
        try container.encode(textConfig, forKey: .textConfig)
        try container.encode(modelType, forKey: .modelType)
        try container.encodeIfPresent(modelRepo, forKey: .modelRepo)
        try container.encode(audioTokenId, forKey: .audioTokenId)
        try container.encode(audioStartTokenId, forKey: .audioStartTokenId)
        try container.encode(audioEndTokenId, forKey: .audioEndTokenId)
        try container.encode(supportLanguages, forKey: .supportLanguages)
        try container.encodeIfPresent(timestampTokenId, forKey: .timestampTokenId)
        try container.encodeIfPresent(timestampSegmentTime, forKey: .timestampSegmentTime)
        try container.encodeIfPresent(classifyNum, forKey: .classifyNum)
    }
}

// MARK: - Helper for arbitrary JSON values

public struct StringAnyCodable: Codable, Sendable {
    public let value: AnyCodableValue

    public enum AnyCodableValue: Sendable {
        case bool(Bool)
        case int(Int)
        case double(Double)
        case string(String)
        case array([StringAnyCodable])
        case dictionary([String: StringAnyCodable])
        case null
    }

    public init(_ value: Any) {
        switch value {
        case let b as Bool:
            self.value = .bool(b)
        case let i as Int:
            self.value = .int(i)
        case let d as Double:
            self.value = .double(d)
        case let s as String:
            self.value = .string(s)
        default:
            self.value = .null
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let bool = try? container.decode(Bool.self) {
            value = .bool(bool)
        } else if let int = try? container.decode(Int.self) {
            value = .int(int)
        } else if let double = try? container.decode(Double.self) {
            value = .double(double)
        } else if let string = try? container.decode(String.self) {
            value = .string(string)
        } else if let array = try? container.decode([StringAnyCodable].self) {
            value = .array(array)
        } else if let dictionary = try? container.decode([String: StringAnyCodable].self) {
            value = .dictionary(dictionary)
        } else {
            value = .null
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case .bool(let b):
            try container.encode(b)
        case .int(let i):
            try container.encode(i)
        case .double(let d):
            try container.encode(d)
        case .string(let s):
            try container.encode(s)
        case .array(let a):
            try container.encode(a)
        case .dictionary(let d):
            try container.encode(d)
        case .null:
            try container.encodeNil()
        }
    }
}
