//
//  GLMASRConfig.swift
//  MLXAudioSTT
//
// Created by Prince Canuma on 04/01/2026.
//
import MLXLMCommon

import Foundation

/// Configuration for the Whisper audio encoder.
public struct WhisperConfig: Codable {
    public var modelType: String
    public var activationFunction: String
    public var dModel: Int
    public var encoderAttentionHeads: Int
    public var encoderFfnDim: Int
    public var encoderLayers: Int
    public var encoderLayerdrop: Float
    public var numMelBins: Int
    public var maxSourcePositions: Int
    public var dropout: Float
    public var attentionDropout: Float
    public var activationDropout: Float
    public var initStd: Float
    public var scaleEmbedding: Bool
    public var ropeTraditional: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case activationFunction = "activation_function"
        case hiddenAct = "hidden_act"
        case dModel = "d_model"
        case hiddenSize = "hidden_size"
        case encoderAttentionHeads = "encoder_attention_heads"
        case numAttentionHeads = "num_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
        case intermediateSize = "intermediate_size"
        case encoderLayers = "encoder_layers"
        case numHiddenLayers = "num_hidden_layers"
        case encoderLayerdrop = "encoder_layerdrop"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
        case maxPositionEmbeddings = "max_position_embeddings"
        case dropout
        case attentionDropout = "attention_dropout"
        case activationDropout = "activation_dropout"
        case initStd = "init_std"
        case scaleEmbedding = "scale_embedding"
        case ropeTraditional = "rope_traditional"
    }

    public init(
        modelType: String = "whisper",
        activationFunction: String = "gelu",
        dModel: Int = 1280,
        encoderAttentionHeads: Int = 20,
        encoderFfnDim: Int = 5120,
        encoderLayers: Int = 32,
        encoderLayerdrop: Float = 0.0,
        numMelBins: Int = 128,
        maxSourcePositions: Int = 1500,
        dropout: Float = 0.0,
        attentionDropout: Float = 0.0,
        activationDropout: Float = 0.0,
        initStd: Float = 0.02,
        scaleEmbedding: Bool = false,
        ropeTraditional: Bool = true
    ) {
        self.modelType = modelType
        self.activationFunction = activationFunction
        self.dModel = dModel
        self.encoderAttentionHeads = encoderAttentionHeads
        self.encoderFfnDim = encoderFfnDim
        self.encoderLayers = encoderLayers
        self.encoderLayerdrop = encoderLayerdrop
        self.numMelBins = numMelBins
        self.maxSourcePositions = maxSourcePositions
        self.dropout = dropout
        self.attentionDropout = attentionDropout
        self.activationDropout = activationDropout
        self.initStd = initStd
        self.scaleEmbedding = scaleEmbedding
        self.ropeTraditional = ropeTraditional
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "glmasr_encoder"
        // Try hidden_act first (GLM-ASR style), then activation_function (Whisper style)
        activationFunction = try container.decodeIfPresent(String.self, forKey: .hiddenAct)
            ?? container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
        // Try hidden_size first (GLM-ASR style), then d_model (Whisper style)
        dModel = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? container.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        // Try num_attention_heads first, then encoder_attention_heads
        encoderAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
            ?? container.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 20
        // Try intermediate_size first, then encoder_ffn_dim
        encoderFfnDim = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? container.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? 5120
        // Try num_hidden_layers first, then encoder_layers
        encoderLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? container.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 32
        encoderLayerdrop = try container.decodeIfPresent(Float.self, forKey: .encoderLayerdrop) ?? 0.0
        numMelBins = try container.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        // Try max_position_embeddings first, then max_source_positions
        maxSourcePositions = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? container.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        dropout = try container.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.0
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        activationDropout = try container.decodeIfPresent(Float.self, forKey: .activationDropout) ?? 0.0
        initStd = try container.decodeIfPresent(Float.self, forKey: .initStd) ?? 0.02
        scaleEmbedding = try container.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? false
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(activationFunction, forKey: .activationFunction)
        try container.encode(dModel, forKey: .dModel)
        try container.encode(encoderAttentionHeads, forKey: .encoderAttentionHeads)
        try container.encode(encoderFfnDim, forKey: .encoderFfnDim)
        try container.encode(encoderLayers, forKey: .encoderLayers)
        try container.encode(encoderLayerdrop, forKey: .encoderLayerdrop)
        try container.encode(numMelBins, forKey: .numMelBins)
        try container.encode(maxSourcePositions, forKey: .maxSourcePositions)
        try container.encode(dropout, forKey: .dropout)
        try container.encode(attentionDropout, forKey: .attentionDropout)
        try container.encode(activationDropout, forKey: .activationDropout)
        try container.encode(initStd, forKey: .initStd)
        try container.encode(scaleEmbedding, forKey: .scaleEmbedding)
        try container.encode(ropeTraditional, forKey: .ropeTraditional)
    }
}

/// Configuration for the LLaMA language model.
public struct LlamaConfig: Codable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var hiddenAct: String
    public var headDim: Int?
    public var maxPositionEmbeddings: Int
    public var layerTypes: [String]?
    public var initializerRange: Float
    public var rmsNormEps: Float
    public var slidingWindow: Int?
    public var ropeTraditional: Bool
    public var ropeScaling: [String: AnyCodable]?
    public var ropeTheta: Float
    public var ropeDim: Int
    public var tieWordEmbeddings: Bool
    public var attentionBias: Bool
    public var mlpBias: Bool
    public var padTokenId: Int
    public var eosTokenId: [Int]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case hiddenAct = "hidden_act"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerTypes = "layer_types"
        case initializerRange = "initializer_range"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
        case ropeDim = "rope_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case padTokenId = "pad_token_id"
        case eosTokenId = "eos_token_id"
    }

    public init(
        modelType: String = "llama",
        vocabSize: Int = 59264,
        hiddenSize: Int = 2048,
        intermediateSize: Int = 6144,
        numHiddenLayers: Int = 28,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 4,
        hiddenAct: String = "silu",
        headDim: Int? = nil,
        maxPositionEmbeddings: Int = 8192,
        layerTypes: [String]? = nil,
        initializerRange: Float = 0.02,
        rmsNormEps: Float = 1e-5,
        slidingWindow: Int? = nil,
        ropeTraditional: Bool = false,
        ropeScaling: [String: AnyCodable]? = nil,
        ropeTheta: Float = 10000.0,
        ropeDim: Int = 128,
        tieWordEmbeddings: Bool = false,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        padTokenId: Int = 59260,
        eosTokenId: [Int] = [59246, 59253, 59255]
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.hiddenAct = hiddenAct
        self.headDim = headDim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.layerTypes = layerTypes ?? Array(repeating: "full_attention", count: numHiddenLayers)
        self.initializerRange = initializerRange
        self.rmsNormEps = rmsNormEps
        self.slidingWindow = slidingWindow
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.ropeTheta = ropeTheta
        self.ropeDim = ropeDim
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "llama"
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 59264
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 28
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 4
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        let layers = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        layerTypes = layers ?? Array(repeating: "full_attention", count: numHiddenLayers)
        initializerRange = try container.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeScaling = try container.decodeIfPresent([String: AnyCodable].self, forKey: .ropeScaling)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        ropeDim = try container.decodeIfPresent(Int.self, forKey: .ropeDim) ?? 128
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 59260
        eosTokenId = try container.decodeIfPresent([Int].self, forKey: .eosTokenId) ?? [59246, 59253, 59255]
    }
}

/// Configuration for the GLM-ASR model.
public struct GLMASRModelConfig: Codable {
    public var modelType: String
    public var whisperConfig: WhisperConfig
    public var lmConfig: LlamaConfig

    // Adapter configuration
    public var adapterType: String
    public var mergeFactor: Int
    public var mlpAdapterAct: String

    // Audio processing
    public var useRope: Bool
    public var maxWhisperLength: Int
    public var maxLength: Int
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case whisperConfig = "whisper_config"
        case lmConfig = "lm_config"
        case adapterType = "adapter_type"
        case mergeFactor = "merge_factor"
        case mlpAdapterAct = "mlp_adapter_act"
        case useRope = "use_rope"
        case maxWhisperLength = "max_whisper_length"
        case maxLength = "max_length"
        // Note: perLayerQuantization is NOT in CodingKeys - decoded from BaseConfiguration
    }

    public init(
        modelType: String = "glmasr",
        whisperConfig: WhisperConfig = WhisperConfig(),
        lmConfig: LlamaConfig = LlamaConfig(),
        adapterType: String = "mlp",
        mergeFactor: Int = 4,
        mlpAdapterAct: String = "gelu",
        useRope: Bool = true,
        maxWhisperLength: Int = 1500,
        maxLength: Int = 65536,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.modelType = modelType
        self.whisperConfig = whisperConfig
        self.lmConfig = lmConfig
        self.adapterType = adapterType
        self.mergeFactor = mergeFactor
        self.mlpAdapterAct = mlpAdapterAct
        self.useRope = useRope
        self.maxWhisperLength = maxWhisperLength
        self.maxLength = maxLength
        self.perLayerQuantization = perLayerQuantization
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "glmasr"
        whisperConfig = try container.decodeIfPresent(WhisperConfig.self, forKey: .whisperConfig) ?? WhisperConfig()
        lmConfig = try container.decodeIfPresent(LlamaConfig.self, forKey: .lmConfig) ?? LlamaConfig()
        adapterType = try container.decodeIfPresent(String.self, forKey: .adapterType) ?? "mlp"
        mergeFactor = try container.decodeIfPresent(Int.self, forKey: .mergeFactor) ?? 4
        mlpAdapterAct = try container.decodeIfPresent(String.self, forKey: .mlpAdapterAct) ?? "gelu"
        useRope = try container.decodeIfPresent(Bool.self, forKey: .useRope) ?? true
        maxWhisperLength = try container.decodeIfPresent(Int.self, forKey: .maxWhisperLength) ?? 1500
        maxLength = try container.decodeIfPresent(Int.self, forKey: .maxLength) ?? 65536

        // Decode quantization from BaseConfiguration
        let baseConfig = try? BaseConfiguration(from: decoder)
        perLayerQuantization = baseConfig?.perLayerQuantization
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(whisperConfig, forKey: .whisperConfig)
        try container.encode(lmConfig, forKey: .lmConfig)
        try container.encode(adapterType, forKey: .adapterType)
        try container.encode(mergeFactor, forKey: .mergeFactor)
        try container.encode(mlpAdapterAct, forKey: .mlpAdapterAct)
        try container.encode(useRope, forKey: .useRope)
        try container.encode(maxWhisperLength, forKey: .maxWhisperLength)
        try container.encode(maxLength, forKey: .maxLength)
        // Note: perLayerQuantization is NOT encoded (handled by BaseConfiguration)
    }
}

/// A type-erased Codable container for arbitrary JSON values.
public struct AnyCodable: Codable {
    public let value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self.value = NSNull()
        } else if let bool = try? container.decode(Bool.self) {
            self.value = bool
        } else if let int = try? container.decode(Int.self) {
            self.value = int
        } else if let double = try? container.decode(Double.self) {
            self.value = double
        } else if let string = try? container.decode(String.self) {
            self.value = string
        } else if let array = try? container.decode([AnyCodable].self) {
            self.value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            self.value = dict.mapValues { $0.value }
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unable to decode value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case is NSNull:
            try container.encodeNil()
        case let bool as Bool:
            try container.encode(bool)
        case let int as Int:
            try container.encode(int)
        case let double as Double:
            try container.encode(double)
        case let string as String:
            try container.encode(string)
        case let array as [Any]:
            try container.encode(array.map { AnyCodable($0) })
        case let dict as [String: Any]:
            try container.encode(dict.mapValues { AnyCodable($0) })
        default:
            throw EncodingError.invalidValue(value, EncodingError.Context(codingPath: container.codingPath, debugDescription: "Unable to encode value"))
        }
    }
}
