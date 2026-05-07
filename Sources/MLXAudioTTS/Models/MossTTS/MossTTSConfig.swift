import Foundation
import MLXAudioCore

public struct MossQwen3Config: Decodable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var maxPositionEmbeddings: Int
    public var ropeTheta: Float
    public var tieWordEmbeddings: Bool
    public var padTokenID: Int?
    public var bosTokenID: Int?
    public var eosTokenID: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case padTokenID = "pad_token_id"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
    }

    public init(
        modelType: String = "qwen3",
        vocabSize: Int = 155_648,
        hiddenSize: Int = 4_096,
        numHiddenLayers: Int = 36,
        intermediateSize: Int = 12_288,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int = 8,
        headDim: Int = 128,
        rmsNormEps: Float = 1e-6,
        maxPositionEmbeddings: Int = 40_960,
        ropeTheta: Float = 1_000_000,
        tieWordEmbeddings: Bool = false,
        padTokenID: Int? = nil,
        bosTokenID: Int? = nil,
        eosTokenID: Int? = nil
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.tieWordEmbeddings = tieWordEmbeddings
        self.padTokenID = padTokenID
        self.bosTokenID = bosTokenID
        self.eosTokenID = eosTokenID
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
        self.vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 155_648
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4_096
        self.numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 36
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12_288
        self.numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        self.numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        self.headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / numAttentionHeads)
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 40_960
        var ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        if ropeTheta == nil,
           let ropeParams = try c.decodeIfPresent([String: FlexibleJSONValue].self, forKey: .ropeParameters),
           let value = ropeParams["rope_theta"]?.floatValue {
            ropeTheta = value
        }
        self.ropeTheta = ropeTheta ?? 1_000_000
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID)
        self.bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID)
        self.eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID)
    }
}

public struct MossTTSConfig: Decodable, Sendable {
    public var modelType: String
    public var modelPath: String?
    public var languageConfig: MossQwen3Config
    public var initializerRange: Float
    public var nVQ: Int
    public var audioVocabSize: Int
    public var audioUserSlotTokenID: Int
    public var audioAssistantGenSlotTokenID: Int
    public var audioAssistantDelaySlotTokenID: Int
    public var audioStartTokenID: Int
    public var audioEndTokenID: Int
    public var audioPadCode: Int
    public var padTokenID: Int
    public var imStartTokenID: Int
    public var imEndTokenID: Int
    public var samplingRate: Int
    public var audioTokenizerPretrainedNameOrPath: String?
    public var additionalMLPFFNHiddenSize: Int?
    public var localFFNHiddenSize: Int?
    public var localHiddenSize: Int?
    public var localNumLayers: Int?

    public var hiddenSize: Int { languageConfig.hiddenSize }
    public var vocabSize: Int { languageConfig.vocabSize }
    public var isLocalTransformer: Bool {
        additionalMLPFFNHiddenSize != nil
            && localFFNHiddenSize != nil
            && localHiddenSize != nil
            && localNumLayers != nil
    }
    public var usesDialogueScenePrompt: Bool {
        nVQ == 16
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modelPath = "model_path"
        case languageConfig = "language_config"
        case initializerRange = "initializer_range"
        case nVQ = "n_vq"
        case audioVocabSize = "audio_vocab_size"
        case audioUserSlotTokenID = "audio_user_slot_token_id"
        case audioAssistantGenSlotTokenID = "audio_assistant_gen_slot_token_id"
        case audioAssistantDelaySlotTokenID = "audio_assistant_delay_slot_token_id"
        case audioStartTokenID = "audio_start_token_id"
        case audioEndTokenID = "audio_end_token_id"
        case audioPadCode = "audio_pad_code"
        case padTokenID = "pad_token_id"
        case imStartTokenID = "im_start_token_id"
        case imEndTokenID = "im_end_token_id"
        case samplingRate = "sampling_rate"
        case sampleRate = "sample_rate"
        case audioTokenizerPretrainedNameOrPath = "audio_tokenizer_pretrained_name_or_path"
        case additionalMLPFFNHiddenSize = "additional_mlp_ffn_hidden_size"
        case localFFNHiddenSize = "local_ffn_hidden_size"
        case localHiddenSize = "local_hidden_size"
        case localNumLayers = "local_num_layers"
    }

    public init(
        modelType: String = "moss_tts_delay",
        modelPath: String? = nil,
        languageConfig: MossQwen3Config = MossQwen3Config(),
        initializerRange: Float = 0.02,
        nVQ: Int = 32,
        audioVocabSize: Int = 1_024,
        audioUserSlotTokenID: Int = 151_654,
        audioAssistantGenSlotTokenID: Int = 151_656,
        audioAssistantDelaySlotTokenID: Int = 151_662,
        audioStartTokenID: Int = 151_652,
        audioEndTokenID: Int = 151_653,
        audioPadCode: Int = 1_024,
        padTokenID: Int = 151_643,
        imStartTokenID: Int = 151_644,
        imEndTokenID: Int = 151_645,
        samplingRate: Int = 24_000,
        audioTokenizerPretrainedNameOrPath: String? = nil,
        additionalMLPFFNHiddenSize: Int? = nil,
        localFFNHiddenSize: Int? = nil,
        localHiddenSize: Int? = nil,
        localNumLayers: Int? = nil
    ) {
        self.modelType = modelType
        self.modelPath = modelPath
        self.languageConfig = languageConfig
        self.initializerRange = initializerRange
        self.nVQ = nVQ
        self.audioVocabSize = audioVocabSize
        self.audioUserSlotTokenID = audioUserSlotTokenID
        self.audioAssistantGenSlotTokenID = audioAssistantGenSlotTokenID
        self.audioAssistantDelaySlotTokenID = audioAssistantDelaySlotTokenID
        self.audioStartTokenID = audioStartTokenID
        self.audioEndTokenID = audioEndTokenID
        self.audioPadCode = audioPadCode
        self.padTokenID = padTokenID
        self.imStartTokenID = imStartTokenID
        self.imEndTokenID = imEndTokenID
        self.samplingRate = samplingRate
        self.audioTokenizerPretrainedNameOrPath = audioTokenizerPretrainedNameOrPath
        self.additionalMLPFFNHiddenSize = additionalMLPFFNHiddenSize
        self.localFFNHiddenSize = localFFNHiddenSize
        self.localHiddenSize = localHiddenSize
        self.localNumLayers = localNumLayers
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "moss_tts_delay"
        self.modelPath = try c.decodeIfPresent(String.self, forKey: .modelPath)
        self.languageConfig = try c.decodeIfPresent(MossQwen3Config.self, forKey: .languageConfig) ?? MossQwen3Config()
        self.initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        self.nVQ = try c.decodeIfPresent(Int.self, forKey: .nVQ) ?? 32
        self.audioVocabSize = try c.decodeIfPresent(Int.self, forKey: .audioVocabSize) ?? 1_024
        self.audioUserSlotTokenID = try c.decodeIfPresent(Int.self, forKey: .audioUserSlotTokenID) ?? 151_654
        self.audioAssistantGenSlotTokenID = try c.decodeIfPresent(Int.self, forKey: .audioAssistantGenSlotTokenID) ?? 151_656
        self.audioAssistantDelaySlotTokenID = try c.decodeIfPresent(Int.self, forKey: .audioAssistantDelaySlotTokenID) ?? 151_662
        self.audioStartTokenID = try c.decodeIfPresent(Int.self, forKey: .audioStartTokenID) ?? 151_652
        self.audioEndTokenID = try c.decodeIfPresent(Int.self, forKey: .audioEndTokenID) ?? 151_653
        self.audioPadCode = try c.decodeIfPresent(Int.self, forKey: .audioPadCode) ?? 1_024
        self.padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 151_643
        self.imStartTokenID = try c.decodeIfPresent(Int.self, forKey: .imStartTokenID) ?? 151_644
        self.imEndTokenID = try c.decodeIfPresent(Int.self, forKey: .imEndTokenID) ?? 151_645
        self.samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate)
            ?? c.decodeIfPresent(Int.self, forKey: .sampleRate)
            ?? 24_000
        self.audioTokenizerPretrainedNameOrPath = try c.decodeIfPresent(String.self, forKey: .audioTokenizerPretrainedNameOrPath)
        self.additionalMLPFFNHiddenSize = try c.decodeIfPresent(Int.self, forKey: .additionalMLPFFNHiddenSize)
        self.localFFNHiddenSize = try c.decodeIfPresent(Int.self, forKey: .localFFNHiddenSize)
        self.localHiddenSize = try c.decodeIfPresent(Int.self, forKey: .localHiddenSize)
        self.localNumLayers = try c.decodeIfPresent(Int.self, forKey: .localNumLayers)
    }

    public func localTransformerConfig() throws -> MossQwen3Config {
        guard let localHiddenSize,
              let localFFNHiddenSize,
              let localNumLayers
        else {
            throw AudioGenerationError.invalidInput("local transformer configuration is not initialized")
        }
        var config = languageConfig
        config.hiddenSize = localHiddenSize
        config.intermediateSize = localFFNHiddenSize
        config.numHiddenLayers = localNumLayers
        return config
    }
}

public struct MossTTSGenerationConfig: Decodable, Sendable {
    public var maxNewTokens: Int?
    public var temperature: Float?
    public var topP: Float?
    public var topK: Int?
    public var repetitionPenalty: Float?

    enum CodingKeys: String, CodingKey {
        case maxNewTokens = "max_new_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
    }

    public static func fromFileIfPresent(_ url: URL) -> MossTTSGenerationConfig {
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode(MossTTSGenerationConfig.self, from: data)
        else {
            return MossTTSGenerationConfig()
        }
        return decoded
    }

    public init(
        maxNewTokens: Int? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        repetitionPenalty: Float? = nil
    ) {
        self.maxNewTokens = maxNewTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
    }
}

private enum FlexibleJSONValue: Decodable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let value = try? c.decode(Int.self) {
            self = .int(value)
        } else if let value = try? c.decode(Double.self) {
            self = .double(value)
        } else if let value = try? c.decode(Bool.self) {
            self = .bool(value)
        } else {
            self = .string(try c.decode(String.self))
        }
    }

    var floatValue: Float? {
        switch self {
        case .int(let value): Float(value)
        case .double(let value): Float(value)
        case .string(let value): Float(value)
        case .bool, .null: nil
        }
    }
}
