import Foundation
import MLXLMCommon

public struct FishTextConfig: Codable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var nLayer: Int
    public var nHead: Int
    public var dim: Int
    public var intermediateSize: Int
    public var nLocalHeads: Int
    public var headDim: Int
    public var ropeBase: Float
    public var normEps: Float
    public var maxSeqLen: Int
    public var dropout: Float
    public var tieWordEmbeddings: Bool
    public var attentionQKVBias: Bool
    public var attentionOBias: Bool
    public var attentionQKNorm: Bool
    public var useGradientCheckpointing: Bool
    public var initializerRange: Float

    public var resolvedLocalHeads: Int {
        nLocalHeads > 0 ? nLocalHeads : nHead
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case nLayer = "n_layer"
        case nHead = "n_head"
        case dim
        case intermediateSize = "intermediate_size"
        case nLocalHeads = "n_local_heads"
        case headDim = "head_dim"
        case ropeBase = "rope_base"
        case normEps = "norm_eps"
        case maxSeqLen = "max_seq_len"
        case dropout
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionQKVBias = "attention_qkv_bias"
        case attentionOBias = "attention_o_bias"
        case attentionQKNorm = "attention_qk_norm"
        case useGradientCheckpointing = "use_gradient_checkpointing"
        case initializerRange = "initializer_range"
    }

    public init(
        modelType: String = "fish_qwen3",
        vocabSize: Int = 155_776,
        nLayer: Int = 36,
        nHead: Int = 32,
        dim: Int = 2_560,
        intermediateSize: Int = 9_728,
        nLocalHeads: Int = 8,
        headDim: Int = 128,
        ropeBase: Float = 1_000_000,
        normEps: Float = 1e-6,
        maxSeqLen: Int = 32_768,
        dropout: Float = 0,
        tieWordEmbeddings: Bool = true,
        attentionQKVBias: Bool = false,
        attentionOBias: Bool = false,
        attentionQKNorm: Bool = true,
        useGradientCheckpointing: Bool = false,
        initializerRange: Float = 0.01976423537605237
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.nLayer = nLayer
        self.nHead = nHead
        self.dim = dim
        self.intermediateSize = intermediateSize
        self.nLocalHeads = nLocalHeads
        self.headDim = headDim
        self.ropeBase = ropeBase
        self.normEps = normEps
        self.maxSeqLen = maxSeqLen
        self.dropout = dropout
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionQKVBias = attentionQKVBias
        self.attentionOBias = attentionOBias
        self.attentionQKNorm = attentionQKNorm
        self.useGradientCheckpointing = useGradientCheckpointing
        self.initializerRange = initializerRange
    }
}

public struct FishAudioDecoderConfig: Codable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var nLayer: Int
    public var nHead: Int
    public var dim: Int
    public var intermediateSize: Int
    public var nLocalHeads: Int
    public var headDim: Int
    public var ropeBase: Float
    public var normEps: Float
    public var maxSeqLen: Int
    public var dropout: Float
    public var tieWordEmbeddings: Bool
    public var attentionQKVBias: Bool
    public var attentionOBias: Bool
    public var attentionQKNorm: Bool
    public var useGradientCheckpointing: Bool
    public var initializerRange: Float
    public var textDim: Int
    public var numCodebooks: Int

    public var resolvedLocalHeads: Int {
        nLocalHeads > 0 ? nLocalHeads : nHead
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case nLayer = "n_layer"
        case nHead = "n_head"
        case dim
        case intermediateSize = "intermediate_size"
        case nLocalHeads = "n_local_heads"
        case headDim = "head_dim"
        case ropeBase = "rope_base"
        case normEps = "norm_eps"
        case maxSeqLen = "max_seq_len"
        case dropout
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionQKVBias = "attention_qkv_bias"
        case attentionOBias = "attention_o_bias"
        case attentionQKNorm = "attention_qk_norm"
        case useGradientCheckpointing = "use_gradient_checkpointing"
        case initializerRange = "initializer_range"
        case textDim = "text_dim"
        case numCodebooks = "num_codebooks"
    }

    public init(
        modelType: String = "fish_qwen3_audio_decoder",
        vocabSize: Int = 4_096,
        nLayer: Int = 4,
        nHead: Int = 32,
        dim: Int = 2_560,
        intermediateSize: Int = 9_728,
        nLocalHeads: Int = 8,
        headDim: Int = 128,
        ropeBase: Float = 1_000_000,
        normEps: Float = 1e-6,
        maxSeqLen: Int = 11,
        dropout: Float = 0,
        tieWordEmbeddings: Bool = false,
        attentionQKVBias: Bool = false,
        attentionOBias: Bool = false,
        attentionQKNorm: Bool = false,
        useGradientCheckpointing: Bool = false,
        initializerRange: Float = 0.01976423537605237,
        textDim: Int = 2_560,
        numCodebooks: Int = 10
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.nLayer = nLayer
        self.nHead = nHead
        self.dim = dim
        self.intermediateSize = intermediateSize
        self.nLocalHeads = nLocalHeads
        self.headDim = headDim
        self.ropeBase = ropeBase
        self.normEps = normEps
        self.maxSeqLen = maxSeqLen
        self.dropout = dropout
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionQKVBias = attentionQKVBias
        self.attentionOBias = attentionOBias
        self.attentionQKNorm = attentionQKNorm
        self.useGradientCheckpointing = useGradientCheckpointing
        self.initializerRange = initializerRange
        self.textDim = textDim
        self.numCodebooks = numCodebooks
    }
}

public struct FishSpeechConfig: Decodable, Sendable {
    public var modelType: String
    public var modelPath: String?
    public var dtype: String
    public var padTokenID: Int
    public var eosTokenID: Int
    public var audioPadTokenID: Int
    public var semanticStartTokenID: Int
    public var semanticEndTokenID: Int
    public var sampleRate: Int
    public var textConfig: FishTextConfig
    public var audioDecoderConfig: FishAudioDecoderConfig
    public var quantization: BaseConfiguration.Quantization?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modelPath = "model_path"
        case dtype
        case padTokenID = "pad_token_id"
        case eosTokenID = "eos_token_id"
        case audioPadTokenID = "audio_pad_token_id"
        case semanticStartTokenID = "semantic_start_token_id"
        case semanticEndTokenID = "semantic_end_token_id"
        case sampleRate = "sample_rate"
        case textConfig = "text_config"
        case audioDecoderConfig = "audio_decoder_config"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(
        modelType: String = "fish_speech",
        modelPath: String? = nil,
        dtype: String = "bfloat16",
        padTokenID: Int = 151_669,
        eosTokenID: Int = 151_645,
        audioPadTokenID: Int = 151_677,
        semanticStartTokenID: Int = 151_678,
        semanticEndTokenID: Int = 155_773,
        sampleRate: Int = 44_100,
        textConfig: FishTextConfig = FishTextConfig(),
        audioDecoderConfig: FishAudioDecoderConfig = FishAudioDecoderConfig(),
        quantization: BaseConfiguration.Quantization? = nil,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.modelType = modelType
        self.modelPath = modelPath
        self.dtype = dtype
        self.padTokenID = padTokenID
        self.eosTokenID = eosTokenID
        self.audioPadTokenID = audioPadTokenID
        self.semanticStartTokenID = semanticStartTokenID
        self.semanticEndTokenID = semanticEndTokenID
        self.sampleRate = sampleRate
        self.textConfig = textConfig
        self.audioDecoderConfig = audioDecoderConfig
        self.quantization = quantization
        self.perLayerQuantization = perLayerQuantization
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "fish_speech"
        self.modelPath = try container.decodeIfPresent(String.self, forKey: .modelPath)
        self.dtype = try container.decodeIfPresent(String.self, forKey: .dtype) ?? "bfloat16"
        self.padTokenID = try container.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 151_669
        self.eosTokenID = try container.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 151_645
        self.audioPadTokenID = try container.decodeIfPresent(Int.self, forKey: .audioPadTokenID) ?? 151_677
        self.semanticStartTokenID =
            try container.decodeIfPresent(Int.self, forKey: .semanticStartTokenID) ?? 151_678
        self.semanticEndTokenID =
            try container.decodeIfPresent(Int.self, forKey: .semanticEndTokenID) ?? 155_773
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 44_100
        self.textConfig =
            try container.decodeIfPresent(FishTextConfig.self, forKey: .textConfig) ?? FishTextConfig()
        self.audioDecoderConfig =
            try container.decodeIfPresent(FishAudioDecoderConfig.self, forKey: .audioDecoderConfig)
            ?? FishAudioDecoderConfig()

        let baseConfig = try? BaseConfiguration(from: decoder)
        let globalQuant = try container.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantization)
        let altGlobalQuant = try container.decodeIfPresent(
            BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
        self.quantization = globalQuant ?? altGlobalQuant
        self.perLayerQuantization = baseConfig?.perLayerQuantization
    }
}
