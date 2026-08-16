import Foundation

public struct MoonshineConfig: Decodable, Sendable {
    public let modelType: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let encoderNumHiddenLayers: Int
    public let decoderNumHiddenLayers: Int
    public let encoderNumAttentionHeads: Int
    public let decoderNumAttentionHeads: Int
    public let encoderNumKeyValueHeads: Int
    public let decoderNumKeyValueHeads: Int
    public let encoderHiddenAct: String
    public let decoderHiddenAct: String
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let partialRotaryFactor: Float
    public let ropeTheta: Float
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let decoderStartTokenId: Int
    public let tieWordEmbeddings: Bool
    public let padHeadDimToMultipleOf: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case encoderNumHiddenLayers = "encoder_num_hidden_layers"
        case decoderNumHiddenLayers = "decoder_num_hidden_layers"
        case encoderNumAttentionHeads = "encoder_num_attention_heads"
        case decoderNumAttentionHeads = "decoder_num_attention_heads"
        case encoderNumKeyValueHeads = "encoder_num_key_value_heads"
        case decoderNumKeyValueHeads = "decoder_num_key_value_heads"
        case encoderHiddenAct = "encoder_hidden_act"
        case decoderHiddenAct = "decoder_hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case decoderStartTokenId = "decoder_start_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case padHeadDimToMultipleOf = "pad_head_dim_to_multiple_of"
    }

    public init(
        modelType: String = "moonshine",
        vocabSize: Int = 32_768,
        hiddenSize: Int = 288,
        intermediateSize: Int = 1_152,
        encoderNumHiddenLayers: Int = 6,
        decoderNumHiddenLayers: Int = 6,
        encoderNumAttentionHeads: Int = 8,
        decoderNumAttentionHeads: Int = 8,
        encoderNumKeyValueHeads: Int? = nil,
        decoderNumKeyValueHeads: Int? = nil,
        encoderHiddenAct: String = "gelu",
        decoderHiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 512,
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        partialRotaryFactor: Float = 0.9,
        ropeTheta: Float = 10_000,
        bosTokenId: Int = 1,
        eosTokenId: Int = 2,
        decoderStartTokenId: Int = 1,
        tieWordEmbeddings: Bool = true,
        padHeadDimToMultipleOf: Int? = nil
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.encoderNumHiddenLayers = encoderNumHiddenLayers
        self.decoderNumHiddenLayers = decoderNumHiddenLayers
        self.encoderNumAttentionHeads = encoderNumAttentionHeads
        self.decoderNumAttentionHeads = decoderNumAttentionHeads
        self.encoderNumKeyValueHeads = encoderNumKeyValueHeads ?? encoderNumAttentionHeads
        self.decoderNumKeyValueHeads = decoderNumKeyValueHeads ?? decoderNumAttentionHeads
        self.encoderHiddenAct = encoderHiddenAct
        self.decoderHiddenAct = decoderHiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.decoderStartTokenId = decoderStartTokenId
        self.tieWordEmbeddings = tieWordEmbeddings
        self.padHeadDimToMultipleOf = padHeadDimToMultipleOf
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "moonshine"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32_768
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 288
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1_152
        encoderNumHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .encoderNumHiddenLayers) ?? 6
        decoderNumHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .decoderNumHiddenLayers) ?? 6
        encoderNumAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .encoderNumAttentionHeads) ?? 8
        decoderNumAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .decoderNumAttentionHeads) ?? 8
        encoderNumKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .encoderNumKeyValueHeads) ?? encoderNumAttentionHeads
        decoderNumKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .decoderNumKeyValueHeads) ?? decoderNumAttentionHeads
        encoderHiddenAct = try c.decodeIfPresent(String.self, forKey: .encoderHiddenAct) ?? "gelu"
        decoderHiddenAct = try c.decodeIfPresent(String.self, forKey: .decoderHiddenAct) ?? "silu"
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 512
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        attentionDropout = try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.9
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        decoderStartTokenId = try c.decodeIfPresent(Int.self, forKey: .decoderStartTokenId) ?? 1
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        padHeadDimToMultipleOf = try c.decodeIfPresent(Int.self, forKey: .padHeadDimToMultipleOf)
    }
}
