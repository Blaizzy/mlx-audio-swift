import Foundation

public struct T5Config: Codable {
    public let vocabSize: Int
    public let dModel: Int
    public let dFf: Int
    public let dKv: Int
    public let numHeads: Int
    public let numLayers: Int
    public let numDecoderLayers: Int
    public let relativeAttentionNumBuckets: Int
    public let relativeAttentionMaxDistance: Int
    public let layerNormEpsilon: Float
    public let feedForwardProj: String
    public let tieWordEmbeddings: Bool
    public let decoderStartTokenId: Int
    public let eosTokenId: Int
    public let padTokenId: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case dModel = "d_model"
        case dFf = "d_ff"
        case dKv = "d_kv"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case numDecoderLayers = "num_decoder_layers"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case relativeAttentionMaxDistance = "relative_attention_max_distance"
        case layerNormEpsilon = "layer_norm_epsilon"
        case feedForwardProj = "feed_forward_proj"
        case tieWordEmbeddings = "tie_word_embeddings"
        case decoderStartTokenId = "decoder_start_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }

    public var innerDim: Int { dKv * numHeads }

    public static func load(from directory: URL) throws -> T5Config {
        let url = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(T5Config.self, from: data)
    }
}
