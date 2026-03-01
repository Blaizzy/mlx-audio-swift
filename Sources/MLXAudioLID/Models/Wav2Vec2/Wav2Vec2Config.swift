import Foundation

public struct Wav2Vec2LIDConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let classifierProjSize: Int
    public let numLabels: Int?
    public let convDim: [Int]
    public let convStride: [Int]
    public let convKernel: [Int]
    public let numConvPosEmbeddings: Int
    public let numConvPosEmbeddingGroups: Int
    public let id2label: [String: String]?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case classifierProjSize = "classifier_proj_size"
        case numLabels = "num_labels"
        case convDim = "conv_dim"
        case convStride = "conv_stride"
        case convKernel = "conv_kernel"
        case numConvPosEmbeddings = "num_conv_pos_embeddings"
        case numConvPosEmbeddingGroups = "num_conv_pos_embedding_groups"
        case id2label = "id2label"
    }
}
