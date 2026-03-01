import Foundation

/// Configuration for the ECAPA-TDNN language identification model.
///
/// Matches the SpeechBrain `speechbrain/lang-id-voxlingua107-ecapa` format
/// with defaults for VoxLingua107 (107 languages).
public struct EcapaTdnnConfig: Codable, Sendable {
    public var nMels: Int
    public var channels: Int
    public var kernelSizes: [Int]
    public var dilations: [Int]
    public var attentionChannels: Int
    public var res2netScale: Int
    public var seChannels: Int
    public var embeddingDim: Int
    public var classifierHiddenDim: Int
    public var numClasses: Int
    public var id2label: [String: String]?

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case channels
        case kernelSizes = "kernel_sizes"
        case dilations
        case attentionChannels = "attention_channels"
        case res2netScale = "res2net_scale"
        case seChannels = "se_channels"
        case embeddingDim = "embedding_dim"
        case classifierHiddenDim = "classifier_hidden_dim"
        case numClasses = "num_classes"
        case id2label = "id2label"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        nMels = try c.decodeIfPresent(Int.self, forKey: .nMels) ?? 60
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 1024
        kernelSizes = try c.decodeIfPresent([Int].self, forKey: .kernelSizes) ?? [5, 3, 3, 3, 1]
        dilations = try c.decodeIfPresent([Int].self, forKey: .dilations) ?? [1, 2, 3, 4, 1]
        attentionChannels = try c.decodeIfPresent(Int.self, forKey: .attentionChannels) ?? 128
        res2netScale = try c.decodeIfPresent(Int.self, forKey: .res2netScale) ?? 8
        seChannels = try c.decodeIfPresent(Int.self, forKey: .seChannels) ?? 128
        embeddingDim = try c.decodeIfPresent(Int.self, forKey: .embeddingDim) ?? 256
        classifierHiddenDim = try c.decodeIfPresent(Int.self, forKey: .classifierHiddenDim) ?? 512
        id2label = try c.decodeIfPresent([String: String].self, forKey: .id2label)
        numClasses = try c.decodeIfPresent(Int.self, forKey: .numClasses) ?? id2label?.count ?? 107
    }

    public init(
        nMels: Int = 60,
        channels: Int = 1024,
        kernelSizes: [Int] = [5, 3, 3, 3, 1],
        dilations: [Int] = [1, 2, 3, 4, 1],
        attentionChannels: Int = 128,
        res2netScale: Int = 8,
        seChannels: Int = 128,
        embeddingDim: Int = 256,
        classifierHiddenDim: Int = 512,
        numClasses: Int = 107,
        id2label: [String: String]? = nil
    ) {
        self.nMels = nMels
        self.channels = channels
        self.kernelSizes = kernelSizes
        self.dilations = dilations
        self.attentionChannels = attentionChannels
        self.res2netScale = res2netScale
        self.seChannels = seChannels
        self.embeddingDim = embeddingDim
        self.classifierHiddenDim = classifierHiddenDim
        self.numClasses = numClasses
        self.id2label = id2label
    }
}
