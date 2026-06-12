import Foundation

public struct NemoPredictNetworkConfig: Codable, Sendable {
    public let predHidden: Int
    public let predRnnLayers: Int
    public let rnnHiddenSize: Int?

    enum CodingKeys: String, CodingKey {
        case predHidden = "pred_hidden"
        case predRnnLayers = "pred_rnn_layers"
        case rnnHiddenSize = "rnn_hidden_size"
    }

    public init(predHidden: Int, predRnnLayers: Int, rnnHiddenSize: Int? = nil) {
        self.predHidden = predHidden
        self.predRnnLayers = predRnnLayers
        self.rnnHiddenSize = rnnHiddenSize
    }
}

public struct NemoJointNetworkConfig: Codable, Sendable {
    public let jointHidden: Int
    public let activation: String
    public let encoderHidden: Int
    public let predHidden: Int

    enum CodingKeys: String, CodingKey {
        case jointHidden = "joint_hidden"
        case activation
        case encoderHidden = "encoder_hidden"
        case predHidden = "pred_hidden"
    }

    public init(jointHidden: Int, activation: String, encoderHidden: Int, predHidden: Int) {
        self.jointHidden = jointHidden
        self.activation = activation
        self.encoderHidden = encoderHidden
        self.predHidden = predHidden
    }
}

public struct NemoPredictConfig: Codable, Sendable {
    public let blankAsPad: Bool
    public let vocabSize: Int
    public let prednet: NemoPredictNetworkConfig

    enum CodingKeys: String, CodingKey {
        case blankAsPad = "blank_as_pad"
        case vocabSize = "vocab_size"
        case prednet
    }

    public init(blankAsPad: Bool, vocabSize: Int, prednet: NemoPredictNetworkConfig) {
        self.blankAsPad = blankAsPad
        self.vocabSize = vocabSize
        self.prednet = prednet
    }
}

public struct NemoJointConfig: Codable, Sendable {
    public let numClasses: Int
    public let vocabulary: [String]
    public let jointnet: NemoJointNetworkConfig
    public let numExtraOutputs: Int

    enum CodingKeys: String, CodingKey {
        case numClasses = "num_classes"
        case vocabulary
        case jointnet
        case numExtraOutputs = "num_extra_outputs"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numClasses = try container.decode(Int.self, forKey: .numClasses)
        vocabulary = try container.decode([String].self, forKey: .vocabulary)
        jointnet = try container.decode(NemoJointNetworkConfig.self, forKey: .jointnet)
        numExtraOutputs = try container.decodeIfPresent(Int.self, forKey: .numExtraOutputs) ?? 0
    }

    public init(
        numClasses: Int,
        vocabulary: [String],
        jointnet: NemoJointNetworkConfig,
        numExtraOutputs: Int = 0
    ) {
        self.numClasses = numClasses
        self.vocabulary = vocabulary
        self.jointnet = jointnet
        self.numExtraOutputs = numExtraOutputs
    }
}
