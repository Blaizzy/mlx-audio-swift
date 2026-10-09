import Foundation

public struct LasrEncoderConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let intermediateSize: Int
    public let hiddenAct: String
    public let convKernelSize: Int
    public let convolutionBias: Bool
    public let numMelBins: Int
    public let subsamplingConvChannels: Int
    public let subsamplingConvKernelSize: Int
    public let subsamplingConvStride: Int
    public let dropout: Float
    public let attentionDropout: Float
    public let activationDropout: Float
    public let layerNormEps: Float
    public let batchNormMomentum: Float
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let ropeTheta: Float
    public let ropeType: String
    public let convResidualWeights: [Float]
    public let feedForwardResidualWeights: [Float]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case convKernelSize = "conv_kernel_size"
        case convolutionBias = "convolution_bias"
        case numMelBins = "num_mel_bins"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case subsamplingConvKernelSize = "subsampling_conv_kernel_size"
        case subsamplingConvStride = "subsampling_conv_stride"
        case dropout
        case attentionDropout = "attention_dropout"
        case activationDropout = "activation_dropout"
        case layerNormEps = "layer_norm_eps"
        case batchNormMomentum = "batch_norm_momentum"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
        case ropeParameters = "rope_parameters"
        case convResidualWeights = "conv_residual_weights"
        case feedForwardResidualWeights = "feed_forward_residual_weights"
    }

    enum RopeCodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
    }

    public init(
        hiddenSize: Int = 512,
        numHiddenLayers: Int = 17,
        numAttentionHeads: Int = 8,
        numKeyValueHeads: Int = 8,
        intermediateSize: Int = 2_048,
        hiddenAct: String = "silu",
        convKernelSize: Int = 32,
        convolutionBias: Bool = false,
        numMelBins: Int = 128,
        subsamplingConvChannels: Int = 256,
        subsamplingConvKernelSize: Int = 5,
        subsamplingConvStride: Int = 2,
        dropout: Float = 0.1,
        attentionDropout: Float = 0.1,
        activationDropout: Float = 0.1,
        layerNormEps: Float = 1e-6,
        batchNormMomentum: Float = 0.01,
        maxPositionEmbeddings: Int = 10_000,
        attentionBias: Bool = false,
        ropeTheta: Float = 10_000,
        ropeType: String = "default",
        convResidualWeights: [Float] = [2.0, 1.0],
        feedForwardResidualWeights: [Float] = [1.5, 0.5]
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.intermediateSize = intermediateSize
        self.hiddenAct = hiddenAct
        self.convKernelSize = convKernelSize
        self.convolutionBias = convolutionBias
        self.numMelBins = numMelBins
        self.subsamplingConvChannels = subsamplingConvChannels
        self.subsamplingConvKernelSize = subsamplingConvKernelSize
        self.subsamplingConvStride = subsamplingConvStride
        self.dropout = dropout
        self.attentionDropout = attentionDropout
        self.activationDropout = activationDropout
        self.layerNormEps = layerNormEps
        self.batchNormMomentum = batchNormMomentum
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionBias = attentionBias
        self.ropeTheta = ropeTheta
        self.ropeType = ropeType
        self.convResidualWeights = convResidualWeights
        self.feedForwardResidualWeights = feedForwardResidualWeights
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 17
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2_048
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 32
        convolutionBias = try c.decodeIfPresent(Bool.self, forKey: .convolutionBias) ?? false
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        subsamplingConvChannels = try c.decodeIfPresent(Int.self, forKey: .subsamplingConvChannels) ?? 256
        subsamplingConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .subsamplingConvKernelSize) ?? 5
        subsamplingConvStride = try c.decodeIfPresent(Int.self, forKey: .subsamplingConvStride) ?? 2
        dropout = try c.decodeIfPresent(Float.self, forKey: .dropout) ?? 0.1
        attentionDropout = try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.1
        activationDropout = try c.decodeIfPresent(Float.self, forKey: .activationDropout) ?? 0.1
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        batchNormMomentum = try c.decodeIfPresent(Float.self, forKey: .batchNormMomentum) ?? 0.01
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 10_000
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false

        var decodedRopeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        var decodedRopeType = try c.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
        if c.contains(.ropeParameters) {
            let rope = try c.nestedContainer(keyedBy: RopeCodingKeys.self, forKey: .ropeParameters)
            decodedRopeTheta = try rope.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? decodedRopeTheta
            decodedRopeType = try rope.decodeIfPresent(String.self, forKey: .ropeType) ?? decodedRopeType
        }
        ropeTheta = decodedRopeTheta
        ropeType = decodedRopeType
        convResidualWeights = try c.decodeIfPresent([Float].self, forKey: .convResidualWeights) ?? [2.0, 1.0]
        feedForwardResidualWeights = try c.decodeIfPresent([Float].self, forKey: .feedForwardResidualWeights) ?? [1.5, 0.5]
    }
}

public struct LasrCTCConfig: Decodable, Sendable {
    public let vocabSize: Int
    public let encoderConfig: LasrEncoderConfig
    public let ctcLossReduction: String
    public let ctcZeroInfinity: Bool
    public let padTokenId: Int
    public let initializerRange: Float
    public let modelType: String

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case encoderConfig = "encoder_config"
        case ctcLossReduction = "ctc_loss_reduction"
        case ctcZeroInfinity = "ctc_zero_infinity"
        case padTokenId = "pad_token_id"
        case initializerRange = "initializer_range"
        case modelType = "model_type"
    }

    public init(
        vocabSize: Int = 512,
        encoderConfig: LasrEncoderConfig = LasrEncoderConfig(),
        ctcLossReduction: String = "mean",
        ctcZeroInfinity: Bool = true,
        padTokenId: Int = 0,
        initializerRange: Float = 0.02,
        modelType: String = "lasr"
    ) {
        self.vocabSize = vocabSize
        self.encoderConfig = encoderConfig
        self.ctcLossReduction = ctcLossReduction
        self.ctcZeroInfinity = ctcZeroInfinity
        self.padTokenId = padTokenId
        self.initializerRange = initializerRange
        self.modelType = modelType
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 512
        encoderConfig = try c.decodeIfPresent(LasrEncoderConfig.self, forKey: .encoderConfig) ?? LasrEncoderConfig()
        ctcLossReduction = try c.decodeIfPresent(String.self, forKey: .ctcLossReduction) ?? "mean"
        ctcZeroInfinity = try c.decodeIfPresent(Bool.self, forKey: .ctcZeroInfinity) ?? true
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
        initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "lasr"
    }
}
