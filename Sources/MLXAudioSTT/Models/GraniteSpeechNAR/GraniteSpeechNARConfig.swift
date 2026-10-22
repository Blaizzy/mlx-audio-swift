import Foundation
import MLXAudioCore

public struct GraniteSpeechNAREncoderConfig: Codable {
    public var conformer: GraniteSpeechEncoderConfig
    public var bpeOutputDim: Int
    public var bpePoolingWindow: Int
    public var selfConditioningLayer: Int

    enum CodingKeys: String, CodingKey {
        case bpeOutputDim = "bpe_output_dim"
        case bpePoolingWindow = "bpe_pooling_window"
        case selfConditioningLayer = "self_conditioning_layer"
        case numLayers = "num_layers"
        case outputDim = "output_dim"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        conformer = try GraniteSpeechEncoderConfig(from: decoder)
        conformer.numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 16
        conformer.outputDim = try c.decodeIfPresent(Int.self, forKey: .outputDim) ?? 348
        bpeOutputDim = try c.decodeIfPresent(Int.self, forKey: .bpeOutputDim) ?? 100352
        bpePoolingWindow = try c.decodeIfPresent(Int.self, forKey: .bpePoolingWindow) ?? 4
        selfConditioningLayer = try c.decodeIfPresent(Int.self, forKey: .selfConditioningLayer) ?? 8
    }
    public func encode(to encoder: Encoder) throws {
        try conformer.encode(to: encoder)
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(bpeOutputDim, forKey: .bpeOutputDim)
        try c.encode(bpePoolingWindow, forKey: .bpePoolingWindow)
        try c.encode(selfConditioningLayer, forKey: .selfConditioningLayer)
    }
}

public struct GraniteSpeechNARProjectorConfig: Codable {
    public var numLayers: Int
    public var numEncoderLayers: Int
    public var hiddenSize: Int
    public var numHeads: Int
    public var blockSize: Int
    public var downsampleRate: Int
    public var encoderDim: Int
    public var llmDim: Int
    public var mlpRatio: Int
    public var mlpBias: Bool
    public var attentionBias: Bool
    public var layerNormEps: Float

    enum CodingKeys: String, CodingKey {
        case numLayers = "num_layers"
        case numEncoderLayers = "num_encoder_layers"
        case hiddenSize = "hidden_size"
        case numHeads = "num_heads"
        case blockSize = "block_size"
        case downsampleRate = "downsample_rate"
        case encoderDim = "encoder_dim"
        case llmDim = "llm_dim"
        case mlpRatio = "mlp_ratio"
        case mlpBias = "mlp_bias"
        case attentionBias = "attn_bias"
        case layerNormEps = "layernorm_eps"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? 2
        numEncoderLayers = try c.decodeIfPresent(Int.self, forKey: .numEncoderLayers) ?? 4
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 32
        blockSize = try c.decodeIfPresent(Int.self, forKey: .blockSize) ?? 15
        downsampleRate = try c.decodeIfPresent(Int.self, forKey: .downsampleRate) ?? 5
        encoderDim = try c.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 1024
        llmDim = try c.decodeIfPresent(Int.self, forKey: .llmDim) ?? 2048
        mlpRatio = try c.decodeIfPresent(Int.self, forKey: .mlpRatio) ?? 2
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
    }
}

public struct GraniteSpeechNARConfig: Codable {
    public var modelType: String
    public var encoderConfig: GraniteSpeechNAREncoderConfig
    public var projectorConfig: GraniteSpeechNARProjectorConfig
    public var textConfig: GraniteSpeechTextConfig
    public var blankTokenId: Int
    public var minEditSequenceLength: Int
    public var encoderLayerIndices: [Int]
    public var scaleProjectedEmbeddings: Bool
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case encoderConfig = "encoder_config"
        case projectorConfig = "projector_config"
        case textConfig = "text_config"
        case blankTokenId = "blank_token_id"
        case minEditSequenceLength = "min_edit_sequence_length"
        case encoderLayerIndices = "encoder_layer_indices"
        case scaleProjectedEmbeddings = "scale_projected_embeddings"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "granite_speech_nar"
        encoderConfig =
            try c.decodeIfPresent(GraniteSpeechNAREncoderConfig.self, forKey: .encoderConfig)
            ?? GraniteSpeechNAREncoderConfig(from: decoder)
        projectorConfig =
            try c.decodeIfPresent(GraniteSpeechNARProjectorConfig.self, forKey: .projectorConfig)
            ?? GraniteSpeechNARProjectorConfig(from: decoder)
        textConfig =
            try c.decodeIfPresent(GraniteSpeechTextConfig.self, forKey: .textConfig)
            ?? GraniteSpeechTextConfig(from: decoder)
        blankTokenId = try c.decodeIfPresent(Int.self, forKey: .blankTokenId) ?? 100257
        minEditSequenceLength = try c.decodeIfPresent(Int.self, forKey: .minEditSequenceLength) ?? 8
        encoderLayerIndices =
            try c.decodeIfPresent([Int].self, forKey: .encoderLayerIndices) ?? [4, 8, 12, -1]
        scaleProjectedEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .scaleProjectedEmbeddings) ?? true
        perLayerQuantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
    }

    func validate() throws {
        let e = encoderConfig.conformer
        let p = projectorConfig
        let t = textConfig
        guard e.numLayers > 0, e.hiddenDim > 0, e.inputDim == 160,
            e.numHeads > 0, e.dimHead > 0, e.contextSize > 0,
            e.maxPosEmb >= e.contextSize - 1, e.convKernelSize > 0,
            encoderConfig.bpePoolingWindow > 0,
            (1 ... e.numLayers).contains(encoderConfig.selfConditioningLayer),
            p.numLayers > 0, p.numHeads > 0, p.hiddenSize > 0, p.hiddenSize % p.numHeads == 0,
            p.blockSize > 0, p.downsampleRate > 0, p.blockSize % p.downsampleRate == 0,
            p.numEncoderLayers == encoderLayerIndices.count, p.encoderDim == e.hiddenDim,
            encoderLayerIndices.allSatisfy({ $0 >= -(e.numLayers + 1) && $0 <= e.numLayers }),
            t.hiddenSize == p.llmDim, t.numAttentionHeads > 0, t.numKeyValueHeads > 0,
            t.hiddenSize % t.numAttentionHeads == 0, t.numAttentionHeads % t.numKeyValueHeads == 0,
            t.vocabSize == encoderConfig.bpeOutputDim, (0 ..< t.vocabSize).contains(blankTokenId),
            t.tieWordEmbeddings, t.embeddingMultiplier > 0, t.logitsScaling > 0,
            minEditSequenceLength > 0
        else {
            throw STTError.modelNotInitialized("Inconsistent Granite Speech NAR configuration")
        }
    }
}
