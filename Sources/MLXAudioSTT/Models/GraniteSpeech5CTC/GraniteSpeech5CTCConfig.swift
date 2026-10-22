import Foundation

public struct GraniteSpeech5CTCEncoderConfig: Codable, Sendable {
    public var vocabSize = 16384
    public var hiddenSize = 1024
    public var intermediateSize = 4096
    public var numHiddenLayers = 16
    public var numAttentionHeads = 8
    public var numKeyValueHeads = 8
    public var numMelBins = 80
    public var headDim = 128
    public var maxPositionEmbeddings = 512
    public var contextSize = 128
    public var convKernelSize = 7
    public var convExpansionFactor = 2
    public var subsampleLayers = [0, 1]
    public var attentionBias = true
    public var hiddenAct = "silu"
    public init() {}
    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size", hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers", numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads", numMelBins = "num_mel_bins", headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings", contextSize = "context_size"
        case convKernelSize = "conv_kernel_size", convExpansionFactor = "conv_expansion_factor"
        case subsampleLayers = "subsample_layers", attentionBias = "attention_bias", hiddenAct = "hidden_act"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 16384
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 16
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 8
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 80
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / max(1, numAttentionHeads))
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 512
        contextSize = try c.decodeIfPresent(Int.self, forKey: .contextSize) ?? 128
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 7
        convExpansionFactor = try c.decodeIfPresent(Int.self, forKey: .convExpansionFactor) ?? 2
        subsampleLayers = try c.decodeIfPresent([Int].self, forKey: .subsampleLayers) ?? [0, 1]
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
    }
}

public struct GraniteSpeech5CTCConfig: Codable, Sendable {
    public var modelType = "granite_speech5_ctc"
    public var vocabSize = 16384
    public var encoderConfig = GraniteSpeech5CTCEncoderConfig()
    public var padTokenId = 0
    public var tieWordEmbeddings = true
    public init() {}
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", vocabSize = "vocab_size", encoderConfig = "encoder_config"
        case padTokenId = "pad_token_id", tieWordEmbeddings = "tie_word_embeddings"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "granite_speech5_ctc"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 16384
        if let encoder = try c.decodeIfPresent(GraniteSpeech5CTCEncoderConfig.self, forKey: .encoderConfig) {
            encoderConfig = encoder
        } else {
            encoderConfig = .init()
            encoderConfig.vocabSize = vocabSize
        }
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
    public func validate() throws {
        let e = encoderConfig
        guard vocabSize > 0, e.vocabSize == vocabSize, (0..<vocabSize).contains(padTokenId),
              e.hiddenSize > 0, e.numAttentionHeads > 0, e.headDim > 0,
              e.hiddenSize == e.numAttentionHeads * e.headDim,
              e.numKeyValueHeads == e.numAttentionHeads, e.intermediateSize > 0,
              e.numHiddenLayers > 0, e.numMelBins > 0, e.contextSize > 0,
              e.maxPositionEmbeddings >= e.contextSize - 1,
              e.convKernelSize > 0, e.convKernelSize % 2 == 1, e.convExpansionFactor > 0,
              Set(e.subsampleLayers).count == e.subsampleLayers.count,
              e.subsampleLayers.allSatisfy({ (0..<e.numHiddenLayers).contains($0) }),
              e.hiddenAct == "silu", tieWordEmbeddings else {
            throw STTError.invalidInput("Unsupported or inconsistent Granite Speech 5 TurboCTC configuration")
        }
    }
}
