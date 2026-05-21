import Foundation
import MLXLMCommon

private func decodeStringEncodedInts(
    from container: KeyedDecodingContainer<MiMoV2ASRConfig.CodingKeys>,
    key: MiMoV2ASRConfig.CodingKeys,
    defaultValue: String
) throws -> String {
    if let stringValue = try container.decodeIfPresent(String.self, forKey: key) {
        return stringValue
    }
    if let intValue = try container.decodeIfPresent(Int.self, forKey: key) {
        return String(intValue)
    }
    return defaultValue
}

public struct MiMoV2ASRAudioConfig: Codable, Sendable {
    public let tokenizerVersion: String?
    public let speechVocabSize: String
    public let speechZeroembIdx: String
    public let groupSize: Int
    public let audioChannels: Int
    public let inputLocalLayers: Int
    public let inputLocalDim: Int
    public let inputFullAttention: Bool
    public let inputLocalAttentionHeads: Int
    public let inputLocalHeadDim: Int
    public let inputLocalIntermediateSize: Int
    public let inputLocalHiddenDropout: Float
    public let outputHiddenSize: Int
    public let ropeTheta: Float
    public let partialRotaryFactor: Float
    public let projectionLayers: Int
    public let addPostNorm: Bool
    public let audioSegmentSize: Int

    enum CodingKeys: String, CodingKey {
        case tokenizerVersion = "tokenizer_version"
        case speechVocabSize = "speech_vocab_size"
        case speechZeroembIdx = "speech_zeroemb_idx"
        case groupSize = "group_size"
        case audioChannels = "audio_channels"
        case inputLocalLayers = "input_local_layers"
        case inputLocalDim = "input_local_dim"
        case inputFullAttention = "input_full_attention"
        case inputLocalAttentionHeads = "input_local_attn_heads"
        case inputLocalHeadDim = "input_local_head_dim"
        case inputLocalIntermediateSize = "input_local_intermediate_size"
        case inputLocalHiddenDropout = "input_local_hidden_dropout"
        case outputHiddenSize = "out_hidden_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case projectionLayers = "projection_layers"
        case addPostNorm = "add_post_norm"
        case audioSegmentSize = "audio_segment_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        tokenizerVersion = try container.decodeIfPresent(String.self, forKey: .tokenizerVersion)
        speechVocabSize = try container.decodeIfPresent(String.self, forKey: .speechVocabSize) ?? "1025-1025-129-129-129-129-129-129"
        speechZeroembIdx = try container.decodeIfPresent(String.self, forKey: .speechZeroembIdx) ?? "1024-1024-128-128-128-128-128-128"
        groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 4
        audioChannels = try container.decodeIfPresent(Int.self, forKey: .audioChannels) ?? 8
        inputLocalLayers = try container.decodeIfPresent(Int.self, forKey: .inputLocalLayers) ?? 6
        inputLocalDim = try container.decodeIfPresent(Int.self, forKey: .inputLocalDim) ?? 1024
        inputFullAttention = try container.decodeIfPresent(Bool.self, forKey: .inputFullAttention) ?? true
        inputLocalAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .inputLocalAttentionHeads) ?? 64
        inputLocalHeadDim = try container.decodeIfPresent(Int.self, forKey: .inputLocalHeadDim) ?? 16
        inputLocalIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .inputLocalIntermediateSize) ?? 4096
        inputLocalHiddenDropout = try container.decodeIfPresent(Float.self, forKey: .inputLocalHiddenDropout) ?? 0.1
        outputHiddenSize = try container.decodeIfPresent(Int.self, forKey: .outputHiddenSize) ?? 4096
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 640_000
        partialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        projectionLayers = try container.decodeIfPresent(Int.self, forKey: .projectionLayers) ?? 1
        addPostNorm = try container.decodeIfPresent(Bool.self, forKey: .addPostNorm) ?? true
        audioSegmentSize = try container.decodeIfPresent(Int.self, forKey: .audioSegmentSize) ?? 6000
    }

    public var parsedSpeechVocabSizes: [Int] {
        speechVocabSize.split(separator: "-").compactMap { Int($0) }
    }

    public var parsedSpeechEmptyIDs: [Int] {
        speechZeroembIdx.split(separator: "-").compactMap { Int($0) }
    }
}

public struct MiMoV2ASRConfig: Decodable, Sendable {
    public let architectures: [String]
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let vocabSize: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let useCache: Bool
    public let tieWordEmbeddings: Bool

    public let groupSize: Int
    public let audioChannels: Int
    public let delayPattern: String
    public let speechVocabSize: String
    public let speechZeroembIdx: String
    public let localDim: Int
    public let localLayers: Int
    public let localAttentionHeads: Int
    public let localFFNDim: Int
    public let localAttentionDropout: Float
    public let inputLocalLayers: Int
    public let inputLocalDim: Int
    public let inputFullAttention: Bool
    public let nRVQ: Int
    public let addInputLocalTransformer: Bool
    public let addSpeechSOSPEOSP: Bool
    public let audioConfig: MiMoV2ASRAudioConfig?

    public var quantization: BaseConfiguration.Quantization?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case architectures
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case vocabSize = "vocab_size"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case useCache = "use_cache"
        case tieWordEmbeddings = "tie_word_embeddings"
        case groupSize = "group_size"
        case audioChannels = "audio_channels"
        case delayPattern = "delay_pattern"
        case speechVocabSize = "speech_vocab_size"
        case speechZeroembIdx = "speech_zeroemb_idx"
        case localDim = "local_dim"
        case localLayers = "local_layers"
        case localAttentionHeads = "local_attn_heads"
        case localFFNDim = "local_ffn_dim"
        case localAttentionDropout = "local_attn_dropout"
        case inputLocalLayers = "input_local_layers"
        case inputLocalDim = "input_local_dim"
        case inputFullAttention = "input_full_attention"
        case nRVQ = "n_rvq"
        case addInputLocalTransformer = "add_input_local_transformer"
        case addSpeechSOSPEOSP = "add_speech_sosp_eosp"
        case audioConfig = "audio_config"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen2"
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11_008
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 36
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151_680
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 640_000
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        useCache = try container.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 4
        audioChannels = try container.decodeIfPresent(Int.self, forKey: .audioChannels) ?? 8
        delayPattern = try decodeStringEncodedInts(
            from: container,
            key: .delayPattern,
            defaultValue: "0-1-2-3-4-5-6-7"
        )
        speechVocabSize = try decodeStringEncodedInts(
            from: container,
            key: .speechVocabSize,
            defaultValue: "1025-1025-129-129-129-129-129-129"
        )
        speechZeroembIdx = try decodeStringEncodedInts(
            from: container,
            key: .speechZeroembIdx,
            defaultValue: "1024-1024-128-128-128-128-128-128"
        )
        localDim = try container.decodeIfPresent(Int.self, forKey: .localDim) ?? 1024
        localLayers = try container.decodeIfPresent(Int.self, forKey: .localLayers) ?? 16
        localAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .localAttentionHeads) ?? 64
        localFFNDim = try container.decodeIfPresent(Int.self, forKey: .localFFNDim) ?? 4096
        localAttentionDropout = try container.decodeIfPresent(Float.self, forKey: .localAttentionDropout) ?? 0.1
        inputLocalLayers = try container.decodeIfPresent(Int.self, forKey: .inputLocalLayers) ?? 6
        inputLocalDim = try container.decodeIfPresent(Int.self, forKey: .inputLocalDim) ?? 1024
        inputFullAttention = try container.decodeIfPresent(Bool.self, forKey: .inputFullAttention) ?? true
        nRVQ = try container.decodeIfPresent(Int.self, forKey: .nRVQ) ?? 20
        addInputLocalTransformer = try container.decodeIfPresent(Bool.self, forKey: .addInputLocalTransformer) ?? true
        addSpeechSOSPEOSP = try container.decodeIfPresent(Bool.self, forKey: .addSpeechSOSPEOSP) ?? false
        audioConfig = try container.decodeIfPresent(MiMoV2ASRAudioConfig.self, forKey: .audioConfig)

        let baseConfig = try? BaseConfiguration(from: decoder)
        let globalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantization)
        let altGlobalQuant = try container.decodeIfPresent(BaseConfiguration.Quantization.self, forKey: .quantizationConfig)
        quantization = globalQuant ?? altGlobalQuant
        perLayerQuantization = baseConfig?.perLayerQuantization
    }

    public var architectureName: String? {
        architectures.first
    }

    public var isMiMoV2ASR: Bool {
        architectureName == "MiMoV2ASRForCausalLM"
    }

    public var parsedSpeechVocabSizes: [Int] {
        speechVocabSize.split(separator: "-").compactMap { Int($0) }
    }

    public var parsedSpeechEmptyIDs: [Int] {
        speechZeroembIdx.split(separator: "-").compactMap { Int($0) }
    }

    public var parsedDelayPattern: [Int] {
        delayPattern.split(separator: "-").compactMap { Int($0) }
    }

    public var activeSpeechCodebookSizes: [Int] {
        Array(parsedSpeechVocabSizes.prefix(audioChannels))
    }
}
