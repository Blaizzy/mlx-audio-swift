import Foundation
import MLXAudioCore

public struct Qwen2AudioEncoderConfig: Codable, Sendable {
    public var dModel: Int
    public var encoderLayers: Int
    public var encoderAttentionHeads: Int
    public var encoderFFNDim: Int
    public var numMelBins: Int
    public var maxSourcePositions: Int

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFFNDim = "encoder_ffn_dim"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 32
        encoderAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 20
        encoderFFNDim = try c.decodeIfPresent(Int.self, forKey: .encoderFFNDim) ?? 5120
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
    }
}

public struct Qwen2AudioConfig: Codable, Sendable {
    public var modelType: String
    public var audioConfig: Qwen2AudioEncoderConfig
    public var textConfig: Qwen2Configuration
    public var audioTokenId: Int
    public var eosTokenId: Int
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfig = "audio_config"
        case textConfig = "text_config"
        case audioTokenId = "audio_token_index"
        case eosTokenId = "eos_token_id"
    }
    private enum ExtraKeys: String, CodingKey { case audioTokenId = "audio_token_id" }
    private enum TextKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case eosTokenId = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen2_audio"
        audioConfig =
            try c.decodeIfPresent(Qwen2AudioEncoderConfig.self, forKey: .audioConfig)
            ?? Qwen2AudioEncoderConfig(from: decoder)
        let t = try (c.contains(.textConfig) ? c.superDecoder(forKey: .textConfig) : decoder)
            .container(keyedBy: TextKeys.self)
        // The published config omits Qwen2's layer/head sizes; these are Qwen2-Audio's defaults.
        textConfig = Qwen2Configuration(
            hiddenSize: try t.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096,
            hiddenLayers: try t.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32,
            intermediateSize: try t.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008,
            attentionHeads: try t.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32,
            rmsNormEps: try t.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            vocabularySize: try t.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 156032,
            kvHeads: try t.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 32,
            ropeTheta: try t.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000,
            ropeTraditional: try t.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false,
            ropeScaling: try t.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling),
            tieWordEmbeddings: try t.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        )
        let extra = try decoder.container(keyedBy: ExtraKeys.self)
        audioTokenId =
            try c.decodeIfPresent(Int.self, forKey: .audioTokenId)
            ?? extra.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151646
        eosTokenId =
            try c.decodeIfPresent(Int.self, forKey: .eosTokenId)
            ?? t.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 151645
        perLayerQuantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
    }

    func validate() throws {
        let a = audioConfig
        let t = textConfig
        guard a.dModel >= 4, a.dModel % 2 == 0, a.encoderAttentionHeads > 0,
            a.dModel % a.encoderAttentionHeads == 0, a.encoderLayers > 0, a.encoderFFNDim > 0,
            a.numMelBins > 0, a.maxSourcePositions > 0,
            t.hiddenSize > 0, t.hiddenLayers > 0, t.intermediateSize > 0,
            t.attentionHeads > 0, t.kvHeads > 0, t.hiddenSize % t.attentionHeads == 0,
            t.attentionHeads % t.kvHeads == 0,
            (0 ..< t.vocabularySize).contains(audioTokenId),
            (0 ..< t.vocabularySize).contains(eosTokenId)
        else {
            throw STTError.modelNotInitialized("Inconsistent Qwen2-Audio configuration")
        }
    }
}
