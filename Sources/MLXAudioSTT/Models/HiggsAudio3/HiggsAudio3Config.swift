import Foundation
import MLXAudioCore

public struct HiggsAudio3EncoderConfig: Codable {
    public var numMelBins: Int = 128
    public var encoderLayers: Int = 32
    public var encoderAttentionHeads: Int = 20
    public var encoderFfnDim: Int = 5120
    public var dModel: Int = 1280
    public var maxSourcePositions: Int = 1500

    enum CodingKeys: String, CodingKey {
        case numMelBins = "num_mel_bins", encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads", encoderFfnDim = "encoder_ffn_dim"
        case dModel = "d_model", maxSourcePositions = "max_source_positions"
    }
    public init() {}
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 32
        encoderAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 20
        encoderFfnDim = try c.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? 5120
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
    }
}

public struct HiggsAudio3Config: Codable {
    public var modelType: String = "higgs_audio_3"
    public var audioEncoderConfig = HiggsAudio3EncoderConfig()
    public var textConfig = Qwen3TextConfig(hiddenSize: 2048, intermediateSize: 6144,
        maxPositionEmbeddings: 32768)
    public var projectorTemporalDownsample: Int = 2
    public var audioInTokenIndex: Int = 151672
    public var chunkSizeSeconds: Float = 4
    public var sampleRate: Int = 16000
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", audioEncoderConfig = "audio_encoder_config"
        case textConfig = "text_config", projectorTemporalDownsample = "projector_temporal_downsample"
        case audioInTokenIndex = "audio_in_token_idx", chunkSizeSeconds = "chunk_size_seconds"
        case sampleRate = "sample_rate"
    }
    public init() {}
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "higgs_audio_3"
        audioEncoderConfig = try c.decodeIfPresent(HiggsAudio3EncoderConfig.self, forKey: .audioEncoderConfig) ?? HiggsAudio3EncoderConfig()
        textConfig = try c.decodeIfPresent(Qwen3TextConfig.self, forKey: .textConfig)
            ?? Qwen3TextConfig(hiddenSize: 2048, intermediateSize: 6144, maxPositionEmbeddings: 32768)
        projectorTemporalDownsample = try c.decodeIfPresent(Int.self, forKey: .projectorTemporalDownsample) ?? 2
        audioInTokenIndex = try c.decodeIfPresent(Int.self, forKey: .audioInTokenIndex) ?? 151672
        chunkSizeSeconds = try c.decodeIfPresent(Float.self, forKey: .chunkSizeSeconds) ?? 4
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16000
        perLayerQuantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(audioEncoderConfig, forKey: .audioEncoderConfig)
        try c.encode(textConfig, forKey: .textConfig)
        try c.encode(projectorTemporalDownsample, forKey: .projectorTemporalDownsample)
        try c.encode(audioInTokenIndex, forKey: .audioInTokenIndex)
        try c.encode(chunkSizeSeconds, forKey: .chunkSizeSeconds)
        try c.encode(sampleRate, forKey: .sampleRate)
    }
}
