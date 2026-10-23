// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import Foundation
import MLXAudioCore

public struct VibeVoiceTokenizerConfig: Codable, Sendable {
    public var channels = 1
    public var causal = true
    public var vaeDim: Int
    public var encoderNFilters = 32
    public var encoderRatios = [8, 5, 5, 4, 2, 2]
    public var encoderDepths = [3, 3, 3, 3, 3, 3, 8]
    public var convBias = true
    public var disableLastNorm = true
    public var layernorm = "RMSNorm"
    public var layernormEps: Float = 1e-5
    public var mixerLayer = "depthwise_conv"
    public var layerScaleInitValue: Float = 1e-6
    public var hopLength: Int { encoderRatios.reduce(1, *) }

    enum CodingKeys: String, CodingKey {
        case channels, causal, layernorm
        case vaeDim = "vae_dim"
        case encoderNFilters = "encoder_n_filters"
        case encoderRatios = "encoder_ratios"
        case encoderDepths = "encoder_depths"
        case convBias = "conv_bias"
        case disableLastNorm = "disable_last_norm"
        case layernormEps = "layernorm_eps"
        case mixerLayer = "mixer_layer"
        case layerScaleInitValue = "layer_scale_init_value"
    }
    public init(vaeDim: Int = 64) { self.vaeDim = vaeDim }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 1
        causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? true
        vaeDim = try c.decodeIfPresent(Int.self, forKey: .vaeDim) ?? 64
        encoderNFilters = try c.decodeIfPresent(Int.self, forKey: .encoderNFilters) ?? 32
        encoderRatios = try c.decodeIfPresent([Int].self, forKey: .encoderRatios) ?? encoderRatios
        if let depths = try? c.decode(String.self, forKey: .encoderDepths) {
            encoderDepths = depths.split(separator: "-").compactMap { Int($0) }
        } else {
            encoderDepths =
                try c.decodeIfPresent([Int].self, forKey: .encoderDepths) ?? encoderDepths
        }
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? true
        disableLastNorm = try c.decodeIfPresent(Bool.self, forKey: .disableLastNorm) ?? true
        layernorm = try c.decodeIfPresent(String.self, forKey: .layernorm) ?? "RMSNorm"
        layernormEps = try c.decodeIfPresent(Float.self, forKey: .layernormEps) ?? 1e-5
        mixerLayer = try c.decodeIfPresent(String.self, forKey: .mixerLayer) ?? "depthwise_conv"
        layerScaleInitValue =
            try c.decodeIfPresent(Float.self, forKey: .layerScaleInitValue) ?? 1e-6
    }
}

public struct VibeVoiceConfig: Codable {
    public var modelType = "vibevoice"
    public var acousticTokenizerConfig = VibeVoiceTokenizerConfig()
    public var semanticTokenizerConfig = VibeVoiceTokenizerConfig(vaeDim: 128)
    public var decoderConfig = Qwen2Configuration(
        hiddenSize: 3584, hiddenLayers: 28, intermediateSize: 18944, attentionHeads: 28,
        rmsNormEps: 1e-6, vocabularySize: 152064, kvHeads: 4)
    public var acousticVaeDim = 64
    public var semanticVaeDim = 128
    public var sampleRate = 24000
    public var speechTokenCompressRatio = 3200
    public var normalizeAudio = true
    public var chunkFrames: Int?
    public var lookaheadFrames: Int?
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case acousticTokenizerConfig = "acoustic_tokenizer_config"
        case semanticTokenizerConfig = "semantic_tokenizer_config"
        case decoderConfig = "decoder_config"
        case acousticVaeDim = "acoustic_vae_dim"
        case semanticVaeDim = "semantic_vae_dim"
        case sampleRate = "sample_rate"
        case targetSampleRate = "target_sample_rate"
        case speechTokenCompressRatio = "speech_tok_compress_ratio"
        case normalizeAudio = "normalize_audio"
        case chunkFrames = "chunk_frames"
        case lookaheadFrames = "lookahead_frames"
    }
    public init() {}
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        acousticTokenizerConfig =
            try c.decodeIfPresent(VibeVoiceTokenizerConfig.self, forKey: .acousticTokenizerConfig)
            ?? acousticTokenizerConfig
        semanticTokenizerConfig =
            try c.decodeIfPresent(VibeVoiceTokenizerConfig.self, forKey: .semanticTokenizerConfig)
            ?? semanticTokenizerConfig
        decoderConfig =
            try c.decodeIfPresent(Qwen2Configuration.self, forKey: .decoderConfig) ?? decoderConfig
        acousticVaeDim =
            try c.decodeIfPresent(Int.self, forKey: .acousticVaeDim)
            ?? acousticTokenizerConfig.vaeDim
        semanticVaeDim =
            try c.decodeIfPresent(Int.self, forKey: .semanticVaeDim)
            ?? semanticTokenizerConfig.vaeDim
        sampleRate =
            try c.decodeIfPresent(Int.self, forKey: .targetSampleRate) ?? c.decodeIfPresent(
                Int.self, forKey: .sampleRate) ?? sampleRate
        speechTokenCompressRatio =
            try c.decodeIfPresent(Int.self, forKey: .speechTokenCompressRatio)
            ?? speechTokenCompressRatio
        normalizeAudio = try c.decodeIfPresent(Bool.self, forKey: .normalizeAudio) ?? true
        chunkFrames = try c.decodeIfPresent(Int.self, forKey: .chunkFrames)
        lookaheadFrames = try c.decodeIfPresent(Int.self, forKey: .lookaheadFrames)
        perLayerQuantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
        if var q = perLayerQuantization {
            q.perLayerQuantization = Dictionary(
                uniqueKeysWithValues: q.perLayerQuantization.map {
                    (canonicalVibeVoicePath($0.key), $0.value)
                })
            perLayerQuantization = q
        }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(acousticTokenizerConfig, forKey: .acousticTokenizerConfig)
        try c.encode(semanticTokenizerConfig, forKey: .semanticTokenizerConfig)
        try c.encode(decoderConfig, forKey: .decoderConfig)
        try c.encode(acousticVaeDim, forKey: .acousticVaeDim)
        try c.encode(semanticVaeDim, forKey: .semanticVaeDim)
        try c.encode(sampleRate, forKey: .sampleRate)
        try c.encode(speechTokenCompressRatio, forKey: .speechTokenCompressRatio)
        try c.encode(normalizeAudio, forKey: .normalizeAudio)
        try c.encodeIfPresent(chunkFrames, forKey: .chunkFrames)
        try c.encodeIfPresent(lookaheadFrames, forKey: .lookaheadFrames)
    }
    mutating func loadProcessor(_ directory: URL) throws {
        let url = directory.appendingPathComponent("preprocessor_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let json =
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
        sampleRate = json["target_sample_rate"] as? Int ?? sampleRate
        speechTokenCompressRatio =
            json["speech_tok_compress_ratio"] as? Int ?? speechTokenCompressRatio
        normalizeAudio = json["normalize_audio"] as? Bool ?? normalizeAudio
        chunkFrames = json["chunk_frames"] as? Int ?? chunkFrames
        lookaheadFrames = json["lookahead_frames"] as? Int ?? lookaheadFrames
    }
    func validate() throws {
        for c in [acousticTokenizerConfig, semanticTokenizerConfig] {
            guard c.channels == 1, c.causal, c.vaeDim > 0, c.encoderNFilters > 0,
                !c.encoderRatios.isEmpty, c.encoderRatios.allSatisfy({ $0 > 0 }),
                c.encoderDepths.count == c.encoderRatios.count + 1,
                c.encoderDepths.allSatisfy({ $0 >= 0 }),
                c.layernorm == "RMSNorm" || c.layernorm == "LayerNorm",
                c.mixerLayer == "depthwise_conv", c.hopLength == speechTokenCompressRatio
            else {
                throw STTError.modelNotInitialized("Unsupported VibeVoice encoder configuration")
            }
        }
        let d = decoderConfig
        guard acousticVaeDim == acousticTokenizerConfig.vaeDim,
            semanticVaeDim == semanticTokenizerConfig.vaeDim, sampleRate > 0,
            d.hiddenSize > 0, d.hiddenLayers > 0, d.attentionHeads > 0, d.kvHeads > 0,
            d.hiddenSize % d.attentionHeads == 0, d.attentionHeads % d.kvHeads == 0,
            (chunkFrames == nil && lookaheadFrames == nil)
                || ((chunkFrames ?? 0) > 0 && (lookaheadFrames ?? -1) >= 0)
        else { throw STTError.modelNotInitialized("Inconsistent VibeVoice configuration") }
    }
}

func canonicalVibeVoicePath(_ key: String) -> String {
    var path = key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
    path = path.replacingOccurrences(
        of: #"\.downsample_layers\.(\d+)\.0\.conv\.conv\."#, with: ".downsample_layers.$1.conv.",
        options: .regularExpression)
    path = path.replacingOccurrences(of: ".head.conv.conv.", with: ".head.conv.")
    path = path.replacingOccurrences(of: ".mixer.conv.conv.conv.", with: ".mixer.conv.conv.")
    if path.hasPrefix("lm_head.") || path == "lm_head" { return "language_model." + path }
    for part in ["layers", "embed_tokens", "norm"] {
        let prefix = "language_model." + part
        if path == prefix || path.hasPrefix(prefix + ".") {
            return "language_model.model." + path.dropFirst("language_model.".count)
        }
    }
    return path
}
