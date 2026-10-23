import Foundation
import MLXAudioCore

public struct VoxtralAudioConfig: Codable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var intermediateSize: Int
    public var numMelBins: Int
    public var maxSourcePositions: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
        case dModel = "d_model"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
    }

    public init(
        hiddenSize: Int = 1280, numHiddenLayers: Int = 32, numAttentionHeads: Int = 20,
        intermediateSize: Int = 5120, numMelBins: Int = 128, maxSourcePositions: Int = 1500
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.numMelBins = numMelBins
        self.maxSourcePositions = maxSourcePositions
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? c.decodeIfPresent(
                Int.self, forKey: .dModel) ?? 1280
        numHiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? c.decodeIfPresent(
                Int.self, forKey: .numHiddenLayers) ?? 32
        numAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? c.decodeIfPresent(
                Int.self, forKey: .numAttentionHeads) ?? 20
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? c.decodeIfPresent(
                Int.self, forKey: .encoderFfnDim) ?? 5120
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try c.encode(numAttentionHeads, forKey: .numAttentionHeads)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(numMelBins, forKey: .numMelBins)
        try c.encode(maxSourcePositions, forKey: .maxSourcePositions)
    }

    var whisperConfig: WhisperConfig {
        var c = WhisperConfig()
        c.dModel = hiddenSize
        c.encoderLayers = numHiddenLayers
        c.encoderAttentionHeads = numAttentionHeads
        c.encoderFfnDim = intermediateSize
        c.numMelBins = numMelBins
        c.maxSourcePositions = maxSourcePositions
        return c
    }
}

public struct VoxtralConfig: Codable {
    public var modelType: String
    public var audioConfig: VoxtralAudioConfig
    public var textConfig: LlamaConfig
    public var audioTokenId: Int
    public var perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfig = "audio_config"
        case textConfig = "text_config"
        case audioTokenId = "audio_token_id"
    }

    public init(
        audioConfig: VoxtralAudioConfig = .init(),
        textConfig: LlamaConfig = .init(
            vocabSize: 131072, hiddenSize: 3072, intermediateSize: 8192, numHiddenLayers: 30,
            numAttentionHeads: 32, numKeyValueHeads: 8, headDim: 128, maxPositionEmbeddings: 131072,
            ropeTheta: 100_000_000, padTokenId: 11, eosTokenId: [2, 4, 11]), audioTokenId: Int = 24
    ) {
        modelType = "voxtral"
        self.audioConfig = audioConfig
        self.textConfig = textConfig
        self.audioTokenId = audioTokenId
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "voxtral"
        audioConfig =
            try c.decodeIfPresent(VoxtralAudioConfig.self, forKey: .audioConfig) ?? .init()
        textConfig =
            try c.decodeIfPresent(LlamaConfig.self, forKey: .textConfig)
            ?? VoxtralConfig().textConfig
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 24
        perLayerQuantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
        if var quantization = perLayerQuantization {
            quantization.perLayerQuantization = Dictionary(
                uniqueKeysWithValues:
                    quantization.perLayerQuantization.map {
                        (canonicalVoxtralWeightPath($0.key), $0.value)
                    })
            perLayerQuantization = quantization
        }
    }

    func validate() throws {
        let a = audioConfig
        let t = textConfig
        guard a.hiddenSize > 0, a.numHiddenLayers > 0, a.numAttentionHeads > 0,
            a.hiddenSize % a.numAttentionHeads == 0, a.intermediateSize >= a.hiddenSize,
            a.intermediateSize % a.hiddenSize == 0, a.maxSourcePositions > 0,
            a.maxSourcePositions % (a.intermediateSize / a.hiddenSize) == 0, a.numMelBins > 0,
            t.hiddenSize > 0, t.numHiddenLayers > 0, t.intermediateSize > 0,
            t.numAttentionHeads > 0, t.numKeyValueHeads > 0,
            t.numAttentionHeads % t.numKeyValueHeads == 0,
            (t.headDim ?? (t.hiddenSize / t.numAttentionHeads)) > 0,
            (t.headDim ?? (t.hiddenSize / t.numAttentionHeads)) % 2 == 0,
            (0 ..< t.vocabSize).contains(audioTokenId)
        else {
            throw STTError.modelNotInitialized("Inconsistent Voxtral configuration")
        }
    }
}

// MLX-VLM artifacts keep the decoder backbone directly below language_model
// and the output head at the root; mlx-audio artifacts wrap both together.
func canonicalVoxtralWeightPath(_ path: String) -> String {
    if path == "lm_head" || path.hasPrefix("lm_head.") { return "language_model." + path }
    for component in ["embed_tokens", "layers", "norm"] {
        let prefix = "language_model." + component
        if path == prefix || path.hasPrefix(prefix + ".") {
            return "language_model.model." + path.dropFirst("language_model.".count)
        }
    }
    return path
}
