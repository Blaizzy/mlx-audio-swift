import Foundation

public struct WhisperConfig: Codable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var numMelBins: Int

    public var dModel: Int
    public var encoderLayers: Int
    public var encoderAttentionHeads: Int
    public var encoderFfnDim: Int
    public var maxSourcePositions: Int

    public var decoderLayers: Int
    public var decoderAttentionHeads: Int
    public var decoderFfnDim: Int
    public var maxTargetPositions: Int

    public var bosTokenId: Int
    public var eosTokenId: Int
    public var padTokenId: Int
    public var decoderStartTokenId: Int

    public var scaleEmbedding: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case numMelBins = "num_mel_bins"
        case dModel = "d_model"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
        case maxSourcePositions = "max_source_positions"
        case decoderLayers = "decoder_layers"
        case decoderAttentionHeads = "decoder_attention_heads"
        case decoderFfnDim = "decoder_ffn_dim"
        case maxTargetPositions = "max_target_positions"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case decoderStartTokenId = "decoder_start_token_id"
        case scaleEmbedding = "scale_embedding"

        // OpenAI / mlx-whisper layout (mlx-community/whisper-*).
        case nMels = "n_mels"
        case nVocab = "n_vocab"
        case nAudioState = "n_audio_state"
        case nAudioHead = "n_audio_head"
        case nAudioLayer = "n_audio_layer"
        case nAudioCtx = "n_audio_ctx"
        case nTextState = "n_text_state"
        case nTextHead = "n_text_head"
        case nTextLayer = "n_text_layer"
        case nTextCtx = "n_text_ctx"
    }

    public init(
        modelType: String = "whisper",
        vocabSize: Int = 51865,
        numMelBins: Int = 80,
        dModel: Int = 384,
        encoderLayers: Int = 4,
        encoderAttentionHeads: Int = 6,
        encoderFfnDim: Int = 1536,
        maxSourcePositions: Int = 1500,
        decoderLayers: Int = 4,
        decoderAttentionHeads: Int = 6,
        decoderFfnDim: Int = 1536,
        maxTargetPositions: Int = 448,
        bosTokenId: Int = 50257,
        eosTokenId: Int = 50257,
        padTokenId: Int = 50257,
        decoderStartTokenId: Int = 50258,
        scaleEmbedding: Bool = false
    ) {
        self.modelType = modelType
        self.vocabSize = vocabSize
        self.numMelBins = numMelBins
        self.dModel = dModel
        self.encoderLayers = encoderLayers
        self.encoderAttentionHeads = encoderAttentionHeads
        self.encoderFfnDim = encoderFfnDim
        self.maxSourcePositions = maxSourcePositions
        self.decoderLayers = decoderLayers
        self.decoderAttentionHeads = decoderAttentionHeads
        self.decoderFfnDim = decoderFfnDim
        self.maxTargetPositions = maxTargetPositions
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.padTokenId = padTokenId
        self.decoderStartTokenId = decoderStartTokenId
        self.scaleEmbedding = scaleEmbedding
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "whisper"

        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize)
            ?? c.decodeIfPresent(Int.self, forKey: .nVocab) ?? 51865
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins)
            ?? c.decodeIfPresent(Int.self, forKey: .nMels) ?? 80

        let audioState = try c.decodeIfPresent(Int.self, forKey: .nAudioState)
        let textState = try c.decodeIfPresent(Int.self, forKey: .nTextState)
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel)
            ?? audioState ?? textState ?? 384

        encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers)
            ?? c.decodeIfPresent(Int.self, forKey: .nAudioLayer) ?? 4
        encoderAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads)
            ?? c.decodeIfPresent(Int.self, forKey: .nAudioHead) ?? 6
        encoderFfnDim = try c.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? (4 * dModel)
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions)
            ?? c.decodeIfPresent(Int.self, forKey: .nAudioCtx) ?? 1500

        decoderLayers = try c.decodeIfPresent(Int.self, forKey: .decoderLayers)
            ?? c.decodeIfPresent(Int.self, forKey: .nTextLayer) ?? 4
        decoderAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .decoderAttentionHeads)
            ?? c.decodeIfPresent(Int.self, forKey: .nTextHead) ?? 6
        decoderFfnDim = try c.decodeIfPresent(Int.self, forKey: .decoderFfnDim) ?? (4 * dModel)
        maxTargetPositions = try c.decodeIfPresent(Int.self, forKey: .maxTargetPositions)
            ?? c.decodeIfPresent(Int.self, forKey: .nTextCtx) ?? 448

        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 50257
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 50257
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 50257
        decoderStartTokenId = try c.decodeIfPresent(Int.self, forKey: .decoderStartTokenId) ?? 50258
        scaleEmbedding = try c.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? false
    }

    public func encode(to encoder: Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(numMelBins, forKey: .numMelBins)
        try c.encode(dModel, forKey: .dModel)
        try c.encode(encoderLayers, forKey: .encoderLayers)
        try c.encode(encoderAttentionHeads, forKey: .encoderAttentionHeads)
        try c.encode(encoderFfnDim, forKey: .encoderFfnDim)
        try c.encode(maxSourcePositions, forKey: .maxSourcePositions)
        try c.encode(decoderLayers, forKey: .decoderLayers)
        try c.encode(decoderAttentionHeads, forKey: .decoderAttentionHeads)
        try c.encode(decoderFfnDim, forKey: .decoderFfnDim)
        try c.encode(maxTargetPositions, forKey: .maxTargetPositions)
        try c.encode(bosTokenId, forKey: .bosTokenId)
        try c.encode(eosTokenId, forKey: .eosTokenId)
        try c.encode(padTokenId, forKey: .padTokenId)
        try c.encode(decoderStartTokenId, forKey: .decoderStartTokenId)
        try c.encode(scaleEmbedding, forKey: .scaleEmbedding)
    }
}

public struct WhisperGenerationConfig: Codable, Sendable {
    public var isMultilingual: Bool?
    public var decoderStartTokenId: Int?
    public var bosTokenId: Int?
    public var eosTokenId: Int?
    public var padTokenId: Int?
    public var noTimestampsTokenId: Int?
    public var prevSotTokenId: Int?
    public var maxInitialTimestampIndex: Int?
    public var maxLength: Int?
    public var suppressTokens: [Int]?
    public var beginSuppressTokens: [Int]?
    public var langToId: [String: Int]?
    public var taskToId: [String: Int]?

    enum CodingKeys: String, CodingKey {
        case isMultilingual = "is_multilingual"
        case decoderStartTokenId = "decoder_start_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case noTimestampsTokenId = "no_timestamps_token_id"
        case prevSotTokenId = "prev_sot_token_id"
        case maxInitialTimestampIndex = "max_initial_timestamp_index"
        case maxLength = "max_length"
        case suppressTokens = "suppress_tokens"
        case beginSuppressTokens = "begin_suppress_tokens"
        case langToId = "lang_to_id"
        case taskToId = "task_to_id"
    }
}

public enum WhisperAudioConfig {
    public static let sampleRate: Int = 16_000
    public static let nFft: Int = 400
    public static let hopLength: Int = 160
    public static let chunkLengthSeconds: Int = 30
    public static let nFrames: Int = 3_000
    public static var chunkLengthSamples: Int { chunkLengthSeconds * sampleRate }
}
