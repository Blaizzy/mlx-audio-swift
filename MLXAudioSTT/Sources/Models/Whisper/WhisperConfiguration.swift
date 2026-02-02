import Foundation

/// Configuration for Whisper model architecture
/// Values match the Python mlx-audio and OpenAI Whisper implementations
public struct WhisperConfiguration: Codable, Sendable {
    // MARK: - Audio Encoder

    /// Number of mel frequency bins (80 for v1/v2, 128 for v3)
    public var nMels: Int

    /// Audio context length (1500 frames = 30 seconds)
    public var nAudioCtx: Int

    /// Audio encoder hidden dimension
    public var nAudioState: Int

    /// Number of attention heads in audio encoder
    public var nAudioHead: Int

    /// Number of transformer layers in audio encoder
    public var nAudioLayer: Int

    // MARK: - Text Decoder

    /// Vocabulary size
    public var nVocab: Int

    /// Maximum text context length
    public var nTextCtx: Int

    /// Text decoder hidden dimension
    public var nTextState: Int

    /// Number of attention heads in text decoder
    public var nTextHead: Int

    /// Number of transformer layers in text decoder
    public var nTextLayer: Int

    // MARK: - Alignment

    /// Alignment heads for timestamp extraction [(layer, head), ...]
    /// Used by AlignAtt streaming to determine token emission timing
    public var alignmentHeads: [(Int, Int)]

    // MARK: - Codable

    enum CodingKeys: String, CodingKey {
        case nMels = "n_mels"
        case nAudioCtx = "n_audio_ctx"
        case nAudioState = "n_audio_state"
        case nAudioHead = "n_audio_head"
        case nAudioLayer = "n_audio_layer"
        case nVocab = "n_vocab"
        case nTextCtx = "n_text_ctx"
        case nTextState = "n_text_state"
        case nTextHead = "n_text_head"
        case nTextLayer = "n_text_layer"
        case alignmentHeads = "alignment_heads"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        nMels = try container.decode(Int.self, forKey: .nMels)
        nAudioCtx = try container.decode(Int.self, forKey: .nAudioCtx)
        nAudioState = try container.decode(Int.self, forKey: .nAudioState)
        nAudioHead = try container.decode(Int.self, forKey: .nAudioHead)
        nAudioLayer = try container.decode(Int.self, forKey: .nAudioLayer)
        nVocab = try container.decode(Int.self, forKey: .nVocab)
        nTextCtx = try container.decode(Int.self, forKey: .nTextCtx)
        nTextState = try container.decode(Int.self, forKey: .nTextState)
        nTextHead = try container.decode(Int.self, forKey: .nTextHead)
        nTextLayer = try container.decode(Int.self, forKey: .nTextLayer)

        // alignment_heads is optional in HuggingFace configs - fallback to empty (use WhisperAlignmentHeads)
        if let headsArray = try container.decodeIfPresent([[Int]].self, forKey: .alignmentHeads) {
            alignmentHeads = try headsArray.map { head in
                guard head.count == 2 else {
                    throw DecodingError.dataCorruptedError(
                        forKey: .alignmentHeads,
                        in: container,
                        debugDescription: "Each alignment head must have exactly 2 elements (layer, head)"
                    )
                }
                return (head[0], head[1])
            }
        } else {
            alignmentHeads = []
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(nMels, forKey: .nMels)
        try container.encode(nAudioCtx, forKey: .nAudioCtx)
        try container.encode(nAudioState, forKey: .nAudioState)
        try container.encode(nAudioHead, forKey: .nAudioHead)
        try container.encode(nAudioLayer, forKey: .nAudioLayer)
        try container.encode(nVocab, forKey: .nVocab)
        try container.encode(nTextCtx, forKey: .nTextCtx)
        try container.encode(nTextState, forKey: .nTextState)
        try container.encode(nTextHead, forKey: .nTextHead)
        try container.encode(nTextLayer, forKey: .nTextLayer)

        let headsArray = alignmentHeads.map { [$0.0, $0.1] }
        try container.encode(headsArray, forKey: .alignmentHeads)
    }

    // MARK: - Initializer

    public init(
        nMels: Int,
        nAudioCtx: Int,
        nAudioState: Int,
        nAudioHead: Int,
        nAudioLayer: Int,
        nVocab: Int,
        nTextCtx: Int,
        nTextState: Int,
        nTextHead: Int,
        nTextLayer: Int,
        alignmentHeads: [(Int, Int)]
    ) {
        self.nMels = nMels
        self.nAudioCtx = nAudioCtx
        self.nAudioState = nAudioState
        self.nAudioHead = nAudioHead
        self.nAudioLayer = nAudioLayer
        self.nVocab = nVocab
        self.nTextCtx = nTextCtx
        self.nTextState = nTextState
        self.nTextHead = nTextHead
        self.nTextLayer = nTextLayer
        self.alignmentHeads = alignmentHeads
    }
}

// MARK: - Preset Configurations

extension WhisperConfiguration {
    /// Whisper large-v3-turbo configuration
    /// Optimized for speed with 4 decoder layers instead of 32
    public static let largeTurbo = WhisperConfiguration(
        nMels: 128,
        nAudioCtx: 1500,
        nAudioState: 1280,
        nAudioHead: 20,
        nAudioLayer: 32,
        nVocab: 51866,
        nTextCtx: 448,
        nTextState: 1280,
        nTextHead: 20,
        nTextLayer: 4,
        alignmentHeads: [
            (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)
        ]
    )

    /// Whisper large-v3 configuration
    /// Full 32-layer decoder for maximum accuracy
    public static let largeV3 = WhisperConfiguration(
        nMels: 128,
        nAudioCtx: 1500,
        nAudioState: 1280,
        nAudioHead: 20,
        nAudioLayer: 32,
        nVocab: 51866,
        nTextCtx: 448,
        nTextState: 1280,
        nTextHead: 20,
        nTextLayer: 32,
        alignmentHeads: [
            (7, 0), (8, 16), (9, 0), (10, 17), (11, 11), (11, 4),
            (13, 15), (16, 11), (16, 4), (17, 9), (19, 19), (22, 6),
            (23, 12), (24, 1), (25, 4), (26, 14), (27, 9), (29, 13),
            (30, 18), (31, 7)
        ]
    )
}

/// Quantization level for model weights
public enum WhisperQuantization: String, CaseIterable, Sendable {
    case float16      // Default, highest quality
    case int8         // 8-bit, 2x smaller
    case int4         // 4-bit, 4x smaller, recommended
}
