import Foundation

public struct MossFormer2SEConfig: Codable, Sendable {
    public var modelType: String
    public var sampleRate: Int
    public var winLen: Int
    public var winInc: Int
    public var fftLen: Int
    public var numMels: Int
    public var winType: String
    public var preemphasis: Float
    public var inChannels: Int
    public var outChannels: Int
    public var outChannelsFinal: Int
    public var numBlocks: Int
    public var oneTimeDecodeLength: Int
    public var decodeWindow: Int
    public var chunkSeconds: Float
    public var chunkOverlap: Float
    public var autoChunkThreshold: Float
    public var quantizationConfig: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case winLen = "win_len"
        case winInc = "win_inc"
        case fftLen = "fft_len"
        case numMels = "num_mels"
        case winType = "win_type"
        case preemphasis
        case inChannels = "in_channels"
        case outChannels = "out_channels"
        case outChannelsFinal = "out_channels_final"
        case numBlocks = "num_blocks"
        case oneTimeDecodeLength = "one_time_decode_length"
        case decodeWindow = "decode_window"
        case chunkSeconds = "chunk_seconds"
        case chunkOverlap = "chunk_overlap"
        case autoChunkThreshold = "auto_chunk_threshold"
        case quantizationConfig = "quantization_config"
    }

    public init() {
        modelType = "mossformer2_se"
        sampleRate = 48000
        winLen = 1920
        winInc = 384
        fftLen = 1920
        numMels = 60
        winType = "hamming"
        preemphasis = 0.97
        inChannels = 180
        outChannels = 512
        outChannelsFinal = 961
        numBlocks = 24
        oneTimeDecodeLength = 20
        decodeWindow = 4
        chunkSeconds = 4.0
        chunkOverlap = 0.25
        autoChunkThreshold = 60.0
        quantizationConfig = nil
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mossformer2_se"
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 48000
        winLen = try c.decodeIfPresent(Int.self, forKey: .winLen) ?? 1920
        winInc = try c.decodeIfPresent(Int.self, forKey: .winInc) ?? 384
        fftLen = try c.decodeIfPresent(Int.self, forKey: .fftLen) ?? 1920
        numMels = try c.decodeIfPresent(Int.self, forKey: .numMels) ?? 60
        winType = try c.decodeIfPresent(String.self, forKey: .winType) ?? "hamming"
        preemphasis = try c.decodeIfPresent(Float.self, forKey: .preemphasis) ?? 0.97
        inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 180
        outChannels = try c.decodeIfPresent(Int.self, forKey: .outChannels) ?? 512
        outChannelsFinal = try c.decodeIfPresent(Int.self, forKey: .outChannelsFinal) ?? 961
        numBlocks = try c.decodeIfPresent(Int.self, forKey: .numBlocks) ?? 24
        oneTimeDecodeLength = try c.decodeIfPresent(Int.self, forKey: .oneTimeDecodeLength) ?? 20
        decodeWindow = try c.decodeIfPresent(Int.self, forKey: .decodeWindow) ?? 4
        chunkSeconds = try c.decodeIfPresent(Float.self, forKey: .chunkSeconds) ?? 4.0
        chunkOverlap = try c.decodeIfPresent(Float.self, forKey: .chunkOverlap) ?? 0.25
        autoChunkThreshold = try c.decodeIfPresent(Float.self, forKey: .autoChunkThreshold) ?? 60.0
        quantizationConfig = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantizationConfig)
    }
}

public struct QuantizationConfig: Codable, Sendable {
    public var bits: Int
    public var groupSize: Int

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        bits = try c.decodeIfPresent(Int.self, forKey: .bits) ?? 4
        groupSize = try c.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
    }
}
