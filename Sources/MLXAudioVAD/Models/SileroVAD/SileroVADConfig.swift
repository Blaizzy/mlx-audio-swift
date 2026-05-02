import Foundation

public struct SileroVADBranchConfig: Codable, Sendable {
    public var sampleRate: Int
    public var filterLength: Int
    public var hopLength: Int
    public var pad: Int
    public var cutoff: Int
    public var contextSize: Int
    public var chunkSize: Int

    public init(
        sampleRate: Int = 16000,
        filterLength: Int = 256,
        hopLength: Int = 128,
        pad: Int = 64,
        cutoff: Int = 129,
        contextSize: Int = 64,
        chunkSize: Int = 512
    ) {
        self.sampleRate = sampleRate
        self.filterLength = filterLength
        self.hopLength = hopLength
        self.pad = pad
        self.cutoff = cutoff
        self.contextSize = contextSize
        self.chunkSize = chunkSize
    }

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case filterLength = "filter_length"
        case hopLength = "hop_length"
        case pad
        case cutoff
        case contextSize = "context_size"
        case chunkSize = "chunk_size"
    }

    public static let default16k = SileroVADBranchConfig()
    public static let default8k = SileroVADBranchConfig(
        sampleRate: 8000,
        filterLength: 128,
        hopLength: 64,
        pad: 32,
        cutoff: 65,
        contextSize: 32,
        chunkSize: 256
    )
}

public struct SileroVADConfig: Codable, Sendable {
    public var modelType: String
    public var architecture: String
    public var dtype: String
    public var threshold: Float
    public var minSpeechDurationMs: Int
    public var minSilenceDurationMs: Int
    public var speechPadMs: Int
    public var branch16k: SileroVADBranchConfig
    public var branch8k: SileroVADBranchConfig

    public init(
        modelType: String = "silero_vad",
        architecture: String = "silero_vad",
        dtype: String = "float32",
        threshold: Float = 0.5,
        minSpeechDurationMs: Int = 250,
        minSilenceDurationMs: Int = 100,
        speechPadMs: Int = 30,
        branch16k: SileroVADBranchConfig = .default16k,
        branch8k: SileroVADBranchConfig = .default8k
    ) {
        self.modelType = modelType
        self.architecture = architecture
        self.dtype = dtype
        self.threshold = threshold
        self.minSpeechDurationMs = minSpeechDurationMs
        self.minSilenceDurationMs = minSilenceDurationMs
        self.speechPadMs = speechPadMs
        self.branch16k = branch16k
        self.branch8k = branch8k
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architecture
        case dtype
        case threshold
        case minSpeechDurationMs = "min_speech_duration_ms"
        case minSilenceDurationMs = "min_silence_duration_ms"
        case speechPadMs = "speech_pad_ms"
        case branch16k = "branch_16k"
        case branch8k = "branch_8k"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "silero_vad"
        architecture = try c.decodeIfPresent(String.self, forKey: .architecture) ?? "silero_vad"
        dtype = try c.decodeIfPresent(String.self, forKey: .dtype) ?? "float32"
        threshold = try c.decodeIfPresent(Float.self, forKey: .threshold) ?? 0.5
        minSpeechDurationMs = try c.decodeIfPresent(Int.self, forKey: .minSpeechDurationMs) ?? 250
        minSilenceDurationMs = try c.decodeIfPresent(Int.self, forKey: .minSilenceDurationMs) ?? 100
        speechPadMs = try c.decodeIfPresent(Int.self, forKey: .speechPadMs) ?? 30
        branch16k = try c.decodeIfPresent(SileroVADBranchConfig.self, forKey: .branch16k) ?? .default16k
        branch8k = try c.decodeIfPresent(SileroVADBranchConfig.self, forKey: .branch8k) ?? .default8k
    }
}
