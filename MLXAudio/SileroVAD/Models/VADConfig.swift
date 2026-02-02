import Foundation

// MARK: - VADConfig

public struct VADConfig: Sendable, Equatable {
    public var threshold: Float
    public var minSpeechDurationMs: Int
    public var minSilenceDurationMs: Int
    public var speechPadMs: Int

    public init(
        threshold: Float = 0.5,
        minSpeechDurationMs: Int = 250,
        minSilenceDurationMs: Int = 100,
        speechPadMs: Int = 30
    ) {
        precondition(threshold >= 0.0 && threshold <= 1.0, "threshold must be in [0, 1]")
        precondition(minSpeechDurationMs >= 0, "minSpeechDurationMs must be non-negative")
        precondition(minSilenceDurationMs >= 0, "minSilenceDurationMs must be non-negative")
        precondition(speechPadMs >= 0, "speechPadMs must be non-negative")

        self.threshold = threshold
        self.minSpeechDurationMs = minSpeechDurationMs
        self.minSilenceDurationMs = minSilenceDurationMs
        self.speechPadMs = speechPadMs
    }

    /// Balanced for general use
    public static let `default` = VADConfig()

    /// Lower threshold, catches more speech
    /// Use: quiet speakers, recording, transcription
    public static let sensitive = VADConfig(
        threshold: 0.35,
        minSpeechDurationMs: 200,
        minSilenceDurationMs: 150,
        speechPadMs: 50
    )

    /// Higher threshold, fewer false positives
    /// Use: noisy environments, voice commands
    public static let strict = VADConfig(
        threshold: 0.65,
        minSpeechDurationMs: 300,
        minSilenceDurationMs: 80,
        speechPadMs: 20
    )

    /// Quick response for voice assistants
    public static let conversation = VADConfig(
        threshold: 0.5,
        minSpeechDurationMs: 200,
        minSilenceDurationMs: 50,
        speechPadMs: 30
    )
}

// MARK: - VADAudioFormat

public enum VADAudioFormat {
    public static let sampleRate: Int = 16000
    public static let chunkSamples: Int = 512
    public static var chunkDuration: TimeInterval {
        Double(chunkSamples) / Double(sampleRate)
    }
    public static let valueRange: ClosedRange<Float> = -1.0...1.0
}

// MARK: - VADResult

public struct VADResult: Sendable, Equatable, Hashable {
    public let probability: Float
    public let isSpeech: Bool
    public let timestamp: TimeInterval

    public init(probability: Float, isSpeech: Bool, timestamp: TimeInterval) {
        self.probability = probability
        self.isSpeech = isSpeech
        self.timestamp = timestamp
    }
}

// MARK: - SpeechSegment

public struct SpeechSegment: Sendable, Equatable, Comparable, Hashable {
    public let start: TimeInterval
    public let end: TimeInterval

    public var duration: TimeInterval { end - start }

    public init(start: TimeInterval, end: TimeInterval) {
        precondition(end >= start, "SpeechSegment end (\(end)) must be >= start (\(start))")
        self.start = start
        self.end = end
    }

    public static func < (lhs: SpeechSegment, rhs: SpeechSegment) -> Bool {
        lhs.start < rhs.start
    }
}
