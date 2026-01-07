import Foundation
import MLX

/// Result from processing a single chunk
public struct ChunkResult: Sendable {
    public let text: String
    public let tokens: [Int]
    public let timeRange: ClosedRange<TimeInterval>
    public let confidence: Float
    public let words: [WordTimestamp]?

    public init(
        text: String,
        tokens: [Int],
        timeRange: ClosedRange<TimeInterval>,
        confidence: Float,
        words: [WordTimestamp]? = nil
    ) {
        self.text = text
        self.tokens = tokens
        self.timeRange = timeRange
        self.confidence = confidence
        self.words = words
    }
}

public struct WordTimestamp: Sendable, Equatable {
    public let word: String
    public let start: TimeInterval
    public let end: TimeInterval
    public let confidence: Float

    public init(word: String, start: TimeInterval, end: TimeInterval, confidence: Float) {
        self.word = word
        self.start = start
        self.end = end
        self.confidence = confidence
    }
}

/// Partial result available when cancelled or failed mid-stream
public struct PartialTranscriptionResult: Sendable {
    public let text: String
    public let processedDuration: TimeInterval
    public let totalDuration: TimeInterval
    public let completedChunks: Int
    public let totalChunks: Int

    public init(
        text: String,
        processedDuration: TimeInterval,
        totalDuration: TimeInterval,
        completedChunks: Int,
        totalChunks: Int
    ) {
        self.text = text
        self.processedDuration = processedDuration
        self.totalDuration = totalDuration
        self.completedChunks = completedChunks
        self.totalChunks = totalChunks
    }
}

/// Resource governance for long audio processing
public struct ProcessingLimits: Sendable {
    public var maxConcurrentChunks: Int
    public var chunkTimeout: TimeInterval
    public var totalTimeout: TimeInterval?
    public var maxMemoryMB: Int?
    public var abortOnFirstFailure: Bool

    public init(
        maxConcurrentChunks: Int = 4,
        chunkTimeout: TimeInterval = 60,
        totalTimeout: TimeInterval? = nil,
        maxMemoryMB: Int? = nil,
        abortOnFirstFailure: Bool = false
    ) {
        self.maxConcurrentChunks = maxConcurrentChunks
        self.chunkTimeout = chunkTimeout
        self.totalTimeout = totalTimeout
        self.maxMemoryMB = maxMemoryMB
        self.abortOnFirstFailure = abortOnFirstFailure
    }

    public static let `default` = ProcessingLimits()

    public static let conservative = ProcessingLimits(
        maxConcurrentChunks: 2,
        chunkTimeout: 30,
        maxMemoryMB: 1024
    )

    public static let aggressive = ProcessingLimits(
        maxConcurrentChunks: 8,
        chunkTimeout: 120,
        maxMemoryMB: 4096
    )
}

/// Cancellation behavior configuration
public struct CancellationPolicy: Sendable {
    public var emitPartialOnCancel: Bool
    public var gracePeriod: TimeInterval
    public var waitForCurrentChunk: Bool

    public init(
        emitPartialOnCancel: Bool = true,
        gracePeriod: TimeInterval = 1.0,
        waitForCurrentChunk: Bool = true
    ) {
        self.emitPartialOnCancel = emitPartialOnCancel
        self.gracePeriod = gracePeriod
        self.waitForCurrentChunk = waitForCurrentChunk
    }

    public static let `default` = CancellationPolicy()
}

/// Progress update during transcription
public struct TranscriptionProgress: Sendable {
    public let text: String
    public let words: [WordTimestamp]?
    public let isFinal: Bool
    public let processedDuration: TimeInterval
    public let audioDuration: TimeInterval
    public let chunkIndex: Int
    public let totalChunks: Int

    public var progress: Float {
        guard audioDuration > 0 else { return 0 }
        return Float(processedDuration / audioDuration)
    }

    public init(
        text: String,
        words: [WordTimestamp]?,
        isFinal: Bool,
        processedDuration: TimeInterval,
        audioDuration: TimeInterval,
        chunkIndex: Int,
        totalChunks: Int
    ) {
        self.text = text
        self.words = words
        self.isFinal = isFinal
        self.processedDuration = processedDuration
        self.audioDuration = audioDuration
        self.chunkIndex = chunkIndex
        self.totalChunks = totalChunks
    }
}

/// Final transcription result
public struct TranscriptionResult: Sendable {
    public let text: String
    public let words: [WordTimestamp]?
    public let duration: TimeInterval
    public let language: String?

    public init(
        text: String,
        words: [WordTimestamp]?,
        duration: TimeInterval,
        language: String? = nil
    ) {
        self.text = text
        self.words = words
        self.duration = duration
        self.language = language
    }
}

/// How chunks relate to each other for context
public enum TranscriptionMode: Sendable {
    case independent
    case sequential
}
