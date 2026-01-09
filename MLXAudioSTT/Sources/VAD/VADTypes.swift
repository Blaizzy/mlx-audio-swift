import Foundation

/// A detected speech segment with timing and confidence
public struct SpeechSegment: Sendable, Equatable {
    public let start: TimeInterval
    public let end: TimeInterval
    public let confidence: Float

    public var duration: TimeInterval { end - start }

    public init(start: TimeInterval, end: TimeInterval, confidence: Float) {
        self.start = start
        self.end = end
        self.confidence = confidence
    }
}

/// Configuration for segmenting speech probabilities
public struct VADSegmentConfig: Sendable {
    public var minSpeechDuration: TimeInterval
    public var minSilenceDuration: TimeInterval
    public var maxSegmentDuration: TimeInterval
    public var speechPadding: TimeInterval

    public init(
        minSpeechDuration: TimeInterval = 0.25,
        minSilenceDuration: TimeInterval = 0.3,
        maxSegmentDuration: TimeInterval = 30.0,
        speechPadding: TimeInterval = 0.1
    ) {
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.maxSegmentDuration = maxSegmentDuration
        self.speechPadding = speechPadding
    }

    public static let `default` = VADSegmentConfig()
}

/// Errors from VAD processing
public enum VADError: Error, Sendable, Equatable {
    case sampleRateMismatch(expected: Int, got: Int)
    case modelLoadFailed(String)
    case modelOutputMissing
    case audioTooShort(minimum: TimeInterval)
}

extension VADError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .sampleRateMismatch(let expected, let got):
            return "Sample rate mismatch: expected \(expected), got \(got)"
        case .modelLoadFailed(let message):
            return "Failed to load VAD model: \(message)"
        case .modelOutputMissing:
            return "VAD model output is missing"
        case .audioTooShort(let minimum):
            return "Audio too short: minimum \(String(format: "%.1f", minimum))s required"
        }
    }
}
