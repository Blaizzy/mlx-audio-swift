import Foundation
import MLX

/// Protocol for Voice Activity Detection providers
public protocol VADProvider: Sendable {
    /// Detect speech segments in audio
    func detectSpeech(in audio: MLXArray, sampleRate: Int) async throws -> [SpeechSegment]

    /// Get frame-level speech probabilities
    func speechProbabilities(in audio: MLXArray, sampleRate: Int) async throws -> [(time: TimeInterval, probability: Float)]

    /// Reset internal state (for stateful models like Silero LSTM)
    func reset() async
}

// MARK: - Segmentation Utilities

/// Convert frame-level probabilities to speech segments
public func segmentFromProbabilities(
    _ probabilities: [(time: TimeInterval, probability: Float)],
    threshold: Float,
    config: VADSegmentConfig = .default
) -> [SpeechSegment] {
    var segments: [SpeechSegment] = []
    var segmentStart: TimeInterval?
    var maxConfidence: Float = 0

    for (time, prob) in probabilities {
        if prob >= threshold {
            if segmentStart == nil {
                segmentStart = max(0, time - config.speechPadding)
            }
            maxConfidence = max(maxConfidence, prob)
        } else if let start = segmentStart {
            let end = time + config.speechPadding
            let duration = end - start

            if duration >= config.minSpeechDuration {
                segments.append(SpeechSegment(
                    start: start,
                    end: end,
                    confidence: maxConfidence
                ))
            }
            segmentStart = nil
            maxConfidence = 0
        }
    }

    // Handle segment at end of audio
    if let start = segmentStart, let lastTime = probabilities.last?.time {
        let duration = lastTime - start
        if duration >= config.minSpeechDuration {
            segments.append(SpeechSegment(
                start: start,
                end: lastTime,
                confidence: maxConfidence
            ))
        }
    }

    return mergeCloseSegments(segments, minSilence: config.minSilenceDuration)
}

/// Merge segments that are closer than minSilence
public func mergeCloseSegments(
    _ segments: [SpeechSegment],
    minSilence: TimeInterval
) -> [SpeechSegment] {
    guard !segments.isEmpty else { return [] }

    var merged: [SpeechSegment] = []
    var current = segments[0]

    for next in segments.dropFirst() {
        if next.start - current.end < minSilence {
            // Merge
            current = SpeechSegment(
                start: current.start,
                end: next.end,
                confidence: max(current.confidence, next.confidence)
            )
        } else {
            merged.append(current)
            current = next
        }
    }
    merged.append(current)

    return merged
}

/// Split long segments at specified maximum duration
public func splitLongSegments(
    _ segments: [SpeechSegment],
    maxDuration: TimeInterval
) -> [SpeechSegment] {
    var result: [SpeechSegment] = []

    for segment in segments {
        if segment.duration <= maxDuration {
            result.append(segment)
        } else {
            // Split into chunks
            var start = segment.start
            while start < segment.end {
                let end = min(start + maxDuration, segment.end)
                result.append(SpeechSegment(
                    start: start,
                    end: end,
                    confidence: segment.confidence
                ))
                start = end
            }
        }
    }

    return result
}
