import Foundation

// MARK: - DiscardReason

public enum DiscardReason: Sendable, Equatable {
    case tooShort(duration: TimeInterval)
}

// MARK: - SpeechEvent

public enum SpeechEvent: Sendable, Equatable {
    case speechStarted(at: TimeInterval)
    case speechEnded(at: TimeInterval, duration: TimeInterval)
    case speechDiscarded(reason: DiscardReason)
}

// MARK: - SpeechDetector

/// Detects speech start/end events from VAD probability stream.
/// Separate from VADIterator for composition flexibility.
///
/// State machine:
/// - IDLE: Waiting for consecutive speech frames (>= minSpeechChunks)
/// - SPEAKING: Speech confirmed, tracking duration
/// - SILENCE_PENDING: Silence detected, waiting for minSilenceDuration
///
/// Edge cases:
/// - Speech < minSpeechDuration: Returns `speechDiscarded(.tooShort)`
/// - Pause < minSilenceDuration: Treated as continuous speech
/// - Probability = threshold: Classified as speech (>=)
///
/// - Note: NOT thread-safe. Use one instance per stream.
public final class SpeechDetector {
    private enum State {
        case idle
        case pending(firstSpeechAt: TimeInterval, consecutiveCount: Int)
        case speaking(startedAt: TimeInterval, lastSpeechAt: TimeInterval)
        case silencePending(speechStartedAt: TimeInterval, lastSpeechAt: TimeInterval, silenceCount: Int)
    }

    private let config: VADConfig
    private var state: State = .idle

    private let minSpeechChunks: Int
    private let minSilenceChunks: Int

    private var minSpeechDuration: TimeInterval {
        Double(config.minSpeechDurationMs) / 1000.0
    }

    private var speechPad: TimeInterval {
        Double(config.speechPadMs) / 1000.0
    }

    public init(config: VADConfig = .default) {
        self.config = config
        self.minSpeechChunks = max(1, Int(ceil(Double(config.minSpeechDurationMs) / 1000.0 / VADAudioFormat.chunkDuration)))
        self.minSilenceChunks = max(1, Int(ceil(Double(config.minSilenceDurationMs) / 1000.0 / VADAudioFormat.chunkDuration)))
    }

    /// Feed a VAD result and get an optional speech event.
    /// - Parameter result: The VAD result from VADIterator
    /// - Returns: A speech event if a state transition occurred, nil otherwise
    public func feed(_ result: VADResult) -> SpeechEvent? {
        let isSpeech = result.probability >= config.threshold
        let timestamp = result.timestamp

        switch state {
        case .idle:
            if isSpeech {
                state = .pending(firstSpeechAt: timestamp, consecutiveCount: 1)
            }
            return nil

        case .pending(let firstSpeechAt, let count):
            if isSpeech {
                let newCount = count + 1
                if newCount >= minSpeechChunks {
                    let paddedStart = max(0, firstSpeechAt - speechPad)
                    state = .speaking(startedAt: paddedStart, lastSpeechAt: timestamp)
                    return .speechStarted(at: paddedStart)
                } else {
                    state = .pending(firstSpeechAt: firstSpeechAt, consecutiveCount: newCount)
                }
            } else {
                state = .idle
            }
            return nil

        case .speaking(let speechStartedAt, _):
            if isSpeech {
                state = .speaking(startedAt: speechStartedAt, lastSpeechAt: timestamp)
                return nil
            }
            state = .silencePending(speechStartedAt: speechStartedAt, lastSpeechAt: timestamp, silenceCount: 1)
            return nil

        case .silencePending(let speechStartedAt, let lastSpeechAt, let silenceCount):
            if isSpeech {
                state = .speaking(startedAt: speechStartedAt, lastSpeechAt: timestamp)
                return nil
            }

            let newSilenceCount = silenceCount + 1
            if newSilenceCount >= minSilenceChunks {
                // End is at last speech frame + chunk duration + padding
                let speechEndTime = lastSpeechAt + VADAudioFormat.chunkDuration
                let paddedEnd = speechEndTime + speechPad
                let speechDuration = paddedEnd - speechStartedAt
                state = .idle

                if speechDuration >= minSpeechDuration {
                    return .speechEnded(at: paddedEnd, duration: speechDuration)
                } else {
                    return .speechDiscarded(reason: .tooShort(duration: speechDuration))
                }
            }
            state = .silencePending(speechStartedAt: speechStartedAt, lastSpeechAt: lastSpeechAt, silenceCount: newSilenceCount)
            return nil
        }
    }

    /// Finalize detection at end of audio stream.
    /// Call this when no more audio chunks will be provided.
    /// - Parameter timestamp: The end timestamp of the audio stream
    /// - Returns: Final speech event if speech was ongoing, nil otherwise
    public func finalize(at timestamp: TimeInterval) -> SpeechEvent? {
        switch state {
        case .idle, .pending:
            state = .idle
            return nil

        case .speaking(let speechStartedAt, let lastSpeechAt),
             .silencePending(let speechStartedAt, let lastSpeechAt, _):
            let speechEndTime = max(lastSpeechAt + VADAudioFormat.chunkDuration, timestamp)
            let paddedEnd = speechEndTime + speechPad
            let speechDuration = paddedEnd - speechStartedAt
            state = .idle

            if speechDuration >= minSpeechDuration {
                return .speechEnded(at: paddedEnd, duration: speechDuration)
            } else {
                return .speechDiscarded(reason: .tooShort(duration: speechDuration))
            }
        }
    }

    public func reset() {
        state = .idle
    }
}
