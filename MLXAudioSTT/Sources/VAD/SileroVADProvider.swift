import Foundation
import MLX
import SileroVAD

/// VAD provider using Silero VAD model for high-quality speech detection
public final class SileroVADProvider: VADProvider, @unchecked Sendable {
    private var session: VADSession?
    private let config: SileroVADConfig
    private let segmentConfig: VADSegmentConfig
    private let lock = NSLock()

    public struct SileroVADConfig: Sendable {
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
            self.threshold = threshold
            self.minSpeechDurationMs = minSpeechDurationMs
            self.minSilenceDurationMs = minSilenceDurationMs
            self.speechPadMs = speechPadMs
        }

        public static let `default` = SileroVADConfig()
        public static let sensitive = SileroVADConfig(threshold: 0.35)
        public static let strict = SileroVADConfig(threshold: 0.65)
    }

    public init(
        config: SileroVADConfig = .default,
        segmentConfig: VADSegmentConfig = .default
    ) {
        self.config = config
        self.segmentConfig = segmentConfig
    }

    public func detectSpeech(in audio: MLXArray, sampleRate: Int) async throws -> [SpeechSegment] {
        guard sampleRate == VADSession.sampleRate else {
            throw VADError.sampleRateMismatch(expected: VADSession.sampleRate, got: sampleRate)
        }

        let session = try await getOrCreateSession()

        let sileroConfig = SileroVAD.VADConfig(
            threshold: config.threshold,
            minSpeechDurationMs: config.minSpeechDurationMs,
            minSilenceDurationMs: config.minSilenceDurationMs,
            speechPadMs: config.speechPadMs
        )

        let sileroSegments = try session.getSpeechTimestamps(audio, config: sileroConfig)

        var segments = sileroSegments.map { segment in
            SpeechSegment(
                start: segment.start,
                end: segment.end,
                confidence: 1.0
            )
        }

        if segmentConfig.maxSegmentDuration < .infinity {
            segments = splitLongSegments(segments, maxDuration: segmentConfig.maxSegmentDuration)
        }

        return segments
    }

    public func speechProbabilities(
        in audio: MLXArray,
        sampleRate: Int
    ) async throws -> [(time: TimeInterval, probability: Float)] {
        guard sampleRate == VADSession.sampleRate else {
            throw VADError.sampleRateMismatch(expected: VADSession.sampleRate, got: sampleRate)
        }

        let session = try await getOrCreateSession()
        let sileroConfig = SileroVAD.VADConfig(threshold: config.threshold)
        let iterator = session.makeIterator(config: sileroConfig)

        let chunkSize = 512
        let totalSamples = audio.shape[0]
        var probabilities: [(time: TimeInterval, probability: Float)] = []

        var offset = 0
        while offset < totalSamples {
            let remainingSamples = totalSamples - offset
            let chunk: MLXArray

            if remainingSamples >= chunkSize {
                chunk = audio[offset..<(offset + chunkSize)]
            } else {
                let partialChunk = audio[offset...]
                let padding = MLXArray.zeros([chunkSize - remainingSamples])
                chunk = concatenated([partialChunk, padding], axis: 0)
            }

            let result = try iterator.process(chunk)
            probabilities.append((time: result.timestamp, probability: result.probability))

            offset += chunkSize
        }

        return probabilities
    }

    public func reset() async {
        lock.lock()
        session = nil
        lock.unlock()
    }

    private func getOrCreateSession() async throws -> VADSession {
        lock.lock()
        if let existing = session {
            lock.unlock()
            return existing
        }
        lock.unlock()

        let newSession = try await VADSession.make()

        lock.lock()
        if let existing = session {
            lock.unlock()
            return existing
        }
        session = newSession
        lock.unlock()
        return newSession
    }
}
