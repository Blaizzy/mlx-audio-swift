import Foundation
@preconcurrency import MLX

/// Mock implementation of ChunkTranscriber for testing
public final class MockChunkTranscriber: ChunkTranscriber, @unchecked Sendable {
    public struct TranscribeCall: @unchecked Sendable {
        public let audio: MLXArray
        public let sampleRate: Int
        public let previousTokens: [Int]?
        public let timestamp: Date
    }

    private let lock = NSLock()
    private var _fixedResult: ChunkResult?
    private var _sequentialResults: [ChunkResult] = []
    private var _transcribeDelay: TimeInterval = 0
    private var _shouldFail: Bool = false
    private var _errorToThrow: Error = MockTranscriberError.simulatedFailure
    private var _transcribeCalls: [TranscribeCall] = []
    private var _sequentialIndex: Int = 0

    public var fixedResult: ChunkResult? {
        get { lock.withLock { _fixedResult } }
        set { lock.withLock { _fixedResult = newValue } }
    }

    public var sequentialResults: [ChunkResult] {
        get { lock.withLock { _sequentialResults } }
        set { lock.withLock { _sequentialResults = newValue; _sequentialIndex = 0 } }
    }

    public var transcribeDelay: TimeInterval {
        get { lock.withLock { _transcribeDelay } }
        set { lock.withLock { _transcribeDelay = newValue } }
    }

    public var shouldFail: Bool {
        get { lock.withLock { _shouldFail } }
        set { lock.withLock { _shouldFail = newValue } }
    }

    public var errorToThrow: Error {
        get { lock.withLock { _errorToThrow } }
        set { lock.withLock { _errorToThrow = newValue } }
    }

    public var transcribeCalls: [TranscribeCall] {
        lock.withLock { _transcribeCalls }
    }

    public var transcribeCallCount: Int {
        lock.withLock { _transcribeCalls.count }
    }

    public init() {}

    public func transcribe(
        audio: MLXArray,
        sampleRate: Int,
        previousTokens: [Int]?
    ) async throws -> ChunkResult {
        let call = TranscribeCall(
            audio: audio,
            sampleRate: sampleRate,
            previousTokens: previousTokens,
            timestamp: Date()
        )
        lock.withLock { _transcribeCalls.append(call) }

        let delay = lock.withLock { _transcribeDelay }
        if delay > 0 {
            try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
        }

        let shouldFail = lock.withLock { _shouldFail }
        if shouldFail {
            let error = lock.withLock { _errorToThrow }
            throw error
        }

        return lock.withLock {
            if let fixed = _fixedResult {
                return fixed
            }

            if !_sequentialResults.isEmpty {
                let result = _sequentialResults[_sequentialIndex % _sequentialResults.count]
                _sequentialIndex += 1
                return result
            }

            return ChunkResult(
                text: "Mock transcription",
                tokens: [],
                timeRange: 0...30,
                confidence: 1.0,
                words: nil
            )
        }
    }

    public func transcribeStreaming(
        audio: MLXArray,
        sampleRate: Int,
        previousTokens: [Int]?,
        timeOffset: TimeInterval
    ) -> AsyncThrowingStream<ChunkPartialResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let result = try await self.transcribe(
                        audio: audio,
                        sampleRate: sampleRate,
                        previousTokens: previousTokens
                    )

                    let adjustedTimestamp = (result.timeRange.lowerBound + timeOffset)...(result.timeRange.upperBound + timeOffset)
                    let partialResult = ChunkPartialResult(
                        text: result.text,
                        timestamp: adjustedTimestamp,
                        isFinal: true
                    )
                    continuation.yield(partialResult)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func reset() {
        lock.withLock {
            _transcribeCalls.removeAll()
            _sequentialIndex = 0
        }
    }
}

public enum MockTranscriberError: Error, Sendable {
    case simulatedFailure
    case timeout
    case invalidAudio
}
