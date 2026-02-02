import Foundation
import MLX

/// Mock implementation of VADProvider for testing
public final class MockVADProvider: VADProvider, @unchecked Sendable {
    private let lock = NSLock()
    private var _segments: [SpeechSegment] = []
    private var _probabilities: [(time: TimeInterval, probability: Float)] = []
    private var _shouldFail: Bool = false
    private var _errorToThrow: VADError = .modelLoadFailed("Mock failure")
    private var _detectSpeechCalls: Int = 0
    private var _speechProbabilitiesCalls: Int = 0
    private var _resetCalls: Int = 0

    public var segments: [SpeechSegment] {
        get { lock.withLock { _segments } }
        set { lock.withLock { _segments = newValue } }
    }

    public var probabilities: [(time: TimeInterval, probability: Float)] {
        get { lock.withLock { _probabilities } }
        set { lock.withLock { _probabilities = newValue } }
    }

    public var shouldFail: Bool {
        get { lock.withLock { _shouldFail } }
        set { lock.withLock { _shouldFail = newValue } }
    }

    public var errorToThrow: VADError {
        get { lock.withLock { _errorToThrow } }
        set { lock.withLock { _errorToThrow = newValue } }
    }

    public var detectSpeechCallCount: Int {
        lock.withLock { _detectSpeechCalls }
    }

    public var speechProbabilitiesCallCount: Int {
        lock.withLock { _speechProbabilitiesCalls }
    }

    public var resetCallCount: Int {
        lock.withLock { _resetCalls }
    }

    public init() {}

    public init(segments: [SpeechSegment]) {
        self._segments = segments
    }

    public init(probabilities: [(time: TimeInterval, probability: Float)]) {
        self._probabilities = probabilities
    }

    public func detectSpeech(in audio: MLXArray, sampleRate: Int) async throws -> [SpeechSegment] {
        lock.withLock { _detectSpeechCalls += 1 }

        let shouldFail = lock.withLock { _shouldFail }
        if shouldFail {
            let error = lock.withLock { _errorToThrow }
            throw error
        }

        return lock.withLock { _segments }
    }

    public func speechProbabilities(in audio: MLXArray, sampleRate: Int) async throws -> [(time: TimeInterval, probability: Float)] {
        lock.withLock { _speechProbabilitiesCalls += 1 }

        let shouldFail = lock.withLock { _shouldFail }
        if shouldFail {
            let error = lock.withLock { _errorToThrow }
            throw error
        }

        return lock.withLock { _probabilities }
    }

    public func reset() async {
        lock.withLock { _resetCalls += 1 }
    }

    public func resetCallCounts() {
        lock.withLock {
            _detectSpeechCalls = 0
            _speechProbabilitiesCalls = 0
            _resetCalls = 0
        }
    }
}
