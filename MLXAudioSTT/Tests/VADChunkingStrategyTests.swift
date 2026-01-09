import Foundation
import MLX
import Testing
@testable import MLXAudioSTT

struct VADChunkingStrategyTests {
    let sampleRate = 16000

    // MARK: - VAD Segment Processing Tests

    @Test func processesVADSegmentsCorrectly() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.5, end: 2.5, confidence: 0.9),
            SpeechSegment(start: 5.0, end: 8.0, confidence: 0.95),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.sequentialResults = [
            ChunkResult(text: "First segment", tokens: [1, 2], timeRange: 0...2, confidence: 0.9),
            ChunkResult(text: "Second segment", tokens: [3, 4], timeRange: 0...3, confidence: 0.95),
        ]

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.1,
            parallelProcessing: false,
            packSegments: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 10])

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        #expect(results.count == 2)
        #expect(results[0].text == "First segment")
        #expect(results[1].text == "Second segment")

        #expect(mockVAD.detectSpeechCallCount == 1)
        #expect(mockTranscriber.transcribeCallCount == 2)
    }

    // MARK: - Parallel Processing Tests

    @Test func parallelProcessingWhenEnabled() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 1.0, confidence: 0.9),
            SpeechSegment(start: 2.0, end: 3.0, confidence: 0.9),
            SpeechSegment(start: 4.0, end: 5.0, confidence: 0.9),
            SpeechSegment(start: 6.0, end: 7.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.transcribeDelay = 0.05
        mockTranscriber.fixedResult = ChunkResult(
            text: "chunk",
            tokens: [],
            timeRange: 0...1,
            confidence: 0.9
        )

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.0,
            parallelProcessing: true,
            packSegments: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 10])
        let limits = ProcessingLimits(maxConcurrentChunks: 4)

        let startTime = Date()
        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: limits,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }
        let elapsed = Date().timeIntervalSince(startTime)

        #expect(results.count == 4)
        #expect(mockTranscriber.transcribeCallCount == 4)

        // With parallel processing (4 concurrent), should complete faster than sequential
        // Sequential would take ~0.2s (4 * 0.05s), parallel should be closer to 0.05s
        #expect(elapsed < 0.15)
    }

    // MARK: - Sequential Fallback Tests

    @Test func sequentialFallbackWhenParallelDisabled() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 1.0, confidence: 0.9),
            SpeechSegment(start: 2.0, end: 3.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.sequentialResults = [
            ChunkResult(text: "First", tokens: [1], timeRange: 0...1, confidence: 0.9),
            ChunkResult(text: "Second", tokens: [2], timeRange: 0...1, confidence: 0.9),
        ]

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.0,
            parallelProcessing: false,
            packSegments: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 5])

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        #expect(results.count == 2)
        #expect(results[0].text == "First")
        #expect(results[1].text == "Second")

        // Verify sequential order by checking timestamps
        let calls = mockTranscriber.transcribeCalls
        #expect(calls[0].timestamp <= calls[1].timestamp)
    }

    @Test func sequentialFallbackWhenMaxConcurrencyIsOne() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 1.0, confidence: 0.9),
            SpeechSegment(start: 2.0, end: 3.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.fixedResult = ChunkResult(
            text: "chunk",
            tokens: [],
            timeRange: 0...1,
            confidence: 0.9
        )

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.0,
            parallelProcessing: true,
            packSegments: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 5])
        let limits = ProcessingLimits(maxConcurrentChunks: 1)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: limits,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        #expect(results.count == 2)
    }

    // MARK: - No Speech Detected Tests

    @Test func handlesNoSpeechDetected() async throws {
        let mockVAD = MockVADProvider(segments: [])

        let mockTranscriber = MockChunkTranscriber()

        let strategy = VADChunkingStrategy(vadProvider: mockVAD)

        let audio = MLXArray.zeros([sampleRate * 5])

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        #expect(results.isEmpty)
        #expect(mockTranscriber.transcribeCallCount == 0)
        #expect(mockVAD.detectSpeechCallCount == 1)
    }

    // MARK: - Segment Merging Tests

    @Test func mergesSmallSegments() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 0.5, confidence: 0.9),
            SpeechSegment(start: 0.6, end: 1.0, confidence: 0.9),
            SpeechSegment(start: 1.1, end: 1.5, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.fixedResult = ChunkResult(
            text: "merged chunk",
            tokens: [],
            timeRange: 0...1.5,
            confidence: 0.9
        )

        let config = VADChunkingStrategy.VADConfig(
            targetChunkDuration: 30.0,
            minSpeechDuration: 0.1,
            speechPadding: 0.2,
            parallelProcessing: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 5])

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        // Small segments should be merged into one
        #expect(results.count == 1)
        #expect(mockTranscriber.transcribeCallCount == 1)
    }

    // MARK: - Large Segment Splitting Tests

    @Test func splitsLargeSegments() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 45.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.fixedResult = ChunkResult(
            text: "split chunk",
            tokens: [],
            timeRange: 0...30,
            confidence: 0.9
        )

        let config = VADChunkingStrategy.VADConfig(
            maxChunkDuration: 30.0,
            minSpeechDuration: 0.1,
            speechPadding: 0.0,
            parallelProcessing: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 50])

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        // 45s segment should be split into 2 chunks (30s + 15s)
        #expect(results.count == 2)
        #expect(mockTranscriber.transcribeCallCount == 2)
    }

    // MARK: - VAD Failure Tests

    @Test func handlesVADFailure() async throws {
        let mockVAD = MockVADProvider()
        mockVAD.shouldFail = true
        mockVAD.errorToThrow = .modelLoadFailed("Test failure")

        let mockTranscriber = MockChunkTranscriber()

        let strategy = VADChunkingStrategy(vadProvider: mockVAD)

        let audio = MLXArray.zeros([sampleRate * 5])

        var caughtError: Error?
        do {
            for try await _ in strategy.process(
                audio: audio,
                sampleRate: sampleRate,
                transcriber: mockTranscriber,
                limits: .default,
                telemetry: nil,
                options: .default
            ) {}
        } catch {
            caughtError = error
        }

        #expect(caughtError != nil)
        if let chunkingError = caughtError as? ChunkingError {
            switch chunkingError {
            case .vadFailed:
                break
            default:
                Issue.record("Expected vadFailed error but got \(chunkingError)")
            }
        }
    }

    // MARK: - Timeout Tests

    @Test func respectsChunkTimeout() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 1.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.transcribeDelay = 5.0
        mockTranscriber.fixedResult = ChunkResult(
            text: "never",
            tokens: [],
            timeRange: 0...1,
            confidence: 0.9
        )

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.0,
            parallelProcessing: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 5])
        let limits = ProcessingLimits(chunkTimeout: 0.1)

        var caughtError: Error?
        do {
            for try await _ in strategy.process(
                audio: audio,
                sampleRate: sampleRate,
                transcriber: mockTranscriber,
                limits: limits,
                telemetry: nil,
                options: .default
            ) {}
        } catch {
            caughtError = error
        }

        #expect(caughtError != nil)
        if let chunkingError = caughtError as? ChunkingError {
            switch chunkingError {
            case .chunkTimeout(let index, _):
                #expect(index == 0)
            default:
                Issue.record("Expected chunkTimeout error but got \(chunkingError)")
            }
        }
    }

    @Test func respectsTotalTimeout() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 1.0, confidence: 0.9),
            SpeechSegment(start: 2.0, end: 3.0, confidence: 0.9),
            SpeechSegment(start: 4.0, end: 5.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.transcribeDelay = 0.3
        mockTranscriber.fixedResult = ChunkResult(
            text: "chunk",
            tokens: [],
            timeRange: 0...1,
            confidence: 0.9
        )

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.0,
            parallelProcessing: false,
            packSegments: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 10])
        let limits = ProcessingLimits(chunkTimeout: 10, totalTimeout: 0.5)

        var caughtError: Error?
        do {
            for try await _ in strategy.process(
                audio: audio,
                sampleRate: sampleRate,
                transcriber: mockTranscriber,
                limits: limits,
                telemetry: nil,
                options: .default
            ) {}
        } catch {
            caughtError = error
        }

        #expect(caughtError != nil)
        if let chunkingError = caughtError as? ChunkingError {
            switch chunkingError {
            case .totalTimeoutExceeded:
                break
            default:
                Issue.record("Expected totalTimeoutExceeded error but got \(chunkingError)")
            }
        }
    }

    // MARK: - Time Ordering Tests

    @Test func resultsYieldedInTimeOrder() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 1.0, confidence: 0.9),
            SpeechSegment(start: 2.0, end: 3.0, confidence: 0.9),
            SpeechSegment(start: 4.0, end: 5.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.sequentialResults = [
            ChunkResult(text: "First", tokens: [], timeRange: 0...1, confidence: 0.9),
            ChunkResult(text: "Second", tokens: [], timeRange: 0...1, confidence: 0.9),
            ChunkResult(text: "Third", tokens: [], timeRange: 0...1, confidence: 0.9),
        ]

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.0,
            parallelProcessing: true,
            packSegments: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 10])
        let limits = ProcessingLimits(maxConcurrentChunks: 4)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: limits,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        #expect(results.count == 3)

        // Verify time ordering
        for i in 1..<results.count {
            #expect(results[i].timeRange.lowerBound >= results[i - 1].timeRange.lowerBound)
        }
    }

    // MARK: - Property Tests

    @Test func strategyProperties() {
        let mockVAD = MockVADProvider()
        let strategy = VADChunkingStrategy(vadProvider: mockVAD)

        #expect(strategy.name == "vad")
        #expect(strategy.transcriptionMode == .independent)
    }

    @Test func defaultConfiguration() {
        let config = VADChunkingStrategy.VADConfig.default

        #expect(config.targetChunkDuration == 30.0)
        #expect(config.maxChunkDuration == 30.0)
        #expect(config.minSpeechDuration == 0.5)
        #expect(config.speechPadding == 0.2)
        #expect(config.parallelProcessing == true)
    }

    // MARK: - Word Timestamp Adjustment Tests

    @Test func adjustsWordTimestampsCorrectly() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 5.0, end: 7.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.fixedResult = ChunkResult(
            text: "Hello world",
            tokens: [1, 2],
            timeRange: 0...2,
            confidence: 0.9,
            words: [
                WordTimestamp(word: "Hello", start: 0.0, end: 0.5, confidence: 0.9),
                WordTimestamp(word: "world", start: 0.6, end: 1.0, confidence: 0.9),
            ]
        )

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.0,
            parallelProcessing: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 10])

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        #expect(results.count == 1)
        #expect(results[0].words?.count == 2)

        // Word timestamps should be adjusted by segment start (5.0s)
        if let words = results[0].words {
            #expect(words[0].start == 5.0)
            #expect(words[0].end == 5.5)
            #expect(words[1].start == 5.6)
            #expect(words[1].end == 6.0)
        }
    }

    // MARK: - Empty Audio Tests

    @Test func handlesEmptyAudio() async throws {
        let mockVAD = MockVADProvider(segments: [])
        let mockTranscriber = MockChunkTranscriber()

        let strategy = VADChunkingStrategy(vadProvider: mockVAD)
        let audio = MLXArray([Float]())

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        #expect(results.isEmpty)
    }

    // MARK: - Transcription Failure Tests

    @Test func handlesTranscriptionFailureWithAbortOnFirstFailure() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 1.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.shouldFail = true
        mockTranscriber.errorToThrow = MockTranscriberError.simulatedFailure

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.0,
            parallelProcessing: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 5])
        let limits = ProcessingLimits(abortOnFirstFailure: true)

        var caughtError: Error?
        do {
            for try await _ in strategy.process(
                audio: audio,
                sampleRate: sampleRate,
                transcriber: mockTranscriber,
                limits: limits,
                telemetry: nil,
                options: .default
            ) {}
        } catch {
            caughtError = error
        }

        #expect(caughtError != nil)
        if let chunkingError = caughtError as? ChunkingError {
            switch chunkingError {
            case .chunkTranscriptionFailed:
                break
            default:
                Issue.record("Expected chunkTranscriptionFailed error but got \(chunkingError)")
            }
        }
    }

    @Test func continuesOnTranscriptionFailureWithoutAbortFlag() async throws {
        let mockVAD = MockVADProvider(segments: [
            SpeechSegment(start: 0.0, end: 1.0, confidence: 0.9),
            SpeechSegment(start: 2.0, end: 3.0, confidence: 0.9),
        ])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.shouldFail = true
        mockTranscriber.errorToThrow = MockTranscriberError.simulatedFailure

        let config = VADChunkingStrategy.VADConfig(
            speechPadding: 0.0,
            parallelProcessing: false,
            packSegments: false
        )
        let strategy = VADChunkingStrategy(vadProvider: mockVAD, config: config)

        let audio = MLXArray.zeros([sampleRate * 5])
        let limits = ProcessingLimits(abortOnFirstFailure: false)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: limits,
            telemetry: nil,
            options: .default
        ) {
            results.append(result)
        }

        // Should still yield empty results for failed chunks
        #expect(results.count == 2)
        #expect(results[0].text.isEmpty)
        #expect(results[1].text.isEmpty)
    }
}
