import Foundation
import MLX
import Testing
@testable import MLXAudioSTT

struct SequentialChunkingStrategyTests {
    let sampleRate = 16000

    @Test func processesAudioSequentially() async throws {
        let mock = MockChunkTranscriber()
        mock.sequentialResults = [
            ChunkResult(text: "First chunk", tokens: [1, 2, 3], timeRange: 0...10, confidence: 0.9),
            ChunkResult(text: "Second chunk", tokens: [4, 5, 6], timeRange: 0...10, confidence: 0.9),
            ChunkResult(text: "Third chunk", tokens: [7, 8, 9], timeRange: 0...10, confidence: 0.9),
        ]

        let audio = MLXArray.zeros([sampleRate * 90])
        let strategy = SequentialChunkingStrategy(config: .init(maxChunkDuration: 30.0))
        let limits = ProcessingLimits(chunkTimeout: 10)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mock,
            limits: limits,
            telemetry: nil
        ) {
            results.append(result)
        }

        #expect(results.count == 3)
        #expect(results[0].text == "First chunk")
        #expect(results[1].text == "Second chunk")
        #expect(results[2].text == "Third chunk")

        #expect(mock.transcribeCallCount == 3)

        let calls = mock.transcribeCalls
        #expect(calls[0].timestamp <= calls[1].timestamp)
        #expect(calls[1].timestamp <= calls[2].timestamp)
    }

    @Test func passesPreviousTokensWhenConfigured() async throws {
        let mock = MockChunkTranscriber()
        mock.sequentialResults = [
            ChunkResult(text: "First", tokens: [100, 200, 300], timeRange: 0...30, confidence: 0.9),
            ChunkResult(text: "Second", tokens: [400, 500], timeRange: 0...30, confidence: 0.9),
        ]

        let audio = MLXArray.zeros([sampleRate * 60])
        let config = SequentialChunkingStrategy.SequentialConfig(
            conditionOnPreviousText: true,
            maxPreviousTokens: 224,
            maxChunkDuration: 30.0
        )
        let strategy = SequentialChunkingStrategy(config: config)
        let limits = ProcessingLimits(chunkTimeout: 10)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mock,
            limits: limits,
            telemetry: nil
        ) {
            results.append(result)
        }

        #expect(results.count == 2)
        #expect(mock.transcribeCallCount == 2)

        let calls = mock.transcribeCalls
        #expect(calls[0].previousTokens == nil)
        #expect(calls[1].previousTokens == [100, 200, 300])
    }

    @Test func doesNotPassPreviousTokensWhenDisabled() async throws {
        let mock = MockChunkTranscriber()
        mock.sequentialResults = [
            ChunkResult(text: "First", tokens: [100, 200], timeRange: 0...30, confidence: 0.9),
            ChunkResult(text: "Second", tokens: [300, 400], timeRange: 0...30, confidence: 0.9),
        ]

        let audio = MLXArray.zeros([sampleRate * 60])
        let config = SequentialChunkingStrategy.SequentialConfig(
            conditionOnPreviousText: false,
            maxChunkDuration: 30.0
        )
        let strategy = SequentialChunkingStrategy(config: config)
        let limits = ProcessingLimits(chunkTimeout: 10)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mock,
            limits: limits,
            telemetry: nil
        ) {
            results.append(result)
        }

        #expect(results.count == 2)

        let calls = mock.transcribeCalls
        #expect(calls[0].previousTokens == nil)
        #expect(calls[1].previousTokens == nil)
    }

    @Test func seeksBasedOnTimestampInResult() async throws {
        let mock = MockChunkTranscriber()
        mock.sequentialResults = [
            ChunkResult(
                text: "First",
                tokens: [1],
                timeRange: 0...20,
                confidence: 0.9,
                words: [
                    WordTimestamp(word: "First", start: 0, end: 15, confidence: 0.9)
                ]
            ),
            ChunkResult(
                text: "Second",
                tokens: [2],
                timeRange: 0...25,
                confidence: 0.9,
                words: [
                    WordTimestamp(word: "Second", start: 0, end: 20, confidence: 0.9)
                ]
            ),
        ]

        let audio = MLXArray.zeros([sampleRate * 45])
        let config = SequentialChunkingStrategy.SequentialConfig(maxChunkDuration: 30.0)
        let strategy = SequentialChunkingStrategy(config: config)
        let limits = ProcessingLimits(chunkTimeout: 10)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mock,
            limits: limits,
            telemetry: nil
        ) {
            results.append(result)
        }

        #expect(results.count == 2)
        #expect(results[0].timeRange.lowerBound == 0)
        #expect(results[1].timeRange.lowerBound == 15)
    }

    @Test func adjustsTimestampsToAbsolutePositions() async throws {
        let mock = MockChunkTranscriber()
        mock.sequentialResults = [
            ChunkResult(
                text: "First",
                tokens: [1],
                timeRange: 0...30,
                confidence: 0.9,
                words: [
                    WordTimestamp(word: "Hello", start: 5, end: 10, confidence: 0.9),
                    WordTimestamp(word: "World", start: 15, end: 25, confidence: 0.9),
                ]
            ),
            ChunkResult(
                text: "Second",
                tokens: [2],
                timeRange: 0...15,
                confidence: 0.9,
                words: [
                    WordTimestamp(word: "Test", start: 2, end: 8, confidence: 0.9),
                ]
            ),
        ]

        let audio = MLXArray.zeros([sampleRate * 60])
        let config = SequentialChunkingStrategy.SequentialConfig(maxChunkDuration: 30.0)
        let strategy = SequentialChunkingStrategy(config: config)
        let limits = ProcessingLimits(chunkTimeout: 10)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mock,
            limits: limits,
            telemetry: nil
        ) {
            results.append(result)
        }

        #expect(results.count == 2)

        #expect(results[0].timeRange.lowerBound == 0)
        #expect(results[0].words?[0].start == 5)
        #expect(results[0].words?[1].end == 25)

        #expect(results[1].timeRange.lowerBound == 25)
        #expect(results[1].words?[0].start == 27)
        #expect(results[1].words?[0].end == 33)
    }

    @Test func handlesTimeout() async throws {
        let mock = MockChunkTranscriber()
        mock.transcribeDelay = 5.0
        mock.fixedResult = ChunkResult(text: "Never", tokens: [], timeRange: 0...30, confidence: 0.9)

        let audio = MLXArray.zeros([sampleRate * 30])
        let strategy = SequentialChunkingStrategy()
        let limits = ProcessingLimits(chunkTimeout: 0.1)

        var caughtError: Error?
        do {
            for try await _ in strategy.process(
                audio: audio,
                sampleRate: sampleRate,
                transcriber: mock,
                limits: limits,
                telemetry: nil
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
        } else {
            Issue.record("Expected ChunkingError but got \(String(describing: caughtError))")
        }
    }

    @Test func handlesTotalTimeout() async throws {
        let mock = MockChunkTranscriber()
        mock.transcribeDelay = 0.5
        mock.fixedResult = ChunkResult(text: "Chunk", tokens: [], timeRange: 0...30, confidence: 0.9)

        let audio = MLXArray.zeros([sampleRate * 120])
        let strategy = SequentialChunkingStrategy(config: .init(maxChunkDuration: 30.0))
        let limits = ProcessingLimits(chunkTimeout: 10, totalTimeout: 0.8)

        var caughtError: Error?
        var resultCount = 0
        do {
            for try await _ in strategy.process(
                audio: audio,
                sampleRate: sampleRate,
                transcriber: mock,
                limits: limits,
                telemetry: nil
            ) {
                resultCount += 1
            }
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

    @Test func skipsLowConfidenceResults() async throws {
        let mock = MockChunkTranscriber()
        mock.sequentialResults = [
            ChunkResult(text: "Good", tokens: [1], timeRange: 0...30, confidence: 0.9),
            ChunkResult(text: "Bad", tokens: [2], timeRange: 0...30, confidence: 0.3),
            ChunkResult(text: "Good again", tokens: [3], timeRange: 0...30, confidence: 0.8),
        ]

        let audio = MLXArray.zeros([sampleRate * 90])
        let config = SequentialChunkingStrategy.SequentialConfig(
            noSpeechThreshold: 0.6,
            maxChunkDuration: 30.0
        )
        let strategy = SequentialChunkingStrategy(config: config)
        let limits = ProcessingLimits(chunkTimeout: 10)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mock,
            limits: limits,
            telemetry: nil
        ) {
            results.append(result)
        }

        #expect(results.count == 2)
        #expect(results[0].text == "Good")
        #expect(results[1].text == "Good again")
    }

    @Test func respectsMaxPreviousTokensLimit() async throws {
        let mock = MockChunkTranscriber()
        let longTokens = Array(1...300)
        mock.sequentialResults = [
            ChunkResult(text: "First", tokens: longTokens, timeRange: 0...30, confidence: 0.9),
            ChunkResult(text: "Second", tokens: [1000], timeRange: 0...30, confidence: 0.9),
        ]

        let audio = MLXArray.zeros([sampleRate * 60])
        let config = SequentialChunkingStrategy.SequentialConfig(
            conditionOnPreviousText: true,
            maxPreviousTokens: 100,
            maxChunkDuration: 30.0
        )
        let strategy = SequentialChunkingStrategy(config: config)
        let limits = ProcessingLimits(chunkTimeout: 10)

        var results: [ChunkResult] = []
        for try await result in strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mock,
            limits: limits,
            telemetry: nil
        ) {
            results.append(result)
        }

        let calls = mock.transcribeCalls
        #expect(calls[1].previousTokens?.count == 100)
        #expect(calls[1].previousTokens == Array(201...300))
    }

    @Test func strategyProperties() {
        let strategy = SequentialChunkingStrategy()
        #expect(strategy.name == "sequential")
        #expect(strategy.transcriptionMode == .sequential)
    }
}
