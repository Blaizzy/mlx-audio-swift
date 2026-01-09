import Foundation
import MLX
import Testing
@testable import MLXAudioSTT

struct SlidingWindowChunkingStrategyTests {
    let sampleRate = 16000

    // MARK: - Window Size Tests

    @Test func processesAudioInCorrectWindowSizes() async throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 2.0,
            overlapDuration: 0.5,
            mergeStrategy: .simple
        )
        let strategy = SlidingWindowChunkingStrategy(config: config)

        // 5 seconds of audio
        let numSamples = 5 * sampleRate
        let audio = MLXArray.ones([numSamples])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.fixedResult = ChunkResult(
            text: "test",
            tokens: [],
            timeRange: 0...2,
            confidence: 1.0,
            words: nil
        )

        let stream = strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil
        )

        var results: [ChunkResult] = []
        for try await result in stream {
            results.append(result)
        }

        // With 2s window and 1.5s hop (2-0.5), we expect:
        // Chunk 0: 0-2s
        // Chunk 1: 1.5-3.5s
        // Chunk 2: 3-5s (or 3.5-5s)
        #expect(mockTranscriber.transcribeCallCount >= 3)

        // Verify each call received audio of correct size (window duration worth of samples)
        let calls = mockTranscriber.transcribeCalls
        for call in calls.dropLast() {
            let expectedSamples = Int(config.windowDuration * Double(sampleRate))
            #expect(call.audio.shape[0] == expectedSamples)
        }
    }

    @Test func handlesAudioShorterThanWindow() async throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 30.0,
            overlapDuration: 5.0,
            mergeStrategy: .simple
        )
        let strategy = SlidingWindowChunkingStrategy(config: config)

        // 10 seconds of audio (shorter than 30s window)
        let numSamples = 10 * sampleRate
        let audio = MLXArray.ones([numSamples])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.fixedResult = ChunkResult(
            text: "short audio transcription",
            tokens: [],
            timeRange: 0...10,
            confidence: 0.95,
            words: nil
        )

        let stream = strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil
        )

        var results: [ChunkResult] = []
        for try await result in stream {
            results.append(result)
        }

        // Should have exactly one chunk
        #expect(results.count == 1)
        #expect(mockTranscriber.transcribeCallCount == 1)

        // Verify the chunk covers the full audio
        if let result = results.first {
            #expect(result.timeRange.lowerBound == 0.0)
            #expect(abs(result.timeRange.upperBound - 10.0) < 0.01)
        }
    }

    // MARK: - Progressive Results Tests

    @Test func yieldsProgressiveResults() async throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 1.0,
            overlapDuration: 0.2,
            mergeStrategy: .simple
        )
        let strategy = SlidingWindowChunkingStrategy(config: config)

        // 3 seconds of audio
        let numSamples = 3 * sampleRate
        let audio = MLXArray.ones([numSamples])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.sequentialResults = [
            ChunkResult(text: "first chunk", tokens: [], timeRange: 0...1, confidence: 1.0, words: nil),
            ChunkResult(text: "second chunk", tokens: [], timeRange: 0...1, confidence: 1.0, words: nil),
            ChunkResult(text: "third chunk", tokens: [], timeRange: 0...1, confidence: 1.0, words: nil),
            ChunkResult(text: "fourth chunk", tokens: [], timeRange: 0...1, confidence: 1.0, words: nil),
        ]

        let stream = strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil
        )

        var receivedOrder: [String] = []
        for try await result in stream {
            receivedOrder.append(result.text)
        }

        // Verify results are yielded in order
        #expect(receivedOrder.count >= 3)
        #expect(receivedOrder.first?.contains("first") == true)
    }

    // MARK: - Cancellation Tests

    @Test func respectsCancellation() async throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 1.0,
            overlapDuration: 0.2,
            mergeStrategy: .simple
        )
        let strategy = SlidingWindowChunkingStrategy(config: config)

        // 10 seconds of audio
        let numSamples = 10 * sampleRate
        let audio = MLXArray.ones([numSamples])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.transcribeDelay = 0.1
        mockTranscriber.fixedResult = ChunkResult(
            text: "chunk",
            tokens: [],
            timeRange: 0...1,
            confidence: 1.0,
            words: nil
        )

        let task = Task {
            var count = 0
            let stream = strategy.process(
                audio: audio,
                sampleRate: sampleRate,
                transcriber: mockTranscriber,
                limits: .default,
                telemetry: nil
            )
            for try await _ in stream {
                count += 1
            }
            return count
        }

        try await Task.sleep(nanoseconds: 250_000_000)
        task.cancel()

        let result = await task.result
        switch result {
        case .success(let count):
            // Should have processed some but not all chunks
            let expectedTotalChunks = 12
            #expect(count < expectedTotalChunks)
        case .failure(let error):
            // Cancellation error is expected
            #expect(error is CancellationError)
        }
    }

    // MARK: - Timeout Tests

    @Test func respectsTotalTimeout() async throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 1.0,
            overlapDuration: 0.2,
            mergeStrategy: .simple
        )
        let strategy = SlidingWindowChunkingStrategy(config: config)

        // 5 seconds of audio
        let numSamples = 5 * sampleRate
        let audio = MLXArray.ones([numSamples])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.transcribeDelay = 0.3
        mockTranscriber.fixedResult = ChunkResult(
            text: "chunk",
            tokens: [],
            timeRange: 0...1,
            confidence: 1.0,
            words: nil
        )

        let limits = ProcessingLimits(
            maxConcurrentChunks: 1,
            chunkTimeout: 60,
            totalTimeout: 0.5
        )

        let stream = strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: limits,
            telemetry: nil
        )

        var didTimeout = false
        do {
            for try await _ in stream {}
        } catch let error as ChunkingError {
            if case .totalTimeoutExceeded = error {
                didTimeout = true
            }
        }

        #expect(didTimeout)
    }

    // MARK: - Merge Strategy Tests

    @Test func simpleMergeDeduplicatesOverlappingWords() async throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 2.0,
            overlapDuration: 0.5,
            mergeStrategy: .simple
        )
        let strategy = SlidingWindowChunkingStrategy(config: config)

        // 3 seconds of audio
        let numSamples = 3 * sampleRate
        let audio = MLXArray.ones([numSamples])

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.sequentialResults = [
            ChunkResult(
                text: "hello world testing",
                tokens: [],
                timeRange: 0...2,
                confidence: 1.0,
                words: nil
            ),
            ChunkResult(
                text: "testing one two three",
                tokens: [],
                timeRange: 0...2,
                confidence: 1.0,
                words: nil
            ),
        ]

        let stream = strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil
        )

        var results: [ChunkResult] = []
        for try await result in stream {
            results.append(result)
        }

        #expect(results.count >= 2)

        // Second result should have "testing" deduplicated
        if results.count >= 2 {
            #expect(results[1].text == "one two three")
        }
    }

    // MARK: - Edge Cases

    @Test func handlesEmptyAudio() async throws {
        let strategy = SlidingWindowChunkingStrategy()
        let audio = MLXArray([Float]())

        let mockTranscriber = MockChunkTranscriber()

        let stream = strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: mockTranscriber,
            limits: .default,
            telemetry: nil
        )

        var results: [ChunkResult] = []
        for try await result in stream {
            results.append(result)
        }

        #expect(results.isEmpty)
        #expect(mockTranscriber.transcribeCallCount == 0)
    }

    @Test func usesDefaultConfigurationWhenNotSpecified() async throws {
        let strategy = SlidingWindowChunkingStrategy()

        #expect(strategy.name == "slidingWindow")
        #expect(strategy.transcriptionMode == .independent)
    }
}
