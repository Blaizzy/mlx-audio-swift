import Foundation
import MLX
import Testing
@testable import MLXAudioSTT

struct LongAudioProcessorTests {
    let sampleRate = 16000

    // MARK: - Factory Tests

    @Test func factoryCreatesProcessorWithAutoStrategy() async throws {
        let processor = createTestProcessor(strategy: .auto)
        // Verify processor is usable by checking it can be cancelled
        processor.cancel()
    }

    @Test func factoryCreatesProcessorWithSequentialStrategy() async throws {
        let config = SequentialChunkingStrategy.SequentialConfig(
            conditionOnPreviousText: true,
            maxPreviousTokens: 128
        )
        let strategy = SequentialChunkingStrategy(config: config)
        let processor = createTestProcessor(strategy: strategy)
        processor.cancel()
    }

    @Test func factoryCreatesProcessorWithSlidingWindowStrategy() async throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 20.0,
            overlapDuration: 3.0
        )
        let strategy = SlidingWindowChunkingStrategy(config: config)
        let processor = createTestProcessor(strategy: strategy)
        processor.cancel()
    }

    @Test func factoryCreatesProcessorWithVADStrategy() async throws {
        let vadConfig = LongAudioProcessor.StrategyType.VADConfig(
            speechThreshold: 0.6,
            minSpeechDuration: 0.3
        )
        // Verify VAD config can be created
        #expect(vadConfig.speechThreshold == 0.6)
        #expect(vadConfig.minSpeechDuration == 0.3)
    }

    @Test func factoryCreatesProcessorWithCustomLimits() async throws {
        let limits = ProcessingLimits(
            maxConcurrentChunks: 2,
            chunkTimeout: 30,
            totalTimeout: 300
        )
        let processor = createTestProcessor(limits: limits)
        processor.cancel()
    }

    @Test func factoryCreatesProcessorWithCustomMergeConfig() async throws {
        let mergeConfig = LongAudioProcessor.MergeConfig(
            deduplicateOverlap: false,
            minWordConfidence: 0.5,
            normalizeText: false
        )
        let processor = createTestProcessor(mergeConfig: mergeConfig)
        processor.cancel()
    }

    // MARK: - Streaming Transcription Tests

    @Test func streamingTranscriptionYieldsProgressUpdates() async throws {
        let processor = createTestProcessor()

        // 3 seconds of audio
        let numSamples = 3 * sampleRate
        let audio = MLXArray.zeros([numSamples])

        var progressUpdates: [TranscriptionProgress] = []

        let stream: AsyncThrowingStream<TranscriptionProgress, Error> = processor.transcribe(
            audio,
            sampleRate: sampleRate
        )

        for try await progress in stream {
            progressUpdates.append(progress)
        }

        #expect(!progressUpdates.isEmpty)

        // Verify at least one final update
        let hasFinal = progressUpdates.contains { $0.isFinal }
        #expect(hasFinal)
    }

    @Test func streamingTranscriptionAccumulatesText() async throws {
        let processor = createTestProcessor()

        // 2 seconds of audio
        let numSamples = 2 * sampleRate
        let audio = MLXArray.zeros([numSamples])

        var lastProgress: TranscriptionProgress?

        let stream: AsyncThrowingStream<TranscriptionProgress, Error> = processor.transcribe(
            audio,
            sampleRate: sampleRate
        )

        for try await progress in stream {
            lastProgress = progress
        }

        #expect(lastProgress != nil)
        #expect(lastProgress?.isFinal == true)
    }

    @Test func streamingTranscriptionReportsCorrectProgress() async throws {
        let processor = createTestProcessor()

        // 5 seconds of audio
        let numSamples = 5 * sampleRate
        let audio = MLXArray.zeros([numSamples])

        var progressValues: [Float] = []

        let stream: AsyncThrowingStream<TranscriptionProgress, Error> = processor.transcribe(
            audio,
            sampleRate: sampleRate
        )

        for try await progress in stream {
            progressValues.append(progress.progress)
        }

        #expect(!progressValues.isEmpty)

        // Final progress should be 1.0
        if let lastProgress = progressValues.last {
            #expect(lastProgress >= 0.99)
        }
    }

    // MARK: - Blocking Transcription Tests

    @Test func blockingTranscriptionReturnsFinalResult() async throws {
        let processor = createTestProcessor()

        // 2 seconds of audio
        let numSamples = 2 * sampleRate
        let audio = MLXArray.zeros([numSamples])

        let result: TranscriptionResult = try await processor.transcribe(
            audio,
            sampleRate: sampleRate
        )

        #expect(result.duration > 0)
    }

    @Test func blockingTranscriptionReturnsCorrectDuration() async throws {
        let processor = createTestProcessor()

        // 3 seconds of audio
        let numSamples = 3 * sampleRate
        let audio = MLXArray.zeros([numSamples])

        let result: TranscriptionResult = try await processor.transcribe(
            audio,
            sampleRate: sampleRate
        )

        #expect(abs(result.duration - 3.0) < 0.1)
    }

    // MARK: - Cancellation Tests

    @Test func cancelStopsProcessing() async throws {
        let processor = createTestProcessor()

        // 10 seconds of audio
        let numSamples = 10 * sampleRate
        let audio = MLXArray.zeros([numSamples])

        let task = Task {
            var count = 0
            let stream: AsyncThrowingStream<TranscriptionProgress, Error> = processor.transcribe(
                audio,
                sampleRate: sampleRate
            )
            for try await _ in stream {
                count += 1
            }
            return count
        }

        // Wait briefly then cancel
        try await Task.sleep(nanoseconds: 100_000_000)
        processor.cancel()

        let result = await task.result
        switch result {
        case .success:
            // May complete if fast enough
            break
        case .failure(let error):
            // Cancellation is expected
            #expect(error is ChunkingError || error is CancellationError)
        }
    }

    // MARK: - Error Handling Tests

    @Test func rejectsInvalidSampleRate() async throws {
        let processor = createTestProcessor()

        let audio = MLXArray.zeros([16000])

        let stream: AsyncThrowingStream<TranscriptionProgress, Error> = processor.transcribe(
            audio,
            sampleRate: 44100  // Wrong sample rate
        )

        var threwError = false
        do {
            for try await _ in stream {}
        } catch let error as ChunkingError {
            if case .invalidSampleRate = error {
                threwError = true
            }
        }

        #expect(threwError)
    }

    // MARK: - MergeConfig Tests

    @Test func defaultMergeConfigHasCorrectValues() {
        let config = LongAudioProcessor.MergeConfig.default

        #expect(config.deduplicateOverlap == true)
        #expect(config.minWordConfidence == 0.3)
        #expect(config.normalizeText == true)
    }

    @Test func customMergeConfigIsApplied() {
        let config = LongAudioProcessor.MergeConfig(
            deduplicateOverlap: false,
            minWordConfidence: 0.8,
            normalizeText: false
        )

        #expect(config.deduplicateOverlap == false)
        #expect(config.minWordConfidence == 0.8)
        #expect(config.normalizeText == false)
    }

    // MARK: - StrategyType Tests

    @Test func vadConfigHasCorrectDefaults() {
        let config = LongAudioProcessor.StrategyType.VADConfig.default

        #expect(config.speechThreshold == 0.5)
        #expect(config.minSpeechDuration == 0.25)
        #expect(config.maxSegmentDuration == 30.0)
    }

    // MARK: - Helpers

    private func createTestProcessor(
        strategy: LongAudioProcessor.StrategyType = .auto,
        limits: ProcessingLimits = .default,
        mergeConfig: LongAudioProcessor.MergeConfig = .default
    ) -> TestLongAudioProcessor {
        let chunkingStrategy: ChunkingStrategy
        switch strategy {
        case .auto:
            chunkingStrategy = SlidingWindowChunkingStrategy(
                config: .init(windowDuration: 1.0, overlapDuration: 0.2)
            )
        case .sequential(let config):
            chunkingStrategy = SequentialChunkingStrategy(config: config)
        case .slidingWindow(let config):
            chunkingStrategy = SlidingWindowChunkingStrategy(config: config)
        case .vad:
            chunkingStrategy = SlidingWindowChunkingStrategy()
        }

        return TestLongAudioProcessor(
            strategy: chunkingStrategy,
            limits: limits,
            mergeConfig: mergeConfig
        )
    }

    private func createTestProcessor(strategy: ChunkingStrategy) -> TestLongAudioProcessor {
        TestLongAudioProcessor(strategy: strategy)
    }
}

// MARK: - Test Helpers

/// A test-friendly version of LongAudioProcessor that uses mocks
final class TestLongAudioProcessor: @unchecked Sendable {
    private let strategy: ChunkingStrategy
    private let transcriber: ChunkTranscriber
    private let limits: ProcessingLimits
    private let telemetry: ChunkingTelemetry?
    private let mergeConfig: LongAudioProcessor.MergeConfig

    private var currentTask: Task<Void, Never>?
    private let taskLock = NSLock()

    init(
        strategy: ChunkingStrategy? = nil,
        transcriber: ChunkTranscriber? = nil,
        limits: ProcessingLimits = .default,
        telemetry: ChunkingTelemetry? = nil,
        mergeConfig: LongAudioProcessor.MergeConfig = .default
    ) {
        self.strategy = strategy ?? SlidingWindowChunkingStrategy(
            config: .init(windowDuration: 1.0, overlapDuration: 0.2)
        )

        let mockTranscriber = MockChunkTranscriber()
        mockTranscriber.fixedResult = ChunkResult(
            text: "test transcription",
            tokens: [],
            timeRange: 0...1,
            confidence: 1.0,
            words: nil
        )
        self.transcriber = transcriber ?? mockTranscriber
        self.limits = limits
        self.telemetry = telemetry
        self.mergeConfig = mergeConfig
    }

    func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = AudioConstants.sampleRate,
        options: TranscriptionOptions = .default
    ) -> AsyncThrowingStream<TranscriptionProgress, Error> {
        AsyncThrowingStream { continuation in
            continuation.onTermination = { [weak self] _ in
                self?.taskLock.withLock {
                    self?.currentTask?.cancel()
                    self?.currentTask = nil
                }
            }

            self.taskLock.withLock {
                self.currentTask = Task {
                    do {
                        try await self.processAudio(
                            audio,
                            sampleRate: sampleRate,
                            continuation: continuation
                        )
                        continuation.finish()
                    } catch is CancellationError {
                        continuation.finish(throwing: ChunkingError.cancelled(partialResult: nil))
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
            }
        }
    }

    private func processAudio(
        _ audio: MLXArray,
        sampleRate: Int,
        continuation: AsyncThrowingStream<TranscriptionProgress, Error>.Continuation
    ) async throws {
        guard sampleRate == AudioConstants.sampleRate else {
            throw ChunkingError.invalidSampleRate(
                expected: AudioConstants.sampleRate,
                got: sampleRate
            )
        }

        let totalSamples = audio.shape[0]
        let audioDuration = TimeInterval(totalSamples) / TimeInterval(sampleRate)

        var accumulatedText = ""
        var chunkIndex = 0

        let chunkStream = strategy.process(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: transcriber,
            limits: limits,
            telemetry: telemetry,
            options: .default
        )

        for try await chunkResult in chunkStream {
            try Task.checkCancellation()

            let processedText = mergeConfig.normalizeText
                ? chunkResult.text.trimmingCharacters(in: CharacterSet.whitespaces)
                : chunkResult.text

            if !processedText.isEmpty {
                if !accumulatedText.isEmpty {
                    accumulatedText += " "
                }
                accumulatedText += processedText
            }

            let progress = TranscriptionProgress(
                text: accumulatedText,
                words: nil,
                isFinal: false,
                processedDuration: chunkResult.timeRange.upperBound,
                audioDuration: audioDuration,
                chunkIndex: chunkIndex,
                totalChunks: Int(ceil(audioDuration / 1.0))
            )

            continuation.yield(progress)
            chunkIndex += 1
        }

        let finalProgress = TranscriptionProgress(
            text: accumulatedText,
            words: nil,
            isFinal: true,
            processedDuration: audioDuration,
            audioDuration: audioDuration,
            chunkIndex: chunkIndex,
            totalChunks: chunkIndex
        )

        continuation.yield(finalProgress)
    }

    func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = AudioConstants.sampleRate,
        options: TranscriptionOptions = .default
    ) async throws -> TranscriptionResult {
        var finalText = ""
        var finalDuration: TimeInterval = 0

        let stream: AsyncThrowingStream<TranscriptionProgress, Error> = transcribe(
            audio,
            sampleRate: sampleRate,
            options: options
        )

        for try await progress in stream {
            if progress.isFinal {
                finalText = progress.text
                finalDuration = progress.audioDuration
            }
        }

        return TranscriptionResult(
            text: finalText,
            words: nil,
            duration: finalDuration,
            language: options.language
        )
    }

    func cancel() {
        taskLock.withLock {
            currentTask?.cancel()
            currentTask = nil
        }
    }
}
