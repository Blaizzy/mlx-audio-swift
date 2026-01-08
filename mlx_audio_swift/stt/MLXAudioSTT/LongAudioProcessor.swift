import Foundation
import MLX

/// Unified API for transcribing audio of any length
public final class LongAudioProcessor: @unchecked Sendable {
    private let session: WhisperSession
    private let strategy: ChunkingStrategy
    private let mergeConfig: MergeConfig
    private let limits: ProcessingLimits
    private let telemetry: ChunkingTelemetry?
    private let transcriber: ChunkTranscriber

    private var currentTask: Task<Void, Never>?
    private let taskLock = NSLock()

    public struct MergeConfig: Sendable {
        public var deduplicateOverlap: Bool
        public var minWordConfidence: Float
        public var normalizeText: Bool

        public init(
            deduplicateOverlap: Bool = true,
            minWordConfidence: Float = 0.3,
            normalizeText: Bool = true
        ) {
            self.deduplicateOverlap = deduplicateOverlap
            self.minWordConfidence = minWordConfidence
            self.normalizeText = normalizeText
        }

        public static let `default` = MergeConfig()
    }

    public enum StrategyType: Sendable {
        case auto
        case sequential(SequentialChunkingStrategy.SequentialConfig = .default)
        case slidingWindow(SlidingWindowChunkingStrategy.SlidingWindowConfig = .default)
        case vad(VADConfig = .default, vadProvider: VADProviderType = .energy)

        public struct VADConfig: Sendable {
            public var speechThreshold: Float
            public var minSpeechDuration: TimeInterval
            public var maxSegmentDuration: TimeInterval

            public init(
                speechThreshold: Float = 0.5,
                minSpeechDuration: TimeInterval = 0.25,
                maxSegmentDuration: TimeInterval = 30.0
            ) {
                self.speechThreshold = speechThreshold
                self.minSpeechDuration = minSpeechDuration
                self.maxSegmentDuration = maxSegmentDuration
            }

            public static let `default` = VADConfig()
        }
    }

    public enum VADProviderType: Sendable {
        case energy
        case sileroMLX
    }

    private init(
        session: WhisperSession,
        strategy: ChunkingStrategy,
        transcriber: ChunkTranscriber,
        mergeConfig: MergeConfig,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) {
        self.session = session
        self.strategy = strategy
        self.transcriber = transcriber
        self.mergeConfig = mergeConfig
        self.limits = limits
        self.telemetry = telemetry
    }

    /// Wait for the underlying session to be ready (for background loading)
    public func waitUntilReady(timeout: Duration = .seconds(30)) async throws -> Bool {
        try await session.waitUntilReady(timeout: timeout)
    }

    // MARK: - Factory

    public static func create(
        model: WhisperModel = .largeTurbo,
        strategy: StrategyType = .auto,
        mergeConfig: MergeConfig = .default,
        limits: ProcessingLimits = .default,
        telemetry: ChunkingTelemetry? = nil,
        progressHandler: ((WhisperProgress) -> Void)? = nil
    ) async throws -> LongAudioProcessor {
        let session = try await WhisperSession.fromPretrained(
            model: model,
            progressHandler: progressHandler
        )

        let chunkingStrategy = createStrategy(from: strategy)
        let transcriber = WhisperSessionTranscriber(session: session)

        return LongAudioProcessor(
            session: session,
            strategy: chunkingStrategy,
            transcriber: transcriber,
            mergeConfig: mergeConfig,
            limits: limits,
            telemetry: telemetry
        )
    }

    public static func create(
        model: WhisperModel = .largeTurbo,
        loadingOptions: ModelLoadingOptions,
        strategy: StrategyType = .auto,
        mergeConfig: MergeConfig = .default,
        limits: ProcessingLimits = .default,
        telemetry: ChunkingTelemetry? = nil,
        progressHandler: ((WhisperProgress) -> Void)? = nil
    ) async throws -> LongAudioProcessor {
        let session = try await WhisperSession.fromPretrained(
            model: model,
            options: loadingOptions,
            progressHandler: progressHandler
        )

        let chunkingStrategy = createStrategy(from: strategy)
        let transcriber = WhisperSessionTranscriber(session: session)

        return LongAudioProcessor(
            session: session,
            strategy: chunkingStrategy,
            transcriber: transcriber,
            mergeConfig: mergeConfig,
            limits: limits,
            telemetry: telemetry
        )
    }

    public static func create(
        session: WhisperSession,
        strategy: StrategyType = .auto,
        mergeConfig: MergeConfig = .default,
        limits: ProcessingLimits = .default,
        telemetry: ChunkingTelemetry? = nil
    ) -> LongAudioProcessor {
        let chunkingStrategy = createStrategy(from: strategy)
        let transcriber = WhisperSessionTranscriber(session: session)

        return LongAudioProcessor(
            session: session,
            strategy: chunkingStrategy,
            transcriber: transcriber,
            mergeConfig: mergeConfig,
            limits: limits,
            telemetry: telemetry
        )
    }

    private static func createStrategy(from type: StrategyType) -> ChunkingStrategy {
        switch type {
        case .auto:
            return SlidingWindowChunkingStrategy()
        case .sequential(let config):
            return SequentialChunkingStrategy(config: config)
        case .slidingWindow(let config):
            return SlidingWindowChunkingStrategy(config: config)
        case .vad:
            // VAD strategy not yet fully implemented - fallback to sliding window
            return SlidingWindowChunkingStrategy()
        }
    }

    // MARK: - Streaming Transcription

    public func transcribe(
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
                            options: options,
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
        options: TranscriptionOptions,
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
        var currentChunkText = ""
        var accumulatedWords: [WordTimestamp] = []
        var currentChunkIndex = 0
        var totalChunks = estimateTotalChunks(audioDuration: audioDuration)

        let streamingChunkStream = strategy.processStreaming(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: transcriber,
            limits: limits,
            telemetry: telemetry
        )

        for try await streamingResult in streamingChunkStream {
            try Task.checkCancellation()

            if streamingResult.chunkIndex != currentChunkIndex {
                currentChunkIndex = streamingResult.chunkIndex
                currentChunkText = ""
            }

            let processedText = mergeConfig.normalizeText
                ? streamingResult.text.trimmingCharacters(in: .whitespaces)
                : streamingResult.text

            if streamingResult.isPartial {
                currentChunkText = processedText

                let displayText: String
                if accumulatedText.isEmpty {
                    displayText = currentChunkText
                } else {
                    displayText = accumulatedText + " " + currentChunkText
                }

                let progress = TranscriptionProgress(
                    text: displayText,
                    chunkText: currentChunkText,
                    words: accumulatedWords.isEmpty ? nil : accumulatedWords,
                    isFinal: false,
                    isPartial: true,
                    processedDuration: streamingResult.timestamp.upperBound,
                    audioDuration: audioDuration,
                    chunkIndex: currentChunkIndex,
                    totalChunks: totalChunks
                )

                continuation.yield(progress)
            } else {
                if !processedText.isEmpty {
                    if !accumulatedText.isEmpty {
                        accumulatedText += " "
                    }
                    accumulatedText += processedText
                }
                currentChunkText = ""

                let progress = TranscriptionProgress(
                    text: accumulatedText,
                    chunkText: processedText,
                    words: accumulatedWords.isEmpty ? nil : accumulatedWords,
                    isFinal: false,
                    isPartial: false,
                    processedDuration: streamingResult.timestamp.upperBound,
                    audioDuration: audioDuration,
                    chunkIndex: currentChunkIndex,
                    totalChunks: totalChunks
                )

                continuation.yield(progress)
                totalChunks = max(totalChunks, currentChunkIndex + 2)
            }
        }

        let finalProgress = TranscriptionProgress(
            text: accumulatedText,
            words: accumulatedWords.isEmpty ? nil : accumulatedWords,
            isFinal: true,
            isPartial: false,
            processedDuration: audioDuration,
            audioDuration: audioDuration,
            chunkIndex: currentChunkIndex,
            totalChunks: currentChunkIndex + 1
        )

        continuation.yield(finalProgress)
    }

    // MARK: - Blocking Transcription

    public func transcribe(
        _ audio: MLXArray,
        sampleRate: Int = AudioConstants.sampleRate,
        options: TranscriptionOptions = .default
    ) async throws -> TranscriptionResult {
        var finalText = ""
        var finalWords: [WordTimestamp]?
        var finalDuration: TimeInterval = 0

        let stream: AsyncThrowingStream<TranscriptionProgress, Error> = transcribe(
            audio,
            sampleRate: sampleRate,
            options: options
        )

        for try await progress in stream {
            if progress.isFinal {
                finalText = progress.text
                finalWords = progress.words
                finalDuration = progress.audioDuration
            }
        }

        return TranscriptionResult(
            text: finalText,
            words: finalWords,
            duration: finalDuration,
            language: options.language
        )
    }

    // MARK: - Control

    public func cancel() {
        taskLock.withLock {
            currentTask?.cancel()
            currentTask = nil
        }
        session.cancel()
    }

    // MARK: - Helpers

    private func estimateTotalChunks(audioDuration: TimeInterval) -> Int {
        let chunkDuration: TimeInterval = 30.0
        return max(1, Int(ceil(audioDuration / chunkDuration)))
    }

    private func filterDuplicateWords(
        _ newWords: [WordTimestamp],
        existing: [WordTimestamp]
    ) -> [WordTimestamp] {
        guard let lastExisting = existing.last else {
            return newWords
        }

        return newWords.filter { word in
            word.start >= lastExisting.end ||
            word.confidence >= mergeConfig.minWordConfidence
        }
    }
}

// MARK: - WhisperSession Adapter

/// Adapter to use WhisperSession as ChunkTranscriber
final class WhisperSessionTranscriber: ChunkTranscriber, @unchecked Sendable {
    private let session: WhisperSession

    init(session: WhisperSession) {
        self.session = session
    }

    func transcribe(
        audio: MLXArray,
        sampleRate: Int,
        previousTokens: [Int]?
    ) async throws -> ChunkResult {
        var finalText = ""
        var finalTimestamp: ClosedRange<TimeInterval> = 0...0

        let stream: AsyncThrowingStream<StreamingResult, Error> = session.transcribe(
            audio,
            sampleRate: sampleRate,
            options: .default
        )

        for try await result in stream {
            if result.isFinal {
                finalText = result.text
                finalTimestamp = result.timestamp
            }
        }

        return ChunkResult(
            text: finalText,
            tokens: [],
            timeRange: finalTimestamp,
            confidence: 1.0,
            words: nil
        )
    }

    func transcribeStreaming(
        audio: MLXArray,
        sampleRate: Int,
        previousTokens: [Int]?,
        timeOffset: TimeInterval
    ) -> AsyncThrowingStream<ChunkPartialResult, Error> {
        let whisperStream: AsyncThrowingStream<StreamingResult, Error> = session.transcribe(
            audio,
            sampleRate: sampleRate,
            options: .default
        )

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await result in whisperStream {
                        let adjustedTimestamp = (result.timestamp.lowerBound + timeOffset)...(result.timestamp.upperBound + timeOffset)
                        let partialResult = ChunkPartialResult(
                            text: result.text,
                            timestamp: adjustedTimestamp,
                            isFinal: result.isFinal
                        )
                        continuation.yield(partialResult)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
