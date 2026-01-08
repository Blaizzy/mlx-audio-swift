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
        public var deduplicationStrategy: (any DeduplicationStrategy)?
        public var minWordConfidence: Float
        public var normalizeText: Bool

        public init(
            deduplicateOverlap: Bool = true,
            deduplicationStrategy: (any DeduplicationStrategy)? = nil,
            minWordConfidence: Float = 0.3,
            normalizeText: Bool = true
        ) {
            self.deduplicateOverlap = deduplicateOverlap
            self.deduplicationStrategy = deduplicationStrategy
            self.minWordConfidence = minWordConfidence
            self.normalizeText = normalizeText
        }

        public static let `default` = MergeConfig(
            deduplicateOverlap: true,
            deduplicationStrategy: CompositeDeduplicationStrategy()
        )

        /// Create config with smart deduplication using composite strategy
        public static func withSmartDeduplication(overlapEnd: TimeInterval? = nil) -> MergeConfig {
            MergeConfig(
                deduplicateOverlap: true,
                deduplicationStrategy: CompositeDeduplicationStrategy(overlapEnd: overlapEnd)
            )
        }
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
        let effectiveConfig = effectiveMergeConfig(mergeConfig, for: strategy)

        return LongAudioProcessor(
            session: session,
            strategy: chunkingStrategy,
            transcriber: transcriber,
            mergeConfig: effectiveConfig,
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
        let effectiveConfig = effectiveMergeConfig(mergeConfig, for: strategy)

        return LongAudioProcessor(
            session: session,
            strategy: chunkingStrategy,
            transcriber: transcriber,
            mergeConfig: effectiveConfig,
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
        let effectiveConfig = effectiveMergeConfig(mergeConfig, for: strategy)

        return LongAudioProcessor(
            session: session,
            strategy: chunkingStrategy,
            transcriber: transcriber,
            mergeConfig: effectiveConfig,
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
        case .vad(let vadConfig, let providerType):
            let provider: VADProvider = switch providerType {
            case .energy:
                EnergyVADProvider(
                    config: EnergyVADProvider.EnergyVADConfig(
                        speechThreshold: vadConfig.speechThreshold > 0.1 ? 0.02 : 0.01
                    ),
                    segmentConfig: VADSegmentConfig(
                        minSpeechDuration: vadConfig.minSpeechDuration,
                        maxSegmentDuration: vadConfig.maxSegmentDuration
                    )
                )
            case .sileroMLX:
                SileroVADProvider(
                    config: SileroVADProvider.SileroVADConfig(
                        threshold: vadConfig.speechThreshold,
                        minSpeechDurationMs: Int(vadConfig.minSpeechDuration * 1000)
                    ),
                    segmentConfig: VADSegmentConfig(
                        minSpeechDuration: vadConfig.minSpeechDuration,
                        maxSegmentDuration: vadConfig.maxSegmentDuration
                    )
                )
            }
            return VADChunkingStrategy(
                vadProvider: provider,
                config: VADChunkingStrategy.VADConfig(
                    maxChunkDuration: vadConfig.maxSegmentDuration,
                    minSpeechDuration: vadConfig.minSpeechDuration
                )
            )
        }
    }

    private static func effectiveMergeConfig(
        _ mergeConfig: MergeConfig,
        for strategyType: StrategyType
    ) -> MergeConfig {
        // If user explicitly set a deduplication strategy, use it
        // Note: MergeConfig.default now uses CompositeDeduplicationStrategy()
        // We enhance it with the proper overlapEnd from sliding window config
        var config = mergeConfig

        switch strategyType {
        case .auto:
            // Auto uses sliding window with default config (hopDuration = 25.0)
            let swConfig = SlidingWindowChunkingStrategy.SlidingWindowConfig.default
            config.deduplicationStrategy = swConfig.deduplicationStrategy
        case .slidingWindow(let swConfig):
            // Use the deduplication strategy configured in sliding window
            config.deduplicationStrategy = swConfig.deduplicationStrategy
        case .sequential:
            // Sequential doesn't overlap, use Levenshtein for safety
            config.deduplicationStrategy = LevenshteinDeduplicationStrategy()
        case .vad:
            // VAD produces non-overlapping chunks, minimal deduplication needed
            config.deduplicationStrategy = NoOpDeduplicationStrategy()
        }

        return config
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
        var previousChunkEndWords: [String] = []

        let streamingChunkStream = strategy.processStreaming(
            audio: audio,
            sampleRate: sampleRate,
            transcriber: transcriber,
            limits: limits,
            telemetry: telemetry,
            options: options
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
                var textToAccumulate = processedText

                if mergeConfig.deduplicateOverlap {
                    if let strategy = mergeConfig.deduplicationStrategy {
                        let result = strategy.deduplicate(
                            currentText: processedText,
                            previousEndWords: previousChunkEndWords,
                            currentWords: nil
                        )
                        // DEBUG
                        print("[DEBUG] Chunk \(currentChunkIndex): dedup '\(result.method)' removed \(result.wordsRemoved) words")
                        print("[DEBUG] previousEndWords: \(previousChunkEndWords)")
                        print("[DEBUG] processedText: \(processedText.prefix(100))...")
                        print("[DEBUG] textToAccumulate: \(result.text.prefix(100))...")
                        textToAccumulate = result.text
                    } else if !previousChunkEndWords.isEmpty {
                        textToAccumulate = deduplicateOverlapText(
                            processedText,
                            previousEndWords: previousChunkEndWords
                        )
                    }
                }

                if !textToAccumulate.isEmpty {
                    if !accumulatedText.isEmpty {
                        accumulatedText += " "
                    }
                    accumulatedText += textToAccumulate
                }

                let words = processedText.split(separator: " ").map(String.init)
                previousChunkEndWords = Array(words.suffix(10))
                currentChunkText = ""

                let progress = TranscriptionProgress(
                    text: accumulatedText,
                    chunkText: textToAccumulate,
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

    private func deduplicateOverlapText(_ text: String, previousEndWords: [String]) -> String {
        let words = text.split(separator: " ").map(String.init)
        guard !words.isEmpty else { return text }

        var matchLength = 0
        for len in 1...min(previousEndWords.count, words.count) {
            let prevSuffix = previousEndWords.suffix(len)
            let currPrefix = words.prefix(len)

            if prevSuffix.elementsEqual(currPrefix, by: { $0.lowercased() == $1.lowercased() }) {
                matchLength = len
            }
        }

        if matchLength > 0 {
            return words.dropFirst(matchLength).joined(separator: " ")
        }
        return text
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
        previousTokens: [Int]?,
        options: TranscriptionOptions
    ) async throws -> ChunkResult {
        var finalText = ""
        var finalTimestamp: ClosedRange<TimeInterval> = 0...0

        let stream: AsyncThrowingStream<StreamingResult, Error> = session.transcribe(
            audio,
            sampleRate: sampleRate,
            options: options
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
        timeOffset: TimeInterval,
        options: TranscriptionOptions
    ) -> AsyncThrowingStream<ChunkPartialResult, Error> {
        let whisperStream: AsyncThrowingStream<StreamingResult, Error> = session.transcribe(
            audio,
            sampleRate: sampleRate,
            options: options
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
