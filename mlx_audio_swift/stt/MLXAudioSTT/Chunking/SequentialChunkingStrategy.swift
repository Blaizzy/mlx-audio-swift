import Foundation
import MLX

/// Sequential decoding with timestamp-based seeking
/// Best accuracy, no parallelization - follows OpenAI-style seek-based decoding
public final class SequentialChunkingStrategy: ChunkingStrategy, @unchecked Sendable {
    public let name = "sequential"
    public let transcriptionMode = TranscriptionMode.sequential

    private let config: SequentialConfig

    public struct SequentialConfig: Sendable {
        public var conditionOnPreviousText: Bool
        public var maxPreviousTokens: Int
        public var contextResetTemperature: Float
        public var compressionRatioThreshold: Float
        public var logprobThreshold: Float
        public var noSpeechThreshold: Float
        public var initialPrompt: String?
        public var maxChunkDuration: TimeInterval

        public init(
            conditionOnPreviousText: Bool = true,
            maxPreviousTokens: Int = 224,
            contextResetTemperature: Float = 0.5,
            compressionRatioThreshold: Float = 2.4,
            logprobThreshold: Float = -1.0,
            noSpeechThreshold: Float = 0.6,
            initialPrompt: String? = nil,
            maxChunkDuration: TimeInterval = 30.0
        ) {
            self.conditionOnPreviousText = conditionOnPreviousText
            self.maxPreviousTokens = maxPreviousTokens
            self.contextResetTemperature = contextResetTemperature
            self.compressionRatioThreshold = compressionRatioThreshold
            self.logprobThreshold = logprobThreshold
            self.noSpeechThreshold = noSpeechThreshold
            self.initialPrompt = initialPrompt
            self.maxChunkDuration = maxChunkDuration
        }

        public static let `default` = SequentialConfig()
    }

    public init(config: SequentialConfig = .default) {
        self.config = config
    }

    public func process(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) -> AsyncThrowingStream<ChunkResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await processSequentially(
                        audio: audio,
                        sampleRate: sampleRate,
                        transcriber: transcriber,
                        limits: limits,
                        telemetry: telemetry,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    telemetry?.error(error)
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func processStreaming(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?
    ) -> AsyncThrowingStream<ChunkStreamingResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await processSequentiallyStreaming(
                        audio: audio,
                        sampleRate: sampleRate,
                        transcriber: transcriber,
                        limits: limits,
                        telemetry: telemetry,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    telemetry?.error(error)
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func processSequentially(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        continuation: AsyncThrowingStream<ChunkResult, Error>.Continuation
    ) async throws {
        let totalSamples = audio.shape[0]
        let audioDuration = TimeInterval(totalSamples) / TimeInterval(sampleRate)
        let samplesPerChunk = Int(config.maxChunkDuration * Double(sampleRate))

        telemetry?.strategyStarted(name, audioDuration: audioDuration)

        var currentPosition: TimeInterval = 0
        var previousTokens: [Int]? = nil
        var chunkIndex = 0
        let startTime = Date()

        while currentPosition < audioDuration {
            try Task.checkCancellation()

            if let totalTimeout = limits.totalTimeout {
                let elapsed = Date().timeIntervalSince(startTime)
                if elapsed >= totalTimeout {
                    throw ChunkingError.totalTimeoutExceeded(
                        processedDuration: currentPosition,
                        totalDuration: audioDuration
                    )
                }
            }

            let startSample = Int(currentPosition * Double(sampleRate))
            let endSample = min(startSample + samplesPerChunk, totalSamples)
            let chunkEndTime = min(currentPosition + config.maxChunkDuration, audioDuration)
            let timeRange = currentPosition...chunkEndTime

            telemetry?.chunkStarted(index: chunkIndex, timeRange: timeRange)
            let chunkStartTime = Date()

            let chunkAudio = audio[startSample..<endSample]

            let tokensForContext: [Int]?
            if config.conditionOnPreviousText, let tokens = previousTokens {
                tokensForContext = Array(tokens.suffix(config.maxPreviousTokens))
            } else {
                tokensForContext = nil
            }

            let result: ChunkResult
            do {
                result = try await withTimeout(limits.chunkTimeout) {
                    try await transcriber.transcribe(
                        audio: chunkAudio,
                        sampleRate: sampleRate,
                        previousTokens: tokensForContext
                    )
                }
            } catch is TimeoutError {
                throw ChunkingError.chunkTimeout(chunkIndex: chunkIndex, timeRange: timeRange)
            } catch {
                throw ChunkingError.chunkTranscriptionFailed(
                    chunkIndex: chunkIndex,
                    timeRange: timeRange,
                    underlying: error
                )
            }

            let processingDuration = Date().timeIntervalSince(chunkStartTime)

            if result.confidence < config.noSpeechThreshold {
                telemetry?.chunkCompleted(index: chunkIndex, duration: processingDuration, text: "[skipped - low confidence]")
                currentPosition += config.maxChunkDuration
                chunkIndex += 1
                continue
            }

            let adjustedResult = adjustTimestamps(result, offset: currentPosition)

            previousTokens = result.tokens

            let seekTime = calculateSeekTime(from: result, chunkStartTime: currentPosition)
            currentPosition = seekTime

            telemetry?.chunkCompleted(index: chunkIndex, duration: processingDuration, text: adjustedResult.text)
            continuation.yield(adjustedResult)

            chunkIndex += 1
        }

        let totalDuration = Date().timeIntervalSince(startTime)
        telemetry?.strategyCompleted(totalChunks: chunkIndex, totalDuration: totalDuration)
    }

    private func processSequentiallyStreaming(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        continuation: AsyncThrowingStream<ChunkStreamingResult, Error>.Continuation
    ) async throws {
        let totalSamples = audio.shape[0]
        let audioDuration = TimeInterval(totalSamples) / TimeInterval(sampleRate)
        let samplesPerChunk = Int(config.maxChunkDuration * Double(sampleRate))

        telemetry?.strategyStarted(name, audioDuration: audioDuration)

        var currentPosition: TimeInterval = 0
        var chunkIndex = 0
        let startTime = Date()

        while currentPosition < audioDuration {
            try Task.checkCancellation()

            if let totalTimeout = limits.totalTimeout {
                let elapsed = Date().timeIntervalSince(startTime)
                if elapsed >= totalTimeout {
                    throw ChunkingError.totalTimeoutExceeded(
                        processedDuration: currentPosition,
                        totalDuration: audioDuration
                    )
                }
            }

            let startSample = Int(currentPosition * Double(sampleRate))
            let endSample = min(startSample + samplesPerChunk, totalSamples)
            let chunkEndTime = min(currentPosition + config.maxChunkDuration, audioDuration)
            let timeRange = currentPosition...chunkEndTime

            telemetry?.chunkStarted(index: chunkIndex, timeRange: timeRange)
            let chunkStartTime = Date()

            let chunkAudio = audio[startSample..<endSample]
            let timeOffset = currentPosition

            let streamingTranscription = transcriber.transcribeStreaming(
                audio: chunkAudio,
                sampleRate: sampleRate,
                previousTokens: nil,
                timeOffset: timeOffset
            )

            var lastTimestamp: ClosedRange<TimeInterval> = timeRange

            do {
                for try await partialResult in streamingTranscription {
                    try Task.checkCancellation()

                    // Check chunk timeout
                    let chunkElapsed = Date().timeIntervalSince(chunkStartTime)
                    if chunkElapsed >= limits.chunkTimeout {
                        throw TimeoutError(timeout: limits.chunkTimeout)
                    }

                    let streamingResult = ChunkStreamingResult(
                        text: partialResult.text,
                        timestamp: partialResult.timestamp,
                        isPartial: !partialResult.isFinal,
                        isChunkFinal: partialResult.isFinal,
                        chunkIndex: chunkIndex
                    )
                    continuation.yield(streamingResult)

                    if partialResult.isFinal {
                        lastTimestamp = partialResult.timestamp
                    }
                }
            } catch is TimeoutError {
                throw ChunkingError.chunkTimeout(chunkIndex: chunkIndex, timeRange: timeRange)
            } catch {
                throw ChunkingError.chunkTranscriptionFailed(
                    chunkIndex: chunkIndex,
                    timeRange: timeRange,
                    underlying: error
                )
            }

            let processingDuration = Date().timeIntervalSince(chunkStartTime)
            telemetry?.chunkCompleted(index: chunkIndex, duration: processingDuration, text: "")

            currentPosition = lastTimestamp.upperBound
            chunkIndex += 1
        }

        let totalDuration = Date().timeIntervalSince(startTime)
        telemetry?.strategyCompleted(totalChunks: chunkIndex, totalDuration: totalDuration)
    }

    private func adjustTimestamps(_ result: ChunkResult, offset: TimeInterval) -> ChunkResult {
        let adjustedTimeRange = (result.timeRange.lowerBound + offset)...(result.timeRange.upperBound + offset)

        let adjustedWords: [WordTimestamp]?
        if let words = result.words {
            adjustedWords = words.map { word in
                WordTimestamp(
                    word: word.word,
                    start: word.start + offset,
                    end: word.end + offset,
                    confidence: word.confidence
                )
            }
        } else {
            adjustedWords = nil
        }

        return ChunkResult(
            text: result.text,
            tokens: result.tokens,
            timeRange: adjustedTimeRange,
            confidence: result.confidence,
            words: adjustedWords
        )
    }

    private func calculateSeekTime(from result: ChunkResult, chunkStartTime: TimeInterval) -> TimeInterval {
        if let words = result.words, let lastWord = words.last {
            return chunkStartTime + lastWord.end
        }
        return chunkStartTime + result.timeRange.upperBound
    }
}
