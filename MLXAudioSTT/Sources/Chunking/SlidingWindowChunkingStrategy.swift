import Foundation
import MLX

/// Sliding window chunking with fixed overlap
/// Predictable latency, good for real-time processing
public final class SlidingWindowChunkingStrategy: ChunkingStrategy, Sendable {
    public let name = "slidingWindow"
    public let transcriptionMode = TranscriptionMode.independent
    private let config: SlidingWindowConfig

    public struct SlidingWindowConfig: Sendable {
        public var windowDuration: TimeInterval
        public var overlapDuration: TimeInterval
        public var mergeStrategy: MergeStrategy
        public var deduplicationStrategy: (any DeduplicationStrategy)?

        public var hopDuration: TimeInterval { windowDuration - overlapDuration }

        public enum MergeStrategy: Sendable {
            case simple           // Just concatenate, dedupe obvious repeats
            case timestampAlignment  // Align using word timestamps
            case lcs              // Longest common subsequence matching
        }

        public init(
            windowDuration: TimeInterval = 30.0,
            overlapDuration: TimeInterval = 5.0,
            mergeStrategy: MergeStrategy = .timestampAlignment,
            deduplicationStrategy: (any DeduplicationStrategy)? = nil
        ) {
            self.windowDuration = windowDuration
            self.overlapDuration = overlapDuration
            self.mergeStrategy = mergeStrategy
            // Default to composite strategy with overlap end calculated from window config
            self.deduplicationStrategy = deduplicationStrategy ?? CompositeDeduplicationStrategy(
                overlapEnd: windowDuration - overlapDuration
            )
        }

        public static let `default` = SlidingWindowConfig()
    }

    public init(config: SlidingWindowConfig = .default) {
        self.config = config
    }

    public func process(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        options: TranscriptionOptions
    ) -> AsyncThrowingStream<ChunkResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await processAudio(
                        audio: audio,
                        sampleRate: sampleRate,
                        transcriber: transcriber,
                        limits: limits,
                        telemetry: telemetry,
                        options: options,
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
        telemetry: ChunkingTelemetry?,
        options: TranscriptionOptions
    ) -> AsyncThrowingStream<ChunkStreamingResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await processAudioStreaming(
                        audio: audio,
                        sampleRate: sampleRate,
                        transcriber: transcriber,
                        limits: limits,
                        telemetry: telemetry,
                        options: options,
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

    private func processAudio(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        options: TranscriptionOptions,
        continuation: AsyncThrowingStream<ChunkResult, Error>.Continuation
    ) async throws {
        let totalSamples = audio.shape[0]
        let audioDuration = Double(totalSamples) / Double(sampleRate)

        telemetry?.strategyStarted(name, audioDuration: audioDuration)

        let windowSamples = Int(config.windowDuration * Double(sampleRate))
        let hopSamples = Int(config.hopDuration * Double(sampleRate))

        let chunks = calculateChunks(
            totalSamples: totalSamples,
            windowSamples: windowSamples,
            hopSamples: hopSamples,
            sampleRate: sampleRate
        )

        let startTime = Date()
        var previousResult: ChunkResult?

        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()

            if let timeout = limits.totalTimeout {
                let elapsed = Date().timeIntervalSince(startTime)
                if elapsed >= timeout {
                    throw ChunkingError.totalTimeoutExceeded(
                        processedDuration: chunk.timeRange.lowerBound,
                        totalDuration: audioDuration
                    )
                }
            }

            telemetry?.chunkStarted(index: index, timeRange: chunk.timeRange)
            let chunkStartTime = Date()

            let chunkAudio = extractChunk(from: audio, range: chunk.sampleRange)

            let result: ChunkResult
            do {
                result = try await withTimeout(limits.chunkTimeout) {
                    try await transcriber.transcribe(
                        audio: chunkAudio,
                        sampleRate: sampleRate,
                        previousTokens: nil,
                        options: options
                    )
                }
            } catch {
                telemetry?.chunkFailed(index: index, error: error)
                if limits.abortOnFirstFailure {
                    throw ChunkingError.chunkTranscriptionFailed(
                        chunkIndex: index,
                        timeRange: chunk.timeRange,
                        underlying: error
                    )
                }
                continue
            }

            let adjustedResult = ChunkResult(
                text: result.text,
                tokens: result.tokens,
                timeRange: chunk.timeRange,
                confidence: result.confidence,
                words: adjustWordTimestamps(result.words, offset: chunk.timeRange.lowerBound)
            )

            let mergedResult = mergeWithPrevious(
                current: adjustedResult,
                previous: previousResult
            )

            let chunkDuration = Date().timeIntervalSince(chunkStartTime)
            telemetry?.chunkCompleted(index: index, duration: chunkDuration, text: mergedResult.text)

            continuation.yield(mergedResult)
            previousResult = adjustedResult
        }

        let totalDuration = Date().timeIntervalSince(startTime)
        telemetry?.strategyCompleted(totalChunks: chunks.count, totalDuration: totalDuration)
    }

    private func processAudioStreaming(
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        options: TranscriptionOptions,
        continuation: AsyncThrowingStream<ChunkStreamingResult, Error>.Continuation
    ) async throws {
        let totalSamples = audio.shape[0]
        let audioDuration = Double(totalSamples) / Double(sampleRate)

        telemetry?.strategyStarted(name, audioDuration: audioDuration)

        let windowSamples = Int(config.windowDuration * Double(sampleRate))
        let hopSamples = Int(config.hopDuration * Double(sampleRate))

        let chunks = calculateChunks(
            totalSamples: totalSamples,
            windowSamples: windowSamples,
            hopSamples: hopSamples,
            sampleRate: sampleRate
        )

        let startTime = Date()

        for (index, chunk) in chunks.enumerated() {
            try Task.checkCancellation()

            if let timeout = limits.totalTimeout {
                let elapsed = Date().timeIntervalSince(startTime)
                if elapsed >= timeout {
                    throw ChunkingError.totalTimeoutExceeded(
                        processedDuration: chunk.timeRange.lowerBound,
                        totalDuration: audioDuration
                    )
                }
            }

            telemetry?.chunkStarted(index: index, timeRange: chunk.timeRange)
            let chunkStartTime = Date()

            let chunkAudio = extractChunk(from: audio, range: chunk.sampleRange)
            let timeOffset = chunk.timeRange.lowerBound

            let streamingTranscription = transcriber.transcribeStreaming(
                audio: chunkAudio,
                sampleRate: sampleRate,
                previousTokens: nil,
                timeOffset: timeOffset,
                options: options
            )

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
                        chunkIndex: index
                    )
                    continuation.yield(streamingResult)
                }
            } catch {
                telemetry?.chunkFailed(index: index, error: error)
                if limits.abortOnFirstFailure {
                    throw ChunkingError.chunkTranscriptionFailed(
                        chunkIndex: index,
                        timeRange: chunk.timeRange,
                        underlying: error
                    )
                }
                continue
            }

            let chunkDuration = Date().timeIntervalSince(chunkStartTime)
            telemetry?.chunkCompleted(index: index, duration: chunkDuration, text: "")
        }

        let totalDuration = Date().timeIntervalSince(startTime)
        telemetry?.strategyCompleted(totalChunks: chunks.count, totalDuration: totalDuration)
    }

    private struct ChunkInfo {
        let sampleRange: Range<Int>
        let timeRange: ClosedRange<TimeInterval>
    }

    private func calculateChunks(
        totalSamples: Int,
        windowSamples: Int,
        hopSamples: Int,
        sampleRate: Int
    ) -> [ChunkInfo] {
        guard totalSamples > 0 else { return [] }

        var chunks: [ChunkInfo] = []
        var startSample = 0

        while startSample < totalSamples {
            let endSample = min(startSample + windowSamples, totalSamples)
            let startTime = Double(startSample) / Double(sampleRate)
            let endTime = Double(endSample) / Double(sampleRate)

            chunks.append(ChunkInfo(
                sampleRange: startSample..<endSample,
                timeRange: startTime...endTime
            ))

            if endSample >= totalSamples {
                break
            }

            startSample += hopSamples
        }

        return chunks
    }

    private func extractChunk(from audio: MLXArray, range: Range<Int>) -> MLXArray {
        audio[range.lowerBound..<range.upperBound]
    }

    private func adjustWordTimestamps(
        _ words: [WordTimestamp]?,
        offset: TimeInterval
    ) -> [WordTimestamp]? {
        words?.map { word in
            WordTimestamp(
                word: word.word,
                start: word.start + offset,
                end: word.end + offset,
                confidence: word.confidence
            )
        }
    }

    private func mergeWithPrevious(
        current: ChunkResult,
        previous: ChunkResult?
    ) -> ChunkResult {
        guard let previous = previous else { return current }

        switch config.mergeStrategy {
        case .simple:
            return mergeSimple(current: current, previous: previous)
        case .timestampAlignment:
            return mergeWithTimestampAlignment(current: current, previous: previous)
        case .lcs:
            return mergeWithTimestampAlignment(current: current, previous: previous)
        }
    }

    private func mergeSimple(current: ChunkResult, previous: ChunkResult) -> ChunkResult {
        let currentWords = current.text.split(separator: " ").map(String.init)

        guard !currentWords.isEmpty else { return current }

        let previousWords = previous.text.split(separator: " ").map(String.init)
        let lastPreviousWords = Array(previousWords.suffix(5))

        var deduplicatedWords = currentWords
        for word in lastPreviousWords.reversed() {
            if let firstWord = deduplicatedWords.first,
               firstWord.lowercased() == word.lowercased() {
                deduplicatedWords.removeFirst()
            } else {
                break
            }
        }

        let mergedText = deduplicatedWords.joined(separator: " ")

        return ChunkResult(
            text: mergedText,
            tokens: current.tokens,
            timeRange: current.timeRange,
            confidence: current.confidence,
            words: current.words
        )
    }

    private func mergeWithTimestampAlignment(
        current: ChunkResult,
        previous: ChunkResult
    ) -> ChunkResult {
        guard let currentWords = current.words, !currentWords.isEmpty,
              let previousWords = previous.words, !previousWords.isEmpty else {
            return mergeSimple(current: current, previous: previous)
        }

        let overlapStart = current.timeRange.lowerBound
        let overlapEnd = previous.timeRange.upperBound

        guard overlapEnd > overlapStart else {
            return current
        }

        let filteredWords = currentWords.filter { word in
            word.start >= overlapEnd ||
            (word.start >= overlapStart && word.end > overlapEnd)
        }

        let mergedText = filteredWords.map(\.word).joined(separator: " ")

        return ChunkResult(
            text: mergedText,
            tokens: current.tokens,
            timeRange: current.timeRange,
            confidence: current.confidence,
            words: filteredWords.isEmpty ? nil : filteredWords
        )
    }
}
