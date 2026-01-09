import Foundation
import MLX

/// VAD-based chunking with parallel transcription
/// Best for noisy audio, fastest with batching
///
/// Note: VAD produces non-overlapping chunks based on speech boundaries,
/// so deduplication is typically not needed. Use NoOpDeduplicationStrategy
/// or disable deduplication when using this strategy.
///
/// Key feature: Packs multiple short speech segments into ~30s chunks to avoid
/// Whisper's silence detection issues with heavily padded short audio.
public final class VADChunkingStrategy: ChunkingStrategy, Sendable {
    public let name = "vad"
    public let transcriptionMode = TranscriptionMode.independent
    private let config: VADConfig
    private let vadProvider: VADProvider

    /// Represents a packed chunk of concatenated speech segments
    private struct PackedChunk {
        /// Concatenated audio data from multiple speech segments
        let audio: MLXArray
        /// Mapping from position in packed audio to original timeline
        /// Each entry: (packedOffset, originalStart, duration)
        let segmentMappings: [(packedOffset: TimeInterval, originalStart: TimeInterval, duration: TimeInterval)]
        /// Total duration of packed audio
        let totalDuration: TimeInterval
        /// Original time range in source audio (for reporting)
        let originalTimeRange: ClosedRange<TimeInterval>
    }

    public struct VADConfig: Sendable {
        public var targetChunkDuration: TimeInterval
        public var maxChunkDuration: TimeInterval
        public var minSpeechDuration: TimeInterval
        public var speechPadding: TimeInterval
        public var parallelProcessing: Bool
        /// Pack multiple short segments into single chunks (faster-whisper style)
        /// When true, concatenates speech segments to avoid padding short audio
        public var packSegments: Bool

        public init(
            targetChunkDuration: TimeInterval = 30.0,
            maxChunkDuration: TimeInterval = 30.0,
            minSpeechDuration: TimeInterval = 0.5,
            speechPadding: TimeInterval = 0.2,
            parallelProcessing: Bool = true,
            packSegments: Bool = true
        ) {
            self.targetChunkDuration = targetChunkDuration
            self.maxChunkDuration = maxChunkDuration
            self.minSpeechDuration = minSpeechDuration
            self.speechPadding = speechPadding
            self.parallelProcessing = parallelProcessing
            self.packSegments = packSegments
        }

        public static let `default` = VADConfig()
    }

    public init(vadProvider: VADProvider, config: VADConfig = .default) {
        self.vadProvider = vadProvider
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
                    try await processWithVAD(
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
                    try await processWithVADStreaming(
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

    private func processWithVAD(
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

        let speechSegments: [SpeechSegment]
        do {
            speechSegments = try await vadProvider.detectSpeech(in: audio, sampleRate: sampleRate)
        } catch {
            throw ChunkingError.vadFailed(underlying: error)
        }

        guard !speechSegments.isEmpty else {
            telemetry?.vadSegmentsDetected(count: 0, totalSpeechDuration: 0)
            telemetry?.strategyCompleted(totalChunks: 0, totalDuration: 0)
            return
        }

        let processedSegments = prepareSegments(speechSegments, audioDuration: audioDuration)

        let totalSpeechDuration = processedSegments.reduce(0) { $0 + $1.duration }
        telemetry?.vadSegmentsDetected(count: processedSegments.count, totalSpeechDuration: totalSpeechDuration)

        let startTime = Date()

        // Use packed chunk processing when enabled (avoids Whisper silence detection issues)
        if config.packSegments {
            let packedChunks = packSegmentsIntoChunks(processedSegments, audio: audio, sampleRate: sampleRate)

            if config.parallelProcessing && limits.maxConcurrentChunks > 1 {
                try await processPackedInParallel(
                    packedChunks: packedChunks,
                    sampleRate: sampleRate,
                    transcriber: transcriber,
                    limits: limits,
                    telemetry: telemetry,
                    startTime: startTime,
                    audioDuration: audioDuration,
                    options: options,
                    continuation: continuation
                )
            } else {
                try await processPackedSequentially(
                    packedChunks: packedChunks,
                    sampleRate: sampleRate,
                    transcriber: transcriber,
                    limits: limits,
                    telemetry: telemetry,
                    startTime: startTime,
                    audioDuration: audioDuration,
                    options: options,
                    continuation: continuation
                )
            }

            let totalDuration = Date().timeIntervalSince(startTime)
            telemetry?.strategyCompleted(totalChunks: packedChunks.count, totalDuration: totalDuration)
        } else {
            // Legacy per-segment processing
            if config.parallelProcessing && limits.maxConcurrentChunks > 1 {
                try await processInParallel(
                    segments: processedSegments,
                    audio: audio,
                    sampleRate: sampleRate,
                    transcriber: transcriber,
                    limits: limits,
                    telemetry: telemetry,
                    startTime: startTime,
                    audioDuration: audioDuration,
                    options: options,
                    continuation: continuation
                )
            } else {
                try await processSequentially(
                    segments: processedSegments,
                    audio: audio,
                    sampleRate: sampleRate,
                    transcriber: transcriber,
                    limits: limits,
                    telemetry: telemetry,
                    startTime: startTime,
                    audioDuration: audioDuration,
                    options: options,
                    continuation: continuation
                )
            }

            let totalDuration = Date().timeIntervalSince(startTime)
            telemetry?.strategyCompleted(totalChunks: processedSegments.count, totalDuration: totalDuration)
        }
    }

    private func processWithVADStreaming(
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

        let speechSegments: [SpeechSegment]
        do {
            speechSegments = try await vadProvider.detectSpeech(in: audio, sampleRate: sampleRate)
        } catch {
            throw ChunkingError.vadFailed(underlying: error)
        }

        guard !speechSegments.isEmpty else {
            telemetry?.vadSegmentsDetected(count: 0, totalSpeechDuration: 0)
            telemetry?.strategyCompleted(totalChunks: 0, totalDuration: 0)
            return
        }

        let processedSegments = prepareSegments(speechSegments, audioDuration: audioDuration)

        let totalSpeechDuration = processedSegments.reduce(0) { $0 + $1.duration }
        telemetry?.vadSegmentsDetected(count: processedSegments.count, totalSpeechDuration: totalSpeechDuration)

        let startTime = Date()

        for (index, segment) in processedSegments.enumerated() {
            try Task.checkCancellation()

            if let timeout = limits.totalTimeout {
                let elapsed = Date().timeIntervalSince(startTime)
                if elapsed >= timeout {
                    throw ChunkingError.totalTimeoutExceeded(
                        processedDuration: segment.start,
                        totalDuration: audioDuration
                    )
                }
            }

            let timeRange = segment.start...segment.end
            telemetry?.chunkStarted(index: index, timeRange: timeRange)
            let chunkStartTime = Date()

            let startSample = Int(segment.start * Double(sampleRate))
            let endSample = Int(segment.end * Double(sampleRate))
            let clampedEndSample = min(endSample, audio.shape[0])
            let clampedStartSample = min(startSample, clampedEndSample)

            guard clampedEndSample > clampedStartSample else { continue }

            let chunkAudio = audio[clampedStartSample..<clampedEndSample]
            let timeOffset = segment.start

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
            } catch is TimeoutError {
                throw ChunkingError.chunkTimeout(chunkIndex: index, timeRange: timeRange)
            } catch {
                telemetry?.chunkFailed(index: index, error: error)
                if limits.abortOnFirstFailure {
                    throw ChunkingError.chunkTranscriptionFailed(
                        chunkIndex: index,
                        timeRange: timeRange,
                        underlying: error
                    )
                }
                continue
            }

            let chunkDuration = Date().timeIntervalSince(chunkStartTime)
            telemetry?.chunkCompleted(index: index, duration: chunkDuration, text: "")
        }

        let totalDuration = Date().timeIntervalSince(startTime)
        telemetry?.strategyCompleted(totalChunks: processedSegments.count, totalDuration: totalDuration)
    }

    private func prepareSegments(_ segments: [SpeechSegment], audioDuration: TimeInterval) -> [SpeechSegment] {
        let merged = mergeSmallSegments(segments)
        let split = splitLargeSegments(merged)
        return clampSegments(split, audioDuration: audioDuration)
    }

    private func mergeSmallSegments(_ segments: [SpeechSegment]) -> [SpeechSegment] {
        guard !segments.isEmpty else { return [] }

        var result: [SpeechSegment] = []
        var current = segments[0]

        for next in segments.dropFirst() {
            let gap = next.start - current.end
            let combinedDuration = next.end - current.start

            if combinedDuration <= config.targetChunkDuration && gap < config.speechPadding * 2 {
                current = SpeechSegment(
                    start: current.start,
                    end: next.end,
                    confidence: max(current.confidence, next.confidence)
                )
            } else {
                if current.duration >= config.minSpeechDuration {
                    result.append(current)
                }
                current = next
            }
        }

        if current.duration >= config.minSpeechDuration {
            result.append(current)
        }

        return result
    }

    private func splitLargeSegments(_ segments: [SpeechSegment]) -> [SpeechSegment] {
        splitLongSegments(segments, maxDuration: config.maxChunkDuration)
    }

    private func clampSegments(_ segments: [SpeechSegment], audioDuration: TimeInterval) -> [SpeechSegment] {
        segments.map { segment in
            SpeechSegment(
                start: max(0, segment.start - config.speechPadding),
                end: min(audioDuration, segment.end + config.speechPadding),
                confidence: segment.confidence
            )
        }
    }

    // MARK: - Segment Packing (faster-whisper style)

    /// Pack multiple speech segments into chunks by concatenating audio data
    /// This avoids Whisper's silence detection issues with heavily padded short audio
    private func packSegmentsIntoChunks(
        _ segments: [SpeechSegment],
        audio: MLXArray,
        sampleRate: Int
    ) -> [PackedChunk] {
        guard !segments.isEmpty else { return [] }

        var packedChunks: [PackedChunk] = []
        var currentAudioParts: [MLXArray] = []
        var currentMappings: [(packedOffset: TimeInterval, originalStart: TimeInterval, duration: TimeInterval)] = []
        var currentPackedDuration: TimeInterval = 0
        var firstSegmentStart: TimeInterval = segments[0].start
        var lastSegmentEnd: TimeInterval = segments[0].end

        for segment in segments {
            let segmentDuration = segment.duration
            let startSample = Int(segment.start * Double(sampleRate))
            let endSample = min(Int(segment.end * Double(sampleRate)), audio.shape[0])

            guard endSample > startSample else { continue }

            // Check if adding this segment would exceed target duration
            if currentPackedDuration + segmentDuration > config.targetChunkDuration && !currentAudioParts.isEmpty {
                // Finalize current chunk
                let packedAudio = concatenated(currentAudioParts, axis: 0)
                packedChunks.append(PackedChunk(
                    audio: packedAudio,
                    segmentMappings: currentMappings,
                    totalDuration: currentPackedDuration,
                    originalTimeRange: firstSegmentStart...lastSegmentEnd
                ))

                // Start new chunk
                currentAudioParts = []
                currentMappings = []
                currentPackedDuration = 0
                firstSegmentStart = segment.start
            }

            // Add segment to current chunk
            let segmentAudio = audio[startSample..<endSample]
            currentAudioParts.append(segmentAudio)
            currentMappings.append((
                packedOffset: currentPackedDuration,
                originalStart: segment.start,
                duration: segmentDuration
            ))
            currentPackedDuration += segmentDuration
            lastSegmentEnd = segment.end
        }

        // Finalize last chunk
        if !currentAudioParts.isEmpty {
            let packedAudio = concatenated(currentAudioParts, axis: 0)
            packedChunks.append(PackedChunk(
                audio: packedAudio,
                segmentMappings: currentMappings,
                totalDuration: currentPackedDuration,
                originalTimeRange: firstSegmentStart...lastSegmentEnd
            ))
        }

        return packedChunks
    }

    /// Remap timestamp from packed audio position to original audio position
    private func remapTimestamp(
        _ packedTime: TimeInterval,
        mappings: [(packedOffset: TimeInterval, originalStart: TimeInterval, duration: TimeInterval)]
    ) -> TimeInterval {
        // Find which segment this time falls into
        for mapping in mappings.reversed() {
            if packedTime >= mapping.packedOffset {
                let offsetInSegment = packedTime - mapping.packedOffset
                return mapping.originalStart + min(offsetInSegment, mapping.duration)
            }
        }
        // Fallback to first segment
        return mappings.first?.originalStart ?? 0
    }

    // MARK: - Packed Chunk Processing

    private func processPackedSequentially(
        packedChunks: [PackedChunk],
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        startTime: Date,
        audioDuration: TimeInterval,
        options: TranscriptionOptions,
        continuation: AsyncThrowingStream<ChunkResult, Error>.Continuation
    ) async throws {
        for (index, packedChunk) in packedChunks.enumerated() {
            try Task.checkCancellation()

            if let timeout = limits.totalTimeout {
                let elapsed = Date().timeIntervalSince(startTime)
                if elapsed >= timeout {
                    throw ChunkingError.totalTimeoutExceeded(
                        processedDuration: packedChunk.originalTimeRange.lowerBound,
                        totalDuration: audioDuration
                    )
                }
            }

            telemetry?.chunkStarted(index: index, timeRange: packedChunk.originalTimeRange)
            let chunkStartTime = Date()

            let result = try await transcribePackedChunk(
                packedChunk: packedChunk,
                index: index,
                sampleRate: sampleRate,
                transcriber: transcriber,
                limits: limits,
                telemetry: telemetry,
                options: options
            )

            let chunkDuration = Date().timeIntervalSince(chunkStartTime)
            telemetry?.chunkCompleted(index: index, duration: chunkDuration, text: result.text)
            continuation.yield(result)
        }
    }

    private func processPackedInParallel(
        packedChunks: [PackedChunk],
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        startTime: Date,
        audioDuration: TimeInterval,
        options: TranscriptionOptions,
        continuation: AsyncThrowingStream<ChunkResult, Error>.Continuation
    ) async throws {
        let indexedResults = try await withThrowingTaskGroup(
            of: (index: Int, result: ChunkResult).self
        ) { group in
            var activeCount = 0
            var chunkIndex = 0
            var collected: [(index: Int, result: ChunkResult)] = []

            while chunkIndex < packedChunks.count || activeCount > 0 {
                if let timeout = limits.totalTimeout {
                    let elapsed = Date().timeIntervalSince(startTime)
                    if elapsed >= timeout {
                        group.cancelAll()
                        throw ChunkingError.totalTimeoutExceeded(
                            processedDuration: 0,
                            totalDuration: audioDuration
                        )
                    }
                }

                while activeCount < limits.maxConcurrentChunks && chunkIndex < packedChunks.count {
                    let packedChunk = packedChunks[chunkIndex]
                    let currentIndex = chunkIndex

                    telemetry?.chunkStarted(index: currentIndex, timeRange: packedChunk.originalTimeRange)

                    group.addTask {
                        let chunkStartTime = Date()
                        let result = try await self.transcribePackedChunk(
                            packedChunk: packedChunk,
                            index: currentIndex,
                            sampleRate: sampleRate,
                            transcriber: transcriber,
                            limits: limits,
                            telemetry: nil,
                            options: options
                        )
                        let chunkDuration = Date().timeIntervalSince(chunkStartTime)
                        telemetry?.chunkCompleted(index: currentIndex, duration: chunkDuration, text: result.text)
                        return (index: currentIndex, result: result)
                    }

                    activeCount += 1
                    chunkIndex += 1
                }

                if activeCount > 0 {
                    if let indexedResult = try await group.next() {
                        collected.append(indexedResult)
                        activeCount -= 1
                    }
                }
            }

            return collected
        }

        let sortedResults = indexedResults.sorted { $0.index < $1.index }
        for indexed in sortedResults {
            continuation.yield(indexed.result)
        }
    }

    private func transcribePackedChunk(
        packedChunk: PackedChunk,
        index: Int,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        options: TranscriptionOptions
    ) async throws -> ChunkResult {
        let timeRange = packedChunk.originalTimeRange

        guard packedChunk.audio.shape[0] > 0 else {
            return ChunkResult(
                text: "",
                tokens: [],
                timeRange: timeRange,
                confidence: 0,
                words: nil
            )
        }

        let result: ChunkResult
        do {
            result = try await withTimeout(limits.chunkTimeout) {
                // Transcribe packed audio with timeOffset = 0 since we'll remap timestamps
                try await transcriber.transcribe(
                    audio: packedChunk.audio,
                    sampleRate: sampleRate,
                    previousTokens: nil,
                    options: options
                )
            }
        } catch is TimeoutError {
            throw ChunkingError.chunkTimeout(chunkIndex: index, timeRange: timeRange)
        } catch {
            telemetry?.chunkFailed(index: index, error: error)
            if limits.abortOnFirstFailure {
                throw ChunkingError.chunkTranscriptionFailed(
                    chunkIndex: index,
                    timeRange: timeRange,
                    underlying: error
                )
            }
            return ChunkResult(
                text: "",
                tokens: [],
                timeRange: timeRange,
                confidence: 0,
                words: nil
            )
        }

        // Remap word timestamps from packed audio positions to original timeline
        let remappedWords = result.words?.map { word in
            WordTimestamp(
                word: word.word,
                start: remapTimestamp(word.start, mappings: packedChunk.segmentMappings),
                end: remapTimestamp(word.end, mappings: packedChunk.segmentMappings),
                confidence: word.confidence
            )
        }

        return ChunkResult(
            text: result.text,
            tokens: result.tokens,
            timeRange: timeRange,
            confidence: result.confidence,
            words: remappedWords
        )
    }

    // MARK: - Legacy Segment Processing

    private func processSequentially(
        segments: [SpeechSegment],
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        startTime: Date,
        audioDuration: TimeInterval,
        options: TranscriptionOptions,
        continuation: AsyncThrowingStream<ChunkResult, Error>.Continuation
    ) async throws {
        for (index, segment) in segments.enumerated() {
            try Task.checkCancellation()

            if let timeout = limits.totalTimeout {
                let elapsed = Date().timeIntervalSince(startTime)
                if elapsed >= timeout {
                    throw ChunkingError.totalTimeoutExceeded(
                        processedDuration: segment.start,
                        totalDuration: audioDuration
                    )
                }
            }

            let timeRange = segment.start...segment.end
            telemetry?.chunkStarted(index: index, timeRange: timeRange)
            let chunkStartTime = Date()

            let result = try await transcribeSegment(
                segment: segment,
                index: index,
                audio: audio,
                sampleRate: sampleRate,
                transcriber: transcriber,
                limits: limits,
                telemetry: telemetry,
                options: options
            )

            let chunkDuration = Date().timeIntervalSince(chunkStartTime)
            telemetry?.chunkCompleted(index: index, duration: chunkDuration, text: result.text)
            continuation.yield(result)
        }
    }

    private func processInParallel(
        segments: [SpeechSegment],
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        startTime: Date,
        audioDuration: TimeInterval,
        options: TranscriptionOptions,
        continuation: AsyncThrowingStream<ChunkResult, Error>.Continuation
    ) async throws {
        let indexedResults = try await withThrowingTaskGroup(
            of: (index: Int, result: ChunkResult).self
        ) { group in
            var activeCount = 0
            var segmentIndex = 0
            var collected: [(index: Int, result: ChunkResult)] = []

            while segmentIndex < segments.count || activeCount > 0 {
                if let timeout = limits.totalTimeout {
                    let elapsed = Date().timeIntervalSince(startTime)
                    if elapsed >= timeout {
                        group.cancelAll()
                        throw ChunkingError.totalTimeoutExceeded(
                            processedDuration: 0,
                            totalDuration: audioDuration
                        )
                    }
                }

                while activeCount < limits.maxConcurrentChunks && segmentIndex < segments.count {
                    let segment = segments[segmentIndex]
                    let currentIndex = segmentIndex

                    let timeRange = segment.start...segment.end
                    telemetry?.chunkStarted(index: currentIndex, timeRange: timeRange)

                    group.addTask {
                        let chunkStartTime = Date()
                        let result = try await self.transcribeSegment(
                            segment: segment,
                            index: currentIndex,
                            audio: audio,
                            sampleRate: sampleRate,
                            transcriber: transcriber,
                            limits: limits,
                            telemetry: nil,
                            options: options
                        )
                        let chunkDuration = Date().timeIntervalSince(chunkStartTime)
                        telemetry?.chunkCompleted(index: currentIndex, duration: chunkDuration, text: result.text)
                        return (index: currentIndex, result: result)
                    }

                    activeCount += 1
                    segmentIndex += 1
                }

                if activeCount > 0 {
                    if let indexedResult = try await group.next() {
                        collected.append(indexedResult)
                        activeCount -= 1
                    }
                }
            }

            return collected
        }

        let sortedResults = indexedResults.sorted { $0.index < $1.index }
        for indexed in sortedResults {
            continuation.yield(indexed.result)
        }
    }

    private func transcribeSegment(
        segment: SpeechSegment,
        index: Int,
        audio: MLXArray,
        sampleRate: Int,
        transcriber: ChunkTranscriber,
        limits: ProcessingLimits,
        telemetry: ChunkingTelemetry?,
        options: TranscriptionOptions
    ) async throws -> ChunkResult {
        let startSample = Int(segment.start * Double(sampleRate))
        let endSample = Int(segment.end * Double(sampleRate))
        let clampedEndSample = min(endSample, audio.shape[0])
        let clampedStartSample = min(startSample, clampedEndSample)

        guard clampedEndSample > clampedStartSample else {
            return ChunkResult(
                text: "",
                tokens: [],
                timeRange: segment.start...segment.end,
                confidence: 0,
                words: nil
            )
        }

        let chunkAudio = audio[clampedStartSample..<clampedEndSample]
        let timeRange = segment.start...segment.end

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
        } catch is TimeoutError {
            throw ChunkingError.chunkTimeout(chunkIndex: index, timeRange: timeRange)
        } catch {
            telemetry?.chunkFailed(index: index, error: error)
            if limits.abortOnFirstFailure {
                throw ChunkingError.chunkTranscriptionFailed(
                    chunkIndex: index,
                    timeRange: timeRange,
                    underlying: error
                )
            }
            return ChunkResult(
                text: "",
                tokens: [],
                timeRange: timeRange,
                confidence: 0,
                words: nil
            )
        }

        return ChunkResult(
            text: result.text,
            tokens: result.tokens,
            timeRange: timeRange,
            confidence: result.confidence,
            words: adjustWordTimestamps(result.words, offset: segment.start)
        )
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
}
