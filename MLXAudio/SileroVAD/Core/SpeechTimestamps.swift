import Foundation
import MLX

/// State machine for detecting speech segment boundaries in batch processing.
private final class BatchSpeechDetector {
    private let config: VADConfig
    private let minSpeechChunks: Int
    private let minSilenceChunks: Int

    private var isSpeaking = false
    private var speechStartTime: TimeInterval?
    private var speechChunkCount = 0
    private var silenceChunkCount = 0

    private(set) var segments: [SpeechSegment] = []

    init(config: VADConfig) {
        self.config = config
        self.minSpeechChunks = max(1, Int(ceil(Double(config.minSpeechDurationMs) / 1000.0 / VADAudioFormat.chunkDuration)))
        self.minSilenceChunks = max(1, Int(ceil(Double(config.minSilenceDurationMs) / 1000.0 / VADAudioFormat.chunkDuration)))
    }

    func process(_ result: VADResult) {
        if result.isSpeech {
            silenceChunkCount = 0
            speechChunkCount += 1

            if !isSpeaking && speechChunkCount >= minSpeechChunks {
                isSpeaking = true
                speechStartTime = result.timestamp - Double(speechChunkCount - 1) * VADAudioFormat.chunkDuration
            }
        } else {
            if isSpeaking {
                silenceChunkCount += 1

                if silenceChunkCount >= minSilenceChunks {
                    let endTime = result.timestamp - Double(silenceChunkCount) * VADAudioFormat.chunkDuration + VADAudioFormat.chunkDuration
                    finalizeSegment(endTime: endTime)
                }
            } else {
                speechChunkCount = 0
            }
        }
    }

    func finalize(audioLength: TimeInterval) {
        if isSpeaking, let startTime = speechStartTime {
            segments.append(SpeechSegment(start: startTime, end: audioLength))
            isSpeaking = false
            speechStartTime = nil
        }
    }

    private func finalizeSegment(endTime: TimeInterval) {
        guard let startTime = speechStartTime else { return }
        segments.append(SpeechSegment(start: startTime, end: endTime))
        isSpeaking = false
        speechStartTime = nil
        speechChunkCount = 0
        silenceChunkCount = 0
    }
}

/// Validates audio format for VAD processing.
private func validateAudio(_ audio: MLXArray) throws {
    guard audio.dtype == .float32 else {
        throw VADError.invalidDtype(expected: .float32, got: audio.dtype)
    }

    guard audio.ndim == 1 || (audio.ndim == 2 && audio.shape[0] == 1) else {
        throw VADError.invalidAudioShape(expected: "[samples] or [1, samples]", got: audio.shape)
    }

    eval(audio)
    let minVal = audio.min().item(Float.self)
    let maxVal = audio.max().item(Float.self)

    if minVal < VADAudioFormat.valueRange.lowerBound || maxVal > VADAudioFormat.valueRange.upperBound {
        throw VADError.audioOutOfRange(min: minVal, max: maxVal)
    }
}

/// Applies padding to segments and merges overlapping ones.
private func applyPaddingAndMerge(
    segments: [SpeechSegment],
    speechPadMs: Int,
    audioLength: TimeInterval
) -> [SpeechSegment] {
    guard !segments.isEmpty else { return [] }

    let paddingSeconds = Double(speechPadMs) / 1000.0

    var paddedSegments = segments.map { segment in
        let paddedStart = max(0, segment.start - paddingSeconds)
        let paddedEnd = min(audioLength, segment.end + paddingSeconds)
        return SpeechSegment(start: paddedStart, end: paddedEnd)
    }

    paddedSegments.sort()

    var merged: [SpeechSegment] = []
    var current = paddedSegments[0]

    for i in 1..<paddedSegments.count {
        let next = paddedSegments[i]
        if next.start <= current.end {
            current = SpeechSegment(start: current.start, end: max(current.end, next.end))
        } else {
            merged.append(current)
            current = next
        }
    }
    merged.append(current)

    return merged
}

/// Batch processing to extract speech segments from complete audio.
/// Matches Silero Python API: getSpeechTimestamps()
///
/// - Parameters:
///   - audio: Audio samples, shape [samples] or [1, samples], 16kHz Float32 in [-1, 1]
///   - model: Loaded SileroVADModel instance
///   - config: VAD configuration for thresholds and timing
/// - Returns: Array of speech segments with start/end times
/// - Throws: VADError for invalid input format
public func getSpeechTimestamps(
    audio: MLXArray,
    model: SileroVADModel,
    config: VADConfig = .default
) throws -> [SpeechSegment] {
    try validateAudio(audio)

    let flatAudio: MLXArray
    if audio.ndim == 2 {
        flatAudio = audio.squeezed(axis: 0)
    } else {
        flatAudio = audio
    }

    let totalSamples = flatAudio.shape[0]
    let audioLength = Double(totalSamples) / Double(VADAudioFormat.sampleRate)

    guard totalSamples > 0 else {
        return []
    }

    let chunkSize = VADAudioFormat.chunkSamples
    let iterator = VADIterator(model: model, config: config)
    iterator.validateRange = false

    let detector = BatchSpeechDetector(config: config)

    var offset = 0
    var chunksProcessed = 0
    while offset < totalSamples {
        let remainingSamples = totalSamples - offset
        let chunk: MLXArray

        if remainingSamples >= chunkSize {
            chunk = flatAudio[offset..<(offset + chunkSize)]
        } else {
            let partialChunk = flatAudio[offset...]
            let padding = MLXArray.zeros([chunkSize - remainingSamples])
            chunk = concatenated([partialChunk, padding], axis: 0)
        }

        let result = try iterator.process(chunk)
        detector.process(result)

        offset += chunkSize
        chunksProcessed += 1

        // Periodic memory cleanup for long audio files
        if chunksProcessed % 1000 == 0 {
            eval(MLXArray(0))
        }
    }

    detector.finalize(audioLength: audioLength)

    return applyPaddingAndMerge(
        segments: detector.segments,
        speechPadMs: config.speechPadMs,
        audioLength: audioLength
    )
}
