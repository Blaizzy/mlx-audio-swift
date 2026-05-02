import Foundation
@preconcurrency import MLX

/// Configuration for VAD-based speech segmentation — model-agnostic chunking of an
/// audio buffer into speech regions for downstream transcription.
public struct SpeechSegmentConfig: Sendable {
    public var threshold: Float
    public var minSpeechMs: Int
    public var minSilenceMs: Int
    public var speechPadMs: Int
    public var mergeGapS: Float
    public var maxChunkS: Float

    public init(
        threshold: Float = 0.5,
        minSpeechMs: Int = 250,
        minSilenceMs: Int = 100,
        speechPadMs: Int = 30,
        mergeGapS: Float = 1.0,
        maxChunkS: Float = 30.0
    ) {
        self.threshold = threshold
        self.minSpeechMs = minSpeechMs
        self.minSilenceMs = minSilenceMs
        self.speechPadMs = speechPadMs
        self.mergeGapS = mergeGapS
        self.maxChunkS = maxChunkS
    }
}

private struct SpeechRun {
    var startSample: Int
    var endSample: Int
}

private let BLOCKS_PER_256MS = 8

private func detectSpeechRuns(
    audio: MLXArray,
    sampleRate: Int,
    vadModel: SileroVAD,
    config: SpeechSegmentConfig
) throws -> [SpeechRun] {
    // One Silero probability spans `chunkSamples` audio samples (512 @ 16 kHz,
    // 256 @ 8 kHz). We group BLOCKS_PER_256MS of them into ~256 ms decision blocks,
    // so a block stays 256 ms at either sample rate.
    let chunkSamples = sampleRate == 16000 ? 512 : 256
    let blockSamples = chunkSamples * BLOCKS_PER_256MS
    let blockDurS = Float(blockSamples) / Float(sampleRate)

    let probsMx = try vadModel.predictProba(audio, sampleRate: sampleRate)
    let probs32 = probsMx.asArray(Float.self)   // asArray forces evaluation of the lazy graph
    let n = (probs32.count / BLOCKS_PER_256MS) * BLOCKS_PER_256MS
    if n == 0 { return [] }

    var probs256 = [Float](repeating: 0, count: n / BLOCKS_PER_256MS)
    for i in 0 ..< probs256.count {
        var product: Float = 1
        for k in 0 ..< BLOCKS_PER_256MS {
            product *= (1.0 - probs32[i * BLOCKS_PER_256MS + k])
        }
        probs256[i] = 1.0 - product
    }

    let speechPadBlocks = max(0, Int(Float(config.speechPadMs) / 1000 / blockDurS))
    // Round the minimum-duration thresholds UP: a min of 500 ms must require ≥ 2 blocks
    // (≈ 512 ms), not 1 (≈ 256 ms), so sub-threshold bursts are actually filtered.
    let minSpeechBlocks = max(1, Int((Float(config.minSpeechMs) / 1000 / blockDurS).rounded(.up)))
    let minSilenceBlocks = max(1, Int((Float(config.minSilenceMs) / 1000 / blockDurS).rounded(.up)))

    let actualLen = audio.shape[0]
    var runs: [SpeechRun] = []
    var inSpeech = false
    var segStart = 0
    var lastSpeech = -1
    var silentRun = 0
    for (idx, p) in probs256.enumerated() {
        if p >= config.threshold {
            if !inSpeech {
                segStart = max(0, idx - speechPadBlocks)
                inSpeech = true
            }
            lastSpeech = idx
            silentRun = 0
        } else if inSpeech {
            silentRun += 1
            if silentRun >= minSilenceBlocks {
                let segEnd = min(lastSpeech + 1 + speechPadBlocks, probs256.count)
                if segEnd - segStart >= minSpeechBlocks {
                    let s = segStart * blockSamples
                    let e = min(segEnd * blockSamples, actualLen)
                    if s < e {
                        runs.append(SpeechRun(startSample: s, endSample: e))
                    }
                }
                inSpeech = false
                silentRun = 0
                lastSpeech = -1
            }
        }
    }
    if inSpeech {
        let endIdx = min(probs256.count, lastSpeech + 1 + speechPadBlocks)
        if endIdx - segStart >= minSpeechBlocks {
            let s = segStart * blockSamples
            let e = min(endIdx * blockSamples, actualLen)
            if s < e {
                runs.append(SpeechRun(startSample: s, endSample: e))
            }
        }
    }
    return runs
}

private func splitLong(start: Int, end: Int, maxChunkSamples: Int) -> [(Int, Int)] {
    if end - start <= maxChunkSamples { return [(start, end)] }
    var parts: [(Int, Int)] = []
    var cur = start
    while cur < end {
        let nxt = min(cur + maxChunkSamples, end)
        parts.append((cur, nxt))
        cur = nxt
    }
    return parts
}

private func mergeRuns(
    _ runs: [SpeechRun],
    sampleRate: Int,
    mergeGapS: Float,
    maxChunkS: Float
) -> [SpeechRun] {
    if runs.isEmpty { return runs }
    // Clamp to ≥ 1 sample: a non-positive maxChunkS would make `splitLong` never advance
    // (nxt == cur) and hang. 1 sample is degenerate but terminates.
    let maxChunkSamples = max(1, Int(maxChunkS * Float(sampleRate)))
    let maxGapSamples = Int(mergeGapS * Float(sampleRate))
    var merged: [(Int, Int)] = splitLong(
        start: runs[0].startSample, end: runs[0].endSample, maxChunkSamples: maxChunkSamples
    )
    for r in runs.dropFirst() {
        var prev = merged[merged.count - 1]
        let gap = r.startSample - prev.1
        let newDur = r.endSample - prev.0
        if gap <= maxGapSamples && newDur <= maxChunkSamples {
            prev.1 = r.endSample
            merged[merged.count - 1] = prev
        } else {
            merged.append(contentsOf: splitLong(
                start: r.startSample, end: r.endSample, maxChunkSamples: maxChunkSamples
            ))
        }
    }
    return merged.map { SpeechRun(startSample: $0.0, endSample: $0.1) }
}

/// Split an audio buffer into speech segments with a Silero VAD, returning each segment
/// paired with its start offset (seconds). Silence is dropped; adjacent runs within
/// `mergeGapS` are merged and long runs are split at `maxChunkS`. If no speech is found
/// the whole buffer is returned as a single segment. Model-agnostic — reuse across STT
/// models instead of per-model VAD wrappers.
public func segmentSpeech(
    audio: MLXArray,
    sampleRate: Int,
    vadModel: SileroVAD,
    config: SpeechSegmentConfig = SpeechSegmentConfig()
) throws -> [(MLXArray, Float)] {
    let audio1D = audio.ndim > 1 ? audio.mean(axis: -1) : audio
    let raw = try detectSpeechRuns(
        audio: audio1D, sampleRate: sampleRate, vadModel: vadModel, config: config
    )
    let runs = mergeRuns(
        raw, sampleRate: sampleRate, mergeGapS: config.mergeGapS, maxChunkS: config.maxChunkS
    )
    if runs.isEmpty {
        return [(audio1D, 0)]
    }
    return runs.map { run in
        let chunk = audio1D[run.startSample ..< run.endSample]
        let offsetS = Float(run.startSample) / Float(sampleRate)
        return (chunk, offsetS)
    }
}
