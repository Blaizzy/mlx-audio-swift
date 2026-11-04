import Foundation
import MLX

// Buffered streaming for Nemotron Unified (`chunked_limited_with_rc`) checkpoints.
//
// The Unified encoder has non-causal convolutions and a right context, so it cannot
// carry per-layer caches. As in NeMo's `speech_to_text_streaming_infer_rnnt.py`, each
// step re-runs the mel and the encoder on a sliding buffer of left context + chunk +
// right context and keeps the chunk's frames; greedy RNN-T continues across steps.
// Step k covers audio [k*C, (k+1)*C). It is the last step once (k+1)*C + R reaches
// the end of the audio, and then runs to the end with no right context. A live
// stream runs a non-last step only once it holds more than (k+1)*C + R samples, so
// it segments audio exactly as NeMo segments the finished file.

/// Buffer layout in encoder frames (80 ms each).
public struct NemotronASRBufferedContext: Equatable, Sendable {
    public let left: Int
    public let chunk: Int
    public let right: Int

    public init(left: Int, chunk: Int, right: Int) {
        precondition(left >= 0 && chunk >= 1 && right >= 0, "invalid buffered context")
        self.left = left
        self.chunk = chunk
        self.right = right
    }

    /// NVIDIA's recommended [chunk, right] per latency (chunk + right, ms).
    public static let latencies: [Int: (chunk: Int, right: Int)] = [
        160: (1, 1), 240: (1, 2), 320: (1, 3), 560: (2, 5), 1120: (7, 7), 2080: (13, 13),
    ]
    /// 5.6 s, the left context the model was trained with.
    public static let defaultLeft = 70

    /// The recommended context for the supported latency nearest `latencyMs`
    /// (ties go to the lower latency).
    public static func forLatency(_ latencyMs: Int, left: Int = defaultLeft) -> NemotronASRBufferedContext {
        let best = latencies.keys.min { a, b in
            let da = abs(a - latencyMs), db = abs(b - latencyMs)
            return da == db ? a < b : da < db
        }!
        let (chunk, right) = latencies[best]!
        return NemotronASRBufferedContext(left: left, chunk: chunk, right: right)
    }

    var asArray: [Int] { [left, chunk, right] }
}

/// Turns pushed PCM into prompted encoder frames, one chunk per step.
final class NemotronASRBufferedEncoder {
    private let model: NemotronASRModel
    let context: NemotronASRBufferedContext
    private let language: String?
    private let hop: Int
    let frameSamples: Int
    private let leftSamples: Int
    let chunkSamples: Int
    private let rightSamples: Int

    // `samples[0]` is stream sample `start`.
    private var samples: [Float] = []
    private var start = 0
    private(set) var total = 0
    private var next = 0  // stream sample where the next chunk starts
    private var previousStart = 0  // buffer start of the previous step
    private var reachedFullWindow = false
    /// The encoder compiled for full-size windows, by batch size. Ramp and last
    /// windows run uncompiled: each has its own size and would trace once.
    private var compiledEncoders: [Int: ([MLXArray]) -> [MLXArray]] = [:]
    var compilesFullWindows = true
    private(set) var done = false

    private let melBasis: NemotronASRAudio.MelBasis

    init(model: NemotronASRModel, context: NemotronASRBufferedContext, language: String?) {
        self.model = model
        self.melBasis = NemotronASRAudio.MelBasis(config: model.preprocessConfig)
        self.context = context
        self.language = language
        self.hop = model.preprocessConfig.hopLength
        self.frameSamples = model.encoderConfig.subsamplingFactor * hop
        self.leftSamples = context.left * frameSamples
        self.chunkSamples = context.chunk * frameSamples
        self.rightSamples = context.right * frameSamples
    }

    /// Most windows encoded in one batched pass.
    static let maxBatch = 16

    private struct Window {
        let start: Int
        let end: Int
        let left: Int  // encoder frames before the chunk
    }

    /// Prompted `(1, frames, d)` chunks this push completes, in audio order. Chunks
    /// that become ready together (a step after a slow one, a whole file) encode
    /// their equal-length windows as one batch.
    func push(_ newSamples: [Float], final: Bool) -> [MLXArray] {
        guard !done else { return [] }
        samples.append(contentsOf: newSamples)
        total += newSamples.count

        var ready: [Window] = []
        while total > next + chunkSamples + rightSamples {
            let bufferStart = max(0, next - leftSamples)
            ready.append(Window(
                start: bufferStart, end: next + chunkSamples + rightSamples,
                left: (next - bufferStart) / frameSamples
            ))
            previousStart = bufferStart
            next += chunkSamples
        }
        var outputs = encode(ready)
        if !reachedFullWindow, let last = ready.last,
           last.end - last.start == leftSamples + chunkSamples + rightSamples {
            // Until the left context fills, every step's window is a new size, and
            // MLX's buffer cache keeps each size's buffers. From here on every
            // window has one size, so drop the ramp's buffers once.
            reachedFullWindow = true
            eval(outputs)
            Memory.clearCache()
        }
        if final {
            if total > next {
                // NeMo caps the last buffer at left + chunk + right samples.
                let bufferStart = max(previousStart, total - (leftSamples + chunkSamples + rightSamples))
                let window = Window(start: bufferStart, end: total, left: (next - bufferStart) / frameSamples)
                outputs.append(encodeLast(window))
            }
            done = true
        }
        // The next buffer starts at or after the previous step's start.
        let keep = max(0, next - chunkSamples - leftSamples)
        if keep > start {
            samples.removeFirst(keep - start)
            start = keep
        }
        return outputs
    }

    private func mel(_ window: Window) -> MLXArray {
        let audio = Array(samples[(window.start - start)..<(window.end - start)])
        return NemotronASRAudio.logMelSpectrogram(MLXArray(audio), config: model.preprocessConfig, basis: melBasis)
    }

    private func encode(_ windows: [Window]) -> [MLXArray] {
        var outputs: [MLXArray] = []
        var i = 0
        while i < windows.count {
            // Windows of equal length and left context share every shape and mask.
            var j = i + 1
            while j < windows.count, j - i < Self.maxBatch,
                  windows[j].end - windows[j].start == windows[i].end - windows[i].start,
                  windows[j].left == windows[i].left {
                j += 1
            }
            let group = windows[i..<j]
            let mels = MLX.concatenated(group.map(mel), axis: 0)
            let length = windows[i].end - windows[i].start
            let lengths = MLXArray(Array(repeating: Int32(length / hop), count: group.count))
            let features = mels.asType(model.computeDType)
            let encoded: MLXArray
            if compilesFullWindows, length == leftSamples + chunkSamples + rightSamples {
                encoded = compiledEncoder(batch: group.count)([features, lengths])[0]
            } else {
                encoded = model.encoder(features, lengths: lengths, attContextSize: context.asArray).0
            }
            let left = windows[i].left
            let prompted = model.applyPrompt(encoded[0..., left..<(left + context.chunk), 0...], language: language)
            for b in 0..<group.count {
                outputs.append(prompted[b..<(b + 1)])
            }
            i = j
        }
        return outputs
    }

    private func compiledEncoder(batch: Int) -> ([MLXArray]) -> [MLXArray] {
        if let compiled = compiledEncoders[batch] { return compiled }
        // Weights are compile inputs, so the traced graph reads the live parameters.
        let encoder = model.encoder
        let attContextSize = context.asArray
        let compiled = MLX.compile(inputs: [encoder]) { (arrays: [MLXArray]) -> [MLXArray] in
            [encoder(arrays[0], lengths: arrays[1], attContextSize: attContextSize).0]
        }
        compiledEncoders[batch] = compiled
        return compiled
    }

    private func encodeLast(_ window: Window) -> MLXArray {
        let (encoded, encodedLengths) = model.encoder(
            mel(window).asType(model.computeDType),
            lengths: MLXArray([Int32((window.end - window.start) / hop)]),
            attContextSize: context.asArray
        )
        let count = Int(encodedLengths[0].item(Int32.self)) - window.left
        return model.applyPrompt(encoded[0..., window.left..<(window.left + count), 0...], language: language)
    }
}

extension NemotronASRModel {
    /// Unified checkpoints stream buffered; 3.5 checkpoints are cache-aware.
    public var isBuffered: Bool { encoderConfig.attContextStyle == "chunked_limited_with_rc" }

    /// Offline, Unified runs one full-context pass up to this length, then NeMo's
    /// long-form buffered setting (10 s left, 10 s chunk, 5 s right).
    static let bufferedOfflineMaxSeconds = 60.0
    static let bufferedLongForm = NemotronASRBufferedContext(left: 125, chunk: 125, right: 62)

    func generateBuffered(_ audio: MLXArray, language: String?) -> NemoAlignedResult {
        let sampleRate = preprocessConfig.sampleRate
        if Double(audio.shape[0]) <= Self.bufferedOfflineMaxSeconds * Double(sampleRate) {
            let mel = NemotronASRAudio.logMelSpectrogram(audio, config: preprocessConfig)
            return decode(
                mel: mel,
                language: language,
                attContextSize: [-1, -1, -1],
                validMelFrames: audio.shape[0] / preprocessConfig.hopLength
            )
        }
        let state = NemotronASRStreamRNNTState(blankToken: blankTokenID)
        let encoder = NemotronASRBufferedEncoder(model: self, context: Self.bufferedLongForm, language: language)
        for prompted in encoder.push(audio.asArray(Float.self), final: true) {
            streamRNNTDecode(prompted, state: state, frameSeconds: encoderFrameSeconds)
        }
        return NemoAlignment.sentencesToResult(NemoAlignment.tokensToSentences(state.results))
    }

    var encoderFrameSeconds: Double {
        Double(encoderConfig.subsamplingFactor * preprocessConfig.hopLength) / Double(preprocessConfig.sampleRate)
    }
}
