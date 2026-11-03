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
    private(set) var done = false

    init(model: NemotronASRModel, context: NemotronASRBufferedContext, language: String?) {
        self.model = model
        self.context = context
        self.language = language
        self.hop = model.preprocessConfig.hopLength
        self.frameSamples = model.encoderConfig.subsamplingFactor * hop
        self.leftSamples = context.left * frameSamples
        self.chunkSamples = context.chunk * frameSamples
        self.rightSamples = context.right * frameSamples
    }

    /// Prompted `(1, frames, d)` chunks this push completes, in audio order.
    func push(_ newSamples: [Float], final: Bool) -> [MLXArray] {
        guard !done else { return [] }
        samples.append(contentsOf: newSamples)
        total += newSamples.count

        var outputs: [MLXArray] = []
        while !done {
            let chunkEnd = next + chunkSamples
            if total > chunkEnd + rightSamples {
                let bufferStart = max(0, next - leftSamples)
                outputs.append(encode(from: bufferStart, to: chunkEnd + rightSamples, last: false))
                previousStart = bufferStart
                next = chunkEnd
            } else if final {
                if total > next {
                    // NeMo caps the last buffer at left + chunk + right samples.
                    let bufferStart = max(previousStart, total - (leftSamples + chunkSamples + rightSamples))
                    outputs.append(encode(from: bufferStart, to: total, last: true))
                }
                done = true
            } else {
                break
            }
        }
        // The next buffer starts at or after the previous step's start.
        let keep = max(0, next - chunkSamples - leftSamples)
        if keep > start {
            samples.removeFirst(keep - start)
            start = keep
        }
        return outputs
    }

    private func encode(from bufferStart: Int, to bufferEnd: Int, last: Bool) -> MLXArray {
        let audio = Array(samples[(bufferStart - start)..<(bufferEnd - start)])
        let mel = NemotronASRAudio.logMelSpectrogram(MLXArray(audio), config: model.preprocessConfig)
        let lengths = MLXArray([Int32(audio.count / hop)])
        let (encoded, encodedLengths) = model.encoder(
            mel.asType(model.computeDType), lengths: lengths, attContextSize: context.asArray
        )
        let left = (next - bufferStart) / frameSamples
        let count = last ? Int(encodedLengths[0].item(Int32.self)) - left : context.chunk
        return model.applyPrompt(encoded[0..., left..<(left + count), 0...], language: language)
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
