import Foundation
import MLX
import MLXAudioCore
import MLXNN

// True incremental (online) streaming for Nemotron 3.5 ASR.
//
// The offline `generateStream(...)` computes the mel of the whole utterance up
// front and only then walks the cache-aware encoder. A live caller (e.g. a mic
// feeding 80 ms chunks) instead wants to push audio as it arrives and read text
// back with the model's native chunk delay. This session does exactly that, while
// staying bit-identical to the offline encode.
//
// Why it is bit-identical (transcript == generateStream(wholeAudio)):
//   * The preprocessor uses `normalize: "NA"` (verified on the shipped checkpoint),
//     so each mel frame is an independent function of a fixed sample window — no
//     per-utterance mean/std that would shift as audio grows. preemph is causal.
//   * The STFT centers with `nFft/2` zero-pad, so mel frame m covers original
//     samples [m·hop − nFft/2, m·hop + nFft/2). Frame m is *frozen* — unaffected by
//     future audio — once `m·hop + nFft/2 <= bufferLen`. The session only feeds the
//     encoder frozen, whole chunks; the trailing partial chunk waits for `finish()`,
//     which reproduces the offline right-pad exactly.
//   * The cache-aware encoder + greedy RNN-T already carry all their state in
//     `NemotronASRStreamEncoderState` / `NemotronASRStreamRNNTState`, so resuming
//     across `step` calls reproduces the single-shot walk.
//
// Cost: each `step` computes only the mel frames the encoder can still consume,
// from the samples they cover, so a step costs O(chunk), not O(buffer).

/// Per-stream greedy RNN-T state, carried across chunks / `step` calls.
final class NemotronASRStreamRNNTState {
    var results: [NemoAlignedToken] = []
    var lastToken: Int
    var decoderState: NemoLSTMState?
    var globalTime = 0  // absolute subsampled-frame index, for token timestamps

    // The prediction network output depends only on (lastToken, decoderState),
    // which change only on emission. Blank frames reuse it, along with the
    // proposed LSTM state and the joint's pred projection. Cleared on emission
    // (keying on lastToken alone would reuse a stale output after a repeated token).
    var cachedPrediction: (predProjected: MLXArray, proposedState: NemoLSTMState)?

    init(blankToken: Int) { lastToken = blankToken }
}

/// Frames whose greedy RNN-T token is read back per host sync. Frames after an
/// emission are discarded, so a larger value trades wasted joint work for syncs.
private let nemoRNNTSyncFrames = 4

extension NemoJointNetwork {
    /// Second half of `callAsFunction`, taking the already projected encoder and
    /// prediction outputs, so a caller can reuse a projection across frames.
    func combine(_ encProjected: MLXArray, _ predProjected: MLXArray) -> MLXArray {
        var x = encProjected.expandedDimensions(axis: 2) + predProjected.expandedDimensions(axis: 1)
        switch activationName {
        case "relu":
            x = relu(x)
        case "sigmoid":
            x = sigmoid(x)
        default:
            x = tanh(x)
        }
        return outputProj(x)
    }
}

extension NemotronASRModel {
    /// Greedy RNN-T over one chunk of prompted encoder frames `(1, c, d)`, mutating
    /// `state`. Lifted verbatim from the offline streaming loop so the one-shot and
    /// session paths share a single decoder (SSOT).
    func streamRNNTDecode(
        _ prompted: MLXArray,
        state: NemotronASRStreamRNNTState,
        frameSeconds: Double
    ) {
        let chunkLen = prompted.shape[1]
        var time = 0
        var newSymbols = 0
        // Tokens for frames [specStart, specStart + spec.count), computed against the
        // current cached prediction in one host sync. Each frame still runs its own
        // M=1 joint, as before; only the `.item()` per frame is batched. Valid until
        // the next emission changes the prediction.
        var spec: [Int] = []
        var specStart = 0
        while time < chunkLen {
            let frame = prompted[0..., time..<(time + 1), 0...]
            let prediction: (predProjected: MLXArray, proposedState: NemoLSTMState)
            if let cached = state.cachedPrediction {
                prediction = cached
            } else {
                let currentToken: MLXArray? = state.lastToken == blankTokenID
                    ? nil
                    : MLXArray(Int32(state.lastToken)).reshaped([1, 1]).asType(.int32)
                let decoderOutput = decoder(currentToken, state: state.decoderState)
                prediction = (
                    joint.pred(decoderOutput.0.asType(frame.dtype)),
                    (
                        hidden: decoderOutput.1.hidden?.asType(frame.dtype),
                        cell: decoderOutput.1.cell?.asType(frame.dtype)
                    )
                )
                state.cachedPrediction = prediction
            }
            if time < specStart || time >= specStart + spec.count {
                let end = min(time + nemoRNNTSyncFrames, chunkLen)
                var picks: [MLXArray] = []
                picks.reserveCapacity(end - time)
                for t in time..<end {
                    let f = t == time ? frame : prompted[0..., t..<(t + 1), 0...]
                    picks.append(joint.combine(joint.enc(f), prediction.predProjected).argMax(axis: -1).reshaped([1]))
                }
                spec = MLX.concatenated(picks, axis: 0).asType(.int32).asArray(Int32.self).map { Int($0) }
                specStart = time
            }
            let token = spec[time - specStart]
            let step = NemoDecodingLogic.rnntStep(
                predictedToken: token,
                blankToken: blankTokenID,
                time: time,
                newSymbols: newSymbols,
                maxSymbols: maxSymbols
            )
            if step.emittedToken {
                state.lastToken = token
                state.decoderState = prediction.proposedState
                state.cachedPrediction = nil
                spec = []
                if !NemotronASRTokenizer.isSpecialToken(token, vocabulary: vocabulary) {
                    state.results.append(
                        NemoAlignedToken(
                            id: token,
                            text: NemotronASRTokenizer.decode(tokens: [token], vocabulary: vocabulary),
                            start: Double(state.globalTime + time) * frameSeconds,
                            duration: frameSeconds
                        )
                    )
                }
            }
            time = step.nextTime
            newSymbols = step.nextNewSymbols
        }
        state.globalTime += chunkLen
    }
}

public final class NemotronASRStreamSession {
    /// Text + token ids decoded by a single `step` / `finish` call.
    public struct Delta {
        public let text: String
        public let tokenIds: [Int]
    }

    private let model: NemotronASRModel
    private let language: String?
    private let chunkFrames: Int?
    private let frameSeconds: Double

    // Samples the next step can still need; `rawBuffer[0]` is stream sample `rawOffset`.
    private var rawBuffer: [Float] = []
    private var rawOffset = 0
    private var sampleCount: Int { rawOffset + rawBuffer.count }
    private let melBasis: NemotronASRAudio.MelBasis
    private let encState: NemotronASRStreamEncoderState
    private let rnntState: NemotronASRStreamRNNTState
    private var emittedText = ""
    // Alignment of `rnntState.results`, rebuilt only when a step adds a token.
    private var aligned = NemoAlignment.sentencesToResult([])
    private var cachedSegments: [[String: Any]]?
    private var done = false

    init(model: NemotronASRModel, language: String?, chunkFrames: Int?) {
        self.model = model
        self.language = language
        self.chunkFrames = chunkFrames
        self.encState = NemotronASRStreamEncoderState(layers: model.encoder.layers.count)
        self.rnntState = NemotronASRStreamRNNTState(blankToken: model.blankTokenID)
        self.melBasis = NemotronASRAudio.MelBasis(config: model.preprocessConfig)
        self.frameSeconds = Double(model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength)
            / Double(model.preprocessConfig.sampleRate)
        let norm = model.preprocessConfig.normalize.lowercased()
        precondition(
            norm == "na" || norm == "none",
            "NemotronASRStreamSession requires NA mel normalization (got \(model.preprocessConfig.normalize)); "
                + "per-utterance normalization is not frozen incrementally."
        )
    }

    /// Full transcript decoded so far.
    public var text: String { emittedText }
    /// Sentence-level segments with start/end times, built from the same alignment
    /// as `text`. Matches the offline path's `STTOutput.segments` so `--format srt`
    /// / `vtt` / `json` keep their timestamps when streaming.
    public var segments: [[String: Any]] {
        if let cachedSegments { return cachedSegments }
        let built = aligned.segments
        cachedSegments = built
        return built
    }
    /// Token ids decoded so far.
    public var tokens: [Int] { rnntState.results.map { $0.id } }
    /// Whether `finish()` has been called.
    public var isFinished: Bool { done }

    /// Ingest a chunk of 16 kHz mono samples; returns the text decoded by this call.
    @discardableResult
    public func step(_ samples: [Float]) -> Delta {
        rawBuffer.append(contentsOf: samples)
        return advance(final: false)
    }

    @discardableResult
    public func step(_ samples: MLXArray) -> Delta {
        let mono = samples.ndim > 1 ? samples.mean(axis: -1) : samples
        return step(mono.asType(.float32).asArray(Float.self))
    }

    /// Flush the trailing partial chunk so the final transcript equals
    /// `generateStream(wholeAudio)`. Call once after the last `step`.
    @discardableResult
    public func finish() -> Delta {
        advance(final: true)
    }

    private func advance(final: Bool) -> Delta {
        guard !done else { return Delta(text: "", tokenIds: []) }
        guard sampleCount > 0 else {
            if final { done = true }
            return Delta(text: "", tokenIds: [])
        }

        let config = model.preprocessConfig
        let mel: MLXArray
        let melOffset: Int
        let limit: Int
        if (config.padTo > 0 && sampleCount < config.padTo) || sampleCount < config.nFft {
            // Short buffers are never trimmed, so `rawBuffer` is the whole stream here.
            mel = NemotronASRAudio.logMelSpectrogram(MLXArray(rawBuffer), config: config)  // (1, T, F)
            melOffset = 0
            limit = final ? mel.shape[1] : frozenMelFrames(totalMel: mel.shape[1])
        } else {
            // Only the frames the encoder can still consume: [consumed, limit). The
            // bounds are widened to the RFFT row pairs of the full computation (see
            // `logMelFrames`), so every frame matches the whole-buffer mel bit for bit.
            let totalMel = NemotronASRAudio.frameCount(sampleCount: sampleCount, config: config)
            limit = final ? totalMel : frozenMelFrames(totalMel: totalMel)
            let first = encState.consumed & ~1
            let end = limit % 2 == 0 ? limit : min(limit + 1, totalMel)
            guard encState.consumed < limit else {
                if final { done = true; Memory.clearCache() }
                return Delta(text: "", tokenIds: [])
            }
            // Fewer frames than a chunk: `streamEncodeChunks` would encode nothing.
            if !final && limit - encState.consumed < model.streamChunkMelFrames(chunkFrames: chunkFrames) {
                return Delta(text: "", tokenIds: [])
            }
            let frames = NemotronASRAudio.streamMelFrames(
                rawBuffer, offset: rawOffset, first: first, end: end, config: config, basis: melBasis
            )
            mel = frames.mel
            melOffset = frames.melOffset
        }

        let firstNew = rnntState.results.count
        model.streamEncodeChunks(
            mel,
            language: language,
            limit: limit,
            chunkFrames: chunkFrames,
            flushTail: final,
            state: encState,
            melOffset: melOffset
        ) { prompted in
            model.streamRNNTDecode(prompted, state: rnntState, frameSeconds: frameSeconds)
        }
        if !final && sampleCount >= max(config.nFft, config.padTo)
            && NemotronASRAudio.frameCount(sampleCount: sampleCount, config: config)
                >= NemotronASRAudio.gemmMinRows(config: config) {
            // Once `streamMelFrames` stops needing the whole stream, keep only what the
            // next step's first frame needs: from `consumed` rounded down to even, minus
            // the STFT half-window and one pre-emphasis sample. With hop > nFft that
            // point can lie past the last sample, so clamp it to the buffer.
            let keepFrom = min(
                max(0, (encState.consumed & ~1) * config.hopLength - config.nFft / 2 - 1),
                sampleCount
            )
            if keepFrom > rawOffset {
                rawBuffer.removeFirst(keepFrom - rawOffset)
                rawOffset = keepFrom
            }
        }

        // Bound the lazy graph across steps: materialize the caches the next step
        // resumes from (the in-loop `.item()` already forced this chunk's encoder).
        var live: [MLXArray] = []
        if let mc = encState.melCache { live.append(mc) }
        for c in encState.attnCache where c != nil { live.append(c!) }
        for c in encState.convCache where c != nil { live.append(c!) }
        if !live.isEmpty { MLX.asyncEval(live) }

        // Rebuilding the alignment walks every token so far; with no new token it
        // would be unchanged and the text delta empty.
        var deltaText = ""
        if rnntState.results.count > firstNew {
            aligned = NemoAlignment.sentencesToResult(NemoAlignment.tokensToSentences(rnntState.results))
            cachedSegments = nil
            let fullText = aligned.text
            deltaText = fullText.hasPrefix(emittedText)
                ? String(fullText.dropFirst(emittedText.count))
                : fullText
            emittedText = fullText
        }
        let deltaIds = rnntState.results[firstNew...].map { $0.id }

        if final {
            done = true
            // The session's carried encoder/RNNT state is still live here, so
            // this clear only releases temporary buffers. Callers that drop
            // the session after finish can clear once more to return the
            // entire idle pool to the weight floor.
            Memory.clearCache()
        }
        return Delta(text: deltaText, tokenIds: Array(deltaIds))
    }

    /// Number of mel frames whose STFT window is fully covered by real audio, hence
    /// bit-identical to the final offline mel regardless of future samples. The STFT
    /// centers with `nFft/2` zero-pad, so frame m is frozen iff m·hop + nFft/2 <=
    /// bufferLen. Conservative by construction: an under-count only delays a chunk
    /// by one `step` (latency), never corrupts output.
    private func frozenMelFrames(totalMel: Int) -> Int {
        let hop = model.preprocessConfig.hopLength
        let half = model.preprocessConfig.nFft / 2
        guard sampleCount >= half else { return 0 }
        let largestFrozen = (sampleCount - half) / hop
        return min(totalMel, largestFrozen + 1)
    }
}

public extension NemotronASRModel {
    /// Create an online streaming session. Feed audio with `step(_:)`, then `finish()`.
    ///
    /// - Parameters:
    ///   - language: language prompt (e.g. "ru"); `nil` uses the model default.
    ///   - chunkMs: chunk size in ms; maps to right-context per the latency ladder
    ///     (80→[56,0], 160→[56,1], 320→[56,3], 560→[56,6], 1120→[56,13]). `nil` uses
    ///     the model's native chunk (`default_att_context_size`). Smaller chunks cut
    ///     latency at a modest WER cost — see the model card's chunk-size table.
    func makeStreamSession(
        language: String? = nil,
        chunkMs: Int? = nil
    ) -> NemotronASRStreamSession {
        let chunkFrames: Int?
        if let chunkMs {
            let msPerSubframe = encoderConfig.subsamplingFactor * preprocessConfig.hopLength * 1000
                / preprocessConfig.sampleRate  // 8*160*1000/16000 = 80 ms
            chunkFrames = max(1, Int((Double(chunkMs) / Double(msPerSubframe)).rounded()))
        } else {
            chunkFrames = nil
        }
        return NemotronASRStreamSession(model: self, language: language, chunkFrames: chunkFrames)
    }

    /// Transcribe a whole audio buffer through the online streaming session, feeding
    /// fixed `chunkMs`-sized chunks as a live caller would — instead of the whole-buffer
    /// `generateStream`. `onDelta` receives each newly decoded fragment as it is produced
    /// (use it to render live output); the returned `STTOutput` is the full transcript.
    func transcribeStreaming(
        audio: MLXArray,
        generationParameters: STTGenerateParameters = STTGenerateParameters(),
        chunkMs: Int = 480,
        onDelta: ((String) -> Void)? = nil
    ) -> STTOutput {
        let mono = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        let samples = mono.asType(.float32).asArray(Float.self)
        let chunk = max(1, 16000 * chunkMs / 1000)

        let session = makeStreamSession(language: generationParameters.language)
        let start = CFAbsoluteTimeGetCurrent()

        func emit(_ delta: NemotronASRStreamSession.Delta) {
            guard !delta.text.isEmpty else { return }
            onDelta?(delta.text)
        }
        var idx = 0
        while idx < samples.count {
            let end = min(idx + chunk, samples.count)
            emit(session.step(Array(samples[idx..<end])))
            idx = end
        }
        emit(session.finish())

        let totalTime = CFAbsoluteTimeGetCurrent() - start
        let tokenCount = session.tokens.count
        return STTOutput(
            text: session.text.trimmingCharacters(in: .whitespacesAndNewlines),
            segments: session.segments,
            language: generationParameters.language,
            generationTokens: tokenCount,
            totalTokens: tokenCount,
            generationTps: totalTime > 0 ? Double(tokenCount) / totalTime : 0,
            totalTime: totalTime
        )
    }
}
