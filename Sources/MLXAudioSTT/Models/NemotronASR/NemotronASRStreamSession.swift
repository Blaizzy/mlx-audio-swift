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
// Cost note: v1 recomputes the full mel from the raw buffer each `step` (O(buffer)).
// Mel is ~1% of encode, so this is negligible at utterance scale; an incremental
// mel over a sliding raw window is a future optimization.

/// Per-stream greedy RNN-T state, carried across chunks / `step` calls.
final class NemotronASRStreamRNNTState {
    var results: [NemoAlignedToken] = []
    var lastToken: Int
    var decoderState: NemoLSTMState?
    var globalTime = 0  // absolute subsampled-frame index, for token timestamps

    init(blankToken: Int) { lastToken = blankToken }
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
        while time < chunkLen {
            let frame = prompted[0..., time..<(time + 1), 0...]
            let currentToken: MLXArray? = state.lastToken == blankTokenID
                ? nil
                : MLXArray(Int32(state.lastToken)).reshaped([1, 1]).asType(.int32)
            let decoderOutput = decoder(currentToken, state: state.decoderState)
            let pred = decoderOutput.0.asType(frame.dtype)
            let proposedState: NemoLSTMState = (
                hidden: decoderOutput.1.hidden?.asType(frame.dtype),
                cell: decoderOutput.1.cell?.asType(frame.dtype)
            )
            let jointOutput = joint(frame, pred)
            let token = jointOutput.argMax(axis: -1).item(Int.self)
            let step = NemoDecodingLogic.rnntStep(
                predictedToken: token,
                blankToken: blankTokenID,
                time: time,
                newSymbols: newSymbols,
                maxSymbols: maxSymbols
            )
            if step.emittedToken {
                state.lastToken = token
                state.decoderState = proposedState
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

    private var rawBuffer: [Float] = []
    private let encState: NemotronASRStreamEncoderState
    private let rnntState: NemotronASRStreamRNNTState
    private var emittedText = ""
    private var done = false

    init(model: NemotronASRModel, language: String?, chunkFrames: Int?) {
        self.model = model
        self.language = language
        self.chunkFrames = chunkFrames
        self.encState = NemotronASRStreamEncoderState(layers: model.encoder.layers.count)
        self.rnntState = NemotronASRStreamRNNTState(blankToken: model.blankTokenID)
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
        NemoAlignment.sentencesToResult(
            NemoAlignment.tokensToSentences(rnntState.results)
        ).segments
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
        guard !rawBuffer.isEmpty else {
            if final { done = true }
            return Delta(text: "", tokenIds: [])
        }

        let audio = MLXArray(rawBuffer)
        let mel = NemotronASRAudio.logMelSpectrogram(audio, config: model.preprocessConfig)  // (1, T, F)
        let totalMel = mel.shape[1]
        let limit = final ? totalMel : frozenMelFrames(totalMel: totalMel)

        let firstNew = rnntState.results.count
        model.streamEncodeChunks(
            mel,
            language: language,
            limit: limit,
            chunkFrames: chunkFrames,
            flushTail: final,
            state: encState
        ) { prompted in
            model.streamRNNTDecode(prompted, state: rnntState, frameSeconds: frameSeconds)
        }

        // Bound the lazy graph across steps: materialize the caches the next step
        // resumes from (the in-loop `.item()` already forced this chunk's encoder).
        var live: [MLXArray] = []
        if let mc = encState.melCache { live.append(mc) }
        for c in encState.attnCache where c != nil { live.append(c!) }
        for c in encState.convCache where c != nil { live.append(c!) }
        if !live.isEmpty { MLX.asyncEval(live) }

        let fullText = NemoAlignment.sentencesToResult(
            NemoAlignment.tokensToSentences(rnntState.results)
        ).text
        let deltaText = fullText.hasPrefix(emittedText)
            ? String(fullText.dropFirst(emittedText.count))
            : fullText
        emittedText = fullText
        let deltaIds = rnntState.results[firstNew...].map { $0.id }

        if final { done = true }
        Memory.clearCache()
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
        guard rawBuffer.count >= half else { return 0 }
        let largestFrozen = (rawBuffer.count - half) / hop
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
