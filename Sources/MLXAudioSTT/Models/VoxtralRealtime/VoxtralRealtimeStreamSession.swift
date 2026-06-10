import Foundation
import MLX
import MLXAudioCore

// True incremental (online) streaming for Voxtral Realtime.
//
// The offline `generate(...)` path encodes the entire audio buffer up front and only
// then walks the decoder. This session ingests audio *as it arrives* (e.g. 80 ms mic
// chunks), feeds only newly-frozen conv frames through the transformer encoder with a
// persistent per-layer KV-cache, maintains the decoder KV-cache, and emits tokens with
// the model's native transcription delay — O(1) work per chunk.
//
// Correctness (WER 0 vs offline):
//   * conv stem is causal and `prepareMel` right-pads with zeros, so `convOut[0..<k]`
//     re-derived from a prefix is bit-identical to the offline full encode for every
//     frozen row k. The only unfrozen row is the trailing partial token (chunk ended
//     mid-1280-sample-token) — guarded by `frozenGuardTokens`.
//   * RoPE attention is relative-position invariant, so feeding conv frames in
//     sliding-window-aligned blocks with the cache RESET at each boundary reproduces
//     `encodeChunked` (>sw) exactly; a single un-reset block reproduces `encodeFull`
//     (<=sw). See `feedIncremental`.
//   * `finish()` reproduces the offline tail zero-pad ⇒ final transcript == generate().

/// Persistent incremental-encoder state carried across `step` calls.
struct VoxtralRealtimeStreamEncoderState {
    var caches: [VoxtralRealtimeEncoderKVCache?]
    var blockBase = 0   // absolute conv-frame index where the current sw-block began
    var consumed = 0    // conv frames already fed to the transformer

    init(layers: Int) {
        caches = Array(repeating: nil, count: layers)
    }
}

extension VoxtralRealtimeAudioEncoder {
    /// Feed conv frames `[state.consumed, upTo)` through the transformer incrementally,
    /// resetting the per-layer caches at each `slidingWindow` boundary so the result is
    /// bit-identical to offline `encodeFull` (<=sw) / `encodeChunked` (>sw). Returns the
    /// new transformer-normed frames (pre-downsample).
    func feedIncremental(
        _ convOut: MLXArray,
        upTo: Int,
        state: inout VoxtralRealtimeStreamEncoderState
    ) -> MLXArray {
        let sw = config.slidingWindow
        var pieces: [MLXArray] = []
        while state.consumed < upTo {
            let blockEnd = state.blockBase + sw
            let end = min(upTo, blockEnd)
            let block = convOut[state.consumed..<end, 0...]
            // Block-relative positions: RoPE is relative, so this matches the absolute
            // positions offline uses within each independent sw-block.
            let relStart = state.consumed - state.blockBase
            pieces.append(encodeIncremental(block, startPos: relStart, caches: &state.caches))
            state.consumed = end
            if state.consumed == blockEnd {
                state.caches = Array(repeating: nil, count: transformerLayers.count)
                state.blockBase = blockEnd
            }
        }
        if pieces.isEmpty { return convOut[0..<0, 0...] }
        return pieces.count == 1 ? pieces[0] : MLX.concatenated(pieces, axis: 0)
    }
}

public final class VoxtralRealtimeStreamSession {
    /// Text + token ids decoded by a single `step` / `finish` call.
    public struct Delta {
        public let text: String
        public let tokenIds: [Int]
    }

    private let model: VoxtralRealtimeModel
    private let temperature: Float
    private let maxTokens: Int
    private let transcriptionDelayMs: Int?

    // Only the trailing partial token (chunk ended mid-1280-sample-token) is unfrozen.
    private let frozenGuardTokens = 1

    private var realAudio: [Float] = []
    private var encState: VoxtralRealtimeStreamEncoderState
    private var adapterBuf: MLXArray?
    private var decCache: [VoxtralRealtimeDecoderKVCache?]?
    private var lastLogits: MLXArray?
    private var decPos = 0
    private var promptLength = 0
    private var prefilled = false
    private var done = false

    private var generated: [Int] = []
    private var emittedText = ""

    public init(
        model: VoxtralRealtimeModel,
        temperature: Float = 0.0,
        maxTokens: Int = 4096,
        transcriptionDelayMs: Int? = nil
    ) {
        self.model = model
        self.temperature = temperature
        self.maxTokens = maxTokens
        self.transcriptionDelayMs = transcriptionDelayMs
        self.encState = VoxtralRealtimeStreamEncoderState(
            layers: model.encoder.transformerLayers.count
        )
    }

    /// Full transcript decoded so far.
    public var text: String { emittedText }
    /// Token ids decoded so far (EOS stripped).
    public var tokens: [Int] { generated }
    /// Whether the stream has emitted EOS / hit maxTokens.
    public var isFinished: Bool { done }

    /// Ingest a chunk of 16 kHz mono samples; returns the text decoded by this call.
    @discardableResult
    public func step(_ samples: [Float]) -> Delta {
        realAudio.append(contentsOf: samples)
        return advance(final: false)
    }

    @discardableResult
    public func step(_ samples: MLXArray) -> Delta {
        let flat = samples.ndim > 1 ? samples.mean(axis: -1) : samples
        return step(flat.asType(.float32).asArray(Float.self))
    }

    /// Flush the tail: reproduces the offline zero-pad so the final transcript equals
    /// `generate(...)`. Call once after the last `step`.
    @discardableResult
    public func finish() -> Delta {
        advance(final: true)
    }

    private func advance(final: Bool) -> Delta {
        guard !done else { return Delta(text: "", tokenIds: []) }
        guard !realAudio.isEmpty else { return Delta(text: "", tokenIds: []) }

        let ds = model.config.encoderArgs.downsampleFactor
        let audio = MLXArray(realAudio)
        let (convOut, nAudioTotal, pLen) = model.convStemForAudio(
            audio: audio,
            transcriptionDelayMs: transcriptionDelayMs
        )
        promptLength = pLen

        let realRegion = model.config.nLeftPadTokens + model.numAudioTokens(realAudio.count)
        let emitLimit = final ? nAudioTotal : max(0, min(nAudioTotal, realRegion - frozenGuardTokens))
        let convFreeze = min(convOut.shape[0], emitLimit * ds)

        if convFreeze > encState.consumed {
            let newEnc = model.encoder.feedIncremental(convOut, upTo: convFreeze, state: &encState)
            let rows = model.encoder.downsampleAndProject(newEnc)   // multiple-of-ds ⇒ whole rows
            adapterBuf = adapterBuf == nil ? rows : MLX.concatenated([adapterBuf!, rows], axis: 0)
            freezeEncoderState()
        }

        guard let adapter = adapterBuf else {
            Memory.clearCache()
            return Delta(text: "", tokenIds: [])
        }
        prefillIfNeeded(adapter: adapter)
        let delta = decode(adapter: adapter, upTo: min(emitLimit, adapter.shape[0]))

        Memory.clearCache()
        return delta
    }

    /// Materialise the adapter buffer + encoder caches so the lazy graph stays bounded
    /// across chunks (each chunk would otherwise extend one unbroken graph).
    private func freezeEncoderState() {
        var arrays: [MLXArray] = []
        if let adapterBuf { arrays.append(adapterBuf) }
        for cache in encState.caches {
            if let cache { arrays.append(cache.keys); arrays.append(cache.values) }
        }
        if !arrays.isEmpty { MLX.eval(arrays) }
    }

    private func prefillIfNeeded(adapter: MLXArray) {
        guard !prefilled, adapter.shape[0] >= promptLength else { return }

        let nLeft = model.config.nLeftPadTokens
        let nDelay = promptLength - 1 - nLeft
        let promptIds = [model.config.bosTokenId]
            + Array(repeating: model.config.streamingPadTokenId, count: nLeft + nDelay)
        let promptIdsMX = MLXArray(promptIds.map(Int32.init))
        let promptTextEmbeds = model.decoder.embedTokens(promptIdsMX)

        let prefixEmbeds = adapter[0..<promptLength, 0...] + promptTextEmbeds
        let prefill = model.decoder(prefixEmbeds, startPos: 0, cache: nil)
        lastLogits = model.decoder.logits(prefill.0[prefill.0.shape[0] - 1])
        decCache = prefill.1
        decPos = promptLength
        prefilled = true
        MLX.eval(lastLogits!)
    }

    private func decode(adapter: MLXArray, upTo emitLimit: Int) -> Delta {
        guard prefilled else { return Delta(text: "", tokenIds: []) }

        var newIds: [Int] = []
        // Mirrors the offline `generate` loop exactly (append → check → pop trailing
        // EOS) so the streamed token stream is identical at temperature 0.
        while decPos < emitLimit {
            guard let logits = lastLogits else { break }
            let token = model.sample(logits: logits, temperature: temperature)
            generated.append(token)

            if token == model.config.eosTokenId || generated.count > maxTokens {
                done = true
                if generated.last == model.config.eosTokenId { generated.removeLast() }
                break
            }
            newIds.append(token)

            let tokenEmbed = model.decoder.embedToken(tokenId: token)
            let inputEmbed = decPos < adapter.shape[0]
                ? adapter[decPos] + tokenEmbed
                : tokenEmbed
            let next = model.decoder(
                inputEmbed.expandedDimensions(axis: 0),
                startPos: decPos,
                cache: decCache
            )
            decCache = next.1
            lastLogits = model.decoder.logits(next.0[0])
            decPos += 1
            MLX.eval(lastLogits!)
        }

        let textSoFar = model.decodeStreaming(generated)
        let delta: String
        if textSoFar.hasPrefix(emittedText) {
            delta = String(textSoFar.dropFirst(emittedText.count))
        } else {
            delta = textSoFar
        }
        emittedText = textSoFar
        return Delta(text: delta, tokenIds: newIds)
    }
}

public extension VoxtralRealtimeModel {
    /// Create an online streaming session. Feed audio with `step(_:)`, then `finish()`.
    func makeStreamSession(
        temperature: Float = 0.0,
        maxTokens: Int = 4096,
        transcriptionDelayMs: Int? = nil
    ) -> VoxtralRealtimeStreamSession {
        VoxtralRealtimeStreamSession(
            model: self,
            temperature: temperature,
            maxTokens: maxTokens,
            transcriptionDelayMs: transcriptionDelayMs
        )
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

        let session = makeStreamSession(
            temperature: generationParameters.temperature,
            maxTokens: generationParameters.maxTokens
        )
        let start = CFAbsoluteTimeGetCurrent()

        func emit(_ delta: VoxtralRealtimeStreamSession.Delta) {
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
            language: generationParameters.language,
            generationTokens: tokenCount,
            totalTokens: tokenCount,
            generationTps: totalTime > 0 ? Double(tokenCount) / totalTime : 0,
            totalTime: totalTime
        )
    }
}
