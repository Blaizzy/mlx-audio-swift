import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioCore
@testable import MLXAudioSTT

/// Nemotron Unified (buffered streaming) on tiny random models, small enough to run
/// on any Metal device, so CI runs them without the MLX runtime-test opt-in. The NeMo
/// token-parity test on real weights is `NemotronUnifiedNeMoParityTests`.
struct NemotronUnifiedTests {
    private func unifiedConfigJSON() -> String {
        """
        {
          "model_type": "nemotron_asr",
          "preprocessor": {
            "sample_rate": 16000,
            "features": 16,
            "n_fft": 64,
            "window_size": 0.004,
            "window_stride": 0.002,
            "window": "hann",
            "preemph": 0.97,
            "dither": 0.0,
            "normalize": "per_feature"
          },
          "encoder": {
            "feat_in": 16,
            "n_layers": 1,
            "d_model": 16,
            "n_heads": 2,
            "ff_expansion_factor": 2,
            "subsampling_factor": 4,
            "subsampling_conv_channels": 4,
            "conv_kernel_size": 3,
            "causal_downsampling": false,
            "conv_context_size": [1, 1],
            "conv_norm_type": "batch_norm",
            "self_attention_model": "rel_pos",
            "att_context_style": "chunked_limited_with_rc",
            "att_context_size": [[-1, -1, -1]],
            "pos_emb_max_len": 64,
            "use_bias": true,
            "xscaling": true
          },
          "prompt": {
            "num_prompts": 4,
            "prompt_hidden": 16,
            "prompt_dictionary": {"en-US": 0, "auto": 1}
          },
          "decoder": {
            "pred_hidden": 8,
            "pred_rnn_layers": 1,
            "vocab_size": 6,
            "blank_as_pad": true
          },
          "joint": {
            "joint_hidden": 8,
            "activation": "relu",
            "encoder_hidden": 16,
            "pred_hidden": 8,
            "num_classes": 6
          },
          "vocabulary": ["<unk>", "<en-US>", "▁hello", "▁world", "!", "a"],
          "default_language": "auto",
          "default_att_context_size": [4, 1, 2],
          "max_symbols": 3
        }
        """
    }

    private func unifiedModel() throws -> NemotronASRModel {
        let config = try JSONDecoder().decode(NemotronASRConfig.self, from: Data(unifiedConfigJSON().utf8))
        let model = NemotronASRModel(config)
        eval(model.parameters())
        model.train(false)
        return model
    }

    @Test func unifiedConfigSelectsBufferedStreaming() throws {
        let config = try JSONDecoder().decode(NemotronASRConfig.self, from: Data(unifiedConfigJSON().utf8))
        #expect(config.encoder.attContextStyle == "chunked_limited_with_rc")
        #expect(config.encoder.convContextSize == .explicit(left: 1, right: 1))
        #expect(config.defaultAttContextSize == [4, 1, 2])
        let cases: [(Int, Int, Int)] = [
            (80, 1, 1), (160, 1, 1), (200, 1, 2), (320, 1, 3), (400, 1, 4), (480, 1, 5),
            (560, 2, 5), (1040, 2, 11), (1120, 7, 7), (1200, 7, 8), (2080, 13, 13), (5000, 13, 13),
        ]
        for (latency, chunk, right) in cases {
            #expect(NemotronASRBufferedContext.forLatency(latency).asArray == [70, chunk, right])
        }
        #expect(NemotronASRBufferedContext.forLatency(1120, left: 25).asArray == [25, 7, 7])
    }

    /// Layouts come from callers, so a bad one is an error, not a trap.
    @Test func aBufferedContextRejectsAnInvalidLayout() throws {
        #expect(throws: NemotronASRStreamingError.invalidBufferedContext(left: 70, chunk: 0, right: 2)) {
            try NemotronASRBufferedContext(left: 70, chunk: 0, right: 2)
        }
        #expect(throws: NemotronASRStreamingError.invalidBufferedContext(left: -1, chunk: 1, right: 0)) {
            try NemotronASRBufferedContext(left: -1, chunk: 1, right: 0)
        }
        #expect(try NemotronASRBufferedContext(left: 0, chunk: 1, right: 0).asArray == [0, 1, 0])
    }

    /// An explicit layout on a cache-aware (3.5-style) checkpoint is an error.
    @Test func anExplicitLayoutOnACacheAwareCheckpointThrows() throws {
        let json = unifiedConfigJSON()
            .replacingOccurrences(of: "\"att_context_style\": \"chunked_limited_with_rc\"", with: "\"att_context_style\": \"chunked_limited\"")
            .replacingOccurrences(of: "\"att_context_size\": [[-1, -1, -1]]", with: "\"att_context_size\": [[4, 1]]")
            .replacingOccurrences(of: "\"default_att_context_size\": [4, 1, 2]", with: "\"default_att_context_size\": [4, 1]")
        let model = NemotronASRModel(try JSONDecoder().decode(NemotronASRConfig.self, from: Data(json.utf8)))
        #expect(!model.isBuffered)
        #expect(throws: NemotronASRStreamingError.notBuffered) {
            try model.makeStreamSession(language: nil, bufferedContext: .forLatency(320))
        }
    }

    @Test func chunkedRCVisibilityMatchesNeMo() {
        // left 2, chunk 2, right 1: frame i sees [chunkStart - 2, chunkStart + 2].
        let visible = NemotronASRAttentionMask.chunkedRCVisibility(seqLen: 7, left: 2, chunk: 2, right: 1)
        let rows = visible.asType(.int32).asArray(Int32.self)
        func row(_ i: Int) -> [Int32] { Array(rows[(i * 7)..<(i * 7 + 7)]) }
        #expect(row(0) == [1, 1, 1, 0, 0, 0, 0])
        #expect(row(3) == [1, 1, 1, 1, 1, 0, 0])
        #expect(row(6) == [0, 0, 0, 0, 1, 1, 1])
    }

    /// NeMo masks padding in the subsampling, the attention keys and the conv, so
    /// frames past the valid length cannot change the valid frames' output.
    @Test func unifiedEncoderIgnoresPaddedFrames() throws {
        let model = try unifiedModel()
        let mel = MLXRandom.normal([1, 40, 16], key: MLXRandom.key(3))
        let noise = MLXRandom.normal([1, 12, 16], key: MLXRandom.key(4))
        let padded = MLX.concatenated([mel, noise], axis: 1)
        let lengths = MLXArray([Int32(40)])
        for context in [[-1, -1, -1], [4, 1, 2]] {
            let (a, la) = model.encoder(mel, lengths: lengths, attContextSize: context)
            let (b, lb) = model.encoder(padded, lengths: lengths, attContextSize: context)
            let valid = Int(la[0].item(Int32.self))
            #expect(valid == Int(lb[0].item(Int32.self)))
            let diff = MLX.abs(a[0..., ..<valid, 0...] - b[0..., ..<valid, 0...]).max().item(Float.self)
            #expect(diff < 1e-4)
        }
    }

    /// A left-0 layout leaves a padded chunk with every key masked. A mask value that
    /// overflows to -inf in float16 turns such rows NaN, and later layers spread the
    /// NaN to valid frames (0 x NaN).
    @Test func float16MaskStaysFiniteWhenAChunkIsAllPadding() throws {
        let model = try unifiedModel()
        let half = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                (key, value.dtype.isFloatingPoint ? value.asType(.float16) : value)
            }
        )
        try model.update(parameters: ModuleParameters.unflattened(half), verify: .noUnusedKeys)
        let mel = MLXRandom.normal([1, 40, 16], key: MLXRandom.key(3))
        let noise = MLXRandom.normal([1, 12, 16], key: MLXRandom.key(4))
        let padded = MLX.concatenated([mel, noise], axis: 1).asType(.float16)
        let (out, lengths) = model.encoder(padded, lengths: MLXArray([Int32(40)]), attContextSize: [0, 2, 0])
        let valid = Int(lengths[0].item(Int32.self))
        #expect(valid > 0)
        // x - x is 0 for finite values and NaN for NaN or infinity.
        let frames = out[0..., ..<valid, 0...].asType(.float32)
        #expect(MLX.abs(frames - frames).sum().item(Float.self) == 0)
    }

    @Test func perFeatureMelNormalizesValidFramesAndZeroesTheTail() throws {
        let config = try JSONDecoder().decode(NemotronASRConfig.self, from: Data(unifiedConfigJSON().utf8)).preprocessor
        let samples = 40 * config.hopLength + 7
        let mel = NemotronASRAudio.logMelSpectrogram(syntheticAudio(samples: samples), config: config)[0]
        #expect(mel.shape[0] == 41)  // centered STFT: samples / hop + 1 frames
        #expect(MLX.abs(mel[40]).max().item(Float.self) == 0)
        let mean = MLX.abs(MLX.mean(mel[..<40], axis: 0)).max().item(Float.self)
        #expect(mean < 1e-4)
    }

    /// A stop with under one hop of new audio leaves no frame to decode.
    @Test func unifiedSessionStopsCleanlyOnTinyInput() async throws {
        let model = try unifiedModel()
        let frame = model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength
        for length in [1, 16, 31, 32, 33, 7 * frame + 5] {
            let audio = syntheticAudio(samples: length)
            let whole = try await wholeStreamText(model, audio)
            let (text, _) = unifiedSessionText(model, audio, feed: 61)
            #expect(text == whole)
        }
    }

    @Test func unifiedSessionTakesAnExplicitContext() throws {
        let model = try unifiedModel()
        let samples = syntheticAudio(samples: 20 * 128 + 70).asArray(Float.self)
        let context = try NemotronASRBufferedContext(left: 2, chunk: 2, right: 1)
        func tokens(feed: Int) throws -> [Int] {
            let session = try model.makeStreamSession(language: "en-US", bufferedContext: context)
            var i = 0
            while i < samples.count {
                let e = min(i + feed, samples.count)
                _ = session.step(Array(samples[i..<e]))
                i = e
            }
            _ = session.finish()
            return session.tokens
        }
        #expect(try tokens(feed: 61) == tokens(feed: 1000))
    }

    @Test func promptLanguageResolvesLikeNeMo() throws {
        let model = try unifiedModel()  // prompts: en-US 0, auto 1; default auto
        #expect(model.resolvePromptIndex(nil) == 1)
        #expect(model.resolvePromptIndex("en-US") == 0)
        #expect(model.resolvePromptIndex("en") == 0)
        #expect(model.resolvePromptIndex("EN") == 0)
        #expect(model.resolvePromptIndex("fr") == 1)  // unsupported: default, with a warning
    }

    /// Full-size windows run a compiled encoder; it must agree with the plain one.
    @Test func unifiedCompiledFullWindowsMatchUncompiled() throws {
        let model = try unifiedModel()
        let samples = syntheticAudio(samples: 30 * 128 + 50).asArray(Float.self)
        func chunks(compiled: Bool) -> [MLXArray] {
            let encoder = NemotronASRBufferedEncoder(
                model: model, context: model.defaultBufferedContext, language: "en-US"
            )
            encoder.compilesFullWindows = compiled
            var out: [MLXArray] = []
            var i = 0
            while i < samples.count {
                let e = min(i + 300, samples.count)
                out += encoder.push(Array(samples[i..<e]), final: e == samples.count)
                i = e
            }
            return out
        }
        let plain = chunks(compiled: false)
        let compiled = chunks(compiled: true)
        #expect(plain.count == compiled.count)
        for (a, b) in zip(plain, compiled) {
            #expect(MLX.abs(a - b).max().item(Float.self) < 1e-4)
        }
    }

    private func unifiedSessionText(_ model: NemotronASRModel, _ audio: MLXArray, feed: Int) -> (String, [Int]) {
        let samples = audio.asArray(Float.self)
        let session = model.makeStreamSession(language: "en-US")
        var i = 0
        while i < samples.count {
            let e = min(i + feed, samples.count)
            _ = session.step(Array(samples[i..<e]))
            i = e
        }
        _ = session.finish()
        return (session.text, session.tokens)
    }

    /// The buffered session segments a live stream as NeMo segments the whole file,
    /// whatever the feed size, so it matches `generateStream` token for token.
    @Test func unifiedSessionMatchesGenerateStreamAtAnyFeedSize() async throws {
        let model = try unifiedModel()
        let frame = model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength
        // Lengths ending mid-frame, on a chunk boundary, and on chunk + right context.
        for length in [9 * frame + 37, 12 * frame, 15 * frame, 2 * frame + 1] {
            let audio = syntheticAudio(samples: length)
            let whole = try await wholeStreamText(model, audio)
            let (fine, fineTokens) = unifiedSessionText(model, audio, feed: 61)
            let (coarse, coarseTokens) = unifiedSessionText(model, audio, feed: 5 * frame)
            #expect(fine == whole)
            #expect(fineTokens == coarseTokens)
            #expect(coarse == whole)
        }
    }


    /// Steps `samples` through `session`, `feed` samples at a time. After every step the
    /// text only grows, by exactly the returned delta, as a realtime server relies on.
    private func appendOnlyTokens(
        _ session: NemotronASRStreamSession, _ samples: [Float], feed: Int
    ) -> [Int] {
        var joined = ""
        var i = 0
        while i < samples.count {
            let e = min(i + feed, samples.count)
            let before = session.text
            joined += session.step(Array(samples[i..<e])).text
            #expect(session.text.hasPrefix(before))
            #expect(joined == session.text)
            i = e
        }
        joined += session.finish().text
        #expect(joined == session.text)
        return session.tokens
    }

    /// Chunks of 1, 2 and 7 frames, past the startup ramp into full (compiled) windows.
    @Test func unifiedSessionTextOnlyGrowsAtEveryChunkSize() throws {
        let model = try unifiedModel()
        let frame = model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength
        let samples = syntheticAudio(samples: 40 * frame + 77).asArray(Float.self)
        for latency in [160, 320, 560, 1120] {
            let runs = [61, frame, 23 * frame].map { feed in
                appendOnlyTokens(
                    model.makeStreamSession(language: "en-US", chunkMs: latency), samples, feed: feed
                )
            }
            #expect(runs[1] == runs[0], "latency \(latency) ms")
            #expect(runs[2] == runs[0], "latency \(latency) ms")
        }
    }

    /// The compiled encoders are cached on the model by shape. Encoders with different
    /// layouts and languages, pushed alternately, must each give what they give alone.
    @Test func interleavedUnifiedEncodersDoNotShareState() throws {
        let model = try unifiedModel()
        let frame = model.encoderConfig.subsamplingFactor * model.preprocessConfig.hopLength
        let a = syntheticAudio(samples: 40 * frame + 50).asArray(Float.self)
        let streams: [(context: NemotronASRBufferedContext, language: String, samples: [Float])] = [
            (try NemotronASRBufferedContext(left: 4, chunk: 2, right: 2), "en-US", a),
            (try NemotronASRBufferedContext(left: 4, chunk: 7, right: 7), "auto", Array(a.reversed())),
        ]
        func encoder(_ s: Int) -> NemotronASRBufferedEncoder {
            NemotronASRBufferedEncoder(model: model, context: streams[s].context, language: streams[s].language)
        }
        func push(_ e: NemotronASRBufferedEncoder, _ s: Int, at i: Int) -> [MLXArray] {
            let end = min(i + frame, streams[s].samples.count)
            return e.push(Array(streams[s].samples[i..<end]), final: end == streams[s].samples.count)
        }
        let alone = (0..<2).map { s -> [MLXArray] in
            let e = encoder(s)
            return stride(from: 0, to: a.count, by: frame).flatMap { push(e, s, at: $0) }
        }
        let encoders = [encoder(0), encoder(1)]
        var together: [[MLXArray]] = [[], []]
        for i in stride(from: 0, to: a.count, by: frame) {
            for s in 0..<2 { together[s] += push(encoders[s], s, at: i) }
        }
        for s in 0..<2 {
            #expect(!alone[s].isEmpty)
            #expect(together[s].count == alone[s].count)
            for (x, y) in zip(together[s], alone[s]) {
                #expect(MLX.abs(x - y).max().item(Float.self) < 1e-5)
            }
        }
    }

    private func syntheticAudio(samples: Int) -> MLXArray {
        let values = (0..<samples).map { i in
            0.1 * sin(Float(i) * 0.05) + 0.05 * sin(Float(i) * 0.17)
        }
        return MLXArray(values)
    }

    private func wholeStreamText(_ model: NemotronASRModel, _ audio: MLXArray) async throws -> String {
        var text = ""
        for try await event in model.generateStream(
            audio: audio,
            generationParameters: STTGenerateParameters(language: "en-US")
        ) {
            if case let .result(out) = event { text = out.text }
        }
        return text
    }
}
