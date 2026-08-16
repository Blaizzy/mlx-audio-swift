//  Streaming-vs-offline equivalence for the Voxtral Realtime incremental front end:
//  fed the same audio in arbitrary chunk sizes, it must produce the same mel frames,
//  the same conv rows, and (at temperature 0) the same tokens as the offline
//  one-shot pipeline. Token comparisons are exact. Mel/conv comparisons allow a
//  tiny epsilon: the streamed slabs present bit-identical sample windows, but MLX
//  kernels are not guaranteed per-row deterministic across batch shapes (CI's
//  virtualized Metal shows ~2e-4 log-mel drift that hardware Metal does not).
//
//  Run:
//    xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
//      -only-testing:'MLXAudioTests/VoxtralRealtimeStreamingFrontEndTests' \
//      CODE_SIGNING_ALLOWED=NO

import Foundation
import Testing
import MLX
import MLXNN

@testable import MLXAudioSTT

struct VoxtralRealtimeStreamingFrontEndTests {
    private static let samplesPerToken = 1280

    /// Irregular feed pattern (samples): 100 ms, odd runt, 1.7 s, single sample,
    /// odd runt, 480 ms — cycled until the audio is exhausted.
    private static let chunkSizes = [1600, 173, 27200, 1, 999, 7680]

    /// Deterministic sine sweep + low tone, amplitude < 1.
    private static func sweep(_ count: Int) -> [Float] {
        (0..<count).map { i in
            let t = Float(i) / 16000
            return 0.5 * sin(2 * .pi * (200 + 1500 * t) * t) + 0.25 * sin(2 * .pi * 45 * t)
        }
    }

    private static func chunked(_ samples: [Float]) -> [[Float]] {
        var chunks: [[Float]] = []
        var idx = 0
        var sizeIdx = 0
        while idx < samples.count {
            let end = min(idx + chunkSizes[sizeIdx % chunkSizes.count], samples.count)
            chunks.append(Array(samples[idx..<end]))
            idx = end
            sizeIdx += 1
        }
        return chunks
    }

    /// The `padAudioStreaming` zero pads around the real audio (left tokens,
    /// token alignment, right tokens).
    private static func pads(real: Int, leftTokens: Int, rightTokens: Int) -> (left: Int, right: Int) {
        let align = (samplesPerToken - real % samplesPerToken) % samplesPerToken
        return (leftTokens * samplesPerToken, align + rightTokens * samplesPerToken)
    }

    /// Offline reference: the exact `prepareMel` pipeline (zero-pad + one-shot mel).
    private static func offlineMel(
        real: [Float], leftTokens: Int, rightTokens: Int, filters: MLXArray
    ) -> MLXArray {
        let (left, right) = pads(real: real.count, leftTokens: leftTokens, rightTokens: rightTokens)
        var stream = [Float](repeating: 0, count: left)
        stream += real
        stream += [Float](repeating: 0, count: right)
        return VoxtralRealtimeAudio.computeMelSpectrogram(
            audio: MLXArray(stream), melFilters: filters
        )
    }

    /// Incremental path: feed irregular chunks, then the session's finish() pad.
    /// Returns the per-step mel columns (empty steps included).
    private static func streamedMelSteps(
        real: [Float], leftTokens: Int, rightTokens: Int, filters: MLXArray
    ) -> [MLXArray] {
        let (left, right) = pads(real: real.count, leftTokens: leftTokens, rightTokens: rightTokens)
        var mel = VoxtralRealtimeMelStream(leftPadSamples: left, melFilters: filters)
        var steps = chunked(real).map { mel.append($0) }
        steps.append(mel.append([Float](repeating: 0, count: right + mel.finishTailPadCount)))
        return steps
    }

    private static func maxAbsDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        MLX.abs(a - b).max().item(Float.self)
    }

    // See the header: kernel drift across batch shapes, not algorithmic error.
    private static let melTolerance: Float = 1e-3
    private static let convTolerance: Float = 1e-4

    @Test func melStreamMatchesOfflineAcrossIrregularChunks() {
        let filters = VoxtralRealtimeAudio.computeMelFilters().asType(.float32)
        let real = Self.sweep(41_337)

        let offline = Self.offlineMel(real: real, leftTokens: 32, rightTokens: 13, filters: filters)
        let steps = Self.streamedMelSteps(real: real, leftTokens: 32, rightTokens: 13, filters: filters)
        let streamed = MLX.concatenated(steps.filter { $0.shape[1] > 0 }, axis: 1)

        #expect(offline.shape == streamed.shape)
        #expect(Self.maxAbsDifference(offline, streamed) <= Self.melTolerance)
    }

    @Test func melStreamEmitsOnlyCompletedWindows() {
        let filters = VoxtralRealtimeAudio.computeMelFilters().asType(.float32)
        var mel = VoxtralRealtimeMelStream(leftPadSamples: 1280, melFilters: filters)
        // Seeded carry is 200 (reflect) + 1280 (left pad) samples ⇒ the zero left
        // pad alone already completes 1 + (1480 - 400) / 160 = 7 windows.
        #expect(mel.append([]).shape[1] == 7)
        #expect(mel.framesEmitted == 7)
        // Carry is now 1480 - 7*160 = 360; +100 samples ⇒ 460 ⇒ one more window.
        #expect(mel.append([Float](repeating: 0.25, count: 100)).shape[1] == 1)
        #expect(mel.framesEmitted == 8)
        // 59 more: carry 359 < 400 ⇒ nothing new.
        #expect(mel.append([Float](repeating: 0.25, count: 59)).shape[1] == 0)
        // 41 more completes exactly one window (carry 400).
        #expect(mel.append([Float](repeating: 0.25, count: 41)).shape[1] == 1)
        #expect(mel.framesEmitted == 9)
    }

    @Test func convStemStepMatchesOfflineAcrossIrregularChunks() {
        MLXRandom.seed(42)
        let config = VoxtralRealtimeEncoderConfig(
            dim: 32, nLayers: 0, nHeads: 2, headDim: 8, hiddenDim: 64, nKvHeads: 2
        )
        let encoder = VoxtralRealtimeAudioEncoder(config, decoderDim: 16)
        let filters = VoxtralRealtimeAudio.computeMelFilters().asType(.float32)
        let real = Self.sweep(41_337)

        let offlineRows = encoder.convStem(
            Self.offlineMel(real: real, leftTokens: 32, rightTokens: 13, filters: filters)
        )
        // The streaming pipeline only sees whole-token streams, for which
        // `convStem`'s leading `% downsampleFactor` truncation is a no-op.
        #expect(offlineRows.shape[0] % config.downsampleFactor == 0)

        var state = VoxtralRealtimeConvStemState()
        var pieces: [MLXArray] = []
        for mel in Self.streamedMelSteps(real: real, leftTokens: 32, rightTokens: 13, filters: filters) {
            let rows = encoder.convStemStep(mel, state: &state)
            if rows.shape[0] > 0 { pieces.append(rows) }
        }
        let streamedRows = MLX.concatenated(pieces, axis: 0)

        #expect(offlineRows.shape == streamedRows.shape)
        #expect(Self.maxAbsDifference(offlineRows, streamedRows) <= Self.convTolerance)
    }

    /// End to end at temperature 0 on a tiny random-weight fixture: the streamed
    /// token ids and transcript must equal the offline `generate(...)` exactly,
    /// under irregular (non-token-aligned) feeding.
    @Test func streamSessionMatchesOfflineOnRandomFixture() throws {
        let fixtureDir = try Self.makeRandomFixture()
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let model = try VoxtralRealtimeModel.fromDirectory(fixtureDir)
        let samples = Self.sweep(20_000)
        let params = STTGenerateParameters(maxTokens: 16, temperature: 0.0)

        let offline = model.generate(audio: MLXArray(samples), generationParameters: params)

        let session = model.makeStreamSession(maxTokens: 16)
        for chunk in Self.chunked(samples) {
            _ = session.step(chunk)
        }
        _ = session.finish()

        #expect(session.text == offline.text)
        #expect(session.tokens.count == offline.generationTokens)
    }

    /// Degenerate feed: the whole utterance in a single step(), then finish().
    @Test func singleGiantChunkMatchesOffline() throws {
        let fixtureDir = try Self.makeRandomFixture()
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let model = try VoxtralRealtimeModel.fromDirectory(fixtureDir)
        let samples = Self.sweep(20_000)
        let params = STTGenerateParameters(maxTokens: 16, temperature: 0.0)
        let offline = model.generate(audio: MLXArray(samples), generationParameters: params)

        let session = model.makeStreamSession(maxTokens: 16)
        _ = session.step(samples)
        _ = session.finish()

        #expect(session.text == offline.text)
        #expect(session.tokens.count == offline.generationTokens)
    }

    /// finish() seals the stream: later step() calls change nothing.
    @Test func stepAfterFinishIsIgnored() throws {
        let fixtureDir = try Self.makeRandomFixture()
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let model = try VoxtralRealtimeModel.fromDirectory(fixtureDir)
        let session = model.makeStreamSession(maxTokens: 16)
        _ = session.step(Self.sweep(20_000))
        _ = session.finish()

        let textBefore = session.text
        let tokensBefore = session.tokens
        let delta = session.step(Self.sweep(5_000))

        #expect(delta.text.isEmpty)
        #expect(delta.tokenIds.isEmpty)
        #expect(session.text == textBefore)
        #expect(session.tokens == tokensBefore)
    }

    /// A second finish() must not re-append the final pad or decode further.
    @Test func doubleFinishIsANoOp() throws {
        let fixtureDir = try Self.makeRandomFixture()
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let model = try VoxtralRealtimeModel.fromDirectory(fixtureDir)
        let session = model.makeStreamSession(maxTokens: 16)
        _ = session.step(Self.sweep(20_000))
        _ = session.finish()

        let textBefore = session.text
        let tokensBefore = session.tokens
        let delta = session.finish()

        #expect(delta.text.isEmpty)
        #expect(delta.tokenIds.isEmpty)
        #expect(session.text == textBefore)
        #expect(session.tokens == tokensBefore)
    }

    /// Non-zero transcription delay changes the prompt length and the right pad;
    /// the streamed result must still equal offline exactly.
    @Test func nonZeroTranscriptionDelayMatchesOffline() throws {
        let fixtureDir = try Self.makeRandomFixture(transcriptionDelayMs: 960)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let model = try VoxtralRealtimeModel.fromDirectory(fixtureDir)
        let samples = Self.sweep(20_000)
        let params = STTGenerateParameters(maxTokens: 16, temperature: 0.0)
        let offline = model.generate(audio: MLXArray(samples), generationParameters: params)

        let session = model.makeStreamSession(maxTokens: 16, transcriptionDelayMs: 960)
        for chunk in Self.chunked(samples) {
            _ = session.step(chunk)
        }
        _ = session.finish()

        #expect(session.text == offline.text)
        #expect(session.tokens.count == offline.generationTokens)
    }

    /// With one real encoder transformer layer and enough audio that the conv rows
    /// cross the 64-frame sliding-window boundary twice, the incremental
    /// cache-reset path (`feedIncremental`) must reproduce `encodeChunked` exactly.
    @Test func slidingWindowCrossingMatchesOffline() throws {
        let fixtureDir = try Self.makeRandomFixture(encoderLayers: 1)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let model = try VoxtralRealtimeModel.fromDirectory(fixtureDir)
        // 30 000 samples ⇒ (1 + 24 + 11) tokens ⇒ 144 conv rows > 2 × 64.
        let samples = Self.sweep(30_000)
        let params = STTGenerateParameters(maxTokens: 32, temperature: 0.0)
        let offline = model.generate(audio: MLXArray(samples), generationParameters: params)

        let session = model.makeStreamSession(maxTokens: 32)
        for chunk in Self.chunked(samples) {
            _ = session.step(chunk)
        }
        _ = session.finish()

        #expect(session.text == offline.text)
        #expect(session.tokens.count == offline.generationTokens)
    }

    /// finish() with no audio at all must still transcribe the zero-padded empty
    /// stream, exactly like `generate` over an empty buffer.
    @Test func emptyAudioFinishMatchesOffline() throws {
        let fixtureDir = try Self.makeRandomFixture()
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let model = try VoxtralRealtimeModel.fromDirectory(fixtureDir)
        let params = STTGenerateParameters(maxTokens: 16, temperature: 0.0)
        let offline = model.generate(audio: MLXArray([Float]()), generationParameters: params)

        let session = model.makeStreamSession(maxTokens: 16)
        _ = session.finish()

        #expect(session.text == offline.text)
        #expect(session.tokens.count == offline.generationTokens)
    }

    /// Like `VoxtralRealtimeSTTTests.makeEOSFixture`, but with seeded random
    /// weights so the decoded tokens actually depend on the conv-stem rows.
    /// `encoderLayers` may be 0 (front end only) or 1 (exercises the encoder
    /// transformer + sliding-window cache path too).
    private static func makeRandomFixture(
        transcriptionDelayMs: Int = 0,
        encoderLayers: Int = 0
    ) throws -> URL {
        precondition((0...1).contains(encoderLayers))
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxtral-random-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)

        let configJSON = """
        {
          "model_type": "voxtral_realtime",
          "encoder_args": {
            "dim": 16, "n_layers": \(encoderLayers), "n_heads": 2, "head_dim": 8, "hidden_dim": 32,
            "n_kv_heads": 2, "norm_eps": 1e-5, "rope_theta": 1000000,
            "sliding_window": 64, "causal": true, "use_biases": true, "downsample_factor": 4
          },
          "decoder": {
            "dim": 16, "n_layers": 0, "n_heads": 2, "n_kv_heads": 2, "head_dim": 8,
            "hidden_dim": 32, "vocab_size": 8, "norm_eps": 1e-5, "rope_theta": 1000000,
            "sliding_window": 64, "tied_embeddings": true,
            "ada_rms_norm_t_cond": false, "ada_rms_norm_t_cond_dim": 4
          },
          "audio_encoding_args": {
            "sampling_rate": 16000, "frame_rate": 12.5, "num_mel_bins": 128,
            "hop_length": 160, "window_size": 400, "global_log_mel_max": 1.5
          },
          "transcription_delay_ms": \(transcriptionDelayMs), "bos_token_id": 1, "eos_token_id": 0,
          "streaming_pad_token_id": 2, "n_left_pad_tokens": 1
        }
        """
        try configJSON.write(
            to: fixtureDir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)

        let tekkenJSON = """
        {
          "vocab": [
            {"token_bytes":"YQ=="},{"token_bytes":"Yg=="},{"token_bytes":"Yw=="},
            {"token_bytes":"ZA=="},{"token_bytes":"ZQ=="},{"token_bytes":"Zg=="},
            {"token_bytes":"Zw=="},{"token_bytes":"aA=="}
          ],
          "config":{"default_num_special_tokens":0},"special_tokens":[]
        }
        """
        try tekkenJSON.write(
            to: fixtureDir.appendingPathComponent("tekken.json"), atomically: true, encoding: .utf8)

        MLXRandom.seed(7)
        var weights: [String: MLXArray] = [
            "encoder.conv_layers_0_conv.conv.weight": MLXRandom.normal([16, 3, 128]) * 0.2,
            "encoder.conv_layers_0_conv.conv.bias": MLXRandom.normal([16]) * 0.1,
            "encoder.conv_layers_1_conv.conv.weight": MLXRandom.normal([16, 3, 16]) * 0.2,
            "encoder.conv_layers_1_conv.conv.bias": MLXRandom.normal([16]) * 0.1,
            "encoder.transformer_norm.weight": MLXArray.ones([16], type: Float.self),
            "encoder.audio_language_projection_0.weight": MLXRandom.normal([16, 64]) * 0.2,
            "encoder.audio_language_projection_2.weight": MLXRandom.normal([16, 16]) * 0.2,
            "decoder.tok_embeddings.weight": MLXRandom.normal([8, 16]) * 0.5,
            "decoder.norm.weight": MLXArray.ones([16], type: Float.self),
        ]
        if encoderLayers == 1 {
            let layer = "encoder.transformer_layers.0"
            weights["\(layer).attention_norm.weight"] = MLXArray.ones([16], type: Float.self)
            weights["\(layer).attention.wq.weight"] = MLXRandom.normal([16, 16]) * 0.2
            weights["\(layer).attention.wq.bias"] = MLXRandom.normal([16]) * 0.1
            weights["\(layer).attention.wk.weight"] = MLXRandom.normal([16, 16]) * 0.2
            weights["\(layer).attention.wv.weight"] = MLXRandom.normal([16, 16]) * 0.2
            weights["\(layer).attention.wv.bias"] = MLXRandom.normal([16]) * 0.1
            weights["\(layer).attention.wo.weight"] = MLXRandom.normal([16, 16]) * 0.2
            weights["\(layer).attention.wo.bias"] = MLXRandom.normal([16]) * 0.1
            weights["\(layer).ffn_norm.weight"] = MLXArray.ones([16], type: Float.self)
            weights["\(layer).feed_forward_w1.weight"] = MLXRandom.normal([32, 16]) * 0.2
            weights["\(layer).feed_forward_w3.weight"] = MLXRandom.normal([32, 16]) * 0.2
            weights["\(layer).feed_forward_w2.weight"] = MLXRandom.normal([16, 32]) * 0.2
            weights["\(layer).feed_forward_w2.bias"] = MLXRandom.normal([16]) * 0.1
        }
        try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))
        return fixtureDir
    }
}
