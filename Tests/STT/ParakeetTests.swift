import Foundation
import Testing
import MLX
import MLXNN

@testable import MLXAudioCore
@testable import MLXAudioSTT

struct ParakeetSTTTests {

    @Test func variantResolutionAndTypedParsing() throws {
        let tdtJSON = """
        {
          "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
          "model_defaults": {"tdt_durations": [0, 1, 2]},
          "preprocessor": {
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hann",
            "features": 80,
            "n_fft": 512,
            "dither": 0.0
          },
          "encoder": {
            "feat_in": 80,
            "n_layers": 2,
            "d_model": 32,
            "n_heads": 4,
            "ff_expansion_factor": 2,
            "subsampling_factor": 4,
            "self_attention_model": "rel_pos",
            "subsampling": "dw_striding",
            "conv_kernel_size": 15,
            "subsampling_conv_channels": 32,
            "pos_emb_max_len": 512
          },
          "decoder": {
            "blank_as_pad": true,
            "vocab_size": 4,
            "prednet": {
              "pred_hidden": 32,
              "pred_rnn_layers": 1
            }
          },
          "joint": {
            "num_classes": 4,
            "vocabulary": ["▁", "a", "b", "."],
            "jointnet": {
              "joint_hidden": 32,
              "activation": "relu",
              "encoder_hidden": 32,
              "pred_hidden": 32
            }
          },
          "decoding": {
            "model_type": "tdt",
            "durations": [0, 1, 2],
            "greedy": {"max_symbols": 10}
          }
        }
        """

        let raw = try JSONDecoder().decode(ParakeetRawConfig.self, from: Data(tdtJSON.utf8))
        let variant = try ParakeetVariantResolver.resolve(raw)
        #expect(variant == .tdt)

        let typed = try ParakeetConfigParser.parseTDT(raw)
        #expect(typed.preprocessor.sampleRate == 16000)
        #expect(typed.encoder.subsampling == "dw_striding")
        #expect(typed.decoding.durations == [0, 1, 2])
        #expect(typed.decoding.greedy?.maxSymbols == 10)
    }

    @Test func ctcVariantResolution() throws {
        let ctcJSON = """
        {
          "target": "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE",
          "preprocessor": {
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hann",
            "features": 80,
            "n_fft": 512,
            "dither": 0.0
          },
          "encoder": {
            "feat_in": 80,
            "n_layers": 2,
            "d_model": 32,
            "n_heads": 4,
            "ff_expansion_factor": 2,
            "subsampling_factor": 4,
            "self_attention_model": "rel_pos",
            "subsampling": "dw_striding",
            "conv_kernel_size": 15,
            "subsampling_conv_channels": 32,
            "pos_emb_max_len": 512
          },
          "decoder": {
            "feat_in": 32,
            "num_classes": 4,
            "vocabulary": ["▁", "a", "b", "."]
          },
          "decoding": {"greedy": {"max_symbols": 8}}
        }
        """

        let raw = try JSONDecoder().decode(ParakeetRawConfig.self, from: Data(ctcJSON.utf8))
        #expect(try ParakeetVariantResolver.resolve(raw) == .ctc)
        let typed = try ParakeetConfigParser.parseCTC(raw)
        #expect(typed.decoder.featIn == 32)
        #expect(typed.decoder.vocabulary.count == 4)
    }

    @Test func tokenizerDecodesSentencePieceMarker() {
        let vocab = ["▁", "h", "e", "l", "o", "."]
        let text = ParakeetTokenizer.decode(tokens: [0, 1, 2, 3, 3, 4, 5], vocabulary: vocab)
        #expect(text == " hello.")
    }

    @Test func alignmentSentenceAndMergeUtilities() throws {
        let tokens: [ParakeetAlignedToken] = [
            .init(id: 1, text: "Hi", start: 0.0, duration: 0.2),
            .init(id: 2, text: ".", start: 0.2, duration: 0.1),
            .init(id: 3, text: " Next", start: 0.5, duration: 0.2),
            .init(id: 4, text: "!", start: 0.7, duration: 0.1),
        ]
        let sentences = ParakeetAlignment.tokensToSentences(tokens)
        #expect(sentences.count == 2)
        #expect(sentences[0].text == "Hi.")
        #expect(sentences[1].text == " Next!")

        let a: [ParakeetAlignedToken] = [
            .init(id: 1, text: " a", start: 0.0, duration: 0.2),
            .init(id: 2, text: " b", start: 0.2, duration: 0.2),
            .init(id: 3, text: " c", start: 0.4, duration: 0.2),
        ]
        let b: [ParakeetAlignedToken] = [
            .init(id: 2, text: " b", start: 0.21, duration: 0.2),
            .init(id: 3, text: " c", start: 0.41, duration: 0.2),
            .init(id: 4, text: " d", start: 0.61, duration: 0.2),
        ]
        let mergedContiguous = try ParakeetAlignment.mergeLongestContiguous(a, b, overlapDuration: 0.6)
        #expect(mergedContiguous.map(\.id) == [1, 2, 3, 4])

        let mergedLCS = ParakeetAlignment.mergeLongestCommonSubsequence(a, b, overlapDuration: 0.6)
        #expect(mergedLCS.map(\.id) == [1, 2, 3, 4])
    }

    @Test func melPreprocessingProducesExpectedShape() {
        let config = ParakeetPreprocessConfig(
            sampleRate: 16000,
            normalize: "per_feature",
            windowSize: 0.02,
            windowStride: 0.01,
            window: "hann",
            features: 80,
            nFft: 512,
            dither: 0,
            padTo: 0,
            padValue: 0,
            preemph: 0.97
        )

        let audio = MLXArray(Array(repeating: Float(0.0), count: 16000))
        let mel = ParakeetAudio.logMelSpectrogram(audio, config: config)

        #expect(mel.ndim == 3)
        #expect(mel.shape[0] == 1)
        #expect(mel.shape[2] == 80)
        #expect(mel.shape[1] > 0)
    }

    @Test func deterministicRNNTAndTDTControlFlow() {
        let rnntBlank = 10
        let rnntStep1 = ParakeetDecodingLogic.rnntStep(
            predictedToken: rnntBlank,
            blankToken: rnntBlank,
            time: 5,
            newSymbols: 2,
            maxSymbols: 4
        )
        #expect(rnntStep1.nextTime == 6)
        #expect(rnntStep1.nextNewSymbols == 0)
        #expect(rnntStep1.emittedToken == false)

        let rnntStep2 = ParakeetDecodingLogic.rnntStep(
            predictedToken: 2,
            blankToken: rnntBlank,
            time: 8,
            newSymbols: 3,
            maxSymbols: 4
        )
        #expect(rnntStep2.nextTime == 9)
        #expect(rnntStep2.nextNewSymbols == 0)
        #expect(rnntStep2.emittedToken == true)

        let tdtStep1 = ParakeetDecodingLogic.tdtStep(
            predictedToken: 1,
            blankToken: 5,
            decisionIndex: 1,
            durations: [0, 2, 4],
            time: 10,
            newSymbols: 0,
            maxSymbols: 4
        )
        #expect(tdtStep1.nextTime == 12)
        #expect(tdtStep1.nextNewSymbols == 0)
        #expect(tdtStep1.jump == 2)
        #expect(tdtStep1.emittedToken == true)

        let tdtStep2 = ParakeetDecodingLogic.tdtStep(
            predictedToken: 1,
            blankToken: 5,
            decisionIndex: 0,
            durations: [0, 2, 4],
            time: 3,
            newSymbols: 3,
            maxSymbols: 4
        )
        #expect(tdtStep2.nextTime == 4)  // zero-duration + max_symbols fallback
        #expect(tdtStep2.nextNewSymbols == 0)
        #expect(tdtStep2.jump == 0)
    }

    @Test func deterministicCTCCollapseSpans() {
        let spans = ParakeetDecodingLogic.ctcSpans(
            bestTokens: [5, 5, 9, 2, 2, 9, 2, 3, 3],
            blankToken: 9
        )
        #expect(spans == [
            .init(token: 5, startFrame: 0, endFrame: 2),
            .init(token: 2, startFrame: 3, endFrame: 5),
            .init(token: 2, startFrame: 6, endFrame: 7),
            .init(token: 3, startFrame: 7, endFrame: 9),
        ])
    }

    @Test func fromDirectorySmokeTestWithFixtureConfigAndWeights() async throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("parakeet-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)

        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let configJSON = """
        {
          "target": "nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE",
          "preprocessor": {
            "sample_rate": 16000,
            "normalize": "per_feature",
            "window_size": 0.02,
            "window_stride": 0.01,
            "window": "hann",
            "features": 80,
            "n_fft": 512,
            "dither": 0.0
          },
          "encoder": {
            "feat_in": 80,
            "n_layers": 0,
            "d_model": 16,
            "n_heads": 4,
            "ff_expansion_factor": 2,
            "subsampling_factor": 2,
            "self_attention_model": "abs_pos",
            "subsampling": "dw_striding",
            "conv_kernel_size": 15,
            "subsampling_conv_channels": 16,
            "pos_emb_max_len": 128
          },
          "decoder": {
            "feat_in": 16,
            "num_classes": 4,
            "vocabulary": ["▁", "a", "b", "."]
          },
          "decoding": {"greedy": {"max_symbols": 8}}
        }
        """
        try configJSON.write(
            to: fixtureDir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )

        let weights: [String: MLXArray] = [
            "encoder.pre_encode.conv0.weight": MLXArray.zeros([16, 3, 3, 1], type: Float.self),
            "encoder.pre_encode.conv0.bias": MLXArray.zeros([16], type: Float.self),
            "encoder.pre_encode.out.weight": MLXArray.zeros([16, 640], type: Float.self),
            "encoder.pre_encode.out.bias": MLXArray.zeros([16], type: Float.self),
            "decoder.decoder_layers.0.weight": MLXArray.zeros([5, 1, 16], type: Float.self),
            "decoder.decoder_layers.0.bias": MLXArray.zeros([5], type: Float.self),
        ]
        try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

        let model = try ParakeetModel.fromDirectory(fixtureDir)
        let audio = MLXArray(Array(repeating: Float(0), count: 3200))
        let output = model.generate(audio: audio)

        #expect(model.variant == .ctc)
        #expect(model.vocabulary.count == 4)
        #expect(output.text.count >= 0)
    }
}
