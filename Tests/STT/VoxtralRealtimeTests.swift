import Foundation
import Testing
import MLX
import MLXNN

@testable import MLXAudioCore
@testable import MLXAudioSTT

struct VoxtralRealtimeSTTTests {
    @Test func configDecodesNestedAudioEncodingArgs() throws {
        let json = """
        {
          "model_type": "voxtral_realtime",
          "encoder_args": {
            "dim": 1280,
            "audio_encoding_args": {
              "sampling_rate": 16000,
              "num_mel_bins": 128,
              "window_size": 400,
              "hop_length": 160,
              "global_log_mel_max": 1.5
            }
          },
          "decoder": {
            "dim": 3072,
            "n_layers": 2,
            "n_heads": 32,
            "n_kv_heads": 8,
            "head_dim": 128,
            "hidden_dim": 9216,
            "vocab_size": 131072
          }
        }
        """

        let config = try JSONDecoder().decode(VoxtralRealtimeConfig.self, from: Data(json.utf8))
        #expect(config.modelType == "voxtral_realtime")
        #expect(config.audioEncodingArgs.numMelBins == 128)
        #expect(config.decoder.dim == 3072)
        #expect(config.vocabSize == 131072)
    }

    @Test func tokenizerDecodeSkipsSpecialTokens() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxtral-tekken-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let tekkenJSON = """
        {
          "vocab": [
            {"token_bytes": "aGVs"},
            {"token_bytes": "bG8="}
          ],
          "config": {"default_num_special_tokens": 1000},
          "special_tokens": [{"rank": 2}, {"rank": 32}]
        }
        """
        try tekkenJSON.write(
            to: fixtureDir.appendingPathComponent("tekken.json"),
            atomically: true,
            encoding: .utf8
        )

        let tokenizer = try VoxtralRealtimeTokenizer.fromModelDirectory(fixtureDir)
        let text = tokenizer.decode(tokenIds: [1, 1000, 1001, 2, 32])
        #expect(text == "hello")
    }

    @Test func audioMelProducesExpectedShape() {
        let audio = MLXArray(Array(repeating: Float(0), count: 16000))
        let filters = VoxtralRealtimeAudio.computeMelFilters()
        let mel = VoxtralRealtimeAudio.computeMelSpectrogram(
            audio: audio,
            melFilters: filters,
            windowSize: 400,
            hopLength: 160,
            globalLogMelMax: 1.5
        )

        #expect(mel.ndim == 2)
        #expect(mel.shape[0] == 128)
        #expect(mel.shape[1] > 0)
    }

    @Test func sanitizeRemapsAndTransposesConvWeights() {
        let convWeight = MLXArray.zeros([8, 4, 3], type: Float.self)
        let weights: [String: MLXArray] = [
            "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv.weight": convWeight,
            "mm_streams_embeddings.embedding_module.tok_embeddings.weight": MLXArray.zeros([32, 16], type: Float.self),
            "layers.0.feed_forward.w1.weight": MLXArray.zeros([4, 4], type: Float.self),
            "layers.0.ada_rms_norm_t_cond.0.weight": MLXArray.zeros([4, 4], type: Float.self),
        ]

        let sanitized = VoxtralRealtimeModel.sanitize(weights: weights)

        let mappedConv = sanitized["encoder.conv_layers_0_conv.conv.weight"]
        #expect(mappedConv != nil)
        #expect(mappedConv?.shape == [8, 3, 4])
        #expect(sanitized["decoder.tok_embeddings.weight"] != nil)
        #expect(sanitized["decoder.layers.0.feed_forward_w1.weight"] != nil)
        #expect(sanitized["decoder.layers.0.ada_rms_norm_t_cond.ada_down.weight"] != nil)
    }

    @Test func fromDirectoryAndGenerateEOSSmoke() throws {
        let fixtureDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxtral-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fixtureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fixtureDir) }

        let configJSON = """
        {
          "model_type": "voxtral_realtime",
          "encoder_args": {
            "dim": 16,
            "n_layers": 0,
            "n_heads": 2,
            "head_dim": 8,
            "hidden_dim": 32,
            "n_kv_heads": 2,
            "norm_eps": 1e-5,
            "rope_theta": 1000000,
            "sliding_window": 64,
            "causal": true,
            "use_biases": true,
            "downsample_factor": 4
          },
          "decoder": {
            "dim": 16,
            "n_layers": 0,
            "n_heads": 2,
            "n_kv_heads": 2,
            "head_dim": 8,
            "hidden_dim": 32,
            "vocab_size": 8,
            "norm_eps": 1e-5,
            "rope_theta": 1000000,
            "sliding_window": 64,
            "tied_embeddings": true,
            "ada_rms_norm_t_cond": false,
            "ada_rms_norm_t_cond_dim": 4
          },
          "audio_encoding_args": {
            "sampling_rate": 16000,
            "frame_rate": 12.5,
            "num_mel_bins": 128,
            "hop_length": 160,
            "window_size": 400,
            "global_log_mel_max": 1.5
          },
          "transcription_delay_ms": 0,
          "bos_token_id": 1,
          "eos_token_id": 0,
          "streaming_pad_token_id": 2,
          "n_left_pad_tokens": 1
        }
        """
        try configJSON.write(
            to: fixtureDir.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )

        let tekkenJSON = """
        {
          "vocab": [
            {"token_bytes":"YQ=="},
            {"token_bytes":"Yg=="},
            {"token_bytes":"Yw=="},
            {"token_bytes":"ZA=="},
            {"token_bytes":"ZQ=="},
            {"token_bytes":"Zg=="},
            {"token_bytes":"Zw=="},
            {"token_bytes":"aA=="}
          ],
          "config":{"default_num_special_tokens":0},
          "special_tokens":[]
        }
        """
        try tekkenJSON.write(
            to: fixtureDir.appendingPathComponent("tekken.json"),
            atomically: true,
            encoding: .utf8
        )

        // All-zero embeddings force argmax token 0 immediately, matching EOS id above.
        let weights: [String: MLXArray] = [
            "encoder.conv_layers_0_conv.conv.weight": MLXArray.zeros([16, 3, 128], type: Float.self),
            "encoder.conv_layers_0_conv.conv.bias": MLXArray.zeros([16], type: Float.self),
            "encoder.conv_layers_1_conv.conv.weight": MLXArray.zeros([16, 3, 16], type: Float.self),
            "encoder.conv_layers_1_conv.conv.bias": MLXArray.zeros([16], type: Float.self),
            "encoder.transformer_norm.weight": MLXArray.ones([16], type: Float.self),
            "encoder.audio_language_projection_0.weight": MLXArray.zeros([16, 64], type: Float.self),
            "encoder.audio_language_projection_2.weight": MLXArray.zeros([16, 16], type: Float.self),
            "decoder.tok_embeddings.weight": MLXArray.zeros([8, 16], type: Float.self),
            "decoder.norm.weight": MLXArray.ones([16], type: Float.self),
        ]
        try MLX.save(arrays: weights, url: fixtureDir.appendingPathComponent("model.safetensors"))

        let model = try VoxtralRealtimeModel.fromDirectory(fixtureDir)
        let audio = MLXArray(Array(repeating: Float(0), count: 16000))
        let output = model.generate(
            audio: audio,
            generationParameters: STTGenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        #expect(output.promptTokens > 0)
        #expect(output.generationTokens == 0)
        #expect(output.totalTokens == output.promptTokens)
        #expect(output.text == "")
    }
}
