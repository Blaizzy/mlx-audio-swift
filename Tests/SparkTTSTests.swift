import Foundation
@preconcurrency import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
import Tokenizers

@testable import MLXAudioTTS

/// Exercise the real generation and codec paths without downloading a checkpoint.
private func makeTinySparkModel() throws -> SparkModel {
    let backboneConfig = try JSONDecoder().decode(Qwen2Configuration.self, from: Data("""
    {
      "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
      "num_attention_heads": 2, "num_key_value_heads": 1,
      "rms_norm_eps": 0.000001, "vocab_size": 3, "tie_word_embeddings": false
    }
    """.utf8))
    let backbone = Qwen2Model(backboneConfig)

    // Make greedy decoding produce one global token followed by semantic tokens.
    // There is deliberately no stop token: generation must obey maxTokens.
    var weights = Dictionary(uniqueKeysWithValues: backbone.parameters().flattened().map {
        ($0.0, MLXArray.zeros($0.1.shape))
    })
    weights["model.embed_tokens.weight"] = MLXArray([
        Float(1), 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0,
    ]).reshaped([3, 8])
    weights["model.norm.weight"] = MLXArray.ones([8])
    weights["lm_head.weight"] = MLXArray([
        Float(0), 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0,
    ]).reshaped([3, 8])
    try backbone.update(parameters: ModuleParameters.unflattened(weights), verify: .all)

    let tokenizer = try AutoTokenizer.from(
        tokenizerConfig: ["tokenizer_class": "GPT2Tokenizer", "unk_token": "<unk>", "fuse_unk": true],
        tokenizerData: [
            "model": [
                "type": "BPE", "unk_token": "<unk>", "merges": [],
                "vocab": ["<unk>": 0, "<|bicodec_global_0|>": 1, "<|bicodec_semantic_0|>": 2],
            ],
            "added_tokens": [
                ["id": 0, "content": "<unk>", "special": true],
                ["id": 1, "content": "<|bicodec_global_0|>", "special": true],
                ["id": 2, "content": "<|bicodec_semantic_0|>", "special": true],
            ],
        ])

    let codecConfig = try JSONDecoder().decode(BiCodecConfiguration.self, from: Data("""
    {
      "mel_params": {"sample_rate": 16000},
      "encoder": {
        "input_channels": 8, "vocos_dim": 8, "vocos_intermediate_dim": 16,
        "vocos_num_layers": 1, "out_channels": 8, "sample_ratios": [1, 1]
      },
      "decoder": {"input_channel": 8, "channels": 16, "rates": [2], "kernel_sizes": [4]},
      "quantizer": {"input_dim": 8, "codebook_size": 4, "codebook_dim": 2},
      "speaker_encoder": {"input_dim": 8, "out_dim": 8, "latent_dim": 8, "token_num": 1, "fsq_levels": [4, 4]},
      "prenet": {
        "input_channels": 8, "vocos_dim": 8, "vocos_intermediate_dim": 16,
        "vocos_num_layers": 1, "out_channels": 8, "condition_dim": 8, "sample_ratios": [1, 1]
      }
    }
    """.utf8))
    let codec = SparkBiCodec(codecConfig)
    codec.train(false)
    return SparkModel(
        backbone: backbone, bicodec: codec, tokenizer: tokenizer,
        modelDir: FileManager.default.temporaryDirectory, sampleRate: codecConfig.melParams.sampleRate)
}

@Suite("Spark-TTS tests", .serialized)
struct SparkTTSTests {
    @Test(arguments: [2, 4])
    func sparkTTSGeneratesAudio(maxTokens: Int) async throws {
        let model = try makeTinySparkModel()
        let audio = try await model.generate(
            text: "Hi.", voice: "female",
            refAudio: nil, refText: nil, language: nil,
            generationParameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
        eval(audio)

        // One global token, then maxTokens - 1 semantic tokens, upsampled by 2.
        #expect(audio.shape == [(maxTokens - 1) * 2])
        let samples = audio.asType(.float32).asArray(Float.self)
        #expect(samples.allSatisfy { $0.isFinite })
        #expect(samples.contains { $0 != 0 })
    }

    @Test func sparkModelResolution() {
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Spark-TTS-0.5B-bf16") == "spark")
    }

    @Test func sparkClonePromptEmbedsSpeakerTokens() {
        let globalOnly = SparkPrompt.clone(
            text: "Hello there.", refText: nil, globalTokenIds: [12, 5], semanticTokenIds: nil)
        #expect(globalOnly == "<|task_tts|><|start_content|>Hello there.<|end_content|>"
            + "<|start_global_token|><|bicodec_global_12|><|bicodec_global_5|><|end_global_token|>")

        let withRef = SparkPrompt.clone(
            text: "Say this.", refText: "Reference.", globalTokenIds: [3], semanticTokenIds: [7, 8])
        #expect(withRef == "<|task_tts|><|start_content|>Reference.Say this.<|end_content|>"
            + "<|start_global_token|><|bicodec_global_3|><|end_global_token|>"
            + "<|start_semantic_token|><|bicodec_semantic_7|><|bicodec_semantic_8|>")
    }
}
