import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioTTS

private func tinyRealtimeConfig() -> VibeVoiceRealtimeConfig {
    var c = VibeVoiceRealtimeConfig()
    c.decoderConfig = Qwen2Configuration(
        hiddenSize: 64, hiddenLayers: 2, intermediateSize: 128, attentionHeads: 2, rmsNormEps: 1e-6,
        vocabularySize: 128, kvHeads: 1)
    c.ttsBackboneNumHiddenLayers = 1
    c.acousticVaeDim = 4
    c.acousticTokenizerConfig.vaeDim = 4
    c.acousticTokenizerConfig.encoderNFilters = 4
    c.acousticTokenizerConfig.decoderNFilters = 4
    c.acousticTokenizerConfig.encoderRatios = [2]
    c.acousticTokenizerConfig.encoderDepths = [1, 1]
    c.diffusionHeadConfig.hiddenSize = 64
    c.diffusionHeadConfig.latentSize = 4
    c.diffusionHeadConfig.headLayers = 1
    c.diffusionHeadConfig.ddpmNumSteps = 10
    c.diffusionHeadConfig.ddpmNumInferenceSteps = 2
    return c
}
private func tinyRealtimeVoice(_ c: VibeVoiceRealtimeConfig) -> [String: MLXArray] {
    var values: [String: MLXArray] = [:]
    for key in ["lm_hidden", "tts_lm_hidden", "neg_tts_lm_hidden"] {
        values[key] = .zeros([1, 3, c.decoderConfig.hiddenSize])
    }
    for prefix in ["lm", "tts_lm", "neg_tts_lm"] {
        for kind in ["key", "value"] { values["\(prefix)_\(kind)_0"] = .zeros([1, 1, 3, 32]) }
    }
    return values
}
@Suite(.serialized)
struct VibeVoiceRealtimeTests {
    @Test func validatesConfigurationAndDispatch() throws {
        let config = tinyRealtimeConfig()
        try config.validate()
        var invalid = config
        invalid.ttsBackboneNumHiddenLayers = 3
        #expect(throws: (any Error).self) { try invalid.validate() }
        #expect(
            TTS.resolveModelType(modelRepo: "mlx-community/VibeVoice-Realtime-0.5B-4bit")
                == "vibevoice_streaming")
        let alias = try JSONDecoder().decode(
            VibeVoiceRealtimeConfig.self, from: Data(#"{"sampling_rate":22050}"#.utf8))
        #expect(alias.sampleRate == 22050)
    }
    @Test func voicesHaveIndependentGenerationCaches() throws {
        let c = tinyRealtimeConfig()
        let tensors = tinyRealtimeVoice(c)
        let voice = try VibeVoiceRealtimeVoice(tensors: tensors, config: c)
        let a = voice.makeCache("lm", layers: 1)
        let b = voice.makeCache("lm", layers: 1)
        _ = a[0].update(keys: .ones([1, 1, 1, 32]), values: .ones([1, 1, 1, 32]))
        eval(a[0].state)
        #expect(a[0].offset == 4)
        #expect(b[0].offset == 3)
        #expect(b[0].state[0].sum().item(Float.self) == 0)
        #expect(tensors["lm_key_0"]!.sum().item(Float.self) == 0)
        var missing = tensors
        missing.removeValue(forKey: "tts_lm_key_0")
        #expect(throws: (any Error).self) {
            try VibeVoiceRealtimeVoice(tensors: missing, config: c)
        }
    }
    @Test func localQuantizedLoadingAndGeneration() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("voices"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let c = tinyRealtimeConfig()
        let model = VibeVoiceRealtimeModel(c)
        var weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { key, value in
                (key, MLXArray.zeros(value.shape))
            })
        weights["speech_scaling_factor"] = MLXArray(Float(1))
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        quantize(model: model) { path, _ in
            path.hasPrefix("language_model.layers.") ? (32, 4, .affine) : nil
        }
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))
        try MLX.save(
            arrays: tinyRealtimeVoice(c),
            url: directory.appendingPathComponent("voices/en-Carter_man.safetensors"))
        let config: [String: Any] = [
            "model_type": "vibevoice_streaming", "tts_backbone_num_hidden_layers": 1,
            "acoustic_vae_dim": 4,
            "decoder_config": [
                "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
                "num_attention_heads": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-6,
                "vocab_size": 128,
            ],
            "acoustic_tokenizer_config": [
                "vae_dim": 4, "encoder_n_filters": 4, "decoder_n_filters": 4, "encoder_ratios": [2],
                "encoder_depths": "1-1",
            ],
            "diffusion_head_config": [
                "hidden_size": 64, "latent_size": 4, "head_layers": 1, "ddpm_num_steps": 10,
                "ddpm_num_inference_steps": 2,
            ],
            "quantization": ["bits": 4, "group_size": 32],
        ]
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json"))
        let tokenizer: [String: Any] = [
            "model": [
                "type": "BPE", "vocab": ["<unk>": 0, "Hello": 1], "merges": [],
                "unk_token": "<unk>",
            ], "added_tokens": [["id": 0, "content": "<unk>", "special": true]],
        ]
        try JSONSerialization.data(withJSONObject: tokenizer).write(
            to: directory.appendingPathComponent("tokenizer.json"))
        try Data(#"{"tokenizer_class":"PreTrainedTokenizerFast","unk_token":"<unk>"}"#.utf8).write(
            to: directory.appendingPathComponent("tokenizer_config.json"))
        let loaded = try await TTS.loadModel(modelRepo: directory.path)
        #expect(loaded is VibeVoiceRealtimeModel)
        let concrete = try #require(loaded as? VibeVoiceRealtimeModel)
        #expect(concrete.availableVoices == ["en-Carter_man"])
        #expect(
            Dictionary(uniqueKeysWithValues: concrete.parameters().flattened())[
                "language_model.layers.0.self_attn.q_proj.weight"]!.dtype == .uint32)
        let audio = try await concrete.generate(
            text: "Hello", voice: nil, refAudio: nil, refText: nil, language: nil,
            generationParameters: GenerateParameters(maxTokens: 3, seed: 1))
        #expect(audio.shape == [6])
        #expect(isFinite(audio).all().item(Bool.self))
        var streamedSamples: [Float] = []
        var chunkCount = 0
        var generationTokens = 0
        let stream = concrete.generateStream(
            text: "Hello", voice: nil, refAudio: nil, refText: nil, language: nil,
            generationParameters: GenerateParameters(maxTokens: 3, seed: 1), streamingInterval: 0)
        for try await event in stream {
            switch event {
            case .audio(let chunk):
                streamedSamples.append(contentsOf: chunk.asArray(Float.self))
                chunkCount += 1
            case .info(let info):
                generationTokens = info.generationTokenCount
                #expect(info.promptTokenCount > 0)
            default: break
            }
        }
        #expect(chunkCount == 3)
        #expect(generationTokens == 3)
        #expect(streamedSamples == audio.asArray(Float.self))
        #expect(throws: (any Error).self) { try concrete.loadVoice("../en-Carter_man") }
    }
}
