import Foundation
import Testing
import MLX
import MLXNN
@testable import MLXAudioCore
@testable import MLXAudioTTS
import MLXAudioCodecs

@Suite(.serialized)
struct MingDenseTests {
    @Test func textFragmentsPreserveTagsAndAbbreviations() {
        let parts = MingText.fragments("[Laughter] Dr. Smith paid 3.14 dollars, and then walked home. 我今天特别高兴。", maxLength: 7)
        #expect(parts.count > 1)
        #expect(parts.contains { $0.contains("[Laughter]") })
        #expect(parts.contains { $0.contains("3.14") })
        #expect(parts.allSatisfy { MingText.semanticLength($0) <= 7 })
        #expect(MingText.addPeriod("Hello,") == "Hello.")
        #expect(MingText.addPeriod("你好") == "你好。")
        #expect(MingText.isVoice("US_James"))
        #expect(!MingText.isVoice("alloy"))
        let caption = MingText.instruction(emotion: "高兴", voice: "US_James")
        #expect(caption.contains("\"情感\": \"高兴\""))
        #expect(!MingText.isStyle(caption))
        #expect(MingText.isStyle(MingText.instruction(style: "warm")))
    }
    @Test func nonstandardGuidanceAndFixedNoise() {
        let config = MingTransformerConfiguration(hiddenSize: 8, depth: 1, numHeads: 2)
        let flow = MingFlowLoss(latentDim: 4, conditionDim: 8, config: config, ming3: true)
        let condition = MLXArray.ones([1, 1, 8]), history = MLXArray.zeros([1, 1, 4])
        let noise = MLXArray.ones([1, 1, 4]) * 0.1
        let a = flow.sample(condition, history: history, patchSize: 1, steps: 2, cfg: 0,
                            noise: noise, sdeNoise: MLXArray.zeros([2, 1, 1, 4]))
        let b = flow.sample(condition, history: history, patchSize: 1, steps: 2, cfg: 2,
                            noise: noise, sdeNoise: MLXArray.zeros([2, 1, 1, 4]))
        #expect(a.shape == [1, 1, 4])
        #expect(all(isFinite(a)).item(Bool.self))
        #expect(max(abs(a - b)).item(Float.self) > 1e-5)
        #expect(mingFlowTimesteps(5).asArray(Float.self) == [0, 0.0625, 0.125, 0.25, 0.5, 1])
    }
    @Test func aggregatorQuantizedRoundTrip() throws {
        let config = MingTransformerConfiguration(hiddenSize: 64, depth: 1, numHeads: 4)
        let first = MingLatentAggregator(latentDim: 64, outputDim: 64, config: config)
        quantize(model: first, groupSize: 32, bits: 4)
        #expect(first.parameters().flattened().contains { $0.0 == "x_embedder.scales" })
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("weights.safetensors")
        try save(arrays: Dictionary(uniqueKeysWithValues: first.parameters().flattened()), url: file)
        let second = MingLatentAggregator(latentDim: 64, outputDim: 64, config: config)
        quantize(model: second, groupSize: 32, bits: 4)
        try second.update(parameters: ModuleParameters.unflattened(loadArrays(url: file)), verify: .all)
        let x = MLXArray.ones([1, 2, 64])
        #expect(max(abs(first(x) - second(x))).item(Float.self) < 1e-6)
    }
    @Test func transformerAudioShapeAndSlidingAttention() throws {
        let qwen = Qwen2Configuration(hiddenSize: 8, hiddenLayers: 1, intermediateSize: 16,
            attentionHeads: 2, rmsNormEps: 1e-6, vocabularySize: 1, kvHeads: 1)
        let backbone = MingAudioBackboneConfiguration(backbone: qwen, inputDim: 4, outputDim: 4,
            hopSize: 4, latentDim: 2, slidingWindow: 3)
        let vae = MingAudioVAE(.init(encoder: backbone, decoder: backbone, sampleRate: 100, patchSize: 2))
        let latent = vae.encode(MLXArray.ones([1, 17]), noise: MLXArray.zeros([1, 3, 2]))
        #expect(latent.shape == [1, 3, 2])
        let wave = vae.decode(latent)
        #expect(wave.shape == [1, 24])
        #expect(all(isFinite(wave)).item(Bool.self))
        let stack = Qwen2ModelInner(qwen, slidingWindow: 3)
        let weights = Dictionary(uniqueKeysWithValues: stack.parameters().flattened().map { key, value in
            let tensor: MLXArray
            if key.hasSuffix("layernorm.weight") || key == "norm.weight" {
                tensor = MLXArray.ones(value.shape)
            } else {
                tensor = (sin(MLXArray(0..<value.size).asType(.float32) * 0.37) * 0.07).reshaped(value.shape)
            }
            return (key, tensor)
        })
        try stack.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let embeddings = sin(MLXArray(0..<56).asType(.float32) * 0.17).reshaped(1, 7, 8)
        let cache: [KVCache] = [KVCacheSimple()]
        let whole = stack(inputEmbeddings: embeddings)
        eval(whole)
        let chunks = (0..<7).map { index in
            let output = stack(inputEmbeddings: embeddings[0..., index..<(index + 1), 0...], cache: cache)
            eval(output)
            return output
        }
        // Full-sequence and vector SDPA kernels differ slightly in their reductions.
        #expect(max(abs(whole - concatenated(chunks, axis: 1))).item(Float.self) < 3e-4)

    }
    @Test func rawSpeakerMappingAndFrontend() {
        let raw = ["onnx::Conv_4423": MLXArray.ones([32, 1, 3, 3]),
                   "onnx::Conv_4462": MLXArray.ones([128, 128, 1]),
                   "xvector.block1.tdnnd1.nonlinear1.batchnorm.weight": MLXArray.ones([128])]
        let clean = MingCAMPPlus.sanitize(raw)
        #expect(clean["head.conv1.weight"]?.shape == [32, 3, 3, 1])
        #expect(clean["blocks.0.layers.0.linear1.weight"]?.shape == [128, 1, 128])
        #expect(clean["blocks.0.layers.0.nonlinear1.0.weight"] != nil)
        let features = MingCAMPPlus.features(sin(MLXArray(0..<1600).asType(.float32) * 0.2))
        #expect(features.shape == [8, 80])
        #expect(max(abs(features.mean(axis: 0))).item(Float.self) < 1e-4)
    }
    @Test func configVariantsAndMoeRejection() throws {
        let backbone: [String: Any] = ["hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
            "num_attention_heads": 2, "num_key_value_heads": 1, "rms_norm_eps": 1e-6, "vocab_size": 1,
            "use_sliding_window": true, "sliding_window": 3, "max_window_layers": 0]
        var root: [String: Any] = ["model_type": "dense", "llm_config": backbone,
            "audio_tokenizer_config": ["sample_rate": 44100, "patch_size": 4,
                "enc_kwargs": ["backbone": backbone, "input_dim": 4, "latent_dim": 2],
                "dec_kwargs": ["backbone": backbone, "output_dim": 4, "latent_dim": 2]],
            "ditar_config": ["patch_size": 1, "history_patch_size": 1], "aggregator_config": [:]]
        let decoder = JSONDecoder()
        let ming3 = try decoder.decode(MingDenseConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        #expect(ming3.isMing3)
        #expect(ming3.audio.encoder.useSlidingWindow)
        #expect(ming3.audio.encoder.maxWindowLayers == 0)
        root["ditar_config"] = ["patch_size": 4, "history_patch_size": 32]
        let omni = try decoder.decode(MingDenseConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        #expect(!omni.isMing3)
        var moe = backbone; moe["num_experts"] = 64; root["llm_config"] = moe
        #expect(throws: DecodingError.self) {
            _ = try decoder.decode(MingDenseConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        }
        #expect(TTS.resolveModelType(modelRepo: "inclusionAI/Ming-omni-tts-0.5B") == "dense")
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1"))
    func checkpointReferenceAndFragmentStreaming() async throws {
        let source = ProcessInfo.processInfo.environment["MLXAUDIO_MING_REPO"]
            ?? "mlx-community/Ming-3.0-tts-0.5B-bf16"
        let shared = try await TTS.loadModel(modelRepo: source)
        let model = try #require(shared as? MingDenseModel)
        #expect(model.sampleRate == 44100)
        let file = try #require(Bundle.module.url(forResource: "conversational_a", withExtension: "wav", subdirectory: "media"))
        let (_, human) = try loadAudioArray(from: file, sampleRate: model.sampleRate)
        let reference = human[0..<min(human.size, model.sampleRate * 2)].materialized()
        let cloned = try await model.generate(text: "Hello world.", voice: nil, refAudio: reference,
            refText: model.configuration.isMing3 ? "Coffee's story likely begins in Ethiopia." : nil, language: "en",
            generationParameters: GenerateParameters(maxTokens: 60, temperature: 0, seed: 42))
        #expect(cloned.size > model.sampleRate / 4)
        #expect(all(isFinite(cloned)).item(Bool.self))
        #expect(sqrt(mean(cloned.square())).item(Float.self) > 1e-4)
        model.options.fragmentLength = 2
        var chunks = 0, samples = 0
        for try await event in model.generateStream(text: "Hello world. Good morning.", voice: "UK_Charlotte",
            refAudio: nil, refText: nil, language: "en",
            generationParameters: GenerateParameters(maxTokens: 60, temperature: 0, seed: 42)) {
            if case .audio(let audio) = event {
                #expect(audio.ndim == 1 && audio.size > 0)
                #expect(all(isFinite(audio)).item(Bool.self))
                chunks += 1; samples += audio.size
            }
        }
        #expect(chunks == (model.configuration.isMing3 ? 2 : 1))
        #expect(samples > model.sampleRate / 2)
    }
}
