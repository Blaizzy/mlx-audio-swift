import Foundation
import Testing
import MLX
import MLXNN
@testable import MLXAudioCore
@testable import MLXAudioTTS

@Suite(.serialized)
struct MingTinyTests {
    func textConfig() -> MingTinyTextConfiguration {
        var c = MingTinyTextConfiguration()
        c.hiddenSize = 64; c.intermediateSize = 96; c.numHiddenLayers = 2; c.vocabSize = 128
        c.numAttentionHeads = 2; c.headDim = 32; c.layerGroupSize = 2
        c.qLoraRank = 32; c.kvLoraRank = 32; c.qkNopeHeadDim = 16; c.qkRopeHeadDim = 16; c.vHeadDim = 32
        c.numExperts = 4; c.nGroup = 2; c.topKGroup = 1; c.numExpertsPerToken = 2
        c.moeIntermediateSize = 64; c.moeSharedExpertIntermediateSize = 64
        return c
    }
    @Test func taskControlsAndPromptSlots() throws {
        var options = MingTinyOptions()
        let caption = try MingTinyPrompt.instruction(task: .ip, voice: "US_James", options: options)
        #expect(caption?.contains("\"IP\": \"US_James\"") == true)
        options.emotion = "happy"
        let emotion = try MingTinyPrompt.instruction(task: .emotion, voice: "", options: options)
        #expect(emotion?.contains("\"情感\": \"高兴\"") == true)
        #expect(throws: MingTinyError.self) {
            try MingTinyPrompt.instruction(task: .seed, voice: "", options: options)
        }
        var encoded: [String] = []
        let result = MingTinyPrompt.build(encode: { encoded.append($0); return [7] }, speech: .init(),
            text: "Hello", instruction: caption, language: nil, speakers: 1, promptText: nil, promptSteps: 2)
        #expect(result.positions == [2])
        #expect(result.ids[2] == 157164)
        #expect(Array(result.ids.suffix(3)) == [157165, 157164, 157164])
        #expect(encoded.contains("<role>ASSISTANT</role>"))
        try MingTinyPrompt.validatePodcast("speaker_1:Hello\nspeaker_2:Hi", count: 2)
        #expect(throws: MingTinyError.self) { try MingTinyPrompt.validatePodcast("speaker_1:Hello", count: 2) }
    }
    @Test func routedQuantizationRoundTripAndCache() throws {
        let c = textConfig()
        let first = MingTinyBackbone(c)
        // Deterministic generated parameters make this a stable cache test.
        let tensors = Dictionary(uniqueKeysWithValues: first.parameters().flattened().map { key, value in
            let array = key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") || key.hasSuffix("o_norm.weight")
                ? MLXArray.ones(value.shape)
                : (sin(MLXArray(0..<value.size).asType(.float32) * 0.17) * 0.05).reshaped(value.shape)
            return (key, array)
        })
        try first.update(parameters: ModuleParameters.unflattened(tensors), verify: .all)
        quantize(model: first, groupSize: 32, bits: 4) { path, _ in !path.hasSuffix(".gate") }
        #expect(first.layers[1].mlp is MingTinySparseMoE)
        #expect(first.parameters().flattened().contains { $0.0 == "layers.1.mlp.switch_mlp.up_proj.scales" })
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("tiny.safetensors")
        try save(arrays: Dictionary(uniqueKeysWithValues: first.parameters().flattened()), url: url)
        let second = MingTinyBackbone(c)
        quantize(model: second, groupSize: 32, bits: 4) { path, _ in !path.hasSuffix(".gate") }
        try second.update(parameters: ModuleParameters.unflattened(loadArrays(url: url)), verify: .all)
        let x = sin(MLXArray(0..<256).asType(.float32) * 0.13).reshaped(1, 4, 64)
        let whole = first(x, cache: first.makeCache())
        #expect(max(abs(whole - second(x, cache: second.makeCache()))).item(Float.self) < 1e-6)
        let cache = second.makeCache()
        var chunks: [MLXArray] = []
        for i in 0..<4 { let y = second(x[0..., i..<(i + 1), 0...], cache: cache); eval(y); chunks.append(y) }
        // Quantized vector and matrix reductions differ: matching Python gives 1.70e-4.
        #expect(max(abs(whole - concatenated(chunks, axis: 1))).item(Float.self) < 3e-4)
    }
    @Test func nativeFlowGuidanceAndFinalEulerStep() {
        let flow = MingTinyFlow(latentDim: 4, conditionDim: 8,
            config: .init(hiddenSize: 8, depth: 1, numHeads: 2))
        let h = MLXArray.ones([1, 1, 8]), history = MLXArray.zeros([1, 1, 4])
        let noise = MLXArray.ones([1, 1, 4]) * 0.1
        let zeros = MLXArray.zeros([2, 1, 1, 4])
        let a = flow.sample(h, history: history, patchSize: 1, steps: 2, cfg: 0, sigma: 1,
            temperature: 1, noise: noise, sdeNoise: zeros)
        let perturbation = MLXArray.zeros([2, 1, 1, 4]); perturbation[1] = MLXArray.ones([1, 1, 4]) * 100
        let b = flow.sample(h, history: history, patchSize: 1, steps: 2, cfg: 0, sigma: 1,
            temperature: 1, noise: noise, sdeNoise: perturbation)
        #expect(a.dtype == .float32)
        #expect(max(abs(a - b)).item(Float.self) == 0)
        let guided = flow.sample(h, history: history, patchSize: 1, steps: 2, cfg: 2, sigma: 0,
            temperature: 0, noise: noise, sdeNoise: zeros)
        #expect(max(abs(a - guided)).item(Float.self) > 1e-5)
    }
    @Test func loaderDispatchAndContractRejection() throws {
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Ming-3.0-tts-tiny-4bit") == "lingv3_tts")
        let c = try JSONDecoder().decode(MingTinySpeechConfiguration.self,
            from: Data("{\"sampling_contract\":\"native_fp32\"}".utf8))
        #expect(c.audioPatchTokenId == 157164)
        let qwen = Qwen2Configuration(hiddenSize: 8, hiddenLayers: 1, intermediateSize: 16,
            attentionHeads: 2, rmsNormEps: 1e-6, vocabularySize: 1, kvHeads: 1)
        let audio = MingAudioBackboneConfiguration(backbone: qwen, inputDim: 4, outputDim: 4,
            hopSize: 4, latentDim: 64)
        var config = MingTinyConfiguration(text: textConfig(), audio: .init(encoder: audio, decoder: audio),
            flow: .init(hiddenSize: 8, depth: 1, numHeads: 2),
            aggregator: .init(hiddenSize: 8, depth: 1, numHeads: 2))
        config.speech.samplingContract = "dense"
        #expect(throws: MingTinyError.self) { try MingTinyModel(config: config) }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MLXAUDIO_MING_TINY_MODEL"] != nil,
                   "Set MLXAUDIO_MING_TINY_MODEL to a downloaded checkpoint directory."))
    func checkpointOfflineGenerationAndReference() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MLXAUDIO_MING_TINY_MODEL"])
        let loaded = try await TTS.loadModel(modelRepo: path)
        let model = try #require(loaded as? MingTinyModel)
        let parameters = GenerateParameters(maxTokens: 64, temperature: 0, seed: 42)
        let text = "Hello from Ming three."
        let audio = try await model.generate(text: text, voice: "US_James", refAudio: nil,
            refText: nil, language: "en", generationParameters: parameters)
        #expect(model.sampleRate == 44100)
        #expect(audio.ndim == 1 && all(isFinite(audio)).item(Bool.self))
        #expect(audio.size == model.lastDecodeSteps * 3528)
        let cloned = try await model.generate(text: "This is the next sentence.", refAudio: audio,
            refText: text, language: "en", generationParameters: parameters)
        #expect(cloned.ndim == 1 && cloned.size > 0 && all(isFinite(cloned)).item(Bool.self))
        var count = 0
        for try await event in model.generateStream(text: text, voice: "US_James", refAudio: nil,
            refText: nil, language: "en", generationParameters: parameters) {
            if case .audio(let chunk) = event {
                #expect(chunk.size > 0 && all(isFinite(chunk)).item(Bool.self)); count += 1
            }
        }
        #expect(count == 1)
    }
}
