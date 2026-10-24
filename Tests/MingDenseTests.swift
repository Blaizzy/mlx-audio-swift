import Foundation
import Testing
import MLX
import MLXNN
@testable import MLXAudioCore
@testable import MLXAudioTTS
import MLXAudioCodecs

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
    @Test func transformerAudioShapeAndSlidingAttention() {
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
        let embeddings = MLXRandom.normal([1, 7, 8])
        let cache: [KVCache] = [KVCacheSimple()]
        let whole = stack(inputEmbeddings: embeddings)
        let chunks = (0..<7).map { stack(inputEmbeddings: embeddings[0..., $0..<($0 + 1), 0...], cache: cache) }
        #expect(max(abs(whole - concatenated(chunks, axis: 1))).item(Float.self) < 1e-4)
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
}
