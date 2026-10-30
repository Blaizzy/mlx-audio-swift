import Foundation
import Testing
import MLX
import MLXNN
@testable import MLXAudioCore
@testable import MLXAudioTTS

@Suite(.serialized)
struct MingOmniTests {
    private func textConfiguration() -> MingOmniTextConfiguration {
        var c = MingOmniTextConfiguration()
        c.hiddenSize = 8; c.intermediateSize = 16; c.numHiddenLayers = 1; c.vocabSize = 16
        c.numAttentionHeads = 2; c.numKeyValueHeads = 1; c.numExperts = 4
        c.numExpertsPerToken = 2; c.numSharedExperts = 1; c.moeIntermediateSize = 8
        return c
    }
    private func configuration() -> MingOmniConfiguration {
        let q = Qwen2Configuration(hiddenSize: 8, hiddenLayers: 1, intermediateSize: 16,
            attentionHeads: 2, rmsNormEps: 1e-6, vocabularySize: 1, kvHeads: 1)
        let b = MingAudioBackboneConfiguration(backbone: q, inputDim: 4, outputDim: 4,
            hopSize: 4, latentDim: 2, slidingWindow: 3)
        return MingOmniConfiguration(text: textConfiguration(),
            audio: .init(encoder: b, decoder: b, sampleRate: 100, patchSize: 2),
            flow: .init(hiddenSize: 8, depth: 1, numHeads: 2, patchSize: 2, historyPatchSize: 4),
            aggregator: .init(hiddenSize: 8, depth: 1, numHeads: 2))
    }
    private func binaryWeights(_ module: Module) throws {
        let weights = Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { key, value in
            (key, key.hasSuffix("layernorm.weight") || key == "norm.weight"
                ? MLXArray.ones(value.shape)
                : ((MLXArray(0..<value.size) % 17).asType(.float32) - 8).reshaped(value.shape) / 64)
        })
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    }
    @Test func bailingRoutingAndThreeStreamPositionsMatchCPUGolden() throws {
        try Device.withDefaultDevice(.cpu) {
            let c = textConfiguration()
            let layer = MingOmniDecoderLayer(c, index: 0)
            try binaryWeights(layer)
            let x = ((MLXArray(0..<24) % 13).asType(.float32) - 6).reshaped(1, 3, 8) / 16
            let pos = MLXArray([0, 1, 2, 1, 2, 3, 2, 3, 4], [3, 1, 3])
            // CPU mlx-audio reference, exact binary weights, 3D RoPE sections [1,1].
            var custom = c
            custom.ropeScaling = try JSONDecoder().decode(MingOmniTextConfiguration.RotaryConfiguration.self,
                from: Data(#"{"type":"3D","mrope_section":[1,1]}"#.utf8))
            let rotatedLayer = MingOmniDecoderLayer(custom, index: 0)
            try binaryWeights(rotatedLayer)
            let expected: [Float] = [-0.39748469,-0.27415296,-0.31883317,-0.15884928,-0.08972396,-0.021298008,-0.06562458,0.09238922,
                0.13072689,0.22068615,0.25883356,0.33144322,0.39736179,-0.35625681,-0.33604378,-0.24510650,
                -0.20564535,-0.07018365,-0.12884921,0.04294666,0.04454618,0.15736917,0.13280471,0.26961279]
            let output = rotatedLayer(x, cache: nil, positions: pos)
            #expect(max(abs(output - MLXArray(expected).reshaped(output.shape))).item(Float.self) < 2e-6)
            let router = try #require(layer.mlp as? MingOmniSparseMoE).gate
            let (indices, weights) = router(x)
            #expect(indices.asArray(Int32.self) == [0, 2, 1, 3, 1, 3])
            let goldenWeights = MLXArray([Float(0.5200088024),0.4799911380,0.5012207031,0.4987792969,0.5009765625,0.4990234375]).reshaped(weights.shape)
            #expect(max(abs(weights - goldenWeights)).item(Float.self) < 2e-7)
        }
    }
    @Test func backboneCacheMatchesFullSequence() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = MingOmniBackbone(textConfiguration())
            try binaryWeights(model)
            let x = ((MLXArray(0..<56) % 13).asType(.float32) - 6).reshaped(1, 7, 8) / 16
            let whole = model(x)
            let cache: [KVCache] = [KVCacheSimple()]
            var pieces: [MLXArray] = []
            for i in 0..<7 {
                let out = model(x[0..., i..<(i + 1), 0...], cache: cache)
                eval(out); pieces.append(out)
            }
            #expect(cache[0].offset == 7)
            #expect(max(abs(whole - concatenated(pieces, axis: 1))).item(Float.self) < 2e-6)
        }
    }
    @Test func quantizedExpertRoundTripAndRawStacking() throws {
        let first = MingSwitchMLP(hidden: 64, intermediate: 64, experts: 4)
        try binaryWeights(first)
        quantize(model: first, groupSize: 32, bits: 4)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("weights.safetensors")
        try save(arrays: Dictionary(uniqueKeysWithValues: first.parameters().flattened()), url: file)
        let second = MingSwitchMLP(hidden: 64, intermediate: 64, experts: 4)
        quantize(model: second, groupSize: 32, bits: 4)
        try second.update(parameters: ModuleParameters.unflattened(loadArrays(url: file)), verify: .all)
        let x = MLXArray.ones([1, 33, 64]) / 8
        let indices = repeated(MLXArray([Int32(3), 1]).reshaped(1, 1, 2), count: 33, axis: 1)
        #expect(max(abs(first(x, indices: indices) - second(x, indices: indices))).item(Float.self) == 0)
        let model = try MingOmniModel(configuration())
        let names = (0..<4).map { "model.model.layers.0.mlp.experts.\($0).gate_proj.weight" }
        let raw = Dictionary(uniqueKeysWithValues: names.enumerated().map { ($0.element, MLXArray.ones([8, 8]) * Float($0.offset)) })
        let sanitized = try model.sanitize(weights: raw)
        #expect(sanitized["model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 8, 8])
        #expect(sanitized["model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?[3, 0, 0].item(Float.self) == 3)
        var incomplete = raw; incomplete.removeValue(forKey: names[2])
        #expect(throws: Error.self) { _ = try model.sanitize(weights: incomplete) }
    }
    @Test func incrementalInterpolationAndISTFTMatchFullDecode() throws {
        let config = configuration().audio
        let vae = MingAudioVAE(config)
        // Keep the actual periodic Hann window while setting exact binary model weights.
        let originalWindow = vae.decoder.head.istft.window
        try binaryWeights(vae)
        try vae.decoder.head.istft.update(parameters: ModuleParameters.unflattened(["window": originalWindow]), verify: .all)
        let x = ((MLXArray(0..<24) % 13).asType(.float32) - 6).reshaped(1, 12, 2) / 16
        let full = vae.decode(x)
        let state = MingAudioDecoderState(configuration: config)
        var pieces: [MLXArray] = []
        for i in 0..<6 {
            let piece = vae.decode(x[0..., (i * 2)..<(i * 2 + 2), 0...], state: state, lastChunk: i == 5)
            eval(piece); pieces.append(piece)
        }
        #expect(pieces[0].dim(1) == 0)
        let stream = concatenated(pieces, axis: 1)
        #expect(stream.shape == full.shape)
        #expect(stream.dim(1) == 96)
        #expect(max(abs(full - stream)).item(Float.self) < 2e-5)
    }
    @Test func legacyFlowMatchesCPUGolden() throws {
        try Device.withDefaultDevice(.cpu) {
            let flow = MingFlowLoss(latentDim: 2, conditionDim: 8,
                config: .init(hiddenSize: 8, depth: 1, numHeads: 2, patchSize: 2, historyPatchSize: 4), ming3: false)
            try binaryWeights(flow)
            let condition = ((MLXArray(0..<8) % 13).asType(.float32) - 6).reshaped(1, 1, 8) / 16
            let noise = MLXArray([Float(0.125), -0.25, 0.375, -0.5]).reshaped(1, 2, 2).transposed(0, 2, 1)
            let output = flow.sample(condition, history: .zeros([1, 4, 2]), patchSize: 2, steps: 2, cfg: 2,
                temperature: 0, noise: noise, sdeNoise: .zeros([2, 1, 2, 2]))
            let golden = MLXArray([Float(-0.059164733), 0.28337961, -0.42397791, -0.58779562]).reshaped(1, 2, 2)
            #expect(max(abs(output - golden)).item(Float.self) < 2e-6)
        }
    }
    @Test func referenceSpeakerSlotsAreExplicit() throws {
        let model = try MingOmniModel(configuration())
        let waveform = MLXArray.ones([100])
        var options = MingOmniSpeechOptions()
        #expect(try model.projectSpeaker(waveform, options: options) == nil)
        options.automaticSpeakerEmbedding = true
        #expect(throws: Error.self) { _ = try model.projectSpeaker(waveform, options: options) }
        options.automaticSpeakerEmbedding = false
        options.speakerEmbedding = Array(repeating: 0.125, count: 192)
        let projected = try #require(try model.projectSpeaker(waveform, options: options))
        let expected = model.speakerHead(MLXArray(options.speakerEmbedding!).reshaped(1, 192))[0]
        #expect(max(abs(projected - expected)).item(Float.self) == 0)
        options.zeroSpeakerEmbedding = true
        let zero = try #require(try model.projectSpeaker(nil, options: options))
        #expect(max(abs(zero)).item(Float.self) == 0)
    }
    @Test func routingAndConfigurationValidation() throws {
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Ming-omni-tts-16.8B-A3B-4bit") == "bailingmm")
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Ming-omni-tts-0.5B") == "dense")
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Ming-3.0-tts-tiny-4bit") == "lingv3_tts")
        var c = textConfiguration(); c.numExpertsPerToken = 5
        #expect(throws: Error.self) { try c.validate() }
        c = textConfiguration(); c.numKeyValueHeads = 3
        #expect(throws: Error.self) { try c.validate() }
        let model = try MingOmniModel(configuration())
        #expect(model.padPrompt(MLXArray.ones([1, 17])).dim(1) == 32)
    }
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MLXAUDIO_ENABLE_NETWORK_TESTS"] == "1"))
    func publishedCheckpointReferenceAndStreaming() async throws {
        let source = ProcessInfo.processInfo.environment["MLXAUDIO_MING_OMNI_REPO"]
            ?? "mlx-community/Ming-omni-tts-16.8B-A3B-4bit"
        let model = try #require(try await TTS.loadModel(modelRepo: source) as? MingOmniModel)
        let parameters = GenerateParameters(maxTokens: 40, temperature: 0, seed: 42)
        let full = try await model.generate(text: "Hello world.", voice: nil, refAudio: nil, refText: nil,
            language: "en", generationParameters: parameters)
        var chunks: [MLXArray] = []
        for try await event in model.generateStream(text: "Hello world.", voice: nil, refAudio: nil,
            refText: nil, language: "en", generationParameters: parameters) {
            if case .audio(let chunk) = event { chunks.append(chunk) }
        }
        #expect(chunks.count > 1)
        let stream = concatenated(chunks)
        #expect(stream.shape == full.shape)
        #expect(max(abs(full - stream)).item(Float.self) < 1e-6)
        let reference = full
        let cloned = try await model.generate(text: "Hello world.", voice: nil, refAudio: reference,
            refText: "Hello world.", language: "en", generationParameters: parameters)
        #expect(cloned.size > model.sampleRate / 4)
        #expect(all(isFinite(cloned)).item(Bool.self))
        #expect(sqrt(mean(cloned.square())).item(Float.self) > 1e-4)
    }
}
