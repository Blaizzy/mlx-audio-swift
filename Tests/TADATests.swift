import Foundation
import Testing
import MLX
import MLXNN
import MLXAudioCore
@testable import MLXAudioTTS

@Suite("TADA", .serialized)
struct TADATests {
    func tiny() -> TADAConfiguration {
        var c = TADAConfiguration()
        c.vocabSize = 16; c.hiddenSize = 8; c.numHiddenLayers = 2
        c.numAttentionHeads = 2; c.numKeyValueHeads = 1; c.headDim = 4
        c.intermediateSize = 16; c.acousticDim = 4; c.numTimeClasses = 8
        c.headLayers = 2; c.headFFNRatio = 2
        return c
    }
    @Test func metadataAndGrayDurations() throws {
        #expect(try TTS.nestedModelType(Data(#"{"model_type":"llama","architectures":["TadaForCausalLM"]}"#.utf8)) == "tada")
        #expect(try TTS.nestedModelType(Data(#"{"model_type":"llama","architectures":["LlamaForCausalLM"]}"#.utf8)) == nil)
        #expect(TTS.resolveModelType(modelRepo: "HumeAI/mlx-tada-1b") == "tada")
        for n in 0 ..< 256 {
            let gray = n ^ (n >> 1)
            let bits = (0 ..< 8).reversed().map { Float((gray >> $0) & 1) * 2 - 1 }
            #expect(tadaGray(MLXArray(bits)) == n)
        }
    }
    @Test func arbitraryNestedDirectory() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("model"), withIntermediateDirectories: true)
        try Data(#"{"model_type":"llama","architectures":["TadaForCausalLM"]}"#.utf8).write(to: directory.appendingPathComponent("model/config.json"))
        #expect(TTS.localModelDirectory(directory.path)?.path == directory.path)
        #expect(try TTS.localModelType(directory) == "tada")
    }
    @Test func segmentBoundaries() {
        let mask = tadaSegmentMask(MLXArray([Int32(1), 0, 1, 0, 1]).reshaped(1, 5))
        #expect(mask.shape == [1, 5, 5])
        #expect(mask.asArray(Bool.self) == [false,true,true,true,true, false,false,false,true,true, false,false,false,true,true, true,false,false,false,false, true,false,false,false,false])
    }
    @Test func monotonicAlignmentAndEmptyText() throws {
        let scores = MLXArray([Float(0), 2, 4, 1, 0, 0, 0, 0, 1, 2, 3, 9]).reshaped(2, 6)
        let result = try tadaMonotonicAlign(scores: scores, audioLength: 2880, sampleRate: 24000)
        #expect(result.positions == [3, 6])
        #expect(result.masks == [0, 0, 1, 0, 0, 1])
        let empty = try tadaMonotonicAlign(scores: .zeros([0, 6]), audioLength: 2880, sampleRate: 24000)
        #expect(empty.positions.isEmpty && empty.masks == Array(repeating: 0, count: 6))
        #expect(throws: TADAError.self) { try tadaMonotonicAlign(scores: .zeros([7, 6]), audioLength: 2880, sampleRate: 24000) }
    }
    @Test func scheduleAndNormalization() {
        #expect(TADAModel.schedule(4, .linear) == [0, 0.25, 0.5, 0.75, 1])
        #expect(TADAModel.schedule(4, .logsnr).first == 0)
        #expect(TADAModel.schedule(4, .logsnr).last == 1)
        #expect(tadaNormalize("HELLO — world! “Yes”: (okay).") == "Hello, world! Yes, okay.")
    }
    @Test func sincResampling() throws {
        let impulse = MLXArray([Float(1), 0, 0, 0, 0, 0])
        let actual = try tadaResample(impulse, from: 24000, to: 16000).asArray(Float.self)
        #expect(actual.count == 4)
        #expect(abs(actual[0] - 0.66) < 1e-6)
        #expect(abs(actual[1] - 0.006227691) < 2e-6)
        #expect(throws: TADAError.self) { try tadaResample(impulse, from: 0, to: 16000) }
    }
    @Test func cachedBackboneMatchesPrefill() throws {
        try Device.withDefaultDevice(.cpu) {
        let c = tiny(), model = TADAModel(c)
        let weights = model.parameters().flattened().map { key, value in
            (key, key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") ? MLXArray.ones(value.shape) : (MLXArray(0 ..< value.size).asType(.float32) % 7 - 3).reshaped(value.shape) / 32)
        }
        try model.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: weights)), verify: .all)
        let ids = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let zero = MLXArray.zeros([1, 3], dtype: .int32)
        var fullCache = Array<(MLXArray, MLXArray)?>(repeating: nil, count: 2)
        let full = model.forward(ids: ids, features: .zeros([1, 3, 4]), masks: zero, before: zero, after: zero, cache: &fullCache)
        var cache = Array<(MLXArray, MLXArray)?>(repeating: nil, count: 2)
        var pieces: [MLXArray] = []
        for i in 0 ..< 3 {
            pieces.append(model.forward(ids: ids[0..., i ..< i + 1], features: .zeros([1, 1, 4]), masks: .zeros([1, 1], dtype: .int32), before: .zeros([1, 1], dtype: .int32), after: .zeros([1, 1], dtype: .int32), cache: &cache))
        }
        #expect(abs(full - concatenated(pieces, axis: 1)).max().item(Float.self) < 1e-5)
        #expect(cache[0]?.0.shape == [1, 1, 3, 4])
        let expected = MLXArray([Float(-1.548230886), -0.686544538, -0.238710567, -0.131647512, 0.356417924, 1.170869350, 1.077846169, -1.548230886])
        #expect(abs(full[0, 0] - expected).max().item(Float.self) < 2e-6)
        }
    }
    @Test func diffusionMatchesCPUReference() throws {
        func output(_ device: Device) throws -> MLXArray {
            try Device.withDefaultDevice(device) {
                let c = tiny(), head = TADADiffusionHead(c)
                let weights = head.parameters().flattened().map { key, value in
                    (key, key.hasSuffix("norm.weight") ? MLXArray.ones(value.shape) : (MLXArray(0 ..< value.size).asType(.float32) % 7 - 3).reshaped(value.shape) / 32)
                }
                try head.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: weights)), verify: .all)
                let result = head((MLXArray(0 ..< 10).asType(.float32) / 16).reshaped(1, 10), time: MLXArray([Float(0.5)]), condition: .ones([1, 8]) / 4)
                eval(result); return result
            }
        }
        let expected = MLXArray([Float(0.313762486), -0.157825589, -0.005806632, -0.085487172, -0.055426404, 0.084288180, -0.093504861, 0.313762486, -0.157825589, -0.005806632]).reshaped(1, 10)
        let cpu = try output(.cpu)
        #expect(abs(cpu - expected).max().item(Float.self) < 2e-6)
        // GPU GEMM may round float32 products on Apple Silicon.
        #expect(abs(try output(.gpu) - cpu).max().item(Float.self) < 1e-3)
    }
    @Test func tinyCodecAndDiffusion() throws {
        let c = tiny(), head = TADADiffusionHead(c)
        let zeros = head.parameters().flattened().map { ($0.0, MLXArray.zeros($0.1.shape)) }
        try head.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: zeros)), verify: .all)
        let out = head(.ones([1, c.latentSize]), time: MLXArray([Float(0.5)]), condition: .ones([1, c.hiddenSize]))
        #expect(out.shape == [1, 10]); #expect(out.max().item(Float.self) == 0)
        let decoder = TADACodecDecoder(hidden: 8, embed: 4, dim: 2, layers: 1, heads: 2, inner: 16, strides: [2, 3])
        let audio = decoder(.ones([1, 3, 4]), masks: MLXArray([Int32(1), 0, 1]).reshaped(1, 3))
        #expect(audio.shape == [1, 17, 1]); #expect(isFinite(audio).all().item(Bool.self))
        let encoder = TADACodecEncoder(hidden: 8, embed: 4, dim: 2, layers: 1, heads: 2, inner: 16, strides: [2, 3])
        let features = encoder.features(.ones([24]), masks: MLXArray([Int32(1), 0, 1]).reshaped(1, 3))
        #expect(features.dim(0) == 1 && features.dim(2) == 4)
        #expect(isFinite(features).all().item(Bool.self))
    }
    @Test func codecStreamingWithholdsProvisionalTail() throws {
        try Device.withDefaultDevice(.cpu) {
            let decoder = TADACodecDecoder(hidden: 8, embed: 4, dim: 2, layers: 1, heads: 2, inner: 16, strides: [2, 3])
            let values = decoder.parameters().flattened().map { key, value in
                (key, key.hasSuffix(".alpha") || key.hasSuffix("norm.weight") ? MLXArray.ones(value.shape) : (MLXArray(0 ..< value.size).asType(.float32) % 7 - 3).reshaped(value.shape) / 32)
            }
            try decoder.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: values)), verify: .all)
            func decode(_ frames: Int, third: Bool) -> MLXArray {
                var features = [Float](repeating: 0, count: frames * 4)
                var masks = [Int32](repeating: 0, count: frames)
                for index in third ? [0, 100, 104] : [0, 100] {
                    masks[index] = 1
                    for channel in 0 ..< 4 { features[index * 4 + channel] = Float(channel + 1) / 8 }
                }
                return decoder(MLXArray(features).reshaped(1, frames, 4), masks: MLXArray(masks).reshaped(1, frames)).reshaped([-1])
            }
            // The provisional prefix has 100 trailing zero frames; the next
            // token replaces that provisional duration with a four-frame gap.
            let partial = decode(201, third: false), full = decode(105, third: true)
            let safe = partial.size - (100 + 32) * 6
            #expect(safe > 0 && safe < full.size)
            #expect(abs(partial[0 ..< safe] - full[0 ..< safe]).max().item(Float.self) < 1e-6)
        }
    }

    @Test func optimalAlignmentAndEncoderMask() throws {
        let scores = MLXArray([Float(0), 9, 10, 0, 0, 100, 0, 0]).reshaped(2, 4)
        #expect(try tadaMonotonicAlign(scores: scores, audioLength: 1920, sampleRate: 24000).positions == [1, 2])
        #expect(try tadaMonotonicAlign(scores: MLXArray([Float(3)]).reshaped(1, 1), audioLength: 480, sampleRate: 24000).positions == [1])
        let mask = tadaEncoderSegmentMask(MLXArray([Int32(0), 1, 0, 1]).reshaped(1, 4))
        #expect(mask.asArray(Bool.self) == [false,true,true,true, false,false,false,true, true,true,false,true, true,true,false,false])
    }
    @Test func scaledRopeMatchesOfficialAtCachedOffset() throws {
        try Device.withDefaultDevice(.cpu) {
            var c = tiny()
            c.ropeScaling = .init(factor: 32, lowFreqFactor: 1, highFreqFactor: 4, originalContext: 8192)
            let model = TADAModel(c)
            let weights = model.parameters().flattened().map { key, value in
                (key, key.hasSuffix("norm.weight") || key.hasSuffix("layernorm.weight") ? MLXArray.ones(value.shape) : (MLXArray(0 ..< value.size).asType(.float32) % 7 - 3).reshaped(value.shape) / 32)
            }
            try model.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: weights)), verify: .all)
            let ids = MLXArray((0 ..< 64).map { Int32($0 % 15 + 1) }).reshaped(1, 64)
            let z = MLXArray.zeros([1, 64], dtype: .int32)
            var cache = Array<(MLXArray, MLXArray)?>(repeating: nil, count: 2)
            let full = model.forward(ids: ids, features: .zeros([1, 64, 4]), masks: z, before: z, after: z, cache: &cache)
            let expected = MLXArray([Float(-1.471399784), -0.520819247, 0.058554396, -0.500193536, -0.007565252, 0.934368789, 1.507054806, -1.471399784])
            #expect(abs(full[0, 63] - expected).max().item(Float.self) < 3e-6)
            cache = Array(repeating: nil, count: 2)
            _ = model.forward(ids: ids[0..., 0 ..< 63], features: .zeros([1, 63, 4]), masks: z[0..., 0 ..< 63], before: z[0..., 0 ..< 63], after: z[0..., 0 ..< 63], cache: &cache)
            let last = model.forward(ids: ids[0..., 63 ..< 64], features: .zeros([1, 1, 4]), masks: z[0..., 63 ..< 64], before: z[0..., 63 ..< 64], after: z[0..., 63 ..< 64], cache: &cache)
            #expect(abs(last[0, 0] - expected).max().item(Float.self) < 3e-6)
        }
    }
    @Test func postNormAlignerLayerMatchesOfficialCPU() throws {
        try Device.withDefaultDevice(.cpu) {
            let layer = TADAW2VEncoderLayer(hiddenSize: 8, numHeads: 2, intermediateSize: 16, eps: 1e-5)
            let weights = layer.parameters().flattened().map { key, value in
                (key, key.hasSuffix("norm.weight") ? MLXArray.ones(value.shape) : (MLXArray(0 ..< value.size).asType(.float32) % 7 - 3).reshaped(value.shape) / 32)
            }
            try layer.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: weights)), verify: .all)
            let input = ((MLXArray(0 ..< 24).asType(.float32) % 11 - 5) / 8).reshaped(1, 3, 8)
            let expected = MLXArray([Float(-1.676309109), -1.241125464, -0.650869608, -0.093981601, 0.291991353, 0.882592678, 1.522545934, 0.871405780])
            #expect(abs(layer(input)[0, 0] - expected).max().item(Float.self) < 3e-6)
        }
    }
    @Test func referenceIsRequired() async throws {
        let model = TADAModel(tiny())
        do { _ = try await model.generate(text: "Hello"); Issue.record("Missing reference should throw") }
        catch { #expect(error.localizedDescription.contains("reference audio")) }
        do {
            for try await _ in model.generateStream(text: "Hello", voice: nil, refAudio: nil, refText: nil, language: nil, generationParameters: .init()) {}
            Issue.record("Missing streaming reference should throw")
        } catch { #expect(error.localizedDescription.contains("Reference audio")) }
    }

}
