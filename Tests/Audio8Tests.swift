import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioTTS

@Suite("Audio8")
struct Audio8Tests {
    @Test func defaultsAndBackboneRejection() throws {
        let c = try JSONDecoder().decode(Audio8Config.self, from: Data("{}".utf8))
        try c.validate()
        #expect(c.codecSampleRate == 44100 && c.codecFrameSize == 2048)
        #expect(c.codecPostIntermediateSize == 1216 && c.codecPostNLocalHeads == 8)
        var hybrid = c; hybrid.slowBackbone = "falcon_h1"
        #expect(throws: (any Error).self) { try hybrid.validate() }
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Audio8-TTS-Preview-0.6b-bf16") == "arktts")
    }
    @Test func localHybridDirectoryIsRejectedBeforeLoadingWeights() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"model_type":"arktts","slow_backbone":"falcon_h1"}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
        do {
            _ = try await TTS.loadModel(modelRepo: directory.path)
            Issue.record("An unsupported Falcon-H1 checkpoint was accepted")
        } catch {
            #expect(error.localizedDescription.contains("Falcon-H1"))
        }
    }
    @Test func promptTextNormalization() {
        #expect(Audio8Model.cleanText("  Hello\n world. \t") == "Hello world.")
        #expect(Audio8Model.referenceText("  A\n reference. ") == "<|speaker:0|>A reference.")
        #expect(Audio8Model.referenceText("<|speaker:2|> Hello") == "<|speaker:2|> Hello")
    }
    @Test func filteringPrecedesTemperatureAndKeepsFirstRank() {
        let logits = MLXArray([Float(3), 2, 1, 0]).reshaped([1, 4])
        let filtered = audio8FilteredScores(logits, topK: 4, topP: 0.9, temperature: 2)
        let values = filtered.asArray(Float.self)
        #expect(values[0] == 1.5 && values[1] == 1)
        #expect(values[2] == -.infinity && values[3] == -.infinity)
        let first = audio8FilteredScores(logits, topK: 1, topP: 0.01, temperature: 1)
        #expect(isFinite(first).asArray(Bool.self) == [true, false, false, false])
        let uniform = MLXArray([Float(0.5), 0.1, 0.2, 0.9]).reshaped([1, 4])
        #expect(audio8RaceSample(filtered, uniform: uniform).item(Int32.self) == 0)
    }
    @Test func causalConvolutionLengthAndNoFutureLeak() throws {
        let conv = Audio8CausalConv(1, 1, kernel: 3, stride: 2)
        try conv.update(parameters: ModuleParameters.unflattened([
            "conv.weight": MLXArray.ones([1, 3, 1]), "conv.bias": MLXArray.zeros([1])]), verify: .all)
        let x = MLX.arange(7, dtype: .float32).reshaped([1, 7, 1])
        let y = conv(x)
        #expect(y.shape == [1, 4, 1])
        #expect(y.asArray(Float.self) == [1, 6, 12, 11])
        let altered = concatenated([x[0..., 0..<4, 0...], MLXArray.zeros([1, 3, 1])], axis: 1)
        #expect(max(abs(conv(altered)[0..., 0..<2, 0...] - y[0..., 0..<2, 0...])).item(Float.self) == 0)
        let up = Audio8CausalTranspose(1, 1, kernel: 4, stride: 2)
        #expect(up(y).shape == [1, 8, 1])
    }
    @Test func codebookRemapPreservesPublishedLayouts() {
        let value = MLXArray.zeros([2, 3, 4])
        let result = Audio8Model.sanitize(["codec.quantizer.upsample.1.0.conv.weight": value])
        #expect(result["codec.quantizer.upsample.1.layers.0.conv.weight"]?.shape == [2, 3, 4])
    }
    @Test func quantizedFeedForwardRoundTrip() throws {
        let first = Audio8FeedForward(dim: 64, intermediate: 64)
        quantize(model: first, groupSize: 32, bits: 4)
        let weights = first.parameters().flattened()
        #expect(weights.contains { $0.0 == "w1.scales" })
        let second = Audio8FeedForward(dim: 64, intermediate: 64)
        quantize(model: second, groupSize: 32, bits: 4)
        try second.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let input = MLX.arange(128, dtype: .float32).reshaped([1, 2, 64]) / 128
        #expect(max(abs(first(input) - second(input))).item(Float.self) == 0)
    }
    @Test func cachedSlowMatchesPrefill() throws {
        var c = Audio8Config()
        c.vocabSize = 40; c.dim = 8; c.fastDim = 8; c.nLayer = 1; c.nFastLayer = 1
        c.nHead = 2; c.nLocalHeads = 1; c.fastNHead = 2; c.fastNLocalHeads = 1
        c.headDim = 4; c.fastHeadDim = 4; c.intermediateSize = 16; c.fastIntermediateSize = 16
        c.codebookSize = 4; c.semanticBeginID = 32; c.semanticEndID = 35; c.eosTokenID = 3
        c.maxSeqLen = 16
        let model = Audio8Backbone(c)
        let weights = model.parameters().flattened().map { key, value -> (String, MLXArray) in
            if key.hasSuffix("norm.weight") { return (key, MLXArray.ones(value.shape)) }
            let data = sin(MLX.arange(value.size, dtype: .float32) * 0.17) * 0.05
            return (key, data.reshaped(value.shape))
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        var ids = Array(repeating: Int32(0), count: 33)
        ids[0] = 1; ids[1] = 32; ids[2] = 33
        let prompt = MLXArray(ids).reshaped([1, 11, 3])
        let (full, _) = model.slow(prompt)
        let cache: [KVCache] = [KVCacheSimple()]
        _ = model.slow(prompt[0..., 0..., 0..<2], cache: cache)
        let (last, _) = model.slow(prompt[0..., 0..., 2..<3], offset: 2, cache: cache)
        // Prefill and single-query Metal attention use different reduction kernels.
        #expect(max(abs(full[0..., 2..<3, 0...] - last)).item(Float.self) < 1e-3)
    }
}
