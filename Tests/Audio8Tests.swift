import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioTTS

@Suite("Audio8")
struct Audio8Tests {
    @Test func defaultsAndHybridConfiguration() throws {
        let c = try JSONDecoder().decode(Audio8Config.self, from: Data("{}".utf8))
        try c.validate()
        #expect(c.codecSampleRate == 44100 && c.codecFrameSize == 2048)
        #expect(c.codecPostIntermediateSize == 1216 && c.codecPostNLocalHeads == 8)
        let hybrid = try JSONDecoder().decode(Audio8Config.self, from: Data(#"{"slow_backbone":"falcon_h1","embedding_multiplier":0.10888671875,"mamba_d_ssm":768,"mamba_n_heads":24,"mamba_d_head":32}"#.utf8))
        try hybrid.validate()
        #expect(hybrid.usesFalconSlow && hybrid.embeddingMultiplier == 0.10888671875)
        var invalid = hybrid; invalid.mambaDSSM = 42
        #expect(throws: (any Error).self) { try invalid.validate() }
        invalid = hybrid; invalid.ssmMultipliers = [1, 1, 2, 1, 1]
        #expect(throws: (any Error).self) { try invalid.validate() }
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Audio8-TTS-Preview-0.1b-bf16") == "arktts")
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Audio8-TTS-Preview-0.6b-bf16") == "arktts")
    }

    private func tinyHybrid() -> Audio8Config {
        var c = Audio8Config()
        c.slowBackbone = "falcon_h1"; c.vocabSize = 40; c.dim = 8; c.fastDim = 8
        c.nLayer = 1; c.nFastLayer = 1; c.nHead = 2; c.nLocalHeads = 1
        c.fastNHead = 2; c.fastNLocalHeads = 1; c.headDim = 4; c.fastHeadDim = 4
        c.intermediateSize = 16; c.fastIntermediateSize = 16; c.codebookSize = 4
        c.semanticBeginID = 32; c.semanticEndID = 35; c.eosTokenID = 3; c.maxSeqLen = 16
        c.mambaDConv = 3; c.mambaDHead = 2; c.mambaDSSM = 4; c.mambaDState = 2
        c.mambaNHeads = 2; c.embeddingMultiplier = 0.25; c.normFastlayerInput = false
        return c
    }

    private func hybridWithBinaryWeights(_ configuration: Audio8Config? = nil) throws -> Audio8Backbone {
        let model = Audio8Backbone(configuration ?? tinyHybrid())
        let weights = model.parameters().flattened().map { key, value -> (String, MLXArray) in
            if key.contains("norm"), key.hasSuffix("weight") { return (key, .ones(value.shape)) }
            if key.hasSuffix("A_log") || key.hasSuffix("dt_bias") { return (key, .zeros(value.shape)) }
            if key.hasSuffix(".D") { return (key, .ones(value.shape) * 0.25) }
            let binary = ((MLX.arange(value.size) % 7).asType(.float32) - 3) / 32
            return (key, binary.reshaped(value.shape))
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        return model
    }

    @Test func hybridCPUReferenceAndCompositeScaling() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try hybridWithBinaryWeights()
            var ids = Array(repeating: Int32(0), count: 33)
            ids[0] = 1; ids[1] = 32; ids[2] = 33; ids[4] = 2; ids[5] = 3
            let prompt = MLXArray(ids).reshaped([1, 11, 3])
            #expect(model.embed(prompt).asArray(Float.self) == [
                -0.0625, -0.03125, 0, 0.03125, 0.0625, 0.09375, -0.09375, -0.0625,
                -0.03125, 0.09375, 0.21875, -0.09375, 0.03125, -0.0625, -0.15625, -0.03125,
                0.03125, 0.15625, 0.0625, -0.03125, -0.125, 0, -0.09375, 0.03125])
            let (logits, hidden) = model.slow(prompt)
            // Python CPU goldens from exactly representable, generated binary weights.
            let expected = MLXArray([Float(0.1786539406), 0.3126541972, -0.1130608320, -0.3452344239, -0.0918479636,
                -0.2435497046, 0.1882997006, 0.2014431804, 0.0210063532, 0.2014828026,
                -0.3050138354, -0.0006508753, 0.0105348453, 0.3113185763, 0.0520081446]).reshaped([1, 3, 5])
            #expect(max(abs(logits - expected)).item(Float.self) < 2e-6)
            let expectedLastHidden = MLXArray([Float(0.2412566543), 1.6153098345, -0.3623270094, 1.2198843956,
                -1.3405461311, -0.0166694932, -1.3569083214, 0.2412566543]).reshaped([1, 1, 8])
            #expect(max(abs(hidden[0..., 2..<3, 0...] - expectedLastHidden)).item(Float.self) < 2e-6)
            let cache = model.makeSlowCache()
            _ = model.slow(prompt[0..., 0..., 0..<2], cache: cache)
            let (last, lastHidden) = model.slow(prompt[0..., 0..., 2..<3], offset: 2, cache: cache)
            #expect(max(abs(last - logits[0..., 2..<3, 0...])).item(Float.self) < 2e-6)
            #expect(max(abs(lastHidden - hidden[0..., 2..<3, 0...])).item(Float.self) < 2e-6)
            let recurrent = (cache[0] as! CacheList)[0] as! ArraysCache
            let state = recurrent[1]!
            #expect(state.shape == [1, 2, 2, 2] && state.dtype == .float32)
            let expectedState = MLXArray([Float(-0.0019461134), -0.0014502321, -0.0012388660, -0.0010128669,
                -0.0000053121, -0.0000570626, -0.0002577688, -0.0001376873]).reshaped(state.shape)
            #expect(max(abs(state - expectedState)).item(Float.self) < 2e-8)
            #expect(recurrent[0]?.shape == [1, 2, 8] && (cache[0] as! CacheList)[1].offset == 3)
        }
    }

    @Test func hybridBF16CPUReference() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = try hybridWithBinaryWeights()
            try model.update(parameters: ModuleParameters.unflattened(model.parameters().flattened().map {
                ($0.0, $0.1.asType(.bfloat16))
            }), verify: .all)
            var ids = Array(repeating: Int32(0), count: 33)
            ids[0] = 1; ids[1] = 32; ids[2] = 33; ids[4] = 2; ids[5] = 3
            let prompt = MLXArray(ids).reshaped([1, 11, 3])
            let (logits, _) = model.slow(prompt)
            let expected = MLXArray([Float(0.1787109375), 0.3125, -0.11279296875, -0.34375, -0.091796875,
                -0.2431640625, 0.1884765625, 0.201171875, 0.0201416015625, 0.201171875,
                -0.3046875, -0.00119781494140625, 0.0101318359375, 0.3125, 0.051513671875]).reshaped([1, 3, 5])
            #expect(max(abs(logits.asType(.float32) - expected)).item(Float.self) == 0)
        }
    }

    @Test func hybridMetalCachedStateAndQuantizedProjections() throws {
        var c = tinyHybrid()
        c.dim = 64; c.fastDim = 64; c.headDim = 32; c.fastHeadDim = 32
        c.intermediateSize = 64; c.fastIntermediateSize = 64; c.mambaDHead = 32; c.mambaDSSM = 64
        let model = try hybridWithBinaryWeights(c)
        quantize(model: model, groupSize: 32, bits: 4, filter: { path, _ in
            path.hasSuffix("in_proj") || path.hasSuffix("out_proj") || path.hasSuffix("semantic_output")
        })
        #expect(model.parameters().flattened().contains { $0.0 == "slow.layers.0.mamba.in_proj.scales" })
        var ids = Array(repeating: Int32(0), count: 33)
        ids[0] = 1; ids[1] = 32; ids[2] = 33
        let prompt = MLXArray(ids).reshaped([1, 11, 3])
        let (full, _) = model.slow(prompt)
        let cache = model.makeSlowCache()
        _ = model.slow(prompt[0..., 0..., 0..<2], cache: cache)
        let (last, _) = model.slow(prompt[0..., 0..., 2..<3], offset: 2, cache: cache)
        let cpu = try Device.withDefaultDevice(.cpu) { () -> MLXArray in
            let copy = try hybridWithBinaryWeights(c)
            quantize(model: copy, groupSize: 32, bits: 4, filter: { path, _ in
                path.hasSuffix("in_proj") || path.hasSuffix("out_proj") || path.hasSuffix("semantic_output")
            })
            try copy.update(parameters: model.parameters(), verify: .all)
            let (fullCPU, _) = copy.slow(prompt)
            let cpuCache = copy.makeSlowCache()
            _ = copy.slow(prompt[0..., 0..., 0..<2], cache: cpuCache)
            let (lastCPU, _) = copy.slow(prompt[0..., 0..., 2..<3], offset: 2, cache: cpuCache)
            #expect(max(abs(lastCPU - fullCPU[0..., 2..<3, 0...])).item(Float.self) < 2e-6)
            eval(fullCPU)
            return fullCPU
        }
        // Quantized Metal prefill and one-token matmuls use different reduction
        // kernels. The CPU architecture check above stays tight.
        #expect(max(abs(full - cpu)).item(Float.self) < 1e-3)
        #expect(max(abs(last - cpu[0..., 2..<3, 0...])).item(Float.self) < 1e-3)
        #expect(all(isFinite(last)).item(Bool.self))
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
        let torch = MLX.arange(12, dtype: .float32).reshaped([3, 1, 4])
        let converted = Audio8Model.sanitize(["model.slow.layers.0.mamba.conv1d.weight": torch])
        #expect(converted["model.slow.layers.0.mamba.conv1d.weight"]?.shape == [3, 4, 1])
        let again = Audio8Model.sanitize(converted)
        #expect(again["model.slow.layers.0.mamba.conv1d.weight"]?.asArray(Float.self) == torch.asArray(Float.self))
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
