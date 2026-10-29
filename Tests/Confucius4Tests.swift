import Foundation
import MLX
import MLXAudioCore
import Testing
@testable import MLXAudioTTS

@Suite("Confucius4 native inference", .serialized)
struct Confucius4Tests {
    func tensor(_ shape: [Int], offset: Int = 0) -> MLXArray {
        MLXArray((0..<shape.reduce(1, *)).map { Float(($0 + offset) % 7 - 3) / 16 }).reshaped(shape)
    }
    func tinyGPT() -> Confucius4T2S {
        var w: [String: MLXArray] = [:]
        for (p, shape) in [("semantic_embedding", [6, 4]), ("semantic_position_embedding.embedding", [16, 4])] { w[p + ".weight"] = tensor(shape) }
        func norm(_ p: String) { w[p + ".weight"] = MLXArray([Float(0.75), 1, 1.25, 1.5]); w[p + ".bias"] = tensor([4]) }
        norm("transformer.ln_f"); norm("final_norm")
        for i in 0..<2 {
            let p = "transformer.h.\(i)."
            norm(p + "ln_1"); norm(p + "ln_2")
            for (n, shape) in [("attn.c_attn", [4, 12]), ("attn.c_proj", [4, 4]), ("mlp.c_fc", [4, 8]), ("mlp.c_proj", [8, 4])] {
                w[p + n + ".weight"] = tensor(shape, offset: i); w[p + n + ".bias"] = tensor([shape[1]], offset: i)
            }
        }
        w["semantic_head.weight"] = tensor([6, 4]); w["semantic_head.bias"] = tensor([6])
        return Confucius4T2S(w, heads: 2, layers: 2, bos: 4, eos: 5)
    }
    @Test func gptMatchesCPUReferenceAndIncrementalCache() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = tinyGPT(), input = tensor([1, 3, 4])
            let (full, _) = try model.transformer(input)
            let expected: [Float] = [-1.34366608, -0.09828973, 0.27509171, 1.86715651, -1.37659073, -0.20502625,
                1.24698114, 0.92684358, -1.34689677, -0.14895317, 0.37579346, 1.82877111]
            #expect(abs(full - MLXArray(expected).reshaped([1, 3, 4])).max().item(Float.self) < 2e-6)
            let logits: [Float] = [0.07652539, -0.65075791, 0.24629170, -0.50137150, 0.41605806, -0.17257011,
                0.03445189, -0.23871960, 0.19929466, -0.62864929, 0.36413741, -0.25949520,
                0.07737923, -0.62921065, 0.24784029, -0.52670264, 0.41830128, -0.17636988]
            #expect(abs(model.head(full) - MLXArray(logits).reshaped([1, 3, 6])).max().item(Float.self) < 2e-6)
            let (_, cache) = try model.transformer(input[0..., 0..<2, 0...])
            let (incremental, next) = try model.transformer(input[0..., 2..<3, 0...], cache: cache)
            #expect(next.count == 2)
            #expect(next[0].0.shape == [1, 2, 3, 2])
            #expect(abs(incremental - full[0..., 2..<3, 0...]).max().item(Float.self) < 2e-6)
        }
    }
    @Test func quantizedLinearAndGPTConvLayouts() {
        let weight = tensor([4, 64])
        let packed = quantized(weight, groupSize: 32, bits: 8)
        let weights = Confucius4Weights(["q.weight": packed.0, "q.scales": packed.1, "q.biases": packed.2!, "q.bias": tensor([4])], groupSize: 32)
        let x = tensor([1, 2, 64])
        let expected = matmul(x, dequantized(packed.0, scales: packed.1, biases: packed.2, groupSize: 32, bits: 8).T) + tensor([4])
        #expect(abs(weights.linear(x, "q") - expected).max().item(Float.self) < 2e-5)
        let plain = Confucius4Weights(["p.weight": weight.T, "p.bias": tensor([4])])
        #expect(abs(plain.linear(x, "p", convLayout: true) - (matmul(x, weight.T) + tensor([4]))).max().item(Float.self) < 1e-6)
    }
    @Test func samplingExcludesNucleusCrossingTokenAndPenalizesBothSigns() {
        let logits: [Float] = [4, 3, 2, -2]
        #expect(Confucius4T2S.sample(logits, generated: [], temperature: 1, topK: 0, topP: 0.7, repetitionPenalty: 1, uniform: 0.99) == 0)
        #expect(Confucius4T2S.sample(logits, generated: [0, 0], temperature: 0, topK: 0, topP: 1, repetitionPenalty: 10, uniform: 0) == 1)
        #expect(Confucius4T2S.sample([-1, -2], generated: [0], temperature: 0, topK: 0, topP: 1, repetitionPenalty: 10, uniform: 0) == 1)
    }
    @Test func conformerMatchesCPUReference() throws {
        try Device.withDefaultDevice(.cpu) {
            var w: [String: MLXArray] = [:]
            func norm(_ p: String, size: Int = 4) {
                w[p + ".weight"] = MLXArray((0..<size).map { Float($0) / 4 + 0.75 }); w[p + ".bias"] = tensor([size])
            }
            func pair(_ p: String, _ shape: [Int]) { w[p + ".weight"] = tensor(shape); w[p + ".bias"] = tensor([shape[0]]) }
            norm("feature_projection.layer_norm", size: 6); pair("feature_projection.projection", [4, 6])
            let p = "encoder.layers.0."
            for n in ["ffn1_layer_norm", "self_attn_layer_norm", "ffn2_layer_norm", "final_layer_norm"] { norm(p + n) }
            for n in ["ffn1", "ffn2"] { pair(p + n + ".intermediate_dense", [8, 4]); pair(p + n + ".output_dense", [4, 8]) }
            for n in ["linear_q", "linear_k", "linear_v", "linear_out"] { pair(p + "self_attn." + n, [4, 4]) }
            w[p + "self_attn.distance_embedding.weight"] = tensor([73, 2])
            for n in ["layer_norm", "depthwise_layer_norm"] { norm(p + "conv_module." + n) }
            for (n, shape) in [("pointwise_conv1", [8, 4, 1]), ("depthwise_conv", [4, 1, 31]), ("pointwise_conv2", [4, 4, 1])] { w[p + "conv_module." + n + ".weight"] = tensor(shape) }
            let model = Confucius4W2VBert(w, heads: 2, layers: 1)
            let output = try model(tensor([1, 4, 6]))
            let expected: [Float] = [0.99050236, -0.02492028, -0.79162800, -1.63117075,
                -0.92246616, 1.52737236, -0.22821619, -0.80976695,
                -1.31127656, -0.40596375, 1.31883097, 1.01140177,
                -0.99388665, -0.69303322, 0.00796083, 2.38027000]
            #expect(abs(output - MLXArray(expected).reshaped([1, 4, 4])).max().item(Float.self) < 3e-6)
        }
    }
    @Test func reflectPaddingAndFeatureStacking() {
        let input = MLXArray([Float(0), 1, 2, 3]).reshaped([1, 4, 1])
        #expect(Confucius4Weights.reflect(input, 2).asArray(Float.self) == [2, 1, 0, 1, 2, 3, 2, 1])
        let audio = sin(MLX.arange(3200).asType(.float32) / 16) / 4
        let mel = MLXArray.ones([257, 80]) / 256
        let window = MLXArray.ones([400])
        let features = Confucius4Features.fbank(audio, mel: mel, window: window)
        #expect(features.shape == [1, 9, 160])
        #expect(all(isFinite(features)).item(Bool.self))
    }
    @Test func rotaryExtensionAndAdaptiveNormalization() {
        let freqs = stacked([MLXArray.ones([2, 2]), MLXArray.zeros([2, 2])], axis: -1)
        let model = Confucius4Acoustic(["decoder.estimator.freqs_cis": freqs,
            "n.norm.weight": MLXArray.ones([4]), "n.modulation.weight": MLXArray.zeros([8, 4]),
            "n.modulation.bias": MLXArray([Float(1), 1, 1, 1, 0, 0, 0, 0])], heads: 1, depth: 1, waveLayers: 1)
        let x = tensor([1, 5, 1, 4])
        #expect(abs(model.rope(x) - x).max().item(Float.self) == 0)
        #expect(model.frequencies.shape == [5, 2, 2])
        let normalized = model.adaptiveNorm(x.reshaped([1, 5, 4]), MLXArray.zeros([1, 4]), "n")
        let expected = x.reshaped([1, 5, 4]) * rsqrt((x * x).mean(axis: -1, keepDims: true).reshaped([1, 5, 1]) + 1e-5)
        #expect(abs(normalized - expected).max().item(Float.self) < 1e-6)
    }
    @Test func polyphaseResamplingMatchesCPUReference() {
        let input: [Float] = [0, 0.25, -0.5, 0.75, -1, 0.5, 0.25, 0]
        let expected: [Float] = [0.13502169, 0.37175655, -0.41528559, -0.13173831,
            0.63767290, -0.34631595, -0.79120767, 0.49283704, 0.62420917,
            -0.16534711, 0.02055285, 0.12995031]
        let result = Confucius4Features.resample(input, from: 16000, to: 22050)
        #expect(result.count == expected.count)
        #expect(zip(result, expected).allSatisfy { abs($0 - $1) < 6e-8 })
        #expect(Confucius4Features.resample(input, from: 16000, to: 16000) == input)
    }
    @Test func tinyAcousticPipelinePreservesPromptAndProducesFiniteAudioFeatures() throws {
        try Device.withDefaultDevice(.cpu) {
            var w: [String: MLXArray] = [:]
            func pair(_ p: String, _ shape: [Int], bias: Bool = true) {
                w[p + ".weight"] = tensor(shape) / 4
                if bias { w[p + ".bias"] = tensor([shape[0]]) / 4 }
            }
            func adaptive(_ p: String) {
                w[p + ".norm.weight"] = MLXArray.ones([4]); pair(p + ".modulation", [8, 4])
                w[p + ".modulation.bias"] = MLXArray([Float(1), 1, 1, 1, 0, 0, 0, 0])
            }
            let p = "decoder.estimator."
            w[p + "freqs_cis"] = stacked([MLXArray.ones([8, 1]), MLXArray.zeros([8, 1])], axis: -1)
            for name in ["t_embedder", "t_embedder2"] { pair(p + name + ".time_mlp.0", [4, 8]); pair(p + name + ".time_mlp.2", [4, 4]) }
            pair(p + "input_embed.mu_projection", [4, 4]); pair(p + "input_embed.proj", [4, 11])
            for i in 0..<3 {
                let q = p + "transformer_blocks.\(i)."
                adaptive(q + "attention_norm"); adaptive(q + "ffn_norm")
                pair(q + "attention.wqkv", [12, 4], bias: false); pair(q + "attention.wo", [4, 4], bias: false)
                pair(q + "feed_forward.w1", [8, 4], bias: false); pair(q + "feed_forward.w3", [8, 4], bias: false)
                pair(q + "feed_forward.w2", [4, 8], bias: false)
                if i == 2 { pair(q + "skip_in_linear", [4, 8]) }
            }
            adaptive(p + "transformer_norm"); pair(p + "skip_linear", [4, 6]); pair(p + "conv1", [4, 4])
            pair(p + "res_projection", [4, 4]); pair(p + "final_layer.adaLN_modulation.1", [8, 4]); pair(p + "final_layer.linear", [4, 4])
            pair(p + "conv2", [2, 4, 1]); pair(p + "wavenet.cond_layer.conv", [16, 4, 1])
            for i in 0..<2 { pair(p + "wavenet.in_layers.\(i).conv", [8, 4, 5]); pair(p + "wavenet.res_skip_layers.\(i).conv", [i == 0 ? 8 : 4, 4, 1]) }
            w["input_embedding.embedding.weight"] = tensor([8, 2]); pair("input_embedding.out_project", [4, 2, 1])
            pair("encoder_proj", [4, 8]); pair("length_regulator.content_in_proj", [4, 4])
            for (ci, gi) in [(0, 1), (3, 4), (6, 7), (9, 10)] {
                pair("length_regulator.model.\(ci)", [4, 4, 3])
                w["length_regulator.model.\(gi).weight"] = MLXArray.ones([4]); w["length_regulator.model.\(gi).bias"] = MLXArray.zeros([4])
            }
            pair("length_regulator.model.12", [4, 4, 1]); w["prompt_cond"] = tensor([1, 1, 4])
            let model = Confucius4Acoustic(w, heads: 2, depth: 3, waveLayers: 2)
            let mu = model.buildMu(codes: MLXArray([Int32(0), 1, 2]).reshaped([1, 3]), latent: tensor([1, 3, 4]), referenceFrames: 2)
            #expect(mu.shape == [1, 7, 4])
            #expect(abs(mu[0..., ..<2] - broadcast(w["prompt_cond"]!, to: [1, 2, 4])).max().item(Float.self) == 0)
            let result = try model.solve(noise: tensor([1, 2, 7]), prompt: tensor([1, 2, 2]), mu: mu, speaker: tensor([1, 3]), steps: 3, guidance: 0.7)
            #expect(result.shape == [1, 2, 7])
            #expect(all(isFinite(result)).item(Bool.self))
            #expect(abs(result[0..., 0..., ..<2]).max().item(Float.self) == 0)
            #expect(abs(result[0..., 0..., 2...]).max().item(Float.self) > 0)
        }
    }
    @Test func configDefaultsRoutingAndLanguagePrompts() throws {
        let config = try JSONDecoder().decode(Confucius4Config.self, from: Data("{}".utf8))
        #expect(config.sampleRate == 22050 && config.quantBits == 8 && config.quantGroupSize == 64)
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/Confucius4-TTS-mlx-int8") == "confucius4")
        #expect(Confucius4Model.prompt(text: "Hello", language: "en") == "You are a helpful assistant. 请用英文朗读接下来的文字:Hello")
        #expect(Confucius4Model.prompt(text: "Hi", language: "unknown").contains("请用英文"))
        #expect(Confucius4Model.prompt(text: "Hi", language: nil).contains("请用越南语"))
    }
    @Test func rejectsIncompleteCheckpoint() throws {
        #expect(throws: (any Error).self) { try Confucius4Checkpoint.validate(w2v: [:], t2s: [:], s2a: [:], config: .init()) }
    }
    @Test func localLoaderRejectsMissingComponents() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try Data("{\"model_type\":\"confucius4\"}".utf8).write(to: folder.appendingPathComponent("config.json"))
        await #expect(throws: (any Error).self) { try await TTS.loadModel(modelRepo: folder.path) }
    }
}
