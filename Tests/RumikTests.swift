import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXAudioTTS
import MLXAudioCore

struct RumikTests {
    static func tiny() -> RumikConfiguration {
        var c = RumikConfiguration()
        c.hiddenSize = 64; c.headDim = 16; c.hiddenLayers = 4
        c.attentionHeads = 4; c.kvHeads = 2; c.intermediateSize = 128
        c.vocabularySize = 128; c.firstUnitID = 32; c.lastUnitID = 95
        c.codebookSize = 8; c.audioEndTokenID = 96; c.slidingWindow = 4
        return c
    }

    private static func fill(_ m: RumikModel) {
        let values = m.parameters().flattened().map { key, p -> (String, MLXArray) in
            // Exact binary fractions keep the generated weights identical on CPU and GPU.
            let values: [Float] = (0..<p.size).map { Float(($0 * 17 + p.size) % 31 - 15) / 256 }
            var v = MLXArray(values)
            if key.contains("norm") && key.hasSuffix("weight") { v = v + 1 }
            return (key, v.reshaped(p.shape))
        }
        m.update(parameters: ModuleParameters.unflattened(values))
    }

    @Test func promptAndCodeMajorFrames() throws {
        let m = RumikModel(Self.tiny())
        #expect(try m.buildPrompt(text: "  नमस्ते <laugh>  ", voice: "IRA", description: " happy ")
                == "<text>Ira: <description=\"happy\"> नमस्ते <laugh><audio>")
        #expect(try m.buildPrompt(text: "<description=\"sad\"> hello", description: "happy")
                == "<text>Ira: <description=\"sad\"> hello<audio>")
        #expect(throws: AudioGenerationError.self) { try m.buildPrompt(text: "hello", voice: "unknown") }
        var partial: [Int32] = []
        #expect(m.pushAudioToken(32 + 3 * 8, frame: &partial) == nil)
        #expect(partial == [3])
        #expect(m.pushAudioToken(32 + 2 * 8 + 2, frame: &partial) == nil)
        #expect(partial.isEmpty)
        let tokens = [32, 33, 40] + (0..<8).map { 32 + 5 * 8 + $0 } + [96]
        let codes = try m.audioTokensToCodes(tokens)
        #expect(codes.shape == [1, 8, 1])
        #expect(codes.asArray(Int32.self) == Array(repeating: 5, count: 8))
        #expect(throws: AudioGenerationError.self) { try m.audioTokensToCodes([32, 33, 96]) }
    }

    @Test func localGlobalCacheMatchesUncachedAcrossWindow() {
        let c = Self.tiny(); let m = RumikModel(c)
        Self.fill(m)
        let tokens = MLXArray((0..<13).map { Int32($0 % 20) }).reshaped(1, 13)
        let full = m(tokens)
        eval(full)
        let cache = m.makeCache()
        #expect(cache[0] is RotatingKVCache)
        #expect(cache[3] is KVCacheSimple)
        let prefill = m(tokens[0..., ..<3], cache: cache)
        eval(prefill)
        var pieces = [prefill]
        for i in 3..<13 {
            let piece = m(tokens[0..., i..<(i+1)], cache: cache)
            eval(piece)
            pieces.append(piece)
        }
        let stepped = concatenated(pieces, axis: 1)
        // Prefill and single-token SDPA kernels also differ slightly in the Python reference.
        #expect(MLX.abs(full - stepped).max().item(Float.self) < 0.002)
        // Python CPU reference with the same generated tiny parameters.
        let expected = MLXArray([Float(-0.46393615), -0.29505327, 0.51430446, -0.09731726,
                                 0.05295812, 0.10113847, 0.30472639, 0.61862969])
        #expect(MLX.abs(stepped[0, -1, ..<8] - expected).max().item(Float.self) < 2e-5)
        #expect(cache.allSatisfy { $0.offset == 13 })
        #expect(m.model.layers.map { $0.attention.isLocal } == [true, true, true, false])
    }

    @Test func quantizedLoadRoundTripAndSlicedHead() throws {
        let c = Self.tiny(); let original = RumikModel(c)
        quantize(model: original, groupSize: 32, bits: 4) { path, _ in !path.hasPrefix("stop_predictor") }
        #expect(original.model.embedding is QuantizedEmbedding)
        #expect(original.model.layers[0].mlp.gate is QuantizedLinear)
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("weights.safetensors")
        try save(arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()), url: file)
        let loaded = RumikModel(c)
        quantize(model: loaded, groupSize: 32, bits: 4) { path, _ in !path.hasPrefix("stop_predictor") }
        try loaded.update(parameters: ModuleParameters.unflattened(try loadArrays(url: file)), verify: .all)
        let input = MLXArray([Int32(2), 7, 9]).reshaped(1, 3)
        #expect(MLX.abs(original(input) - loaded(input)).max().item(Float.self) < 1e-6)
        let h = loaded.model(input)
        let head = RumikAudioHead(embedding: loaded.model.embedding, config: c)
        let all = loaded.model.embedding.asLinear(h)
        let rows = MLXArray((c.firstUnitID...c.lastUnitID).map(Int32.init) + [Int32(c.audioEndTokenID)])
        #expect(MLX.abs(head(h) - all[0..., 0..., rows]).max().item(Float.self) < 1e-5)
        #expect(loaded.stopPredictor.layers[1] is Linear)
        #expect(!(loaded.stopPredictor.layers[1] is QuantizedLinear))
    }

    @Test func cohere2NumericalReference() {
        func output(on device: Device) -> (hidden: MLXArray, logits: MLXArray, stop: MLXArray) {
            Device.withDefaultDevice(device) {
                var c = Self.tiny(); c.logitScale = 0.25
                let m = RumikModel(c)
                Self.fill(m)
                let input = MLXArray([Int32(2), 7, 9, 4, 6, 3]).reshaped(1, 6)
                let hidden = m.model(input)
                let logits = m(input)
                let stop = m.stopProbability(hidden[0..., -1, 0...])
                eval(hidden, logits, stop)
                return (hidden, logits, stop)
            }
        }

        // Python mlx-audio Cohere2/Rumik CPU references using the weights from fill(_:).
        let cpu = output(on: .cpu)
        #expect(cpu.hidden.shape == [1, 6, 64])
        #expect(cpu.logits.shape == [1, 6, 128])
        let expectedHidden = MLXArray([Float(1.70226634), -0.81456667, -0.19158809, -0.40276322,
                                      -0.02495334, 1.82497811, -0.95095789, -0.52419353])
        let hiddenError = MLX.abs(cpu.hidden[0, -1, ..<8] - expectedHidden).max().item(Float.self)
        #expect(hiddenError < 2e-5, "Rumik CPU hidden reference error: \(hiddenError)")
        let expectedLogits = MLXArray([Float(-0.00992873), 0.03266027, 0.11107627, -0.05016552,
                                      0.01744533, -0.02986014, -0.00575606, 0.00077971])
        let logitsError = MLX.abs(cpu.logits[0, -1, ..<8] - expectedLogits).max().item(Float.self)
        #expect(logitsError < 2e-5, "Rumik CPU logits reference error: \(logitsError)")
        #expect(abs(cpu.stop.item(Float.self) - 0.49237058) < 2e-6)

        // Keep architecture parity strict above, then allow GPU matmul rounding,
        // including TF32, when comparing the complete outputs against the CPU result.
        let gpu = output(on: .gpu)
        #expect(gpu.hidden.shape == cpu.hidden.shape)
        #expect(gpu.logits.shape == cpu.logits.shape)
        let gpuHiddenError = MLX.abs(gpu.hidden - cpu.hidden).max().item(Float.self)
        let gpuLogitsError = MLX.abs(gpu.logits - cpu.logits).max().item(Float.self)
        #expect(gpuHiddenError < 3e-3, "Rumik GPU hidden rounding error: \(gpuHiddenError)")
        #expect(gpuLogitsError < 1e-3, "Rumik GPU logits rounding error: \(gpuLogitsError)")
        #expect(MLX.abs(gpu.stop - cpu.stop).max().item(Float.self) < 2e-6)
    }

    @Test func configAndOriginalStopKeys() throws {
        let data = Data("{\"first_unit_id\":10,\"codebook_size\":16}".utf8)
        let config = try JSONDecoder().decode(RumikConfiguration.self, from: data)
        #expect(config.lastUnitID == 137)
        let m = RumikModel(Self.tiny())
        let weights = m.sanitize(weights: ["stop_predictor.1.weight": MLXArray.zeros([64, 64]),
                                          "lm_head.weight": MLXArray.zeros([1]),
                                          "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([1])])
        #expect(Set(weights.keys) == ["stop_predictor.layers.1.weight"])
        #expect(TTS.resolveModelType(modelRepo: "rumik-ai/rumik-oss-1-mlx-8bit") == "rumik_oss")
    }
    @Test func failedGenerationReleasesModel() async {
        var model: RumikModel? = RumikModel(Self.tiny())
        weak var weakModel = model
        do {
            _ = try await model!.generate(text: "Hello")
            Issue.record("An unloaded model should reject generation")
        } catch {}
        model = nil
        #expect(weakModel == nil)
    }

}
