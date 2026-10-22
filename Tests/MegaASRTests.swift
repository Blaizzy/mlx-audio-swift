import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioSTT

private final class MegaTestLinear: Module {
    let projection = Linear(4, 3, bias: false)
}

private final class MegaTestTextCore: Module { let layers = [MegaTestLinear()] }
private final class MegaTestTextRoot: Module { let model = MegaTestTextCore() }

struct MegaASRTests {
    private func fixture() throws -> [String: MLXArray] {
        let path = try #require(
            Bundle.module.url(
                forResource: "mega_asr_reference", withExtension: "safetensors",
                subdirectory: "media"))
        return try MLX.loadArrays(url: path)
    }
    private func compare(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float, name: String) {
        #expect(actual.shape == expected.shape)
        let error = abs(actual.asType(.float32) - expected.asType(.float32))
        print(
            "Mega-ASR \(name) max_abs=\(error.max().item(Float.self)), mean_abs=\(error.mean().item(Float.self))"
        )
        #expect(error.max().item(Float.self) < tolerance)
    }
    @Test func pythonRouterReference() throws {
        let f = try fixture()
        let raw = Dictionary(
            uniqueKeysWithValues: f.compactMap { key, value in
                key.hasPrefix("weights.") ? (String(key.dropFirst(8)), value) : nil
            })
        let router = try MegaASRAudioQualityRouter.fromWeights(raw)
        let audio = try #require(f["audio"])
        compare(
            MegaASRFrontend.features(audio), try #require(f["mel"]), tolerance: 3e-4,
            name: "tiny frontend")
        compare(
            try router.logits(audio: audio), try #require(f["logits"]), tolerance: 2e-5,
            name: "tiny router")
        let expected = softmax(try #require(f["logits"]), axis: -1)[1].item(Float.self)
        #expect(abs(try router.route(audio: audio).degradedProbability - expected) < 2e-5)
        #expect(throws: (any Error).self) { try router.logits(audio: .zeros([0])) }
        #expect(throws: (any Error).self) { try router.logits(audio: .zeros([50000])) }
    }
    @Test func loraRestoresExactOriginalAfterRepeatedToggles() throws {
        let f = try fixture()
        let model = MegaTestLinear()
        let original = try #require(f["lora_base"])
        try model.update(parameters: .unflattened(["projection.weight": original]), verify: .all)
        let factors = [
            "projection.lora_A": try #require(f["lora_A"]),
            "projection.lora_B": try #require(f["lora_B"]),
        ]
        let state = try MegaASRLoRAState(model: model, factors: factors)
        for _ in 0..<20 {
            try state.setActive(true)
            compare(
                model.projection.weight, try #require(f["lora_adapted"]), tolerance: 1e-6,
                name: "BF16 LoRA"
            )
            try state.setActive(true)
            try state.setActive(false)
            #expect(MLX.all(model.projection.weight .== original).item(Bool.self))
        }
        #expect(!state.active)
    }
    @Test func loraComposesPythonTextLayerAliases() throws {
        let f = try fixture()
        let model = MegaTestTextRoot()
        let base = try #require(f["lora_base"])
        let a = try #require(f["lora_A"])
        let b = try #require(f["lora_B"])
        try model.update(
            parameters: .unflattened(["model.layers.0.projection.weight": base]), verify: .all)
        let state = try MegaASRLoRAState(
            model: model,
            factors: [
                "layers.0.projection.lora_A": a, "layers.0.projection.lora_B": b,
                "model.layers.0.projection.lora_A": a * 0.7, "model.layers.0.projection.lora_B": b,
            ])
        try state.setActive(true)
        let expected =
            (base + matmul(b, a).asType(.bfloat16)) + matmul(b, a * 0.7).asType(.bfloat16)
        compare(
            model.model.layers[0].projection.weight, expected, tolerance: 1e-6, name: "aliased LoRA"
        )
        try state.setActive(false)
        #expect(MLX.all(model.model.layers[0].projection.weight .== base).item(Bool.self))
    }

    @Test func sidecarValidationRejectsIncompleteState() throws {
        let model = MegaTestLinear()
        #expect(throws: (any Error).self) {
            try MegaASRLoRAState(model: model, factors: ["projection.lora_A": .ones([2, 4])])
        }
        #expect(throws: (any Error).self) {
            try MegaASRLoRAState(
                model: model,
                factors: ["other.lora_A": .ones([2, 4]), "other.lora_B": .ones([3, 2])])
        }
        #expect(throws: (any Error).self) { try MegaASRAudioQualityRouter.fromWeights([:]) }
        let f = try fixture()
        var raw = Dictionary(
            uniqueKeysWithValues: f.compactMap { key, value in
                key.hasPrefix("weights.") ? (String(key.dropFirst(8)), value) : nil
            })
        raw.removeValue(forKey: "classifier.3.weight")
        #expect(throws: (any Error).self) { try MegaASRAudioQualityRouter.fromWeights(raw) }
    }
    @Test func loaderRequiresSidecarsBeforeAllocatingBackbone() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"model_type":"mega_asr"}"#.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        await #expect(throws: (any Error).self) {
            try await MegaASRModel.fromModelDirectory(directory)
        }
        let config = try JSONDecoder().decode(
            MegaASRConfig.self,
            from: Data(
                #"{"model_type":"mega_asr","router_config":{"nhead":2},"router_weights":"quality.safetensors"}"#
                    .utf8))
        #expect(config.routerHeads == 2)
        #expect(config.routerWeights == "quality.safetensors")
        #expect(config.loraWeights == "extras/lora.safetensors")
    }
}

struct MegaASRCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MEGA_ASR_MODEL_DIR"] != nil))
    func cleanNoisyAndRestoredClean() async throws {
        let env = ProcessInfo.processInfo.environment
        let modelPath = try #require(env["MEGA_ASR_MODEL_DIR"])
        let root = URL(fileURLWithPath: try #require(env["MEGA_ASR_REFERENCE_ROOT"]))
        let model = try #require(
            try await STT.loadModel(modelRepo: modelPath, modelType: "mega_asr") as? MegaASRModel)
        #expect(throws: (any Error).self) {
            try model.transcribe(audio: .zeros([100, 2]), routing: .base)
        }
        var firstClean: String?
        var cleanAudio: MLXArray?
        for name in ["conversational_a", "noisy_audio", "conversational_a"] {
            let directory = root.appendingPathComponent("mega-reference-" + name)
            let reference = try MLX.loadArrays(
                url: directory.appendingPathComponent("reference.safetensors"))
            let metadata = try #require(
                JSONSerialization.jsonObject(
                    with: Data(contentsOf: directory.appendingPathComponent("reference.json")))
                    as? [String: Any])
            let audio = try #require(reference["audio"])
            let mel = MegaASRFrontend.features(audio)
            let melError = abs(mel - (try #require(reference["router_mel"])))
            let logits = try model.router.logits(audio: audio)
            let error = abs(logits - (try #require(reference["router_logits"])))
            let route = try model.router.route(audio: audio)
            print(
                "Mega-ASR checkpoint \(name): frontend_max=\(melError.max().item(Float.self)), router_max=\(error.max().item(Float.self)), probability=\(route.degradedProbability)"
            )
            #expect(melError.max().item(Float.self) < 5e-4)
            #expect(error.max().item(Float.self) < 5e-3)
            #expect(route.usesLoRA == (name == "noisy_audio"))
            let output = try model.transcribe(
                audio: audio,
                generationParameters: .init(maxTokens: 256, topP: 1, language: "English"))
            let expected = try #require(metadata["text"] as? String)
            let rate = wordErrorRate(output.text, expected)
            print("Mega-ASR Swift \(name): \(output.text)")
            print("Mega-ASR Python \(name): \(expected); normalized disagreement=\(rate)")
            #expect(!output.text.isEmpty)
            #expect(rate <= 0.1)
            if name == "conversational_a" {
                if let firstClean {
                    #expect(output.text == firstClean)
                } else {
                    firstClean = output.text
                    cleanAudio = audio
                }
            }
        }
        // Cancellation cannot leave a child decoder running while the next request changes adapters.
        let boxed = UncheckedSendableBox(model)
        let waveform = UncheckedSendableBox(try #require(cleanAudio))
        let consumer = Task {
            for try await _ in boxed.value.transcribeStream(
                audio: waveform.value,
                generationParameters: .init(maxTokens: 256, language: "English"), routing: .robust)
            {
                try Task.checkCancellation()
            }
        }
        try await Task.sleep(for: .milliseconds(20))
        consumer.cancel()
        _ = try? await consumer.value
        let restored = try model.transcribe(
            audio: waveform.value,
            generationParameters: .init(maxTokens: 256, language: "English"), routing: .base)
        #expect(restored.text == firstClean)
        var streamedResult: STTOutput?
        for try await event in model.transcribeStream(
            audio: waveform.value,
            generationParameters: .init(maxTokens: 256, language: "English"), routing: .base)
        {
            if case .result(let result) = event { streamedResult = result }
        }
        #expect(streamedResult?.text == firstClean)
    }

    private func wordErrorRate(_ actual: String, _ expected: String) -> Double {
        func words(_ s: String) -> [String] {
            s.lowercased().components(separatedBy: CharacterSet.alphanumerics.inverted).filter {
                !$0.isEmpty
            }
        }
        let a = words(actual)
        let b = words(expected)
        var previous = Array(0...b.count)
        for (i, x) in a.enumerated() {
            var row = [i + 1]
            for (j, y) in b.enumerated() {
                row.append(min(previous[j + 1] + 1, row[j] + 1, previous[j] + (x == y ? 0 : 1)))
            }
            previous = row
        }
        return Double(previous[b.count]) / Double(max(1, b.count))
    }
}
