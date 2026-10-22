import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioSTT

private final class MegaTestLinear: Module {
    let projection = Linear(4, 3, bias: false)
}

private final class MegaTestTextCore: Module { let layers = [MegaTestLinear()] }
private final class MegaTestTextRoot: Module { let model = MegaTestTextCore() }

struct MegaASRTests {
    private func tinyRouter() -> MegaASRAudioQualityRouter {
        MegaASRAudioQualityRouter(
            dimensions: 4, heads: 2, feedforward: 8, layers: 1, frontendHidden: 2,
            classifierHidden: 2, maxFrames: 16)
    }

    private func tinyRouterWeights() -> [String: MLXArray] {
        Dictionary(
            uniqueKeysWithValues: tinyRouter().parameters().flattened().map { name, value in
                var key = name
                for (native, sidecar) in [
                    ("frontend.conv1.", "frontend.conv.0."),
                    ("frontend.bn1.", "frontend.conv.1."),
                    ("frontend.conv2.", "frontend.conv.4."),
                    ("frontend.bn2.", "frontend.conv.5."),
                    ("classifier.fc1.", "classifier.0."),
                    ("classifier.fc2.", "classifier.3."),
                ] {
                    key = key.replacingOccurrences(of: native, with: sidecar)
                }
                let tensor: MLXArray =
                    key.hasSuffix("running_var") ? .ones(value.shape) : .zeros(value.shape)
                return (key, tensor)
            })
    }

    @Test func routerSelectsByClassifierProbability() throws {
        var weights = tinyRouterWeights()
        weights["classifier.3.bias"] = MLXArray([Float(0), 2])
        let router = try MegaASRAudioQualityRouter.fromWeights(weights, heads: 2)
        let audio = MLXArray.zeros([1600])
        let logits = router.logits(audio: audio)
        #expect(MLX.all(logits .== MLXArray([Float(0), 2])).item(Bool.self))
        let degraded = router.route(audio: audio)
        #expect(abs(degraded.degradedProbability - 0.8807971) < 1e-6)
        #expect(degraded.usesLoRA)

        try router.update(
            parameters: .unflattened(["classifier.fc2.bias": MLXArray([Float(2), 0])]),
            verify: [.noUnusedKeys, .shapeMismatch])
        #expect(!router.route(audio: audio).usesLoRA)
    }

    @Test func loraRestoresExactOriginalAfterRepeatedToggles() throws {
        let model = MegaTestLinear()
        let original = (MLXArray.ones([3, 4]) * 0.01).asType(.bfloat16)
        try model.update(parameters: .unflattened(["projection.weight": original]), verify: .all)
        let factors = [
            "projection.lora_A": MLXArray.ones([2, 4]),
            "projection.lora_B": MLXArray([Float(0.5), 0.5, 1, 1, 2, 2]).reshaped(3, 2),
        ]
        // Updates of 1, 2, and 4 round differently in BF16. Subtracting those updates
        // would lose the original 0.01 values, so each toggle must restore the snapshot.
        let adapted = MLXArray(
            [Float](repeating: 1.0078125, count: 4)
                + [Float](repeating: 2.015625, count: 4)
                + [Float](repeating: 4, count: 4)
        ).reshaped(3, 4).asType(.bfloat16)
        let state = try MegaASRLoRAState(model: model, factors: factors)
        for _ in 0..<20 {
            state.setActive(true)
            #expect(MLX.all(model.projection.weight .== adapted).item(Bool.self))
            state.setActive(true)
            #expect(MLX.all(model.projection.weight .== adapted).item(Bool.self))
            state.setActive(false)
            #expect(MLX.all(model.projection.weight .== original).item(Bool.self))
        }
        #expect(!state.active)
    }

    @Test func loraComposesPythonTextLayerAliases() throws {
        let model = MegaTestTextRoot()
        let base = MLXArray.zeros([3, 4], dtype: .bfloat16)
        let a = MLXArray([Float(1), 0, 0, 0, 0, 1, 0, 0]).reshaped(2, 4)
        let b = MLXArray([Float(1), 2, 3, 4, 5, 6]).reshaped(3, 2)
        try model.update(
            parameters: .unflattened(["model.layers.0.projection.weight": base]), verify: .all)
        let state = try MegaASRLoRAState(
            model: model,
            factors: [
                "layers.0.projection.lora_A": a, "layers.0.projection.lora_B": b,
                "model.layers.0.projection.lora_A": a * 0.5, "model.layers.0.projection.lora_B": b,
            ])
        state.setActive(true)
        let expected = MLXArray([Float(1.5), 3, 0, 0, 4.5, 6, 0, 0, 7.5, 9, 0, 0])
            .reshaped(3, 4).asType(.bfloat16)
        #expect(MLX.all(model.model.layers[0].projection.weight .== expected).item(Bool.self))
        state.setActive(false)
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
        var weights = tinyRouterWeights()
        weights.removeValue(forKey: "classifier.3.weight")
        #expect(throws: (any Error).self) {
            try MegaASRAudioQualityRouter.fromWeights(weights, heads: 2)
        }
    }

    @Test func loaderRequiresSidecarsBeforeAllocatingBackbone() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"model_type":"mega_asr"}"#.utf8).write(
            to: directory.appendingPathComponent("config.json"))
        do {
            _ = try await STT.loadModel(modelRepo: directory.path)
            Issue.record("Expected missing Mega-ASR sidecars")
        } catch STTError.modelNotInitialized(let message) {
            #expect(message.contains("Missing Mega-ASR sidecar"))
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
    private func audio(named name: String) throws -> MLXArray {
        let url = try #require(
            Bundle.module.url(forResource: name, withExtension: "wav", subdirectory: "media"))
        return try loadAudioArray(from: url, sampleRate: 16000).1
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MEGA_ASR_MODEL_DIR"] != nil))
    func cleanNoisyAndRestoredClean() async throws {
        let modelPath = try #require(ProcessInfo.processInfo.environment["MEGA_ASR_MODEL_DIR"])
        #expect(FileManager.default.fileExists(atPath: modelPath))
        let model = try await MegaASRModel.fromModelDirectory(URL(fileURLWithPath: modelPath))
        let sharedModel: any STTGenerationModel = model
        let cleanAudio = try audio(named: "conversational_a")
        var firstClean: String?
        for (name, degraded) in [
            ("conversational_a", false), ("noisy_audio", true), ("conversational_a", false),
        ] {
            let waveform = try audio(named: name)
            let route = model.router.route(audio: waveform)
            #expect(route.usesLoRA == degraded)
            let output = sharedModel.generate(
                audio: waveform,
                generationParameters: .init(maxTokens: 256, topP: 1, language: "English"))
            #expect(!output.text.isEmpty)
            if !degraded {
                if let firstClean {
                    #expect(output.text == firstClean)
                } else {
                    firstClean = output.text
                }
            }
        }
        // Cancellation cannot leave a child decoder running while the next request changes adapters.
        let boxed = UncheckedSendableBox(model)
        let waveform = UncheckedSendableBox(cleanAudio)
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
        let restored = model.transcribe(
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
}
