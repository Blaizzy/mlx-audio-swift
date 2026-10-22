import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioSTT

struct FunASRNanoTests {
    private func fixture() throws -> [String: Any] {
        let url = try #require(Bundle.module.url(forResource: "fun_asr_nano_reference", withExtension: "json", subdirectory: "media"))
        return try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    private func expected(_ name: String, in fixture: [String: Any]) throws -> MLXArray {
        let row = try #require(fixture[name] as? [String: Any])
        let shape = try #require(row["shape"] as? [Int])
        let values = try #require(row["values"] as? [Double])
        return MLXArray(values.map(Float.init)).reshaped(shape)
    }

    private func compare(_ actual: MLXArray, _ expected: MLXArray, stage: String, atol: Float = 5e-4, rtol: Float = 2e-4) {
        #expect(actual.shape == expected.shape)
        let error = abs(actual.asType(.float32) - expected.asType(.float32))
        let maximumError = error.max().item(Float.self)
        let meanError = error.mean().item(Float.self)
        print("FunASRNano \(stage): max_abs=\(maximumError), mean_abs=\(meanError)")
        #expect(MLX.all(error .<= (atol + rtol * abs(expected))).item(Bool.self), "\(stage) exceeded numerical tolerance")
    }

    @Test func configAliasesAndLanguagePrompts() throws {
        let json = #"{"input_size":56,"llm_config":{"hidden_size":8},"audio_encoder_conf":{"sanm_shfit":2},"audio_adaptor_conf":{"downsample_rate":2}}"#
        let config = try JSONDecoder().decode(FunASRNanoConfig.self, from: Data(json.utf8))
        #expect(config.inputSize == 56)
        #expect(config.textConfig.hiddenSize == 8)
        #expect(config.audioEncoder.sanmShift == 2)
        #expect(config.audioAdaptor.downsampleRate == 2)
        #expect(try FunASRNanoModel.promptLanguage("YUE") == "中文")
        #expect(try FunASRNanoModel.promptLanguage("en") == "英文")
        #expect(try FunASRNanoModel.promptLanguage("ja") == "日文")
        #expect(try FunASRNanoModel.promptLanguage("auto") == nil)
        #expect(throws: (any Error).self) { try FunASRNanoModel.promptLanguage("ko") }
        #expect(try FunASRNanoModel.resolveHotwords(["", " MLX "], context: nil) == ["MLX"])
        #expect(try FunASRNanoModel.resolveHotwords([], context: " MLX, Apple ") == ["MLX, Apple"])
        #expect(throws: (any Error).self) { try FunASRNanoModel.resolveHotwords(["MLX"], context: "Apple") }
        #expect(try FunASRNanoModel.promptText(language: "en", hotwords: [], itn: false) == "语音转写成英文，不进行文本规整：")
        #expect(FunASRNanoModel.normalizedText("  hello/sil\n world ") == "hello world")
        #expect([1, 8, 9, 17].map(FunASRNanoAudio.fakeTokenLength) == [1, 1, 2, 3])
    }

    @Test func PythonNumericalReference() throws {
        let fixture = try fixture()
        let json = try JSONSerialization.data(withJSONObject: #require(fixture["config"]))
        let config = try JSONDecoder().decode(FunASRNanoConfig.self, from: json)
        let model = FunASRNanoModel(config)
        var weights: [String: MLXArray] = [:]
        let expectedShapes = try #require(fixture["weights"] as? [String: [Int]])
        #expect(Set(model.parameters().flattened().map(\.0)) == Set(expectedShapes.keys))
        for (key, value) in model.parameters().flattened() {
            #expect(value.shape == expectedShapes[key])
            let phase = Float(key.utf8.reduce(0) { $0 + Int($1) } % 31) / 10
            var weight = sin(arange(value.size, dtype: .float32) * 0.13 + phase).reshaped(value.shape) * 0.05
            if key.hasSuffix("weight"), key.contains("norm") { weight = weight + 1 }
            weights[key] = weight
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let audio = MLXArray(try #require(fixture["audio"] as? [Double]).map(Float.init))
        compare(FunASRNanoAudio.fbank(audio, config: config.frontend), try expected("fbank", in: fixture), stage: "fbank")
        let features = try model.extractFeatures(audio).expandedDimensions(axis: 0)
        compare(features, try expected("features", in: fixture), stage: "features")
        let encoded = model.audioEncoder(features)
        compare(encoded, try expected("encoded", in: fixture), stage: "encoder")
        compare(model.audioAdaptor(encoded), try expected("adapted", in: fixture), stage: "adaptor")
        let ids = MLXArray([Int32(1), 4, 6]).reshaped(1, 3)
        compare(model(inputIds: ids), try expected("logits", in: fixture), stage: "decoder")
        let cache: [KVCache] = [KVCacheSimple()]
        let prefix = model(inputIds: ids[0..., 0..<2], cache: cache)
        eval(prefix)
        let tail = model(inputIds: ids[0..., 2..<3], cache: cache)
        compare(tail, try expected("logits", in: fixture)[0..., 2..<3, 0...], stage: "cached decoder")
    }

    @Test func streamingPropagatesInitializationErrors() async throws {
        let fixture = try fixture()
        let json = try JSONSerialization.data(withJSONObject: #require(fixture["config"]))
        let model = FunASRNanoModel(try JSONDecoder().decode(FunASRNanoConfig.self, from: json))
        do {
            for try await _ in model.generateStream(audio: .zeros([1600]), generationParameters: .init()) { }
            Issue.record("Expected an uninitialized model error")
        } catch let error as STTError {
            #expect(error.localizedDescription.contains("tokenizer"))
        }
    }

    @Test func sanitizerPreservesPreparedLayouts() {
        let upstream = MLXArray.ones([8, 1, 3])
        let native = MLXArray.ones([8, 3, 1])
        let sanitized = FunASRNanoModel.sanitize(weights: [
            "module.audio_encoder.encoders0.0.self_attn.fsmn_block.weight": upstream,
            "audio_encoder.encoders.0.self_attn.fsmn_block.weight": native,
            "llm.lm_head.weight": .ones([32, 8]),
        ])
        #expect(sanitized["audio_encoder.encoders0.0.self_attn.fsmn_block.weight"]?.shape == [8, 3, 1])
        #expect(sanitized["audio_encoder.encoders.0.self_attn.fsmn_block.weight"]?.shape == [8, 3, 1])
        #expect(sanitized["llm.lm_head.weight"] == nil)
    }

    @Test func invalidInputThrows() throws {
        let fixture = try fixture()
        let json = try JSONSerialization.data(withJSONObject: #require(fixture["config"]))
        let model = FunASRNanoModel(try JSONDecoder().decode(FunASRNanoConfig.self, from: json))
        #expect(throws: (any Error).self) { try model.extractFeatures(.zeros([0])) }
        #expect(throws: (any Error).self) { try model.extractFeatures(.zeros([1, 1, 400])) }
        #expect(throws: (any Error).self) { try model.transcribe(audio: .zeros([1600])) }
    }
}

/// Opt-in: requires prepared local weights and Python parity tensors; never downloads in CI.
struct FunASRNanoCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["FUN_ASR_MODEL_DIR"] != nil))
    func checkpointParityAndTranscription() async throws {
        let env = ProcessInfo.processInfo.environment
        let directory = URL(fileURLWithPath: try #require(env["FUN_ASR_MODEL_DIR"]))
        let parity = URL(fileURLWithPath: try #require(env["FUN_ASR_PARITY_DIR"]))
        let reference = try MLX.loadArrays(url: parity.appendingPathComponent("reference.safetensors"))
        let metadata = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: parity.appendingPathComponent("reference.json"))) as? [String: Any])
        let loaded = try await STT.loadModel(modelRepo: directory.path, modelType: "fun_asr_nano")
        let model = try #require(loaded as? FunASRNanoModel)
        let audio = try #require(reference["audio"])
        let features = try model.extractFeatures(audio).expandedDimensions(axis: 0)
        let encoded = model.audioEncoder(features)
        let adapted = model.audioAdaptor(encoded)
        let (ids, embeds) = try model.prepareInputs(audio: audio, language: "en", hotwords: [], itn: true)
        let stages: [(String, MLXArray, Float, Float)] = [
            ("features", features, 0.002, 0.0005), ("encoder", encoded, 0.03, 0.02),
            ("adaptor", adapted, 0.03, 0.02), ("embeddings", embeds, 0.03, 0.02)
        ]
        for (name, actual, atol, rtol) in stages {
            let expected = try #require(reference[name])
            #expect(actual.shape == expected.shape)
            let difference = abs(actual.asType(.float32) - expected.asType(.float32))
            let referenceRMS = sqrt(expected.asType(.float32).square().mean()).item(Float.self)
            let relativeRMS = sqrt(difference.square().mean()).item(Float.self) / max(referenceRMS, 1e-7)
            let a = actual.asType(.float32), b = expected.asType(.float32)
            let cosine = ((a * b).sum() / sqrt(a.square().sum() * b.square().sum())).item(Float.self)
            print("FunASRNano checkpoint \(name): max_abs=\(difference.max().item(Float.self)), mean_abs=\(difference.mean().item(Float.self)), ref_rms=\(referenceRMS), relative_rms=\(relativeRMS), cosine=\(cosine)")
            if name == "features" {
                #expect(MLX.all(difference .<= (atol + rtol * abs(expected))).item(Bool.self))
            } else {
                #expect(relativeRMS < 0.01)
                #expect(cosine > 0.9999)
            }
        }
        #expect(MLX.all(ids .== (try #require(reference["prompt_ids"])) ).item(Bool.self))
        let logits = model(inputEmbeddings: embeds)[0, -1, 0...].asType(.float32)
        let expectedLogits = try #require(reference["logits"])
        let diff = abs(logits - expectedLogits)
        print("FunASRNano checkpoint logits: max_abs=\(diff.max().item(Float.self)), mean_abs=\(diff.mean().item(Float.self))")
        let centeredLogits = logits - logits.mean()
        let centeredExpected = expectedLogits - expectedLogits.mean()
        let centeredError = sqrt((centeredLogits - centeredExpected).square().mean()).item(Float.self)
        let centeredScale = sqrt(centeredExpected.square().mean()).item(Float.self)
        print("FunASRNano checkpoint centered_logits_relative_rms=\(centeredError / max(centeredScale, 1e-7))")
        let relativeLogitRMS = sqrt(diff.square().mean()).item(Float.self)
            / max(sqrt(expectedLogits.square().mean()).item(Float.self), 1e-7)
        let expectedProbability = softmax(expectedLogits)
        let probability = softmax(logits)
        let kl = (expectedProbability * (logSoftmax(expectedLogits) - logSoftmax(logits))).sum().item(Float.self)
        let variation = 0.5 * abs(expectedProbability - probability).sum().item(Float.self)
        print("FunASRNano checkpoint raw_logits_relative_rms=\(relativeLogitRMS), probability_KL=\(kl), total_variation=\(variation)")
        #expect(centeredError / max(centeredScale, 1e-7) < 0.05)
        #expect(kl < 0.001)
        #expect(variation < 0.01)
        #expect(logits.argMax().item(Int.self) == expectedLogits.argMax().item(Int.self))
        let output = try model.transcribe(audio: audio, generationParameters: .init(maxTokens: 256, topP: 1, language: "en"))
        let expectedText = try #require(metadata["transcript"] as? String)
        print("FunASRNano Swift transcript: \(output.text)")
        print("FunASRNano Python transcript: \(expectedText)")
        #expect(!output.text.isEmpty)
        #expect(output.text == expectedText)
        var streamResult: STTOutput?
        for try await event in model.transcribeStream(audio: audio,
            generationParameters: .init(maxTokens: 256, topP: 1, language: "en")) {
            if case .result(let result) = event { streamResult = result }
        }
        #expect(streamResult?.text == output.text)
    }
}
