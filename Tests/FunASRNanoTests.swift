import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing
@testable import MLXAudioSTT

struct FunASRNanoTests {
    private func tinyConfig() -> FunASRNanoConfig {
        FunASRNanoConfig(
            inputSize: 56,
            frontend: .init(nMels: 8),
            audioEncoder: .init(outputSize: 8, attentionHeads: 2, linearUnits: 16,
                numBlocks: 2, tpBlocks: 1, kernelSize: 3),
            audioAdaptor: .init(downsampleRate: 2, ffnDim: 16, llmDim: 8,
                encoderDim: 8, numLayers: 1, attentionHeads: 2),
            textConfig: .init(vocabSize: 32, hiddenSize: 8, intermediateSize: 16,
                numHiddenLayers: 1, numAttentionHeads: 2, numKeyValueHeads: 1, headDim: 4)
        )
    }

    @Test func configAliasesAndLanguagePrompts() throws {
        let json = #"{"input_size":56,"llm_config":{"hidden_size":8},"audio_encoder_conf":{"sanm_shfit":2},"audio_adaptor_conf":{"downsample_rate":2}}"#
        let config = try JSONDecoder().decode(FunASRNanoConfig.self, from: Data(json.utf8))
        #expect(config.inputSize == 56)
        #expect(config.textConfig.hiddenSize == 8)
        #expect(config.audioEncoder.sanmShift == 2)
        #expect(config.audioAdaptor.downsampleRate == 2)
        #expect(FunASRNanoModel.promptLanguage("YUE") == "中文")
        #expect(FunASRNanoModel.promptLanguage("en") == "英文")
        #expect(FunASRNanoModel.promptLanguage("ja") == "日文")
        #expect(FunASRNanoModel.promptLanguage(" English ") == "英文")
        #expect(FunASRNanoModel.promptLanguage("Chinese") == "中文")
        #expect(FunASRNanoModel.promptLanguage("Japanese") == "日文")
        #expect(FunASRNanoModel.promptLanguage("auto") == nil)
        #expect(FunASRNanoModel.normalizedText("  hello/sil\n world ") == "hello world")
        #expect([1, 8, 9, 17].map(FunASRNanoAudio.fakeTokenLength) == [1, 1, 2, 3])
    }

    @Test func tinyModelForwardAndCachedContinuation() throws {
        let model = FunASRNanoModel(tinyConfig())
        model.train(false)
        // Keep the inference comparison independent of random initializers and
        // materialize the same bounded weights before either forward path.
        var weights: [String: MLXArray] = [:]
        for (name, parameter) in model.parameters().flattened() {
            if name.hasSuffix(".bias") {
                weights[name] = .zeros(parameter.shape)
            } else if parameter.ndim == 1 {
                weights[name] = .ones(parameter.shape)
            } else {
                weights[name] = (0.02 * sin(arange(parameter.size, dtype: .float32) * 0.13)).reshaped(parameter.shape)
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        let audio = 0.1 * sin(arange(1600, dtype: .float32) * 0.13)
        let features = model.extractFeatures(audio).expandedDimensions(axis: 0)
        let adapted = model.audioAdaptor(model.audioEncoder(features))
        #expect(adapted.shape == [1, 1, 8])
        #expect(MLX.all(isFinite(adapted)).item(Bool.self))

        let ids = MLXArray([Int32(1), 4, 6]).reshaped(1, 3)
        let logits = model(inputIds: ids)
        #expect(logits.shape == [1, 3, 32])
        #expect(MLX.all(isFinite(logits)).item(Bool.self))
        eval(logits)

        let cache: [KVCache] = [KVCacheSimple()]
        let prefix = model(inputIds: ids[0..., 0..<2], cache: cache)
        eval(prefix)
        #expect(cache[0].offset == 2)
        let continuation = model(inputIds: ids[0..., 2..<3], cache: cache)
        #expect(continuation.shape == [1, 1, 32])
        #expect(cache[0].offset == 3)
        #expect(MLX.all(isFinite(continuation)).item(Bool.self))
        let expected = logits[0..., 2..<3, 0...]
        let difference = continuation - expected
        let maximumError = abs(difference).max().item(Float.self)
        let referenceScale = abs(expected).max().item(Float.self)
        let relativeRMS = sqrt(difference.square().mean()).item(Float.self)
            / max(sqrt(expected.square().mean()).item(Float.self), 1e-7)
        print("FunASRNano cached/full: max_abs=\(maximumError), reference_max_abs=\(referenceScale), relative_rms=\(relativeRMS)")
        #expect(maximumError < 1e-3,
            "Cached/full difference \(maximumError), reference scale \(referenceScale), relative RMS \(relativeRMS)")
    }

    @Test func stereoFeaturesMatchMono() {
        let model = FunASRNanoModel(tinyConfig())
        let audio = 0.1 * sin(arange(1600, dtype: .float32) * 0.13)
        let mono = model.extractFeatures(audio)
        let stereo = model.extractFeatures(stacked([audio, audio], axis: -1))
        #expect(stereo.shape == mono.shape)
        #expect(abs(stereo - mono).max().item(Float.self) < 1e-5)
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

    @Test func hotwordsAndITNControlPrompts() {
        let words = FunASRNanoModel.resolveHotwords(["", " MLX ", " Apple "], context: nil)
        #expect(words == ["MLX", "Apple"])
        #expect(FunASRNanoModel.resolveHotwords([], context: " MLX, Apple ") == ["MLX, Apple"])
        let normalized = FunASRNanoModel.promptText(language: "English", hotwords: words, itn: true)
        let spoken = FunASRNanoModel.promptText(language: "en", hotwords: words, itn: false)
        #expect(normalized.contains("热词列表：[MLX, Apple]"))
        #expect(normalized.hasSuffix("语音转写成英文："))
        #expect(spoken.contains("热词列表：[MLX, Apple]"))
        #expect(spoken.hasSuffix("语音转写成英文，不进行文本规整："))
        #expect(FunASRNanoModel.promptText(language: "en", hotwords: [], itn: false)
            == "语音转写成英文，不进行文本规整：")
    }
}

/// Opt-in: requires prepared local weights; never downloads in CI.
struct FunASRNanoCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["FUN_ASR_MODEL_DIR"] != nil))
    func localCheckpointTranscription() async throws {
        let directory = URL(fileURLWithPath: try #require(ProcessInfo.processInfo.environment["FUN_ASR_MODEL_DIR"]))
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("fun-loader-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let checkpoint = temporary.appendingPathComponent("checkpoint")
        try FileManager.default.createSymbolicLink(at: checkpoint, withDestinationURL: directory)
        let loaded = try await STT.loadModel(modelRepo: checkpoint.path)
        let model = try #require(loaded as? FunASRNanoModel)
        let audioURL = try #require(Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media"))
        let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: model.sampleRate)
        let parameters = STTGenerateParameters(maxTokens: 256, topP: 1, language: "English")
        let output = loaded.generate(audio: audio, generationParameters: parameters)
        #expect(!output.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(output.generationTokens > 0)
        #expect(!model.generate(audio: audio).text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        var streamResult: STTOutput?
        for try await event in loaded.generateStream(audio: audio, generationParameters: parameters) {
            if case .result(let result) = event { streamResult = result }
        }
        #expect(streamResult?.text == output.text)
        let controlled = model.generate(audio: audio, generationParameters: parameters,
            hotwords: ["intention"], itn: false, prefillStepSize: 128)
        #expect(!controlled.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        var controlledStream: STTOutput?
        for try await event in model.generateStream(audio: audio, generationParameters: parameters,
            hotwords: ["intention"], itn: false, prefillStepSize: 128) {
            if case .result(let result) = event { controlledStream = result }
        }
        #expect(controlledStream?.text == controlled.text)
        let contextual = model.generate(audio: audio, generationParameters: parameters,
            context: "intention", itn: false, prefillStepSize: 128)
        #expect(contextual.text == controlled.text)
        #expect(!model.generate(audio: audio, itn: false).text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }
}
