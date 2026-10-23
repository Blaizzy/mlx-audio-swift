import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioSTT

private func tinyVibeVoiceConfig(
    quantized: Bool = false, streaming: Bool = false, tied: Bool = false
) -> VibeVoiceConfig {
    var config = VibeVoiceConfig()
    var audio = VibeVoiceTokenizerConfig(vaeDim: 4)
    audio.encoderNFilters = 4
    audio.encoderRatios = [2, 2]
    audio.encoderDepths = [1, 1, 2]
    config.acousticTokenizerConfig = audio
    audio.vaeDim = 8
    config.semanticTokenizerConfig = audio
    config.acousticVaeDim = 4
    config.semanticVaeDim = 8
    let dim = quantized ? 64 : 8
    config.decoderConfig = Qwen2Configuration(
        hiddenSize: dim, hiddenLayers: 2, intermediateSize: dim * 2,
        attentionHeads: 2, rmsNormEps: 1e-6, vocabularySize: 320, kvHeads: 1,
        tieWordEmbeddings: tied)
    config.speechTokenCompressRatio = 4
    if streaming {
        config.chunkFrames = 3
        config.lookaheadFrames = 1
        config.normalizeAudio = false
    }
    return config
}

private func writeVibeVoiceTokenizer(_ dir: URL, configOnlyDelimiter: Bool = false) throws {
    var vocab = Dictionary(
        uniqueKeysWithValues: (1 ..< 128).map { (String(UnicodeScalar($0)!), $0) })
    vocab["<unk>"] = 0
    let special = [
        "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
        "<|text_chunk_end|>", "<|endoftext|>",
    ]
    var added: [[String: Any]] = [["id": 0, "content": "<unk>", "special": true]]
    for (i, token) in special.enumerated() {
        if !(configOnlyDelimiter && i == 5) {
            vocab[token] = 300 + i
            added.append(["id": 300 + i, "content": token, "special": true])
        }
    }
    let data: [String: Any] = [
        "model": ["type": "BPE", "vocab": vocab, "merges": [], "unk_token": "<unk>"],
        "added_tokens": added,
    ]
    try JSONSerialization.data(withJSONObject: data).write(
        to: dir.appendingPathComponent("tokenizer.json"))
    let config: [String: Any] = [
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
        "eos_token": "<|endoftext|>",
        "added_tokens_decoder": ["305": ["content": "<|text_chunk_end|>", "special": true]],
    ]
    try JSONSerialization.data(withJSONObject: config).write(
        to: dir.appendingPathComponent("tokenizer_config.json"))
}

private func writeTinyVibeVoice(
    _ dir: URL, sourceLayout: Bool = false, quantized: Bool = false,
    streaming: Bool = false, tied: Bool = false, nextToken: Int = 97
) throws {
    let config = tinyVibeVoiceConfig(quantized: quantized, streaming: streaming, tied: tied)
    let model = VibeVoiceASRModel(config)
    var weights = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, MLXArray.zeros($0.1.shape))
        })
    let dim = config.decoderConfig.hiddenSize
    var embedding = MLXArray.ones([320, dim])
    embedding[nextToken] = .ones([dim]) * 2
    weights["language_model.model.embed_tokens.weight"] = embedding
    weights["language_model.model.norm.weight"] = .ones([dim])
    if !tied {
        weights["language_model.lm_head.weight"] = concatenated(
            [.zeros([nextToken, dim]), .ones([1, dim]), .zeros([319 - nextToken, dim])], axis: 0)
    }
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    if quantized {
        quantize(model: model) { path, _ in
            path.hasPrefix("language_model.") ? (32, 4, .affine) : nil
        }
    }
    weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    if sourceLayout {
        weights = Dictionary(
            uniqueKeysWithValues: weights.map { key, array in
                var key = key
                var value = array
                if key.contains("conv") && value.ndim == 3 { value = value.transposed(0, 2, 1) }
                if key.hasPrefix("language_model.lm_head.") {
                    key = String(key.dropFirst("language_model.".count))
                } else {
                    key = key.replacingOccurrences(
                        of: "language_model.model.", with: "language_model.")
                    key = key.replacingOccurrences(
                        of: #"\.downsample_layers\.(\d+)\.conv\."#,
                        with: ".downsample_layers.$1.0.conv.conv.", options: .regularExpression)
                    key = key.replacingOccurrences(of: ".head.conv.", with: ".head.conv.conv.")
                    key = key.replacingOccurrences(
                        of: ".mixer.conv.conv.", with: ".mixer.conv.conv.conv.")
                    key = "model." + key
                }
                return (key, value)
            })
        weights["model.acoustic_tokenizer.decoder.unused.weight"] = .zeros([1])
        weights["model.acoustic_tokenizer.fix_std"] = MLXArray(Float(0.5))
    }
    eval(Array(weights.values))
    try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
    var json =
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as! [String: Any]
    if quantized { json["quantization"] = ["group_size": 32, "bits": 4] }
    if streaming {
        json.removeValue(forKey: "chunk_frames")
        json.removeValue(forKey: "lookahead_frames")
        json["normalize_audio"] = true
        try JSONSerialization.data(withJSONObject: [
            "chunk_frames": 3, "lookahead_frames": 1, "normalize_audio": false,
        ]).write(to: dir.appendingPathComponent("preprocessor_config.json"))
    }
    try JSONSerialization.data(withJSONObject: json).write(
        to: dir.appendingPathComponent("config.json"))
    try writeVibeVoiceTokenizer(dir, configOnlyDelimiter: streaming)
}

@Suite(.serialized)
struct VibeVoiceTests {
    private func directory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            "vibevoice-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    @Test func defaultsAliasesAndDepthFormats() throws {
        let config = try JSONDecoder().decode(VibeVoiceConfig.self, from: Data("{}".utf8))
        try config.validate()
        #expect(config.decoderConfig.hiddenSize == 3584)
        #expect(config.speechTokenCompressRatio == 3200)
        #expect(config.normalizeAudio)
        let encoder = try JSONDecoder().decode(
            VibeVoiceTokenizerConfig.self,
            from: Data(#"{"encoder_depths":"1-2-3","encoder_ratios":[2,2]}"#.utf8))
        #expect(encoder.encoderDepths == [1, 2, 3])
        let roundTrip = try JSONDecoder().decode(
            VibeVoiceConfig.self, from: JSONEncoder().encode(tinyVibeVoiceConfig()))
        try roundTrip.validate()
        var bad = roundTrip
        bad.acousticTokenizerConfig.encoderDepths = [1]
        #expect(throws: STTError.self) { try bad.validate() }
        #expect(STT.inferModelType(from: "microsoft/VibeVoice-ASR-Streaming-1.5B") == "vibevoice")
    }
    @Test func causalConvolutionPadsStridesAndDepthwiseMatchesStandardKernel() throws {
        let conv = VibeVoiceConv(2, 2, kernel: 7, stride: 2, depthwise: true)
        let x = sin(arange(34).asType(.float32)).reshaped(1, 17, 2)
        let expected = conv.conv(
            MLX.padded(x, widths: [.init((0, 0)), .init((5, 1)), .init((0, 0))]))
        let actual = conv(x)
        #expect(actual.shape == [1, 9, 2])
        #expect(abs(actual - expected).max().item(Float.self) < 1e-5)
    }
    @Test func boundedEncoderPreservesFrameAlignmentAndCausalContext() {
        let encoder = VibeVoiceEncoder(tinyVibeVoiceConfig().acousticTokenizerConfig)
        let x = sin(arange(813).asType(.float32) / 17).reshaped(1, -1)
        let full = encoder.forward(x)
        let chunked = encoder(x, chunkSamples: 101)
        #expect(full.shape == [1, 204, 4])
        #expect(full.shape == chunked.shape)
        #expect(abs(full - chunked).max().item(Float.self) < 1e-5)
    }
    @Test func loudnessNormalizationAndStereoInput() {
        let model = VibeVoiceASRModel(tinyVibeVoiceConfig())
        let x = sin(arange(400).asType(.float32) / 17)
        let normalized = model.prepareAudio(x)
        #expect(
            abs(sqrt(mean(normalized * normalized)).item(Float.self) - pow(Float(10), -25 / 20))
                < 1e-5)
        #expect(
            abs(model.prepareAudio(stacked([x, x], axis: -1)) - normalized).max().item(Float.self)
                == 0)
        #expect(abs(model.prepareAudio(.zeros([400]))).max().item(Float.self) == 0)
        let streaming = VibeVoiceASRModel(tinyVibeVoiceConfig(streaming: true))
        #expect(abs(streaming.prepareAudio(x)[0] - x).max().item(Float.self) == 0)
    }
    @Test func offlinePromptContainsDurationContextAndSpeechPositions() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeVibeVoiceTokenizer(dir)
        let model = VibeVoiceASRModel(tinyVibeVoiceConfig())
        model.tokenizer = try await loadSTTTokenizer(from: dir)
        try model.prepareTokenizer()
        let context = VibeVoiceASRModel.context("Names", hotwords: [" MLX ", "", "VibeVoice"])
        #expect(context == "Names\nMLX, VibeVoice")
        let ids = model.buildPrompt(frames: 3, duration: 1.234, context: context)
        let decoded = model.tokenizer!.decode(
            tokens: ids.asArray(Int.self), skipSpecialTokens: false)
        #expect(decoded.contains("1.23 seconds audio"))
        #expect(decoded.contains("Names\nMLX, VibeVoice"))
        #expect(ids.asArray(Int.self).filter { $0 == model.speechPadId }.count == 3)
        let features = arange(24).asType(.float32).reshaped(1, 3, 8)
        let merged = model.mergeEmbeddings(
            MLXArray([Int32(302), 304, 304, 304, 303]).reshaped(1, 5), features: features)
        #expect(abs(merged[0, 1 ..< 4] - features[0]).max().item(Float.self) == 0)
    }
    @Test(arguments: [false, true])
    func strictSharedLoaderSupportsNativeAndPreparedWeights(sourceLayout: Bool) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyVibeVoice(dir, sourceLayout: sourceLayout)
        let common = try await STT.loadModel(modelRepo: dir.path)
        let model = try #require(common as? VibeVoiceASRModel)
        let parameters = STTGenerateParameters(maxTokens: 3)
        #expect(
            common.generate(audio: .zeros([31]), generationParameters: parameters).text == "aaa")
        #expect(
            model.generate(
                audio: .zeros([31]), generationParameters: parameters, prefillStepSize: 2
            ).text == "aaa")
        var streamed = ""
        for try await event in common.generateStream(
            audio: .zeros([31]), generationParameters: parameters)
        {
            if case .token(let text) = event { streamed += text }
        }
        #expect(streamed == "aaa")
        var weights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        eval(Array(weights.values))
        let key = weights.keys.first { $0.contains("acoustic_connector.fc1.weight") }!
        weights.removeValue(forKey: key)
        try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
        await #expect(throws: (any Error).self) {
            try await VibeVoiceASRModel.fromModelDirectory(dir)
        }
    }
    @Test(arguments: [false, true])
    func quantizedLanguageModelSupportsTiedEmbeddingsAndKVCompression(tied: Bool) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyVibeVoice(dir, quantized: true, tied: tied)
        let model = try await VibeVoiceASRModel.fromModelDirectory(dir)
        #expect(model.languageModel.embedTokens(MLXArray([Int32(97)])).dtype.isFloatingPoint)
        #expect(
            model.generate(
                audio: .zeros([31]),
                generationParameters: .init(maxTokens: 2, kvBits: 4, kvGroupSize: 32)
            ).text == "aa")
    }
    @Test func streamingSidecarWindowsPaddingAndPersistentDelimiter() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyVibeVoice(dir, sourceLayout: true, streaming: true)
        let model = try await VibeVoiceASRModel.fromModelDirectory(dir)
        #expect(model.isStreamingModel)
        let common: any STTGenerationModel = model
        #expect(common.sampleRate == 24000)
        #expect(!model.config.normalizeAudio)
        #expect(model.streamingChunkSamples == 12)
        #expect(model.streamingWindowSamples == 16)
        let state = model.makeStreamingState(hotwords: ["MLX"])
        let offset = state.cache[0].offset
        let features = model.encodeSpeech(.zeros([16]))
        #expect(features.shape == [1, 4, 8])
        let part = model.streamingGenerateStep(
            audioFeatures: features, state: state, generationParameters: .init(maxTokens: 2))
        #expect(part == "aa")
        #expect(state.cache[0].offset == offset + 2 + 4 + 2 + 1)
        #expect(state.generationTokens == 2)
        _ = model.streamingGenerateStep(
            audioFeatures: features, state: state, generationParameters: .init(maxTokens: 2))
        #expect(state.cache[0].offset == offset + 18)
        let output = model.generate(audio: .zeros([25]), generationParameters: .init(maxTokens: 2))
        #expect(output.text == "aaaaaa")
        #expect(output.segments?.count == 3)
        #expect(output.generationTokens == 6)
        #expect(abs((output.segments?.last?["end"] as! Double) - 25.0 / 24000) < 1e-8)
        var streamed = ""
        for try await event in model.generateStream(
            audio: .zeros([25]), generationParameters: .init(maxTokens: 2))
        {
            if case .token(let text) = event { streamed += text }
        }
        #expect(streamed == output.text)
    }
    @Test(arguments: [305, 306])
    func streamingStopsAndStillCommitsTheChunkDelimiter(token: Int) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyVibeVoice(dir, streaming: true, nextToken: token)
        let model = try await VibeVoiceASRModel.fromModelDirectory(dir)
        let state = model.makeStreamingState()
        let offset = state.cache[0].offset
        let result = model.streamingGenerateStep(
            audioFeatures: .zeros([1, 4, 8]), state: state,
            generationParameters: .init(maxTokens: 2))
        #expect(result.isEmpty)
        #expect(state.cache[0].offset == offset + 7)
    }
    @Test func streamingLoaderRejectsMissingDelimiterMetadata() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyVibeVoice(dir, streaming: true)
        let path = dir.appendingPathComponent("tokenizer_config.json")
        var json = try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
        json.removeValue(forKey: "added_tokens_decoder")
        try JSONSerialization.data(withJSONObject: json).write(to: path)
        await #expect(throws: STTError.self) { try await VibeVoiceASRModel.fromModelDirectory(dir) }
    }
    @Test func structuredOutputPreservesSpeakersAndTimestamps() {
        let text =
            #"prefix ```json [{"Start time":1.2,"End time":3.4,"Speaker ID":2,"Content":"brackets [in] text"}] ``` suffix"#
        let result = VibeVoiceASRModel.parseTranscription(text)
        #expect(result.count == 1)
        #expect(result[0]["start"] as? Double == 1.2)
        #expect(result[0]["speaker_id"] as? Int == 2)
        #expect(result[0]["text"] as? String == "brackets [in] text")
        #expect(VibeVoiceASRModel.parseTranscription("incomplete [{").isEmpty)
    }
}

@Suite(.serialized)
struct VibeVoiceCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["VIBEVOICE_MODEL_DIR"] != nil))
    func publishedCheckpointTranscribesAndStreams() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["VIBEVOICE_MODEL_DIR"])
        let model = try await STT.loadModel(modelRepo: path)
        #expect(model is VibeVoiceASRModel)
        let url = try #require(
            Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")
        )
        let (_, audio) = try loadAudioArray(from: url, sampleRate: 24000)
        let output = model.generate(audio: audio, generationParameters: .init(maxTokens: 256))
        #expect(output.text.lowercased().contains("intention"))
        var text = ""
        for try await event in model.generateStream(
            audio: audio, generationParameters: .init(maxTokens: 256))
        {
            if case .token(let delta) = event { text += delta }
        }
        #expect(text.trimmingCharacters(in: .whitespacesAndNewlines) == output.text)
    }
}
