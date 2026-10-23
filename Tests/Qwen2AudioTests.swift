import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioSTT

private func tinyQwen2AudioConfig(quantized: Bool = false) throws -> Qwen2AudioConfig {
    let json =
        #"{"audio_token_index":31,"eos_token_id":30,"audio_config":{"d_model":8,"encoder_layers":2,"encoder_attention_heads":2,"encoder_ffn_dim":16,"num_mel_bins":4,"max_source_positions":4},"text_config":{"hidden_size":8,"num_hidden_layers":2,"intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":0.00001,"vocab_size":32,"rope_theta":10000}}"#
    var config = try JSONDecoder().decode(Qwen2AudioConfig.self, from: Data(json.utf8))
    if quantized {
        config.textConfig.hiddenSize = 64
        config.textConfig.intermediateSize = 128
    }
    return config
}

private func writeQwen2AudioTokenizer(_ directory: URL, configOnly: Bool = false) throws {
    var vocab = Dictionary(uniqueKeysWithValues: (2 ..< 27).map { ("token\($0)", $0) })
    vocab["<unk>"] = 0
    vocab["a"] = 1
    let special = ["<|im_start|>", "<|audio_bos|>", "<|audio_eos|>", "<|im_end|>", "<|AUDIO|>"]
    var added: [[String: Any]] = [["id": 0, "content": "<unk>", "special": true]]
    for (i, token) in special.enumerated() {
        if !configOnly { vocab[token] = i + 27 }
        added.append(["id": i + 27, "content": token, "special": true])
    }
    let data: [String: Any] = [
        "model": ["type": "BPE", "vocab": vocab, "merges": [], "unk_token": "<unk>"],
        "added_tokens": configOnly ? [added[0]] : added,
    ]
    try JSONSerialization.data(withJSONObject: data).write(
        to: directory.appendingPathComponent("tokenizer.json"))
    var config: [String: Any] = [
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
        "eos_token": "<|im_end|>",
    ]
    if configOnly {
        config["added_tokens_decoder"] = Dictionary(
            uniqueKeysWithValues: added.map { (String($0["id"] as! Int), $0) })
    }
    try JSONSerialization.data(withJSONObject: config).write(
        to: directory.appendingPathComponent("tokenizer_config.json"))
}

private func writeTinyQwen2Audio(
    _ directory: URL, sourceLayout: Bool = false, quantized: Bool = false
) throws -> Qwen2AudioModel {
    let config = try tinyQwen2AudioConfig(quantized: quantized)
    let model = Qwen2AudioModel(config)
    var weights = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, MLXArray.zeros($0.1.shape))
        })
    // A fixed one-token continuation makes generation length and streaming deterministic.
    let h = config.textConfig.hiddenSize
    weights["language_model.model.embed_tokens.weight"] = .ones([32, h])
    weights["language_model.model.norm.weight"] = .ones([h])
    weights["language_model.lm_head.weight"] = concatenated(
        [MLXArray.zeros([1, h]), .ones([1, h]), .zeros([30, h])], axis: 0)
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    if quantized {
        quantize(model: model) { path, _ in
            path.hasPrefix("language_model.") ? (32, 4, .affine) : nil
        }
    }
    weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    if sourceLayout {
        for key in ["audio_tower.conv1.weight", "audio_tower.conv2.weight"] {
            weights[key] = weights[key]!.transposed(0, 2, 1)
        }
        weights["audio_tower.embed_positions.weight"] = .zeros([4, 8])
    }
    eval(Array(weights.values))
    try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
    var json =
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as! [String: Any]
    if quantized { json["quantization"] = ["group_size": 32, "bits": 4] }
    try JSONSerialization.data(withJSONObject: json).write(
        to: directory.appendingPathComponent("config.json"))
    try writeQwen2AudioTokenizer(directory)
    return model
}

@Suite(.serialized)
struct Qwen2AudioTests {
    private func directory() throws -> URL {
        let result = FileManager.default.temporaryDirectory.appendingPathComponent(
            "qwen2-audio-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: result, withIntermediateDirectories: true)
        return result
    }

    @Test func publishedDefaultsAndAudioTokenAliases() throws {
        for key in ["audio_token_id", "audio_token_index"] {
            let data = Data(
                "{\"\(key)\":151646,\"text_config\":{\"intermediate_size\":11008,\"vocab_size\":156032}}"
                    .utf8)
            let config = try JSONDecoder().decode(Qwen2AudioConfig.self, from: data)
            try config.validate()
            #expect(config.textConfig.hiddenSize == 4096)
            #expect(config.textConfig.hiddenLayers == 32)
            #expect(config.textConfig.attentionHeads == 32)
            #expect(config.textConfig.kvHeads == 32)
            #expect(config.textConfig.ropeTheta == 10000)
            #expect(config.audioTokenId == 151646)
            #expect(config.eosTokenId == 151645)
            #expect(config.audioConfig.numMelBins == 128)
            let roundTrip = try JSONDecoder().decode(
                Qwen2AudioConfig.self, from: JSONEncoder().encode(config))
            #expect(roundTrip.textConfig.hiddenSize == 4096)
        }
        var invalid = try tinyQwen2AudioConfig()
        invalid.audioConfig.encoderAttentionHeads = 3
        #expect(throws: STTError.self) { try invalid.validate() }
    }

    @Test func frontendPadsTruncatesAndUsesFixedTokenCount() throws {
        let config = try tinyQwen2AudioConfig()
        let frontend = Qwen2AudioFrontend(config.audioConfig)
        let short = sin(arange(400).asType(.float32) / 20)
        let paddedWaveform = padded(short, widths: [.init((0, frontend.maxSamples - short.size))])
        let expected = frontend(paddedWaveform, dtype: .float32)
        #expect(expected.shape == [1, 4, 9])
        #expect(frontend.audioTokens == 2)
        #expect(abs(expected - frontend(short, dtype: .float32)).max().item(Float.self) == 0)
        let extended = concatenated([paddedWaveform, MLXArray.ones([400])])
        #expect(abs(expected - frontend(extended, dtype: .float32)).max().item(Float.self) == 0)
        #expect(frontend(short, dtype: .bfloat16).dtype == .bfloat16)
        #expect(MLX.all(isFinite(frontend(.zeros([1]), dtype: .float32))).item(Bool.self))
    }

    @Test func encoderPoolsAfterTransformersAndDropsOddTail() throws {
        let config = try tinyQwen2AudioConfig()
        let model = Qwen2AudioModel(config)
        let features = sin(arange(4 * 9).asType(.float32)).reshaped(1, 4, 9)
        let encoded = model.audioTower(features)
        #expect(encoded.shape == [1, 2, 8])
        #expect(model.getAudioFeatures(features).shape == [1, 2, 8])
        #expect(MLX.all(isFinite(encoded)).item(Bool.self))
        let longer = concatenated([features, features[0..., 0..., 8...]], axis: 2)
        #expect(model.audioTower(longer).shape == [1, 2, 8])
    }

    @Test func sanitizationIsPerTensorAndIdempotent() {
        let raw: [String: MLXArray] = [
            "audio_tower.conv1.weight": .zeros([8, 4, 3]),
            "audio_tower.conv2.weight": .zeros([8, 3, 8]),
            "audio_tower.embed_positions.weight": .zeros([4, 8]),
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": .ones([4]),
            "language_model.lm_head.scales": .ones([32, 1]),
        ]
        let first = Qwen2AudioModel.sanitize(weights: raw)
        let second = Qwen2AudioModel.sanitize(weights: first)
        #expect(first["audio_tower.conv1.weight"]?.shape == [8, 3, 4])
        #expect(first["audio_tower.conv2.weight"]?.shape == [8, 3, 8])
        #expect(first["audio_tower.embed_positions.weight"] == nil)
        #expect(first.keys.sorted() == second.keys.sorted())
        for key in first.keys { #expect(first[key]?.shape == second[key]?.shape) }
    }

    @Test func promptContentNumbersClipsInOrder() {
        #expect(
            Qwen2AudioModel.promptContent(audioTokenCounts: [2, 1], prompt: "Compare them.")
                == "Audio 1: <|audio_bos|><|AUDIO|><|AUDIO|><|audio_eos|>\nAudio 2: <|audio_bos|><|AUDIO|><|audio_eos|>\nCompare them."
        )
        #expect(
            Qwen2AudioModel.promptContent(audioTokenCounts: [1], prompt: nil).hasSuffix(
                "Please transcribe the speech."))
    }

    @Test func mergingFeaturesPreservesTextAndClipOrder() throws {
        let model = Qwen2AudioModel(try tinyQwen2AudioConfig())
        let ids = MLXArray([Int32(1), 31, 31, 2, 31, 3])
        let audioFeatures = arange(3 * 8).asType(.float32).reshaped(1, 3, 8)
        let merged = model.mergeEmbeddings(ids, audioFeatures: audioFeatures)
        #expect(merged.shape == [1, 6, 8])
        #expect(
            abs(merged[0..., 1 ..< 3, 0...] - audioFeatures[0..., ..<2, 0...]).max().item(
                Float.self) == 0)
        #expect(
            abs(merged[0..., 4 ..< 5, 0...] - audioFeatures[0..., 2 ..< 3, 0...]).max().item(
                Float.self) == 0)
        #expect(
            abs(
                merged[0..., 0 ..< 1, 0...]
                    - model.languageModel.embedTokens(MLXArray([Int32(1)]).reshaped(1, 1))
            ).max().item(Float.self) == 0)
    }

    @Test(arguments: [false, true])
    func sharedLocalLoadingAndStreaming(sourceLayout: Bool) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try writeTinyQwen2Audio(dir, sourceLayout: sourceLayout)
        let common = try await STT.loadModel(modelRepo: dir.path)
        let model = try #require(common as? Qwen2AudioModel)
        let audio = MLXArray.zeros([320])
        let parameters = STTGenerateParameters(maxTokens: 3, temperature: 0)
        let result = common.generate(audio: audio, generationParameters: parameters)
        #expect(result.text == "aaa")
        #expect(result.generationTokens == 3)
        #expect(result.segments?.first?["end"] as? Double == 0.02)
        let chunked = model.generate(
            audio: audio, generationParameters: parameters, prompt: nil, prefillStepSize: 3)
        #expect(chunked.text == result.text)
        let (_, embeddings) = model.prepareInputs([audio, audio], prompt: "Compare")
        let prompt = model.buildPrompt(audioTokenCounts: [2, 2], prompt: "Compare")
        #expect(prompt.asArray(Int32.self).filter { $0 == 31 }.count == 4)
        #expect(embeddings.dim(1) == prompt.size)
        var streamed = ""
        var finals = 0
        for try await event in common.generateStream(audio: audio, generationParameters: parameters)
        {
            switch event {
            case .token(let text): streamed += text
            case .info: break
            case .result(let output):
                finals += 1
                #expect(output.text == result.text)
            }
        }
        #expect(finals == 1)
        #expect(streamed == result.text)
        var weights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        eval(Array(weights.values))
        weights.removeValue(forKey: "audio_tower.conv1.weight")
        try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
        await #expect(throws: (any Error).self) {
            try await Qwen2AudioModel.fromModelDirectory(dir)
        }
    }

    @Test func tokenizerLoadsAudioTokensFromConfigurationWithoutChangingFiles() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try writeTinyQwen2Audio(dir)
        try writeQwen2AudioTokenizer(dir, configOnly: true)
        let url = dir.appendingPathComponent("tokenizer.json")
        let original = try Data(contentsOf: url)
        let model = try await Qwen2AudioModel.fromModelDirectory(dir)
        let prompt = model.buildPrompt(audioTokenCounts: [2], prompt: nil)
        #expect(prompt.asArray(Int32.self).filter { $0 == 31 }.count == 2)
        #expect(
            model.generate(audio: .zeros([320]), generationParameters: .init(maxTokens: 2)).text
                == "aa")
        #expect(try Data(contentsOf: url) == original)
    }

    @Test func quantizedLoadingKeepsFeaturesAndTokenEmbeddingsFloatingPoint() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        _ = try writeTinyQwen2Audio(dir, quantized: true)
        let model = try await Qwen2AudioModel.fromModelDirectory(dir)
        let embeddings = model.languageModel.embedTokens(MLXArray([Int32(1)]))
        #expect(embeddings.dtype.isFloatingPoint)
        let output = model.generate(
            audio: .zeros([320]),
            generationParameters: .init(maxTokens: 3, kvBits: 4, kvGroupSize: 32))
        #expect(output.text == "aaa")
        #expect(model.audioTower.conv1.weight.dtype == .float32)
    }
}

@Suite(.serialized)
struct Qwen2AudioCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["QWEN2_AUDIO_MODEL_DIR"] != nil))
    func publishedCheckpointTranscriptionAndPromptedUnderstanding() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["QWEN2_AUDIO_MODEL_DIR"])
        let common = try await STT.loadModel(modelRepo: path)
        let model = try #require(common as? Qwen2AudioModel)
        let url = try #require(
            Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")
        )
        let (_, audio) = try loadAudioArray(from: url, sampleRate: 16000)
        let parameters = STTGenerateParameters(maxTokens: 32, temperature: 0)
        let result = common.generate(audio: audio, generationParameters: parameters)
        #expect(result.text.lowercased().contains("intention"))
        let prompted = model.generate(
            audio: audio, generationParameters: parameters, prompt: "Transcribe the audio.")
        #expect(prompted.text.lowercased().contains("intention"))
        var text = ""
        var finals = 0
        for try await event in common.generateStream(audio: audio, generationParameters: parameters)
        {
            switch event {
            case .token(let delta): text += delta
            case .info: break
            case .result(let output):
                finals += 1
                #expect(output.text == result.text)
            }
        }
        #expect(finals == 1)
        #expect(text == result.text)
    }
}
