import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioSTT

private func tinyVoxtralConfig(quantized: Bool = false) -> VoxtralConfig {
    let dim = quantized ? 64 : 8
    return VoxtralConfig(
        audioConfig: .init(
            hiddenSize: dim, numHiddenLayers: 1, numAttentionHeads: 2,
            intermediateSize: dim * 4, numMelBins: 4, maxSourcePositions: 8),
        textConfig: .init(
            vocabSize: 320, hiddenSize: dim, intermediateSize: dim * 2,
            numHiddenLayers: 2, numAttentionHeads: 2, numKeyValueHeads: 1,
            headDim: quantized ? 32 : 6, ropeTheta: 100_000_000, padTokenId: 11,
            eosTokenId: [2, 4, 11]))
}

private func writeVoxtralTokenizer(_ directory: URL) throws {
    var tokens = (0 ..< 256).map { Data([UInt8($0)]).base64EncodedString() }
    tokens += ["la", "lan", "lang", "en", "fr"].map { Data($0.utf8).base64EncodedString() }
    let json: [String: Any] = [
        "config": [
            "default_num_special_tokens": 35, "default_vocab_size": tokens.count + 35,
            "pattern": #"\S+|\s+"#,
        ],
        "special_tokens": [1, 2, 3, 4, 11, 24, 25, 34].map { ["rank": $0] },
        "vocab": tokens.map { ["token_bytes": $0] },
    ]
    try JSONSerialization.data(withJSONObject: json).write(
        to: directory.appendingPathComponent("tekken.json"))
}

private func writeTinyVoxtral(
    _ directory: URL, sourceLayout: Bool = false,
    quantized: Bool = false, nextToken: Int = 132, flatLanguageModel: Bool = false
) throws {
    let config = tinyVoxtralConfig(quantized: quantized)
    let model = VoxtralModel(config)
    let dim = config.textConfig.hiddenSize
    var weights = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, MLXArray.zeros($0.1.shape))
        })
    weights["language_model.model.embed_tokens.weight"] = .ones([320, dim])
    weights["language_model.model.norm.weight"] = .ones([dim])
    weights["language_model.lm_head.weight"] = concatenated(
        [
            MLXArray.zeros([nextToken, dim]), .ones([1, dim]), .zeros([319 - nextToken, dim]),
        ], axis: 0)
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    if quantized {
        quantize(model: model) { path, _ in
            path == "audio_tower.embed_positions"
                ? nil
                : (
                    32,
                    (path.hasPrefix("audio_tower.") || path == "language_model.lm_head") ? 8 : 4,
                    .affine
                )
        }
    }
    weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    if sourceLayout {
        for key in ["audio_tower.conv1.weight", "audio_tower.conv2.weight"] {
            weights[key] = weights[key]!.transposed(0, 2, 1)
        }
    }
    func artifactPath(_ key: String) -> String {
        guard flatLanguageModel else { return key }
        if key.hasPrefix("language_model.lm_head") {
            return String(key.dropFirst("language_model.".count))
        }
        return key.replacingOccurrences(of: "language_model.model.", with: "language_model.")
    }
    weights = Dictionary(uniqueKeysWithValues: weights.map { (artifactPath($0.key), $0.value) })
    if flatLanguageModel {
        for (key, value) in Array(weights) where key.hasPrefix("language_model.embed_tokens.") {
            weights[String(key.dropFirst("language_model.".count))] = value
        }
    }
    eval(Array(weights.values))
    try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
    var json =
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as! [String: Any]
    if quantized {
        var quantization: [String: Any] = ["group_size": 32, "bits": 4]
        for (path, _) in model.parameters().flattened()
        where path.hasSuffix(".scales") {
            let layer = String(path.dropLast(7))
            quantization[artifactPath(layer)] = [
                "group_size": 32,
                "bits": (layer.hasPrefix("audio_tower.") || layer == "language_model.lm_head")
                    ? 8 : 4,
            ]
        }
        json["quantization"] = quantization
    }
    try JSONSerialization.data(withJSONObject: json).write(
        to: directory.appendingPathComponent("config.json"))
    try writeVoxtralTokenizer(directory)
}

@Suite(.serialized)
struct VoxtralTests {
    private func directory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "voxtral-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    @Test func offlineAndRealtimeRepositoryNamesSelectDifferentModels() {
        #expect(STT.inferModelType(from: "mlx-community/Voxtral-Mini-3B-2507-bf16") == "voxtral")
        #expect(
            STT.inferModelType(from: "mistralai/Voxtral-Mini-4B-Realtime-2602")
                == "voxtral_realtime")
    }

    @Test func configDefaultsAndExplicitHeadDimension() throws {
        let defaults = try JSONDecoder().decode(VoxtralConfig.self, from: Data("{}".utf8))
        try defaults.validate()
        #expect(defaults.textConfig.headDim == 128)
        #expect(defaults.textConfig.hiddenSize == 3072)
        #expect(defaults.audioConfig.maxSourcePositions == 1500)
        let config = tinyVoxtralConfig()
        try config.validate()
        let model = VoxtralModel(config)
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect(
            parameters["language_model.model.layers.0.self_attn.q_proj.weight"]?.shape == [12, 8])
        var invalid = config
        invalid.audioConfig.intermediateSize = 31
        #expect(throws: STTError.self) { try invalid.validate() }
    }

    @Test func frontendPadsToWholeBlocksWithoutTruncating() {
        let frontend = VoxtralAudio(config: tinyVoxtralConfig().audioConfig)
        let samples = frontend.samplesPerChunk
        let short = sin(arange(400).asType(.float32) / 12)
        let padded = MLX.padded(short, widths: [.init((0, samples - short.size))])
        #expect(frontend(short).shape == [1, 16, 4])
        #expect(frontend.tokensPerChunk == 2)
        #expect(abs(frontend(short) - frontend(padded)).max().item(Float.self) == 0)
        #expect(frontend(.zeros([samples])).dim(0) == 1)
        #expect(frontend(.zeros([samples + 1])).dim(0) == 2)
        #expect(frontend(.zeros([samples * 2])).dim(0) == 2)
        let stereo = stacked([short, short], axis: -1)
        #expect(abs(frontend(stereo) - frontend(short)).max().item(Float.self) == 0)
    }

    @Test func tekkenEncodesLanguageAndPreservesUTF8Bytes() throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeVoxtralTokenizer(dir)
        let tokenizer = try TekkenTokenizer.fromModelDirectory(dir)
        #expect(tokenizer.encode("lang:en") == [293, 93, 294])
        #expect(tokenizer.encode("lang:fr") == [293, 93, 295])
        let text = "café 世界\nCapital"
        #expect(tokenizer.decode(tokenIds: [1] + tokenizer.encode(text) + [2, 4, 11, 24]) == text)
        #expect(!VoxtralModel.eosTokenIds.contains(32000))
    }

    @Test func promptsUseOfflineControlTokensAndAlignedAudioPositions() throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeVoxtralTokenizer(dir)
        let model = VoxtralModel(tinyVoxtralConfig())
        model.tokenizer = try TekkenTokenizer.fromModelDirectory(dir)
        #expect(
            model.buildPrompt(audioTokens: 2, language: "en", prompt: nil).asArray(Int32.self) == [
                1, 3, 25, 24, 24, 4, 293, 93, 294, 34,
            ])
        #expect(
            model.buildPrompt(audioTokens: 2, language: nil, prompt: nil).asArray(Int32.self) == [
                1, 3, 25, 24, 24, 4, 34,
            ])
        #expect(
            model.buildPrompt(audioTokens: 2, language: "en", prompt: "a").asArray(Int32.self) == [
                1, 3, 25, 24, 24, 132, 4,
            ])
        let features = arange(16).asType(.float32).reshaped(2, 8)
        let ids = MLXArray([Int32(1), 24, 3, 24, 4])
        let embeddings = model.mergeEmbeddings(ids, audioFeatures: features)
        #expect(abs(embeddings[0, 1] - features[0]).max().item(Float.self) == 0)
        #expect(abs(embeddings[0, 3] - features[1]).max().item(Float.self) == 0)
        #expect(
            abs(embeddings[0, 2] - model.languageModel.embedTokens(MLXArray(Int32(3)))).max().item(
                Float.self) == 0)
    }

    @Test(arguments: [false, true])
    func sharedLoaderReadsBothLayoutsAndUsesOnlyIndexedShards(sourceLayout: Bool) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyVoxtral(dir, sourceLayout: sourceLayout)
        try JSONSerialization.data(withJSONObject: ["weight_map": ["weight": "model.safetensors"]])
            .write(to: dir.appendingPathComponent("model.safetensors.index.json"))
        try MLX.save(
            arrays: ["unrelated": MLXArray.ones([1])],
            url: dir.appendingPathComponent("consolidated.safetensors"))
        let common = try await STT.loadModel(modelRepo: dir.path)
        let model = try #require(common as? VoxtralModel)
        let parameters = STTGenerateParameters(maxTokens: 3)
        let audio = MLXArray.zeros([300])
        let output = common.generate(audio: audio, generationParameters: parameters)
        #expect(output.text == "aaa")
        #expect(output.generationTokens == 3)
        #expect(
            model.generate(audio: audio, generationParameters: parameters, prefillStepSize: 2).text
                == output.text)
        let long = MLXArray.zeros([
            VoxtralAudio(config: model.config.audioConfig).samplesPerChunk + 1
        ])
        let (ids, _) = model.prepareInputs(long, prompt: nil, language: "fr")
        #expect(ids.asArray(Int32.self).filter { $0 == 24 }.count == 4)
        var text = ""
        var finals = 0
        for try await event in common.generateStream(audio: audio, generationParameters: parameters)
        {
            switch event {
            case .token(let delta): text += delta
            case .info: break
            case .result(let result):
                finals += 1
                #expect(result.text == output.text)
            }
        }
        #expect(text == output.text)
        #expect(finals == 1)
        var weights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        eval(Array(weights.values))
        weights.removeValue(forKey: "audio_tower.embed_positions.weight")
        try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
        await #expect(throws: (any Error).self) { try await VoxtralModel.fromModelDirectory(dir) }
    }

    @Test(arguments: [2, 4, 11])
    func eachOfflineStopTokenEndsGeneration(token: Int) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyVoxtral(dir, nextToken: token)
        let model = try await VoxtralModel.fromModelDirectory(dir)
        let output = model.generate(audio: .zeros([300]), generationParameters: .init(maxTokens: 3))
        #expect(output.text.isEmpty)
        #expect(output.generationTokens == 0)
    }

    @Test(arguments: [false, true])
    func mixedQuantizationAndQuantizedKVCacheKeepEmbeddingLookupsFloatingPoint(
        flatLanguageModel: Bool
    ) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyVoxtral(dir, quantized: true, flatLanguageModel: flatLanguageModel)
        let model = try await VoxtralModel.fromModelDirectory(dir)
        #expect(model.languageModel.embedTokens(MLXArray([Int32(1)])).dtype.isFloatingPoint)
        #expect(
            model.generate(
                audio: .zeros([300]),
                generationParameters: .init(maxTokens: 2, kvBits: 4, kvGroupSize: 32)
            ).text == "aa")
        #expect(model.audioTower.layers[0].fc1 is QuantizedLinear)
    }
}

@Suite(.serialized)
struct VoxtralCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["VOXTRAL_MODEL_DIR"] != nil))
    func publishedCheckpointTranscribesAndStreams() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["VOXTRAL_MODEL_DIR"])
        let model = try await STT.loadModel(modelRepo: path)
        #expect(model is VoxtralModel)
        let url = try #require(
            Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")
        )
        let (_, audio) = try loadAudioArray(from: url, sampleRate: 16000)
        let parameters = STTGenerateParameters(maxTokens: 32, language: "en")
        let output = model.generate(audio: audio, generationParameters: parameters)
        #expect(output.text.lowercased().contains("intention"))
        var text = ""
        for try await event in model.generateStream(audio: audio, generationParameters: parameters)
        {
            if case .token(let delta) = event { text += delta }
        }
        #expect(text == output.text)
    }
}
