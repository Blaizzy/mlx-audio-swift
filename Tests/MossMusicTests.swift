import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioSTT

private func tinyMossMusicConfig() -> MossMusicConfig {
    var c = MossMusicConfig()
    c.audioConfig.hiddenSize = 64
    c.audioConfig.outputDim = 32
    c.audioConfig.numMelBins = 8
    c.audioConfig.downsampleHiddenSize = 4
    c.audioConfig.numHiddenLayers = 3
    c.audioConfig.numAttentionHeads = 2
    c.audioConfig.intermediateSize = 128
    c.audioConfig.maxSourcePositions = 8
    c.audioConfig.windowSize = 8
    c.audioConfig.convBatchSize = 2
    c.audioConfig.deepstackLayerIndices = [0, 2]
    c.languageConfig = .init(
        vocabSize: 320, hiddenSize: 64, intermediateSize: 128,
        numHiddenLayers: 3, numAttentionHeads: 2, numKeyValueHeads: 1, headDim: 32,
        maxPositionEmbeddings: 1024, tieWordEmbeddings: false)
    c.adapterHiddenSize = 64
    c.deepstackNumInjectLayers = 2
    c.audioTokenId = 317
    c.audioStartId = 318
    c.audioEndId = 319
    c.eosTokenId = 316
    return c
}

private func writeMossMusicTokenizer(_ dir: URL) throws {
    var vocab: [String: Int] = ["<unk>": 0, "a": 1]
    for id in 2 ..< 315 where !(10 ..< 20).contains(id) { vocab["token\(id)"] = id }
    for digit in 0 ..< 10 { vocab[String(digit)] = digit + 10 }
    let special = ["<|im_start|>", "<|im_end|>", "<|AUDIO|>", "<|audio_bos|>", "<|audio_eos|>"]
    var added: [[String: Any]] = [["id": 0, "content": "<unk>", "special": true]]
    for (i, token) in special.enumerated() {
        vocab[token] = 315 + i
        added.append(["id": 315 + i, "content": token, "special": true])
    }
    try JSONSerialization.data(withJSONObject: [
        "model": ["type": "BPE", "vocab": vocab, "merges": [], "unk_token": "<unk>"],
        "added_tokens": added,
    ])
    .write(to: dir.appendingPathComponent("tokenizer.json"))
    try JSONSerialization.data(withJSONObject: [
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
        "eos_token": "<|im_end|>",
    ])
    .write(to: dir.appendingPathComponent("tokenizer_config.json"))
}

private func mossDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("moss-music-\(UUID())")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}
private func writeTinyMossMusic(_ dir: URL, bits: Int) throws {
    let c = tinyMossMusicConfig()
    let model = MossMusicModel(tinyMossMusicConfig())
    var weights = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, MLXArray.zeros($0.1.shape))
        })
    weights["language_model.embed_tokens.weight"] = .ones([320, 64])
    weights["language_model.norm.weight"] = .ones([64])
    weights["lm_head.weight"] = concatenated([
        MLXArray.zeros([1, 64]), .ones([1, 64]), .zeros([318, 64]),
    ])
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    if bits > 0 {
        quantize(model: model) { path, _ in
            path.hasPrefix("audio_encoder.") ? nil : (32, bits, .affine)
        }
    }
    weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    eval(Array(weights.values))
    try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
    var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(c)) as! [String: Any]
    if bits > 0 { json["quantization"] = ["group_size": 32, "bits": bits] }
    try JSONSerialization.data(withJSONObject: json).write(
        to: dir.appendingPathComponent("config.json"))
    try writeMossMusicTokenizer(dir)
}

@Suite(.serialized)
struct MossMusicTests {
    @Test func configurationAndDispatch() throws {
        let c = try JSONDecoder().decode(MossMusicConfig.self, from: Data("{}".utf8))
        #expect(c.languageConfig.hiddenSize == 4096 && c.languageConfig.numHiddenLayers == 36)
        #expect(c.injectionCount == 3 && c.enableTimeMarkers && c.stripThinking)
        #expect(MossMusicAudioConfig.outputLength(401) == 51)
        #expect(
            STT.inferModelType(from: "mlx-community/MOSS-Music-8B-Thinking-6bit") == "moss_music")
        #expect(
            STT.inferModelType(from: "OpenMOSS-Team/MOSS-Transcribe-Diarize")
                == "moss_transcribe_diarize")
        try tinyMossMusicConfig().validate()
        var invalid = tinyMossMusicConfig()
        invalid.audioConfig.deepstackLayerIndices = [0, 3]
        #expect(throws: (any Error).self) { try invalid.validate() }
    }

    @Test func promptTimeMarkersAndExplicitAudioSpan() async throws {
        let dir = try mossDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeMossMusicTokenizer(dir)
        let tokenizer = try await loadSTTTokenizer(from: dir)
        let p = MossMusicPrompt(config: tinyMossMusicConfig(), tokenizer: tokenizer)
        let tokens = p.audioTokens(count: 126, timeMarkers: true)
        #expect(tokens.filter { $0 == 317 }.count == 126)
        #expect(Array(tokens[25 ... 25]) == [12])
        #expect(Array(tokens.suffix(3)) == [11, 10, 317])
        #expect(p.audioTokens(count: 24, timeMarkers: true) == Array(repeating: 317, count: 24))
        #expect(p.audioTokens(count: 126, timeMarkers: false).count == 126)
        let ids = p.inputIds(audioTokens: 26, prompt: "Transcribe the lyrics.", timeMarkers: true)
        #expect(ids.filter { $0 == 317 }.count == 26)
        #expect(ids.contains(318) && ids.contains(319))
        let explicit = p.inputIds(
            audioTokens: 2,
            prompt: "<|audio_bos|><|AUDIO|><|AUDIO|><|audio_eos|>", timeMarkers: false)
        #expect(explicit == [318, 317, 317, 319])
        let direct = p.inputIds(
            audioTokens: 2, prompt: "Describe this music.", timeMarkers: false,
            enableThinking: false)
        let closing = tokenizer.encode(text: "<think>\n\n</think>\n\n", addSpecialTokens: false)
        #expect(Array(direct.suffix(closing.count)) == closing)
    }

    @Test func vocabularyAndMergesTokenizerFallback() async throws {
        let dir = try mossDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeMossMusicTokenizer(dir)
        let path = dir.appendingPathComponent("tokenizer.json")
        let tokenizer =
            try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as! [String: Any]
        let model = tokenizer["model"] as! [String: Any]
        let added = tokenizer["added_tokens"] as! [[String: Any]]
        try JSONSerialization.data(withJSONObject: model["vocab"]!).write(
            to: dir.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(
            to: dir.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        let configPath = dir.appendingPathComponent("tokenizer_config.json")
        var config =
            try JSONSerialization.jsonObject(with: Data(contentsOf: configPath)) as! [String: Any]
        config["added_tokens_decoder"] = Dictionary(
            uniqueKeysWithValues: added.map { (String($0["id"] as! Int), $0) })
        try JSONSerialization.data(withJSONObject: config).write(to: configPath)
        try FileManager.default.removeItem(at: path)
        let loaded = try await loadMossMusicTokenizer(from: dir)
        #expect(loaded.encode(text: "a12", addSpecialTokens: false) == [1, 11, 12])
        #expect(loaded.convertTokenToId("<|AUDIO|>") == 317)
        #expect(!FileManager.default.fileExists(atPath: path.path))
    }

    @Test func chunkingAndDeepstackLengths() {
        let c = tinyMossMusicConfig()
        let model = MossMusicModel(tinyMossMusicConfig())
        let features = MLXRandom.normal([c.audioConfig.numMelBins, 33])
        let (encoded, deep) = model.encodeAudio(features)
        #expect(encoded.shape == [5, 32])
        #expect(deep.count == 2 && deep.allSatisfy { $0.shape == [5, 32] })
        eval([encoded] + deep)
        let batch = stacked([features[0..., 0 ..< 16], features[0..., 0 ..< 16]])
        let (out, _) = model.audioEncoder(batch, lengths: [16, 1])
        #expect(out.shape == [2, 2, 32])
    }

    @Test func deepstackInjectionAndPrefillCacheAgree() {
        let model = MossMusicModel(tinyMossMusicConfig())
        let embeddings = MLXRandom.normal([1, 9, 64])
        let deep = [MLXRandom.normal([1, 9, 64]), MLXRandom.normal([1, 9, 64])]
        func cache() -> [KVCache] { (0 ..< 3).map { _ in KVCacheSimple() } }
        let full = model.forward(embeddings, cache: cache(), deepstack: deep)
        let state = cache()
        let first = model.forward(
            embeddings[0..., 0 ..< 5], cache: state, deepstack: deep.map { $0[0..., 0 ..< 5] })
        eval(first, state)
        let last = model.forward(
            embeddings[0..., 5...], cache: state, deepstack: deep.map { $0[0..., 5...] })
        #expect(allClose(full, concatenated([first, last], axis: 1), atol: 1e-4).item(Bool.self))
        let without = model.forward(embeddings, cache: cache())
        #expect(abs(full - without).max().item(Float.self) > 0.01)
    }

    @Test func intermediateEmbeddingsOnlyReplaceAudioPositions() {
        let model = MossMusicModel(tinyMossMusicConfig())
        let ids = [2, 317, 12, 317, 3]
        let prepared = model.injectAudio(
            ids: ids, features: .ones([2, 32]), deepstack: [.ones([2, 32]), .ones([2, 32])])
        #expect(prepared.embeddings.shape == [1, 5, 64])
        for deep in prepared.deepstack {
            #expect(deep[0, 0].abs().sum().item(Float.self) == 0)
            #expect(deep[0, 2].abs().sum().item(Float.self) == 0)
            #expect(deep[0, 4].abs().sum().item(Float.self) == 0)
        }
    }

    @Test func thinkingFilterHoldsPartialTagsAndUnfinishedReasoning() {
        #expect(
            MossMusicText.response("<think>private</think>\nA piano.", stripThinking: true)
                == "A piano.")
        #expect(MossMusicText.response("<think>unfinished", stripThinking: true) == "")
        #expect(MossMusicText.response("<thi", stripThinking: true, final: false) == "")
        #expect(MossMusicText.response("A <", stripThinking: true, final: false) == "A")
        #expect(
            MossMusicText.response("<think>raw</think>", stripThinking: false)
                == "<think>raw</think>")
    }

    @Test func timestampParsingAndFallback() {
        let parsed = MossMusicText.segments(
            "[00:01] First line\n[00:03–00:05] Second line\n00:06: Third line", duration: 8)
        #expect(parsed.count == 3)
        #expect(parsed[0]["text"] as? String == "First line")
        #expect(parsed[0]["start"] as? Double == 1 && parsed[0]["end"] as? Double == 3)
        #expect(parsed[1]["end"] as? Double == 5 && parsed[2]["end"] as? Double == 8)
        #expect(MossMusicText.seconds("01:02:03,5") == 3723.5)
        #expect(MossMusicText.seconds("12.5s") == 12.5)
        let fallback = MossMusicText.segments("A quiet piano piece.", duration: 4.25)
        #expect(fallback.count == 1 && fallback[0]["end"] as? Double == 4.25)
        #expect(fallback[0]["kind"] as? String == "text")
    }

    @Test func sourceCheckpointSanitization() {
        let weights = MossMusicModel.sanitize(weights: [
            "audio_encoder.layers.0.q_proj.weight": .zeros([64, 64]),
            "audio_encoder.conv1.weight": .zeros([4, 1, 3, 3]),
            "audio_encoder.embed_positions.inv_timescales": .ones([32]),
        ])
        #expect(weights.count == 2)
        #expect(weights["audio_encoder.layers.0.self_attn.q_proj.weight"]?.shape == [64, 64])
        #expect(weights["audio_encoder.conv1.weight"]?.shape == [4, 3, 3, 1])
    }

    @Test(arguments: [0, 4, 6, 8]) func localLoadingAndSharedGeneration(bits: Int) async throws {
        let dir = try mossDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyMossMusic(dir, bits: bits)
        let model = try await STT.loadModel(modelRepo: dir.path) as! MossMusicModel
        let p = STTGenerateParameters(maxTokens: 3, kvBits: bits > 0 ? 8 : nil, kvGroupSize: 32)
        let audio = MLXArray.zeros([1600])
        let output = model.generate(audio: audio, generationParameters: p, prefillStepSize: 5)
        #expect(output.text == "aaa" && output.generationTokens == 3)
        #expect(output.segments?.first?["end"] as? Double == 0.1)
        var streamed = ""
        var final: STTOutput?
        for try await event in model.generateStream(
            audio: audio, generationParameters: p, prefillStepSize: 5)
        {
            switch event {
            case .token(let text): streamed += text
            case .result(let result): final = result
            default: break
            }
        }
        #expect(streamed == output.text && final?.text == output.text)
    }

    @Test func strictLoaderRejectsMissingMergerWeights() async throws {
        let dir = try mossDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyMossMusic(dir, bits: 0)
        var weights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        weights.removeValue(forKey: "deepstack_audio_merger_list.1.down_proj.weight")
        let replacement = dir.appendingPathComponent("replacement.safetensors")
        try MLX.save(arrays: weights, url: replacement)
        try FileManager.default.removeItem(at: dir.appendingPathComponent("model.safetensors"))
        try FileManager.default.moveItem(
            at: replacement, to: dir.appendingPathComponent("model.safetensors"))
        await #expect(throws: (any Error).self) { try await MossMusicModel.fromModelDirectory(dir) }
    }
}
