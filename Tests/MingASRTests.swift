import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioSTT

private func tinyMingConfig() -> MingASRConfig {
    var c = MingASRConfig()
    c.audioTokenId = 319
    c.audioConfig.numMelBins = 4
    c.audioConfig.hiddenSize = 64
    c.audioConfig.numAttentionHeads = 2
    c.audioConfig.numHiddenLayers = 1
    c.audioConfig.maxSourcePositions = 16
    c.audioConfig.chunkFrames = 16
    c.textConfig.hiddenSize = 64
    c.textConfig.intermediateSize = 128
    c.textConfig.numHiddenLayers = 4
    c.textConfig.vocabSize = 320
    c.textConfig.numAttentionHeads = 2
    c.textConfig.headDim = 32
    c.textConfig.layerGroupSize = 2
    c.textConfig.qLoraRank = 32
    c.textConfig.kvLoraRank = 32
    c.textConfig.qkNopeHeadDim = 16
    c.textConfig.qkRopeHeadDim = 16
    c.textConfig.vHeadDim = 32
    c.textConfig.numExperts = 4
    c.textConfig.numExpertsPerToken = 2
    c.textConfig.nGroup = 2
    c.textConfig.topKGroup = 1
    c.textConfig.moeIntermediateSize = 64
    c.textConfig.moeSharedExpertIntermediateSize = 64
    c.textConfig.maxPositionEmbeddings = 512
    c.textConfig.eosTokenIds = [318]
    return c
}

private func mingDirectory() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ming-\(UUID())")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

private func writeTinyMing(_ dir: URL, quantized: Bool = false, alternateMarkers: Bool = false)
    throws
{
    let config = tinyMingConfig()
    let model = MingASRModel(tinyMingConfig())
    var weights = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, MLXArray.zeros($0.1.shape))
        })
    weights["model.word_embeddings.weight"] = .ones([320, 64])
    weights["model.norm.weight"] = .ones([64])
    weights["lm_head.weight"] = concatenated([
        MLXArray.zeros([1, 64]), .ones([1, 64]), .zeros([318, 64]),
    ])
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    if quantized {
        quantize(model: model) { path, _ in
            path.hasPrefix("model.audio.") || path.hasSuffix(".gate") ? nil : (32, 4, .affine)
        }
    }
    weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    eval(Array(weights.values))
    try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
    var json =
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as! [String: Any]
    if quantized { json["quantization"] = ["group_size": 32, "bits": 4] }
    try JSONSerialization.data(withJSONObject: json).write(
        to: dir.appendingPathComponent("config.json"))
    var vocab = Dictionary(uniqueKeysWithValues: (2 ..< 317).map { ("token\($0)", $0) })
    vocab["<unk>"] = 0
    vocab["a"] = 1
    let markers =
        alternateMarkers
        ? ["<|audio_start|>", "<|audio_end|>", "<|audio_pad|>"]
        : ["<audio>", "</audio>", "<audioPatch>"]
    let special = [markers[0], markers[1], "<|role_end|>", markers[2]]
    var added: [[String: Any]] = [["id": 0, "content": "<unk>", "special": true]]
    for (i, token) in special.enumerated() {
        vocab[token] = 316 + i
        added.append(["id": 316 + i, "content": token, "special": true])
    }
    let tokenizer: [String: Any] = [
        "model": ["type": "BPE", "vocab": vocab, "merges": [], "unk_token": "<unk>"],
        "added_tokens": added,
    ]
    try JSONSerialization.data(withJSONObject: tokenizer).write(
        to: dir.appendingPathComponent("tokenizer.json"))
    try JSONSerialization.data(withJSONObject: [
        "tokenizer_class": "BailingTokenizer", "unk_token": "<unk>",
        "eos_token": "<|role_end|>",
    ])
    .write(to: dir.appendingPathComponent("tokenizer_config.json"))
}

@Suite(.serialized)
struct MingASRTests {
    @Test func configurationAndDispatch() throws {
        let c = try JSONDecoder().decode(MingASRConfig.self, from: Data("{}".utf8))
        #expect(c.textConfig.eosTokenIds == [156892])
        #expect(c.textConfig.usesMLA(3))
        #expect(!c.textConfig.usesMLA(2))
        #expect(c.audioConfig.embeddingLength(3000) == 750)
        #expect(c.audioConfig.tokenCount(3001) == 751)
        #expect(STT.inferModelType(from: "mlx-community/Ming-3.0-asr-tiny-4bit") == "ming_v3_asr")
        let parsed = try JSONDecoder().decode(
            MingTextConfig.self, from: Data(#"{"eos_token_id":[1,2],"q_lora_rank":null}"#.utf8))
        #expect(parsed.qLoraRank == nil && parsed.eosTokenIds == [1, 2])
        var bad = tinyMingConfig()
        bad.textConfig.topKGroup = 3
        #expect(throws: (any Error).self) { try bad.validate() }
        try tinyMingConfig().validate()
    }

    @Test func recurrentKernelMatchesOperationsAndChunking() {
        let shape = [2, 7, 2, 32]
        let q = MLXRandom.normal(shape) / 8
        let k = MLXRandom.normal(shape) / 8
        let v = MLXRandom.normal(shape).asType(.bfloat16)
        let decay = sigmoid(MLXRandom.normal(shape))
        let beta = sigmoid(MLXRandom.normal([2, 7, 2]))
        let state = MLXRandom.normal([2, 2, 32, 32]) / 10
        let actual = MingDeltaKernel.callAsFunction(
            q: q, k: k, v: v, decay: decay, beta: beta, state: state)
        let expected = MingDeltaKernel.reference(
            q: q, k: k, v: v, decay: decay, beta: beta, state: state)
        #expect(allClose(actual.0, expected.0, atol: 2e-6).item(Bool.self))
        #expect(allClose(actual.1, expected.1, atol: 2e-6).item(Bool.self))
        let first = MingDeltaKernel.callAsFunction(
            q: q[0..., 0 ..< 3], k: k[0..., 0 ..< 3], v: v[0..., 0 ..< 3],
            decay: decay[0..., 0 ..< 3], beta: beta[0..., 0 ..< 3], state: state)
        let second = MingDeltaKernel.callAsFunction(
            q: q[0..., 3...], k: k[0..., 3...], v: v[0..., 3...], decay: decay[0..., 3...],
            beta: beta[0..., 3...], state: first.1)
        #expect(
            allClose(actual.0, concatenated([first.0, second.0], axis: 1), atol: 2e-6).item(
                Bool.self))
        #expect(allClose(actual.1, second.1, atol: 2e-6).item(Bool.self))
    }

    @Test func hybridDecoderCacheMatchesFullSequence() {
        let model = MingASRModel(tinyMingConfig())
        let input = MLXRandom.normal([1, 7, 64])
        let full = model.model(input, cache: model.model.makeCache())
        let cache = model.model.makeCache()
        let first = model.model(input[0..., 0 ..< 4], cache: cache)
        eval(first, cache.flatMap { $0.arrays })
        let last = model.model(input[0..., 4...], cache: cache)
        #expect(allClose(full, concatenated([first, last], axis: 1), atol: 1e-4).item(Bool.self))
        #expect(cache[0].recurrent?.dtype == .float32)
        #expect(cache[1].attention.offset == 7)
        #expect(cache[0].convolution[0]?.shape == [1, 3, 64])
    }

    @Test func optionalAttentionLayoutsAndClippedActivation() {
        var c = tinyMingConfig()
        c.textConfig.noKdaLora = false
        c.textConfig.qLoraRank = nil
        c.textConfig.attentionGate = "element_wise"
        c.textConfig.shortConvKernelSize = 1
        let model = MingASRModel(c)
        let cache = model.model.makeCache()
        let result = model.model(.ones([1, 2, 64]), cache: cache)
        #expect(result.shape == [1, 2, 64])
        eval(result)
        #expect(cache[0].convolution[0]?.dim(1) == 0)
        let activated = mingSwiGLU(
            up: MLXArray([Float(-10), 10]), gate: MLXArray([Float(-10), 10]), limit: 2)
        #expect(activated[1].item(Float.self) == 4)
    }

    @Test(arguments: [4, 8]) func packedExpertsMatchDequantizedWeights(bits: Int) {
        let weight = MLXRandom.normal([4, 32, 64])
        let layer = MingQuantizedSwitchLinear(
            weight: weight, groupSize: 32, bits: bits, mode: .affine)
        let unpacked = dequantized(
            layer.weight, scales: layer.scales, biases: layer.biases,
            groupSize: 32, bits: bits)
        let dense = MingSwitchLinear(weight: unpacked)
        let x = MLXRandom.normal([2, 5, 1, 1, 64])
        let indices = MLXArray([Int32(0), 3, 2, 1, 3, 0, 0, 2, 1, 3, 1, 0, 2, 3, 2, 1, 0, 3, 2, 1])
            .reshaped(2, 5, 2)
        #expect(
            allClose(layer(x, indices: indices), dense(x, indices: indices), atol: 5e-5).item(
                Bool.self))
    }

    @Test func sortedExpertBatchingPreservesTokenOrder() {
        let c = tinyMingConfig().textConfig
        let experts = MingSwitchMLP(tinyMingConfig().textConfig, limit: nil)
        let x = MLXRandom.normal([1, 40, c.hiddenSize])
        let indices = MLXArray((0 ..< 80).map { Int32(($0 * 3 + $0 / 5) % 4) }).reshaped(1, 40, 2)
        let batched = experts(x, indices: indices)
        let individual = concatenated(
            (0 ..< 40).map {
                experts(x[0..., $0 ..< ($0 + 1)], indices: indices[0..., $0 ..< ($0 + 1)])
            }, axis: 1)
        // Batched and single-token calls select different Metal matmul kernels.
        let relativeError =
            sqrt((batched - individual).square().mean()) / sqrt(individual.square().mean())
        #expect(relativeError.item(Float.self) < 0.005)
    }

    @Test func routerRespectsGroupsAndUsesUnbiasedWeights() throws {
        let router = MingRouter(tinyMingConfig().textConfig)
        try router.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXArray.zeros([4, 64]), "expert_bias": MLXArray([Float(0), 0, 5, 4]),
            ]), verify: .all)
        let (indices, weights) = router(.ones([1, 3, 64]))
        #expect(indices.min().item(Int.self) >= 2)
        #expect(
            allClose(weights, MLXArray.full(weights.shape, values: MLXArray(Float(1.25)))).item(
                Bool.self))
    }

    @Test func variableAudioChunksMatchLengthContract() {
        let c = tinyMingConfig()
        let model = MingASRModel(tinyMingConfig())
        let features = MLXRandom.normal([35, 4])
        let audio = model.encodeAudio(features)
        #expect(audio.shape == [1, c.audioConfig.tokenCount(35), 64])
        let lengths = sqrt(audio.asType(.float32).square().sum(axis: -1))
        #expect(allClose(lengths, MLXArray.ones(lengths.shape), atol: 1e-5).item(Bool.self))
    }

    @Test func sanitizeOriginalLayoutAndPackExperts() throws {
        let c = tinyMingConfig()
        let model = MingASRModel(tinyMingConfig())
        let sanitized = try model.sanitize(weights: [
            "model.audio.blocks.0.attn.query.weight": .zeros([64, 64]),
            "model.audio.positional_embedding": .zeros([16, 64]),
            "model.audio.conv1.weight": .zeros([64, 4, 3]),
            "model.layers.0.attention.q_conv1d.weight": .zeros([64, 1, 4]),
            "model.layers.4.extra.weight": .zeros([1]), "model.visual.weight": .zeros([1]),
        ])
        #expect(sanitized["model.audio.layers.0.self_attn.q_proj.weight"]?.shape == [64, 64])
        #expect(sanitized["model.audio.embed_positions.weight"]?.shape == [16, 64])
        #expect(sanitized["model.audio.conv1.weight"]?.shape == [64, 3, 4])
        #expect(sanitized["model.layers.0.attention.q_conv1d.weight"]?.shape == [64, 4, 1])
        #expect(sanitized.count == 4)
        var experts: [String: MLXArray] = [:]
        for i in 0 ..< c.textConfig.numExperts {
            experts["model.layers.1.mlp.experts.\(i).gate_proj.weight"] = .full(
                [64, 64], values: MLXArray(Float(i)))
        }
        let packed = try model.sanitize(weights: experts)
        #expect(packed["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 64, 64])
        experts.removeValue(forKey: "model.layers.1.mlp.experts.2.gate_proj.weight")
        #expect(throws: (any Error).self) { try model.sanitize(weights: experts) }
    }

    @Test func transcriptAndNgramHandling() {
        #expect(MingASRModel.decodeTranscript(" English\thello ").text == "hello")
        #expect(MingASRModel.decodeTranscript("Chinese你好").language == "Chinese")
        #expect(MingASRModel.decodeTranscript("plain text").language == nil)
        #expect(MingASRModel.bannedTokens([1, 2, 3, 1, 2], ngramSize: 3) == [3])
        #expect(MingASRModel.bannedTokens([1, 2, 1], ngramSize: 1) == [1, 2])
        #expect(MingASRModel.bannedTokens([1, 2], ngramSize: 0).isEmpty)
    }

    @Test(arguments: [false, true]) func strictLocalLoadingAndGeneration(quantized: Bool)
        async throws
    {
        let dir = try mingDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyMing(dir, quantized: quantized, alternateMarkers: quantized)
        let model = try await STT.loadModel(modelRepo: dir.path) as! MingASRModel
        let audio = MLXArray.zeros([1600])
        let p = STTGenerateParameters(maxTokens: 3, kvBits: quantized ? 8 : nil, kvGroupSize: 32)
        let output = model.generate(audio: audio, generationParameters: p)
        #expect(output.text == "aaa" && output.generationTokens == 3)
        #expect(model.defaultGenerationParameters.maxTokens == 256)
        let ids = model.buildPrompt(audioTokens: 3, language: "en", prompt: nil, hotwords: ["MLX"])
        #expect(ids.filter { $0 == 319 }.count == 3)
        let embedded = model.injectAudio(ids, audio: .ones([1, 3, 64]))
        #expect(embedded.shape == [1, ids.count, 64])
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        #expect((weights["model.layers.1.mlp.switch_mlp.gate_proj.scales"] != nil) == quantized)
        var text = ""
        var result: STTOutput?
        for try await event in model.generateStream(audio: audio, generationParameters: p) {
            switch event {
            case .token(let token): text += token
            case .result(let output): result = output
            default: break
            }
        }
        #expect(text == output.text && result?.text == output.text)
    }

    @Test func loaderRejectsMissingWeights() async throws {
        let dir = try mingDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyMing(dir)
        var weights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        weights.removeValue(forKey: "model.layers.1.mlp.switch_mlp.up_proj.weight")
        let incomplete = dir.appendingPathComponent("incomplete.safetensors")
        try MLX.save(arrays: weights, url: incomplete)
        try FileManager.default.removeItem(at: dir.appendingPathComponent("model.safetensors"))
        try FileManager.default.moveItem(
            at: incomplete, to: dir.appendingPathComponent("model.safetensors"))
        await #expect(throws: (any Error).self) { try await MingASRModel.fromModelDirectory(dir) }
    }
}
