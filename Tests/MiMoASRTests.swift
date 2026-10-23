import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioCodecs
@testable import MLXAudioSTT

private func tinyMiMoASRConfig(quantized: Bool = false, projected: Bool = false) -> MiMoASRConfig {
    var c = MiMoASRConfig()
    let dim = quantized ? 64 : 8
    c.languageModel = .init(
        hiddenSize: dim, hiddenLayers: 2, intermediateSize: dim * 2,
        attentionHeads: 2, rmsNormEps: 1e-6, vocabularySize: 320, kvHeads: 1, ropeTheta: 640000)
    c.audioChannels = 2
    c.groupSize = 2
    c.delayPattern = [0, 1]
    c.speechVocabSizes = [9, 5]
    c.speechEmptyIds = [8, 4]
    c.inputLocalDim = dim
    c.inputLocalLayers = 1
    c.inputLocalHeads = 2
    c.inputLocalIntermediateSize = dim * 2
    c.localDim = projected ? dim * 2 : dim
    c.localLayers = 1
    c.localHeads = 2
    c.localIntermediateSize = c.localDim * 2
    c.emptyId = 304
    c.endOfTurnId = 305
    c.startOfSpeechId = 302
    c.endOfSpeechId = 303
    c.endOfTextSpeechId = 307
    return c
}
private func tinyMiMoASRCodecConfig() -> MiMoAudioTokenizerConfig {
    var c = MiMoAudioTokenizerConfig()
    c.dModel = 16
    c.nMels = 4
    c.nFFT = 32
    c.windowSize = 32
    c.hopLength = 8
    c.encoderLayers = 2
    c.encoderSkipLayerID = 1
    c.encoderAttentionHeads = 2
    c.encoderFFNDim = 32
    c.decoderLayers = 0
    c.decoderAttentionHeads = 2
    c.decoderFFNDim = 32
    c.vocoderDim = 8
    c.vocoderIntermediateDim = 16
    c.vocoderNumLayers = 0
    c.vocoderAttentionHeads = 2
    c.numQuantizers = 2
    c.codebookSize = [8, 4]
    return c
}
private func writeMiMoASRTokenizer(_ dir: URL) throws {
    var vocab = Dictionary(
        uniqueKeysWithValues: (1 ..< 128).map { (String(UnicodeScalar($0)!), $0) })
    vocab["<unk>"] = 0
    let special = [
        "<|im_start|>", "<|im_end|>", "<|sosp|>", "<|eosp|>", "<|empty|>", "<|eot|>",
        "<|endoftext|>", "<|eostm|>",
    ]
    var added: [[String: Any]] = [["id": 0, "content": "<unk>", "special": true]]
    for (i, token) in special.enumerated() {
        vocab[token] = 300 + i
        added.append(["id": 300 + i, "content": token, "special": true])
    }
    let data: [String: Any] = [
        "model": ["type": "BPE", "vocab": vocab, "merges": [], "unk_token": "<unk>"],
        "added_tokens": added,
    ]
    try JSONSerialization.data(withJSONObject: data).write(
        to: dir.appendingPathComponent("tokenizer.json"))
    try JSONSerialization.data(withJSONObject: [
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
        "eos_token": "<|endoftext|>",
    ]).write(to: dir.appendingPathComponent("tokenizer_config.json"))
}
private func writeTinyMiMoASR(
    _ directory: URL, quantized: Bool = false, wrapped: Bool = false, nextToken: Int = 97
) throws {
    var config = tinyMiMoASRConfig(quantized: quantized, projected: wrapped)
    config.modelType = wrapped ? "qwen2" : "mimo_v2_asr"
    let model = MiMoASRModel(config)
    var weights = Dictionary(
        uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, MLXArray.zeros($0.1.shape))
        })
    let dim = config.languageModel.hiddenSize
    weights["model.embed_tokens.weight"] = .ones([320, dim])
    weights["model.norm.weight"] = .ones([dim])
    weights["lm_head.weight"] = concatenated(
        [.zeros([nextToken, dim]), .ones([1, dim]), .zeros([319 - nextToken, dim])], axis: 0)
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    if quantized { quantize(model: model, groupSize: 32, bits: 4) }
    weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
    if wrapped {
        weights = Dictionary(uniqueKeysWithValues: weights.map { ("model." + $0.key, $0.value) })
    }
    eval(Array(weights.values))
    try MLX.save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
    var json =
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as! [String: Any]
    if quantized { json["quantization"] = ["group_size": 32, "bits": 4] }
    try JSONSerialization.data(withJSONObject: json).write(
        to: directory.appendingPathComponent("config.json"))
    try writeMiMoASRTokenizer(directory)
    let codecDirectory = directory.appendingPathComponent("audio-tokenizer")
    try FileManager.default.createDirectory(at: codecDirectory, withIntermediateDirectories: true)
    let codec = try MiMoAudioTokenizer(config: tinyMiMoASRCodecConfig(), loadDecoder: false)
    var codecWeights = Dictionary(
        uniqueKeysWithValues: codec.parameters().flattened().map {
            ($0.0, MLXArray.zeros($0.1.shape))
        })
    // The published ASR companion stores NLC convolutions without an encoder prefix.
    codecWeights = Dictionary(
        uniqueKeysWithValues: codecWeights.map { key, value in
            var key = String(key.dropFirst("encoder.".count))
            key = key.replacingOccurrences(
                of: #"quantizer\.codebooks\.(\d+)\.weight"#,
                with: "quantizer.vq.layers.$1.codebook.embed", options: .regularExpression)
            key = key.replacingOccurrences(of: "down_sample_layer.layers.0.", with: "down_sample.")
            return (key, value)
        })
    codecWeights["position_embedding.inv_freq"] = .ones([4])
    eval(Array(codecWeights.values))
    try MLX.save(
        arrays: codecWeights, url: codecDirectory.appendingPathComponent("model.safetensors"))
    try JSONEncoder().encode(codec.config).write(
        to: codecDirectory.appendingPathComponent("config.json"))
    try JSONSerialization.data(withJSONObject: ["audio_tokenizer_dir": "audio-tokenizer"]).write(
        to: directory.appendingPathComponent("mlx_manifest.json"))
}

@Suite(.serialized)
struct MiMoASRTests {
    private func directory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            "mimo-asr-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    @Test func defaultsNestedAudioAndArrayAliases() throws {
        let defaults = try JSONDecoder().decode(MiMoASRConfig.self, from: Data("{}".utf8))
        try defaults.validate()
        #expect(defaults.languageModel.hiddenLayers == 36)
        #expect(defaults.inputLocalHeads == 64)
        let nested = try JSONDecoder().decode(
            MiMoASRConfig.self,
            from: Data(
                #"{"audio_config":{"input_local_attn_heads":32,"speech_vocab_size":[1025,1025,129,129,129,129,129,129]},"delay_pattern":"0-1-2-3-4-5-6-7"}"#
                    .utf8))
        #expect(nested.inputLocalHeads == 32)
        #expect(nested.delayPattern == Array(0 ..< 8))
        var invalid = tinyMiMoASRConfig()
        invalid.delayPattern = [0]
        #expect(throws: STTError.self) { try invalid.validate() }
        #expect(STT.inferModelType(from: "mlx-community/MiMo-V2.5-ASR-MLX") == "mimo_v2_asr")
    }
    @Test func embeddingOnlyQwenStacksHaveNoUnusedTablesAndAcceptFullAttention() {
        let config = tinyMiMoASRConfig().localConfiguration(input: true)
        let model = Qwen2ModelInner(config, includeEmbedding: false)
        #expect(!model.parameters().flattened().contains { $0.0.hasPrefix("embed_tokens.") })
        let input = sin(arange(32).asType(.float32)).reshaped(1, 4, 8)
        let causal = model(inputEmbeddings: input)
        let full = model(inputEmbeddings: input, mask: .none)
        #expect(full.shape == [1, 4, 8])
        #expect(abs(causal - full).max().item(Float.self) > 1e-6)
    }
    @Test func framePaddingAndInterleavingPreserveCodebookOrder() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeMiMoASRTokenizer(dir)
        let c = tinyMiMoASRConfig()
        let processor = MiMoASRPrompt(config: c, tokenizer: try await loadSTTTokenizer(from: dir))
        let codes = MLXArray([Int32(1), 2, 3, 0, 1, 2]).reshaped(2, 3)
        let padded = processor.paddedCodes(codes)
        #expect(padded.asArray(Int32.self) == [1, 2, 3, 3, 0, 1, 2, 2])
        let text = processor.text([12, 13])
        #expect(text[0].asArray(Int32.self) == [12, -100, 13, -100])
        #expect(text[1].asArray(Int32.self) == [8, 8, 8, 8])
        let prompt = processor(codes: codes, language: "en")
        let ids = prompt[0].asArray(Int32.self)
        let start = try #require(ids.firstIndex(of: Int32(c.startOfSpeechId)))
        #expect(Array(ids[start ..< start + 8]) == [302, -100, 304, -100, 304, -100, 303, -100])
        #expect(
            prompt[1..., (start + 2) ..< (start + 6)].asArray(Int32.self)
                == padded.asArray(Int32.self))
        let decoded = processor.tokenizer.decode(
            tokens: ids.enumerated().filter { $0.offset % c.groupSize == 0 }.map {
                Int($0.element)
            }, skipSpecialTokens: false)
        #expect(decoded.contains("<english>"))
        #expect(MiMoASRPrompt.language("Mandarin") == "zh")
        #expect(MiMoASRPrompt.language("fr") == nil)
    }
    @Test func encoderFramesAndEncoderOnlyCheckpointLoading() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyMiMoASR(dir)
        let folder = dir.appendingPathComponent("audio-tokenizer")
        let codec = try MiMoAudioTokenizer.fromModelDirectory(
            folder, dtype: nil, loadDecoder: false)
        #expect(codec.decoder == nil)
        let rotated = codec.encoder.layers[0].attention.rope(.ones([1, 2, 2, 8]))
        let expected = Float(Foundation.cos(1.0) - Foundation.sin(1.0))
        #expect(abs(rotated[0, 0, 1, 0 ..< 4] - expected).max().item(Float.self) < 1e-6)
        #expect(!codec.parameters().flattened().contains { $0.0.hasPrefix("decoder.") })
        #expect(throws: MiMoAudioTokenizerError.self) {
            try codec.decode(.zeros([2, 3], dtype: .int32))
        }
        let mel = sin(arange(100).asType(.float32)).reshaped(25, 4)
        let full = codec.encoder(mel.expandedDimensions(axis: 0))
        #expect(full.dim(1) == 7)
        #expect(try codec.encodeMels(mel).shape == [2, 7])
    }
    @Test(arguments: [false, true])
    func sharedLoaderReadsManifestAndBothModelLayouts(wrapped: Bool) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyMiMoASR(dir, wrapped: wrapped)
        let common = try await STT.loadModel(modelRepo: dir.path)
        let model = try #require(common as? MiMoASRModel)
        #expect(common.sampleRate == 24000)
        let p = STTGenerateParameters(maxTokens: 3, language: "en")
        let result = common.generate(audio: .zeros([200]), generationParameters: p)
        #expect(result.text == "aaa")
        #expect(result.language == "en")
        #expect(result.generationTokens == 3)
        #expect(
            model.generate(audio: .zeros([200]), generationParameters: p, prefillStepSize: 2).text
                == "aaa")
        var output = ""
        var finals = 0
        for try await event in common.generateStream(audio: .zeros([200]), generationParameters: p)
        {
            if case .token(let text) = event { output += text }
            if case .result(let result) = event {
                finals += 1
                #expect(result.text == "aaa")
            }
        }
        #expect(output == "aaa")
        #expect(finals == 1)
        var weights = try MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))
        eval(Array(weights.values))
        weights.removeValue(forKey: (wrapped ? "model." : "") + "local_transformer.norm.weight")
        try MLX.save(arrays: weights, url: dir.appendingPathComponent("model.safetensors"))
        await #expect(throws: (any Error).self) { try await MiMoASRModel.fromModelDirectory(dir) }
    }
    @Test func quantizationAndCompressedGlobalCache() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyMiMoASR(dir, quantized: true, wrapped: true)
        let model = try await MiMoASRModel.fromModelDirectory(dir)
        #expect(model.lmHead is QuantizedLinear)
        #expect(model.speechGroupDowncast is QuantizedLinear)
        #expect(
            model.generate(
                audio: .zeros([200]),
                generationParameters: .init(maxTokens: 2, kvBits: 4, kvGroupSize: 32)
            ).text == "aa")
    }
    @Test(arguments: [305, 306, 307, 304])
    func stopAndPlaceholderTokensAreNotReturnedAsText(token: Int) async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyMiMoASR(dir, nextToken: token)
        let model = try await MiMoASRModel.fromModelDirectory(dir)
        let output = model.generate(audio: .zeros([200]), generationParameters: .init(maxTokens: 2))
        if token == 304 {
            #expect(!output.text.contains("<|empty|>"))
        } else {
            #expect(output.text.isEmpty)
        }
        #expect(output.generationTokens == (token == 304 ? 2 : 0))
    }
    @Test func delayPatternLocalDecodingMasksPaddingAndSupportsProjection() throws {
        let c = tinyMiMoASRConfig(projected: true)
        let model = MiMoASRModel(c)
        var weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map {
                ($0.0, MLXArray.zeros($0.1.shape))
            })
        weights["speech_embeddings.0.weight"] = .ones([9, 8])
        weights["speech_embeddings.1.weight"] = .ones([5, 8])
        weights["local_transformer.norm.weight"] = .ones([c.localDim])
        for (i, size) in c.speechVocabSizes.enumerated() {
            weights["local_transformer_lm_heads.\(i).weight"] = concatenated(
                [.zeros([size - 1, c.localDim]), .ones([1, c.localDim])], axis: 0)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let result = model.localForward(.ones([1, 1, c.localDim]))
        #expect(result.shape == [1, 2, 2])
        #expect((result .== 0).all().item(Bool.self))
    }
    @Test func missingManifestAndMismatchedCodebooksFailAtLoadTime() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try writeTinyMiMoASR(dir)
        try FileManager.default.removeItem(at: dir.appendingPathComponent("mlx_manifest.json"))
        await #expect(throws: STTError.self) { try await MiMoASRModel.fromModelDirectory(dir) }
        let model = try await MiMoASRModel.fromModelDirectory(
            dir, audioTokenizerDirectory: dir.appendingPathComponent("audio-tokenizer"))
        #expect(model.sampleRate == 24000)
        var codec = tinyMiMoASRCodecConfig()
        codec.samplingRate = 16000
        try JSONEncoder().encode(codec).write(
            to: dir.appendingPathComponent("audio-tokenizer/config.json"))
        await #expect(throws: STTError.self) {
            try await MiMoASRModel.fromModelDirectory(
                dir, audioTokenizerDirectory: dir.appendingPathComponent("audio-tokenizer"))
        }
    }
    @Test func ordinaryQwen2IsNotMisidentifiedAsMiMo() async throws {
        let dir = try directory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(#"{"model_type":"qwen2"}"#.utf8).write(
            to: dir.appendingPathComponent("config.json"))
        await #expect(throws: STTModelError.self) { try await STT.loadModel(modelRepo: dir.path) }
        #expect(MiMoASRModel.cleanText("<english>hello<|empty|> <|eot|>") == "hello")
        #expect(MiMoASRModel.cleanText("hello<eng", partial: true) == "hello")
    }
}

@Suite(.serialized)
struct MiMoASRCheckpointTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["MIMO_ASR_MODEL_DIR"] != nil))
    func publishedCheckpointTranscribesAndStreams() async throws {
        let path = try #require(ProcessInfo.processInfo.environment["MIMO_ASR_MODEL_DIR"])
        let model = try await STT.loadModel(modelRepo: path)
        #expect(model is MiMoASRModel)
        let url = try #require(
            Bundle.module.url(forResource: "intention", withExtension: "wav", subdirectory: "media")
        )
        let (_, audio) = try loadAudioArray(from: url, sampleRate: 24000)
        let result = model.generate(audio: audio, generationParameters: .init(maxTokens: 32))
        #expect(result.text.lowercased().contains("intention"))
        var text = ""
        for try await event in model.generateStream(
            audio: audio, generationParameters: .init(maxTokens: 32))
        {
            if case .token(let delta) = event { text += delta }
        }
        #expect(text == result.text)
    }
}
