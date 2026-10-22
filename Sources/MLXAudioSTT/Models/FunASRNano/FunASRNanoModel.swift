import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXFast
import MLXNN
import Tokenizers

final class FunASRNanoAdaptorAttention: Module {
    @ModuleInfo(key: "linear_q") var query: Linear
    @ModuleInfo(key: "linear_k") var key: Linear
    @ModuleInfo(key: "linear_v") var value: Linear
    @ModuleInfo(key: "linear_out") var output: Linear
    let heads: Int
    let dimensions: Int

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        self.dimensions = dimensions
        _query.wrappedValue = Linear(dimensions, dimensions)
        _key.wrappedValue = Linear(dimensions, dimensions)
        _value.wrappedValue = Linear(dimensions, dimensions)
        _output.wrappedValue = Linear(dimensions, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), d = dimensions / heads
        let q = query(x).reshaped(b, t, heads, d).transposed(0, 2, 1, 3)
        let k = key(x).reshaped(b, t, heads, d).transposed(0, 2, 1, 3)
        let v = value(x).reshaped(b, t, heads, d).transposed(0, 2, 1, 3)
        let y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
            scale: 1 / sqrt(Float(d)), mask: nil)
        return output(y.transposed(0, 2, 1, 3).reshaped(b, t, dimensions))
    }
}

final class FunASRNanoAdaptorBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: FunASRNanoAdaptorAttention
    @ModuleInfo(key: "feed_forward") var feedForward: SenseVoicePositionwiseFeedForward
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    init(dimensions: Int, heads: Int) {
        _attention.wrappedValue = FunASRNanoAdaptorAttention(dimensions: dimensions, heads: heads)
        _feedForward.wrappedValue = SenseVoicePositionwiseFeedForward(idim: dimensions, hiddenUnits: dimensions / 4)
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        _norm2.wrappedValue = LayerNorm(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x))
        return h + feedForward(norm2(h))
    }
}

final class FunASRNanoAudioAdaptor: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "blocks") var blocks: [FunASRNanoAdaptorBlock]
    let downsampleRate: Int

    init(_ config: FunASRNanoAdaptorConfig) {
        downsampleRate = config.downsampleRate
        _linear1.wrappedValue = Linear(config.encoderDim * config.downsampleRate, config.ffnDim)
        _linear2.wrappedValue = Linear(config.ffnDim, config.llmDim)
        _blocks.wrappedValue = (0..<config.numLayers).map { _ in
            FunASRNanoAdaptorBlock(dimensions: config.llmDim, heads: config.attentionHeads)
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let k = downsampleRate, count = (input.dim(1) + k - 1) / k
        var x = input
        if count * k != x.dim(1) {
            x = padded(x, widths: [.init(0), .init((0, count * k - x.dim(1))), .init(0)])
        }
        x = linear2(relu(linear1(x.reshaped(x.dim(0), count, x.dim(2) * k))))
        for block in blocks { x = block(x) }
        return x
    }
}

final class FunASRNanoLanguageModel: Module {
    @ModuleInfo(key: "model") var model: Qwen3ASRTextModel
    init(_ config: Qwen3TextConfig) { _model.wrappedValue = Qwen3ASRTextModel(config) }
}

/// Transcription-only Fun-ASR-Nano. Arrays are mono or time-major stereo PCM at
/// `sampleRate`; file decoding/resampling belongs to MLXAudioCore.
public final class FunASRNanoModel: Module, STTGenerationModel {
    public let config: FunASRNanoConfig
    public var sampleRate: Int { config.frontend.fs }
    @ModuleInfo(key: "audio_encoder") var audioEncoder: SenseVoiceEncoder
    @ModuleInfo(key: "audio_adaptor") var audioAdaptor: FunASRNanoAudioAdaptor
    @ModuleInfo(key: "llm") var llm: FunASRNanoLanguageModel
    public var tokenizer: (any Tokenizer)?

    public init(_ config: FunASRNanoConfig) {
        self.config = config
        _audioEncoder.wrappedValue = SenseVoiceEncoder(config: SenseVoiceConfig(
            inputSize: config.inputSize, encoderConf: config.audioEncoder, frontendConf: config.frontend))
        _audioAdaptor.wrappedValue = FunASRNanoAudioAdaptor(config.audioAdaptor)
        _llm.wrappedValue = FunASRNanoLanguageModel(config.textConfig)
    }

    public var defaultGenerationParameters: STTGenerateParameters {
        .init(maxTokens: config.defaultMaxTokens, topP: 1, language: nil)
    }

    static func promptLanguage(_ language: String?) -> String? {
        guard let language else { return nil }
        let label = language.trimmingCharacters(in: .whitespacesAndNewlines)
        let code = label.lowercased().replacingOccurrences(of: "_", with: "-")
        switch code {
        case "", "auto": return nil
        case "zh", "zh-cn", "zh-hans", "zh-tw", "zh-hant", "cmn", "cjy", "gan", "hak", "hsn", "nan", "wuu", "yue", "chinese": return "中文"
        case "en", "eng", "english": return "英文"
        case "ja", "jpn", "jp", "japanese": return "日文"
        default: return label
        }
    }

    static func resolveHotwords(_ hotwords: [String], context: String?) -> [String] {
        let words = hotwords.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        let context = context?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        precondition(words.isEmpty || context.isEmpty, "Pass either hotwords or context, not both.")
        return words.isEmpty && !context.isEmpty ? [context] : words
    }

    static func promptText(language: String?, hotwords: [String], itn: Bool) -> String {
        var text = ""
        if !hotwords.isEmpty {
            text += "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            text += "热词列表：[\(hotwords.joined(separator: ", "))]\n"
        }
        if let label = promptLanguage(language) { text += "语音转写成\(label)" } else { text += "语音转写" }
        if !itn { text += "，不进行文本规整" }
        return text + "："
    }

    /// Model-specific frontend, without implicit resampling or CMVN.
    public func extractFeatures(_ audio: MLXArray) -> MLXArray {
        precondition((audio.ndim == 1 || audio.ndim == 2) && audio.size > 0,
            "Expected nonempty mono or time-major stereo PCM.")
        let mono = audio.ndim == 2 ? audio.mean(axis: -1) : audio
        let minSamples = config.frontend.fs * config.frontend.frameLength / 1000
        precondition(mono.size >= minSamples, "Audio is shorter than one analysis frame.")
        return FunASRNanoAudio.features(mono, config: config.frontend)
    }

    public func callAsFunction(inputIds: MLXArray? = nil, inputEmbeddings: MLXArray? = nil,
                              cache: [KVCache]? = nil) -> MLXArray {
        let hidden = llm.model(inputIds: inputIds, inputsEmbeds: inputEmbeddings, cache: cache)
        return llm.model.embedTokens.asLinear(hidden)
    }

    func prepareInputs(audio: MLXArray, language: String?, hotwords: [String], itn: Bool) -> (MLXArray, MLXArray) {
        guard let tokenizer else { preconditionFailure("Fun-ASR-Nano tokenizer is not loaded.") }
        let features = extractFeatures(audio)
        let projected = audioAdaptor(audioEncoder(features.expandedDimensions(axis: 0)))
        let fakeLength = FunASRNanoAudio.fakeTokenLength(features.dim(0))
        let prompt = Self.promptText(language: language, hotwords: hotwords, itn: itn)
        let before = tokenizer.encode(text: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt)
        let after = tokenizer.encode(text: "<|im_end|>\n<|im_start|>assistant\n")
        let ids = MLXArray((before + Array(repeating: 0, count: fakeLength) + after).map(Int32.init)).expandedDimensions(axis: 0)
        let embeds = llm.model.embedTokens(ids)
        let n = min(fakeLength, projected.dim(1)), start = before.count
        let merged = concatenated([embeds[0..., 0..<start, 0...], projected[0..., 0..<n, 0...].asType(embeds.dtype),
            embeds[0..., (start + n)..., 0...]], axis: 1)
        return (ids, merged)
    }

    /// Generate a transcription with optional hotword biasing and inverse text normalization.
    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
                         hotwords: [String] = [], context: String? = nil, itn: Bool = true,
                         prefillStepSize: Int = 2048) -> STTOutput {
        generateImpl(audio: audio, parameters: generationParameters ?? defaultGenerationParameters,
            hotwords: hotwords, context: context, itn: itn, prefillStepSize: prefillStepSize, emit: nil)
    }

    private func generateImpl(audio: MLXArray, parameters p: STTGenerateParameters, hotwords: [String],
                              context: String?, itn: Bool, prefillStepSize: Int,
                              emit: ((String) -> Void)?) -> STTOutput {
        guard let tokenizer else { preconditionFailure("Fun-ASR-Nano tokenizer is not loaded.") }
        precondition(prefillStepSize > 0, "Prefill step size must be positive.")
        precondition(p.kvBits == nil, "Fun-ASR-Nano KV cache quantization is not supported.")
        precondition((audio.ndim == 1 || audio.ndim == 2) && audio.size > 0,
            "Expected nonempty mono or time-major stereo PCM.")
        let words = Self.resolveHotwords(hotwords, context: context)
        let chunks = splitAudioIntoChunks(audio, sampleRate: sampleRate, chunkDuration: p.chunkDuration,
            minChunkDuration: p.minChunkDuration)
        let started = Date()
        var texts: [String] = [], segments: [[String: Any]] = [], promptCount = 0, tokenCount = 0
        let sampler: any LogitSampler = p.temperature == 0 ? ArgMaxSampler()
            : TopPSampler(temperature: p.temperature, topP: p.topP, topK: p.topK)
        chunks: for (chunk, offset) in chunks where tokenCount < p.maxTokens {
            if Task.isCancelled { break }
            let (ids, embeddings) = prepareInputs(audio: chunk, language: p.language, hotwords: words, itn: itn)
            let cache: [KVCache] = (0..<config.textConfig.numHiddenLayers).map { _ in KVCacheSimple() }
            promptCount += ids.dim(1)
            var consumed = 0
            while ids.dim(1) - consumed > 1 {
                if Task.isCancelled { break chunks }
                let n = min(prefillStepSize, ids.dim(1) - consumed - 1)
                let logits = self(inputEmbeddings: embeddings[0..., consumed..<(consumed + n), 0...], cache: cache)
                eval(logits)
                consumed += n
            }
            var logits = self(inputEmbeddings: embeddings[0..., consumed..., 0...], cache: cache)[0, -1, 0...]
            var generated: [Int] = []
            var previousText = ""
            let budget = p.maxTokens - tokenCount
            for _ in 0..<budget {
                if Task.isCancelled { break chunks }
                if p.repetitionPenalty != 1, p.repetitionContextSize > 0, !generated.isEmpty {
                    let recent = MLXArray(Array(Set(generated.suffix(p.repetitionContextSize))).map(Int32.init))
                    let values = logits[recent]
                    logits[recent] = MLX.where(values .> 0, values / p.repetitionPenalty, values * p.repetitionPenalty)
                }
                let token = sampler.sample(logits: logits).item(Int.self)
                if token == 151643 || token == 151645 { break }
                generated.append(token)
                if let emit {
                    let text = tokenizer.decode(tokens: generated)
                    if text.hasPrefix(previousText), !text.hasSuffix("\u{FFFD}") {
                        emit(String(text.dropFirst(previousText.count)))
                        previousText = text
                    }
                }
                if generated.count < budget {
                    logits = self(inputIds: MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)[0, -1, 0...]
                }
            }
            let text = Self.normalizedText(tokenizer.decode(tokens: generated))
            texts.append(text)
            tokenCount += generated.count
            segments.append(["text": text, "start": offset, "end": offset + Float(chunk.dim(0)) / Float(sampleRate)])
            if emit != nil, chunks.count > 1 { emit?(" ") }
        }
        let seconds = Date().timeIntervalSince(started)
        return STTOutput(text: texts.filter { !$0.isEmpty }.joined(separator: " "), segments: segments,
            language: p.language, promptTokens: promptCount, generationTokens: tokenCount,
            totalTokens: promptCount + tokenCount, promptTps: Double(promptCount) / max(seconds, 1e-9),
            generationTps: Double(tokenCount) / max(seconds, 1e-9), totalTime: seconds,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    static func normalizedText(_ text: String) -> String {
        text.replacingOccurrences(of: "/sil", with: " ")
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput {
        generate(audio: audio, generationParameters: generationParameters, hotwords: [])
    }

    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters) -> AsyncThrowingStream<STTGeneration, Error> {
        generateStream(audio: audio, generationParameters: generationParameters, hotwords: [])
    }

    /// Streams text tokens from a complete recording. This is not live-input ASR.
    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
                               hotwords: [String] = [], context: String? = nil, itn: Bool = true,
                               prefillStepSize: Int = 2048) -> AsyncThrowingStream<STTGeneration, Error> {
        let parameters = generationParameters ?? defaultGenerationParameters
        let boxedModel = UncheckedSendableBox(self)
        let boxedAudio = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let result = boxedModel.value.generateImpl(audio: boxedAudio.value, parameters: parameters,
                        hotwords: hotwords, context: context, itn: itn, prefillStepSize: prefillStepSize) {
                            continuation.yield(.token($0))
                        }
                    try Task.checkCancellation()
                    continuation.yield(.result(result))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    static func sanitize(weights: [String: MLXArray], tiedEmbeddings: Bool = true) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (raw, value) in weights {
            let key = raw.hasPrefix("module.") ? String(raw.dropFirst(7)) : raw
            if tiedEmbeddings && key == "llm.lm_head.weight" { continue }
            result[key] = key.hasSuffix("fsmn_block.weight") && value.ndim == 3 && value.dim(1) == 1
                ? value.transposed(0, 2, 1) : value
        }
        return result
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> FunASRNanoModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(FunASRNanoConfig.self, from: data)
        guard config.frontend.fs > 0, config.frontend.frameLength > 0, config.frontend.frameShift > 0,
              config.frontend.nMels > 3, config.frontend.lfrM > 0, config.frontend.lfrN > 0,
              config.audioEncoder.attentionHeads > 0, config.audioEncoder.outputSize >= 4,
              config.audioEncoder.outputSize % config.audioEncoder.attentionHeads == 0,
              config.audioEncoder.numBlocks > 0, config.audioEncoder.tpBlocks >= 0,
              config.audioEncoder.kernelSize > 0, config.audioEncoder.sanmShift >= 0,
              config.audioEncoder.sanmShift <= config.audioEncoder.kernelSize / 2,
              config.textConfig.tieWordEmbeddings, config.textConfig.numHiddenLayers > 0,
              config.textConfig.numKeyValueHeads > 0, config.textConfig.headDim > 0,
              config.textConfig.numAttentionHeads % config.textConfig.numKeyValueHeads == 0,
              config.audioAdaptor.downsampleRate > 0,
              config.audioAdaptor.llmDim == config.textConfig.hiddenSize,
              config.audioAdaptor.encoderDim == config.audioEncoder.outputSize,
              config.audioAdaptor.attentionHeads > 0,
              config.audioAdaptor.llmDim % config.audioAdaptor.attentionHeads == 0,
              config.frontend.nMels * config.frontend.lfrM == config.inputSize else {
            throw STTError.invalidInput("Unsupported or inconsistent Fun-ASR-Nano configuration.")
        }
        let tokenizerDirectory = directory.appendingPathComponent(config.tokenizerPath).standardizedFileURL
        guard tokenizerDirectory.path == directory.standardizedFileURL.path
                || tokenizerDirectory.path.hasPrefix(directory.standardizedFileURL.path + "/") else {
            throw STTError.invalidInput("Tokenizer directory must be inside the model directory.")
        }
        let model = FunASRNanoModel(config)
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
        guard !files.isEmpty else { throw STTError.modelNotInitialized("No Fun-ASR-Nano safetensors found.") }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = sanitize(weights: weights)
        if let quantization = try? JSONDecoder().decode(BaseConfiguration.self, from: data).perLayerQuantization {
            quantize(model: model) { path, _ in
                path.hasPrefix("llm.model.") && weights[path + ".scales"] != nil
                    ? quantization.quantization(layer: path)?.asTuple : nil
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        model.train(false)
        eval(model)
        return model
    }

    public static func fromPretrained(_ modelPath: String, cache: HubCache = .default) async throws -> FunASRNanoModel {
        let local = URL(fileURLWithPath: NSString(string: modelPath).expandingTildeInPath)
        if FileManager.default.fileExists(atPath: local.appendingPathComponent("config.json").path) {
            return try await fromModelDirectory(local)
        }
        guard let repo = Repo.ID(rawValue: modelPath) else { throw STTModelError.invalidRepositoryID(modelPath) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors",
            additionalMatchingPatterns: ["Qwen3-0.6B/*"], cache: cache)
        return try await fromModelDirectory(directory)
    }
}
