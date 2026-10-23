import Foundation
import HuggingFace
import MLX
import MLXAudioCodecs
import MLXAudioCore
import MLXNN
import Tokenizers

/// MiMo-V2.5-ASR with discrete speech codes and grouped text generation.
public final class MiMoASRModel: Module, STTGenerationModel {
    public let config: MiMoASRConfig
    @ModuleInfo(key: "model") var languageModel: Qwen2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "speech_embeddings") var speechEmbeddings: [Embedding]
    @ModuleInfo(key: "input_local_transformer") var inputLocal: Qwen2ModelInner
    @ModuleInfo(key: "speech_group_downcast") var speechGroupDowncast: Linear
    @ModuleInfo(key: "hidden_states_downcast") var hiddenStatesDowncast: Linear
    @ModuleInfo(key: "local_transformer") var localTransformer: Qwen2ModelInner
    @ModuleInfo(key: "local_transformer_lm_heads") var localHeads: [Linear]
    @ModuleInfo(key: "speech_embeddings_to_local") var speechEmbeddingsToLocal: Linear?
    var tokenizer: Tokenizer?
    private var _audioTokenizer: UncheckedSendableBox<MiMoAudioTokenizer>?
    public var sampleRate: Int { 24000 }
    public var defaultGenerationParameters: STTGenerateParameters {
        .init(maxTokens: 256, temperature: 0, topP: 0.95)
    }

    public init(_ config: MiMoASRConfig) {
        self.config = config
        _languageModel.wrappedValue = Qwen2ModelInner(config.languageModel)
        _lmHead.wrappedValue = Linear(
            config.languageModel.hiddenSize, config.languageModel.vocabularySize, bias: false)
        _speechEmbeddings.wrappedValue = config.speechVocabSizes.map {
            Embedding(embeddingCount: $0, dimensions: config.inputLocalDim)
        }
        _inputLocal.wrappedValue = Qwen2ModelInner(
            config.localConfiguration(input: true), includeEmbedding: false)
        _speechGroupDowncast.wrappedValue = Linear(
            config.inputLocalDim * config.groupSize, config.languageModel.hiddenSize, bias: false)
        _hiddenStatesDowncast.wrappedValue = Linear(
            config.languageModel.hiddenSize, config.localDim, bias: false)
        _localTransformer.wrappedValue = Qwen2ModelInner(
            config.localConfiguration(input: false), includeEmbedding: false)
        _localHeads.wrappedValue = config.speechVocabSizes.map {
            Linear(config.localDim, $0, bias: false)
        }
        if config.inputLocalDim != config.localDim {
            _speechEmbeddingsToLocal.wrappedValue = Linear(
                config.inputLocalDim, config.localDim, bias: false)
        }
    }
    func encodeAudio(_ audio: MLXArray) -> MLXArray {
        guard let codec = _audioTokenizer?.value else {
            preconditionFailure("MiMo audio tokenizer is not loaded")
        }
        do {
            return try codec.encode(
                audio, sampleRate: sampleRate, numQuantizers: config.audioChannels)
        } catch { preconditionFailure("MiMo audio encoding failed: \(error)") }
    }
    func prompt(codes: MLXArray, language: String?) -> MLXArray {
        guard let tokenizer else { preconditionFailure("MiMo text tokenizer is not loaded") }
        return MiMoASRPrompt(config: config, tokenizer: tokenizer)(codes: codes, language: language)
    }
    func prepareEmbeddings(_ ids: MLXArray) -> MLXArray {
        let c = config
        let batch = ids.dim(0)
        let groups = ids.dim(2) / config.groupSize
        let textIds = ids[0..., 0, .stride(by: c.groupSize)]
        let isSpeech = textIds .== c.emptyId
        var speech = MLXArray.zeros([batch, groups, c.groupSize, c.inputLocalDim], dtype: .float32)
        for i in 0 ..< c.audioChannels {
            let codes = ids[0..., i + 1, 0...].reshaped(batch, groups, c.groupSize)
            let embeddings = speechEmbeddings[i](codes)
            speech =
                speech
                + MLX.where(
                    (codes .== c.speechEmptyIds[i]).expandedDimensions(axis: -1),
                    MLXArray.zeros(like: embeddings), embeddings)
        }
        let mask = isSpeech.expandedDimensions(axes: [-1, -2]).asType(.float32)
        speech = speech * mask
        speech =
            inputLocal(
                inputEmbeddings: speech.reshaped(batch * groups, c.groupSize, c.inputLocalDim),
                mask: c.inputFullAttention ? .none : .causal
            )
            .reshaped(batch, groups, c.groupSize, c.inputLocalDim) * mask
        let acoustic = speechGroupDowncast(speech.reshaped(batch, groups, -1))
        let text = languageModel.embed(textIds)
        return MLX.where(isSpeech.expandedDimensions(axis: -1), MLXArray.zeros(like: text), text)
            + acoustic
    }
    func forward(_ ids: MLXArray, cache: [KVCache]) -> (MLXArray, MLXArray) {
        let hidden = languageModel(inputEmbeddings: prepareEmbeddings(ids), cache: cache)[
            0..., (-1)..., 0...]
        return (lmHead(hidden), hiddenStatesDowncast(hidden))
    }
    func localForward(_ hidden: MLXArray) -> MLXArray {
        let c = config
        let batch = hidden.dim(0)
        let cache: [KVCache] = (0 ..< c.localLayers).map { _ in KVCacheSimple() }
        var codes = MLXArray.zeros([batch, c.groupSize, c.audioChannels], dtype: .int32)
        var input = hidden
        for step in 0 ..< (c.groupSize + (c.delayPattern.max() ?? 0)) {
            let h = localTransformer(inputEmbeddings: input, cache: cache)[0..., (-1)..., 0...]
            input = MLXArray.zeros(like: hidden)
            for channel in 0 ..< c.audioChannels {
                let position = step - c.delayPattern[channel]
                guard (0 ..< c.groupSize).contains(position) else { continue }
                var logits = localHeads[channel](h)[0..., -1, 0...]
                logits[0..., c.speechEmptyIds[channel]] = MLXArray(-Float.infinity)
                let token = argMax(logits, axis: -1).asType(.int32)
                codes[0..., position, channel] = token
                var embedding = speechEmbeddings[channel](token.expandedDimensions(axis: -1))
                if let speechEmbeddingsToLocal { embedding = speechEmbeddingsToLocal(embedding) }
                input = input + embedding
            }
            eval(input, codes, cache)
        }
        return codes
    }
    static func cleanText(_ text: String, partial: Bool = false) -> String {
        let markers = ["<|empty|>", "<|eot|>", "<|eostm|>", "<chinese>", "<english>"]
        var result = text
        for marker in markers { result = result.replacingOccurrences(of: marker, with: "") }
        if partial {
            let pending =
                markers.flatMap { marker in (1 ..< marker.count).map { String(marker.prefix($0)) } }
                .filter { result.hasSuffix($0) }.map(\.count).max() ?? 0
            if pending > 0 { result = String(result.dropLast(pending)) }
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    private func compress(_ cache: inout [KVCache], _ p: STTGenerateParameters) {
        guard let bits = p.kvBits else { return }
        do {
            let affine = try AffineKVCacheConfiguration(
                bits: bits, groupSize: p.kvGroupSize, compressionStart: p.quantizedKVStart)
            _ = try applyKVCacheConfiguration(
                cache: &cache, configuration: .init(strategy: .affine(affine)))
        } catch { preconditionFailure("Invalid MiMo KV-cache settings: \(error)") }
    }
    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        generateImpl(audio, generationParameters, prefillStepSize: 512, emit: nil)
    }
    public func generate(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prefillStepSize: Int = 512
    ) -> STTOutput {
        generateImpl(
            audio, generationParameters ?? defaultGenerationParameters,
            prefillStepSize: prefillStepSize, emit: nil)
    }
    private func generateImpl(
        _ audio: MLXArray, _ p: STTGenerateParameters, prefillStepSize: Int,
        emit: ((String) -> Void)?
    ) -> STTOutput {
        precondition(
            prefillStepSize > 0 && p.maxTokens >= 0 && p.temperature >= 0 && p.repetitionPenalty > 0
        )
        guard let tokenizer else { preconditionFailure("MiMo text tokenizer is not loaded") }
        let started = Date()
        let language = MiMoASRPrompt.language(p.language)
        let ids = prompt(codes: encodeAudio(audio), language: language).expandedDimensions(axis: 0)
        let c = config
        let length = ids.dim(2)
        var cache: [KVCache] = (0 ..< c.languageModel.hiddenLayers).map { _ in KVCacheSimple() }
        let step = prefillStepSize * c.groupSize
        var offset = 0
        while length - offset > step && !Task.isCancelled {
            eval(forward(ids[0..., 0..., offset ..< (offset + step)], cache: cache).0, cache)
            compress(&cache, p)
            offset += step
        }
        var (logits, hidden) = forward(ids[0..., 0..., offset...], cache: cache)
        eval(logits, hidden, cache)
        compress(&cache, p)
        let prefillTime = Date().timeIntervalSince(started)
        let decodeStarted = Date()
        let sampler: any LogitSampler =
            p.temperature == 0
            ? ArgMaxSampler() : TopPSampler(temperature: p.temperature, topP: p.topP, topK: p.topK)
        var generated: [Int] = []
        var emitted = ""
        let stops: Set<Int> = Set([tokenizer.eosTokenId, c.endOfTurnId].compactMap { $0 })
        for iteration in 0 ..< p.maxTokens {
            if Task.isCancelled { break }
            var scores = logits[0, -1, 0...]
            if p.repetitionPenalty != 1 && !generated.isEmpty {
                let indices = MLXArray(
                    Array(Set(generated.suffix(max(1, p.repetitionContextSize)))).map(Int32.init))
                let values = scores[indices]
                scores[indices] = MLX.where(
                    values .> 0, values / p.repetitionPenalty, values * p.repetitionPenalty)
            }
            let token = sampler.sample(logits: scores).item(Int.self)
            if stops.contains(token) { break }
            generated.append(token)
            if emit != nil || p.verbose {
                let raw = tokenizer.decode(tokens: generated, skipSpecialTokens: false)
                let text = Self.cleanText(raw, partial: true)
                if !text.contains("\u{FFFD}"), text.hasPrefix(emitted) {
                    let delta = String(text.dropFirst(emitted.count))
                    if !delta.isEmpty {
                        emit?(delta)
                        if p.verbose { print(delta, terminator: "") }
                    }
                    emitted = text
                }
            }
            if iteration + 1 == p.maxTokens { break }
            let speech: MLXArray
            if token == c.emptyId {
                speech = localForward(hidden)
            } else {
                speech = broadcast(
                    MLXArray(c.speechEmptyIds.map(Int32.init)).reshaped(1, 1, c.audioChannels),
                    to: [1, c.groupSize, c.audioChannels])
            }
            let text = MLXArray.full([1, c.groupSize, 1], values: MLXArray(Int32(token)))
            let next = concatenated([text, speech], axis: -1).transposed(0, 2, 1)
            (logits, hidden) = forward(next, cache: cache)
            eval(logits, hidden, cache)
            compress(&cache, p)
        }
        while generated.last == c.endOfTextSpeechId { generated.removeLast() }
        let text = Self.cleanText(tokenizer.decode(tokens: generated, skipSpecialTokens: false))
        if text.hasPrefix(emitted) {
            let tail = String(text.dropFirst(emitted.count))
            if !tail.isEmpty {
                emit?(tail)
                if p.verbose { print(tail, terminator: "") }
            }
        }
        if p.verbose { print() }
        let duration = Double(audio.dim(0)) / Double(sampleRate)
        let generationTime = Date().timeIntervalSince(decodeStarted)
        return STTOutput(
            text: text, segments: [["text": text, "start": 0.0, "end": duration]],
            language: language,
            promptTokens: length, generationTokens: generated.count,
            totalTokens: length + generated.count,
            promptTps: Double(length) / max(prefillTime, 0.001),
            generationTps: Double(generated.count) / max(generationTime, 0.001),
            totalTime: Date().timeIntervalSince(started),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }
    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error>
    {
        makeStream(audio, generationParameters, prefillStepSize: 512)
    }
    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prefillStepSize: Int = 512
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        makeStream(
            audio, generationParameters ?? defaultGenerationParameters,
            prefillStepSize: prefillStepSize)
    }
    private func makeStream(
        _ audio: MLXArray, _ parameters: STTGenerateParameters, prefillStepSize: Int
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let model = UncheckedSendableBox(self)
        let input = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let output = model.value.generateImpl(
                        input.value, parameters, prefillStepSize: prefillStepSize
                    ) { continuation.yield(.token($0)) }
                    try Task.checkCancellation()
                    continuation.yield(.result(output))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let wrapped = weights["model.model.embed_tokens.weight"] != nil
        return Dictionary(
            uniqueKeysWithValues: weights.compactMap { key, value in
                if key.contains("rotary_emb.inv_freq") { return nil }
                return (wrapped && key.hasPrefix("model.") ? String(key.dropFirst(6)) : key, value)
            })
    }
    public static func fromPretrained(
        _ modelRepo: String = "mlx-community/MiMo-V2.5-ASR-MLX", cache: HubCache = .default
    ) async throws -> MiMoASRModel {
        guard let repo = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory, cache: cache)
    }
    public static func fromModelDirectory(
        _ directory: URL, audioTokenizerDirectory: URL? = nil, cache: HubCache = .default
    ) async throws -> MiMoASRModel {
        let config = try JSONDecoder().decode(
            MiMoASRConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try config.validate()
        let model = MiMoASRModel(config)
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw STTError.modelNotInitialized("No MiMo ASR safetensors found")
        }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = sanitize(weights: weights)
        if let quantization = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                weights[path + ".scales"] == nil
                    ? nil : quantization.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try await loadSTTTokenizer(from: directory)
        let audioDirectory = try await resolveAudioTokenizer(
            directory, override: audioTokenizerDirectory, cache: cache)
        let codec = try MiMoAudioTokenizer.fromModelDirectory(
            audioDirectory, dtype: nil, loadDecoder: false)
        guard codec.sampleRate == model.sampleRate, codec.numQuantizers >= config.audioChannels,
            Array(codec.config.codebookSize.prefix(config.audioChannels)) == config.speechEmptyIds
        else {
            throw STTError.modelNotInitialized(
                "MiMo audio tokenizer does not match the ASR codebooks")
        }
        model._audioTokenizer = UncheckedSendableBox(codec)
        model.train(false)
        eval(model)
        return model
    }
    static func resolveAudioTokenizer(_ directory: URL, override: URL?, cache: HubCache)
        async throws -> URL
    {
        if let override { return override }
        let path = directory.appendingPathComponent("mlx_manifest.json")
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw STTError.modelNotInitialized(
                "MiMo ASR requires mlx_manifest.json or audioTokenizerDirectory")
        }
        let manifest =
            try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any] ?? [:]
        if let relative = manifest["audio_tokenizer_dir"] as? String {
            let candidate = URL(
                fileURLWithPath: relative,
                relativeTo: URL(fileURLWithPath: directory.path, isDirectory: true)
            )
            .standardizedFileURL
            if FileManager.default.fileExists(
                atPath: candidate.appendingPathComponent("config.json").path)
            {
                return candidate
            }
        }
        if let repo = manifest["audio_tokenizer_repo"] as? String, let id = Repo.ID(rawValue: repo)
        {
            return try await ModelUtils.resolveOrDownloadModel(
                repoID: id, requiredExtension: ".safetensors", cache: cache)
        }
        throw STTError.modelNotInitialized("MiMo manifest does not resolve an audio tokenizer")
    }
}
