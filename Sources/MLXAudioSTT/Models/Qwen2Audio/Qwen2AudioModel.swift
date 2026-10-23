import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import Tokenizers

/// Qwen2-Audio transcribes, translates and answers questions about supplied audio clips.
public final class Qwen2AudioModel: Module, STTGenerationModel {
    public let config: Qwen2AudioConfig
    @ModuleInfo(key: "audio_tower") var audioTower: Qwen2AudioEncoder
    @ModuleInfo(key: "multi_modal_projector") var projector: Qwen2AudioProjector
    @ModuleInfo(key: "language_model") public var languageModel: Qwen2Model
    public var tokenizer: Tokenizer?
    private let _frontend: Qwen2AudioFrontend
    public var sampleRate: Int { 16000 }
    public var maximumAudioSamples: Int { _frontend.maxSamples }
    public var defaultGenerationParameters: STTGenerateParameters {
        .init(maxTokens: 4096, temperature: 0, topP: 1)
    }

    public init(_ config: Qwen2AudioConfig) {
        self.config = config
        _audioTower.wrappedValue = Qwen2AudioEncoder(config.audioConfig)
        _projector.wrappedValue = Qwen2AudioProjector(config)
        _languageModel.wrappedValue = Qwen2Model(config.textConfig)
        _frontend = Qwen2AudioFrontend(config.audioConfig)
    }

    public func callAsFunction(
        inputIds: MLXArray, inputEmbeddings: MLXArray? = nil, cache: [KVCache]? = nil
    ) -> MLXArray {
        if let inputEmbeddings {
            return languageModel(inputEmbeddings: inputEmbeddings, cache: cache)
        }
        return languageModel(inputIds, cache: cache)
    }

    func extractFeatures(_ audio: MLXArray) -> MLXArray {
        _frontend(audio, dtype: audioTower.conv1.weight.dtype)
    }

    func getAudioFeatures(_ features: MLXArray) -> MLXArray { projector(audioTower(features)) }

    static func promptContent(audioTokenCounts: [Int], prompt: String?) -> String {
        let clips = audioTokenCounts.enumerated().map { index, count in
            "Audio \(index + 1): <|audio_bos|>" + String(repeating: "<|AUDIO|>", count: count)
                + "<|audio_eos|>"
        }
        return clips.joined(separator: "\n") + "\n" + (prompt ?? "Please transcribe the speech.")
    }

    func buildPrompt(audioTokenCounts: [Int], prompt: String?) -> MLXArray {
        guard let tokenizer else { preconditionFailure("Qwen2-Audio tokenizer is not loaded") }
        let content = Self.promptContent(audioTokenCounts: audioTokenCounts, prompt: prompt)
        let messages: [Tokenizers.Message] = [["role": "user", "content": content]]
        let ids: [Int]
        if let formatted = try? tokenizer.applyChatTemplate(messages: messages) {
            ids = formatted
        } else {
            ids = tokenizer.encode(
                text:
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\(content)<|im_end|>\n<|im_start|>assistant\n"
            )
        }
        precondition(
            ids.filter { $0 == config.audioTokenId }.count == audioTokenCounts.reduce(0, +),
            "Qwen2-Audio tokenizer must preserve each <|AUDIO|> placeholder")
        return MLXArray(ids.map(Int32.init))
    }

    func mergeEmbeddings(_ ids: MLXArray, audioFeatures: MLXArray) -> MLXArray {
        let isAudio = ids .== config.audioTokenId
        let textIds = MLX.where(isAudio, MLXArray(Int32(0)), ids)
        let text = languageModel.embedTokens(textIds.expandedDimensions(axis: 0))
        let indices = clip(
            cumsum(isAudio.asType(.int32)) - 1, min: 0, max: audioFeatures.dim(1) - 1)
        let audio = audioFeatures.asType(text.dtype)[0..., indices, 0...]
        return MLX.where(isAudio.expandedDimensions(axes: [0, 2]), audio, text)
    }

    func prepareInputs(_ audio: [MLXArray], prompt: String?) -> (MLXArray, MLXArray) {
        precondition(!audio.isEmpty, "Expected at least one audio clip")
        var features: [MLXArray] = []
        for clip in audio {
            let projected = getAudioFeatures(extractFeatures(clip))
            eval(projected)
            features.append(projected)
        }
        let ids = buildPrompt(audioTokenCounts: features.map { $0.dim(1) }, prompt: prompt)
        let embeddings = mergeEmbeddings(ids, audioFeatures: concatenated(features, axis: 1))
        eval(embeddings)
        return (ids, embeddings)
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        generateImpl(
            audio: [audio], parameters: generationParameters, prompt: nil, prefillStepSize: 2048,
            minP: 0, emit: nil)
    }

    public func generate(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, prefillStepSize: Int = 2048, minP: Float = 0
    ) -> STTOutput {
        generate(
            audio: [audio], generationParameters: generationParameters, prompt: prompt,
            prefillStepSize: prefillStepSize, minP: minP)
    }

    /// Supply several clips to ask a single question about them; this is one conversation, not a batch.
    public func generate(
        audio: [MLXArray], generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, prefillStepSize: Int = 2048, minP: Float = 0
    ) -> STTOutput {
        generateImpl(
            audio: audio, parameters: generationParameters ?? defaultGenerationParameters,
            prompt: prompt, prefillStepSize: prefillStepSize, minP: minP, emit: nil)
    }

    private func generateImpl(
        audio: [MLXArray], parameters p: STTGenerateParameters, prompt: String?,
        prefillStepSize: Int, minP: Float, emit: ((String) -> Void)?
    ) -> STTOutput {
        guard let tokenizer else { preconditionFailure("Qwen2-Audio tokenizer is not loaded") }
        precondition(
            prefillStepSize > 0 && p.maxTokens >= 0 && p.temperature >= 0 && p.repetitionPenalty > 0
        )
        let start = Date()
        let (ids, embeddings) = prepareInputs(audio, prompt: prompt)
        var cache: [KVCache] = (0 ..< config.textConfig.hiddenLayers).map { _ in KVCacheSimple() }
        let cacheConfiguration: KVCacheConfiguration?
        if let bits = p.kvBits {
            guard
                let affine = try? AffineKVCacheConfiguration(
                    bits: bits, groupSize: p.kvGroupSize, compressionStart: p.quantizedKVStart)
            else {
                preconditionFailure("Invalid Qwen2-Audio KV-cache quantization settings")
            }
            cacheConfiguration = KVCacheConfiguration(strategy: .affine(affine))
        } else {
            cacheConfiguration = nil
        }
        func compressCache() {
            if let cacheConfiguration {
                do {
                    _ = try applyKVCacheConfiguration(
                        cache: &cache, configuration: cacheConfiguration)
                } catch {
                    preconditionFailure("Unsupported Qwen2-Audio KV-cache configuration: \(error)")
                }
            }
        }
        var offset = 0
        while embeddings.dim(1) - offset > prefillStepSize {
            if Task.isCancelled { break }
            let hidden = embeddings[0..., offset ..< (offset + prefillStepSize), 0...]
            eval(languageModel(inputEmbeddings: hidden, cache: cache), cache)
            compressCache()
            offset += prefillStepSize
        }
        var logits = languageModel(inputEmbeddings: embeddings[0..., offset..., 0...], cache: cache)
        eval(logits)
        compressCache()
        let prefillTime = Date().timeIntervalSince(start)
        let generationStart = Date()
        let sampler: any LogitSampler =
            p.temperature == 0
            ? ArgMaxSampler()
            : TopPSampler(temperature: p.temperature, topP: p.topP, topK: p.topK, minP: minP)
        var tokens: [Int] = []
        var emitted = ""
        for _ in 0 ..< p.maxTokens {
            if Task.isCancelled { break }
            var scores = logits[0, -1, 0...]
            if p.repetitionPenalty != 1 && !tokens.isEmpty {
                let recent = Array(Set(tokens.suffix(max(1, p.repetitionContextSize))))
                let indices = MLXArray(recent.map(Int32.init))
                let values = scores[indices]
                scores[indices] = MLX.where(
                    values .> 0, values / p.repetitionPenalty, values * p.repetitionPenalty)
            }
            let token = sampler.sample(logits: scores).item(Int.self)
            if token == (tokenizer.eosTokenId ?? config.eosTokenId) { break }
            tokens.append(token)
            if emit != nil || p.verbose {
                let decoded = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
                if !decoded.contains("\u{FFFD}"), decoded.hasPrefix(emitted) {
                    let delta = String(decoded.dropFirst(emitted.count))
                    if !delta.isEmpty {
                        emit?(delta)
                        if p.verbose { print(delta, terminator: "") }
                    }
                    emitted = decoded
                }
            }
            let next = MLXArray([Int32(token)]).reshaped(1, 1)
            logits = languageModel(next, cache: cache)
            eval(logits)
            compressCache()
        }
        let text = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
        if text.hasPrefix(emitted) {
            let tail = String(text.dropFirst(emitted.count))
            if !tail.isEmpty {
                emit?(tail)
                if p.verbose { print(tail, terminator: "") }
            }
        }
        if p.verbose { print() }
        let generationTime = Date().timeIntervalSince(generationStart)
        let duration = audio.reduce(0.0) { total, clip in
            total + Double(min(clip.dim(0), maximumAudioSamples)) / Double(sampleRate)
        }
        return STTOutput(
            text: text, segments: [["start": 0.0, "end": duration, "text": text]],
            promptTokens: ids.size, generationTokens: tokens.count,
            totalTokens: ids.size + tokens.count,
            promptTps: Double(ids.size) / max(prefillTime, 0.001),
            generationTps: Double(tokens.count) / max(generationTime, 0.001),
            totalTime: Date().timeIntervalSince(start),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error>
    {
        generateStream(audio: [audio], generationParameters: generationParameters)
    }

    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, prefillStepSize: Int = 2048, minP: Float = 0
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        generateStream(
            audio: [audio], generationParameters: generationParameters,
            prompt: prompt, prefillStepSize: prefillStepSize, minP: minP)
    }

    public func generateStream(
        audio: [MLXArray], generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, prefillStepSize: Int = 2048, minP: Float = 0
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let model = UncheckedSendableBox(self)
        let input = UncheckedSendableBox(audio)
        let parameters = generationParameters ?? defaultGenerationParameters
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let result = model.value.generateImpl(
                        audio: input.value, parameters: parameters, prompt: prompt,
                        prefillStepSize: prefillStepSize, minP: minP
                    ) { continuation.yield(.token($0)) }
                    try Task.checkCancellation()
                    continuation.yield(.result(result))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, var value) in weights {
            if key.contains("embed_positions") || key.contains("rotary_emb.inv_freq") { continue }
            if ["audio_tower.conv1.weight", "audio_tower.conv2.weight"].contains(key),
                value.ndim == 3,
                value.dim(1) != 3 && value.dim(2) == 3
            {
                value = value.transposed(0, 2, 1)
            }
            result[key] = value
        }
        return result
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws
        -> Qwen2AudioModel
    {
        guard let repo = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["*.jinja"], cache: cache)
        return try await fromModelDirectory(directory)
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> Qwen2AudioModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen2AudioConfig.self, from: data)
        try config.validate()
        let model = Qwen2AudioModel(config)
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw STTError.modelNotInitialized("No Qwen2-Audio safetensors found")
        }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = sanitize(weights: weights)
        if config.textConfig.tieWordEmbeddings {
            weights = weights.filter { !$0.key.hasPrefix("language_model.lm_head.") }
        }
        if let quantization = config.perLayerQuantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] == nil
                    ? nil : quantization.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.tokenizer = try await loadQwen2AudioTokenizer(from: directory)
        model.audioTower.preparePositions()
        model.train(false)
        eval(model)
        return model
    }
}
