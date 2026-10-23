import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

final class VoxtralProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ config: VoxtralConfig) {
        _linear1.wrappedValue = Linear(
            config.audioConfig.intermediateSize, config.textConfig.hiddenSize, bias: false)
        _linear2.wrappedValue = Linear(
            config.textConfig.hiddenSize, config.textConfig.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(x))) }
}

/// Offline Voxtral transcription and audio understanding.
public final class VoxtralModel: Module, STTGenerationModel {
    public let config: VoxtralConfig
    public static let eosTokenIds: Set<Int> = [2, 4, 11]
    @ModuleInfo(key: "audio_tower") var audioTower: WhisperEncoder
    @ModuleInfo(key: "multi_modal_projector") var projector: VoxtralProjector
    @ModuleInfo(key: "language_model") public var languageModel: GLMASRLanguageModel
    var tokenizer: TekkenTokenizer?
    private let _frontend: VoxtralAudio
    public var sampleRate: Int { 16000 }
    public var defaultGenerationParameters: STTGenerateParameters {
        .init(maxTokens: 128, temperature: 0, topP: 0.95, language: "en")
    }

    public init(_ config: VoxtralConfig) {
        self.config = config
        _audioTower.wrappedValue = WhisperEncoder(config: config.audioConfig.whisperConfig)
        _projector.wrappedValue = VoxtralProjector(config)
        _languageModel.wrappedValue = GLMASRLanguageModel(config: config.textConfig)
        _frontend = VoxtralAudio(config: config.audioConfig)
    }

    func extractFeatures(_ audio: MLXArray) -> MLXArray { _frontend(audio) }

    func getAudioFeatures(_ features: MLXArray) -> MLXArray {
        // Keep independent encoder blocks bounded even for long recordings.
        var chunks: [MLXArray] = []
        for i in 0 ..< features.dim(0) {
            let encoded = audioTower(features[i ..< (i + 1)])
            let projected = projector(encoded.reshaped(-1, config.audioConfig.intermediateSize))
            eval(projected)
            chunks.append(projected)
        }
        return concatenated(chunks, axis: 0)
    }

    func buildPrompt(audioTokens: Int, language: String?, prompt: String?) -> MLXArray {
        guard let tokenizer else { preconditionFailure("Voxtral tokenizer is not loaded") }
        var tokens = [1, 3, 25] + Array(repeating: config.audioTokenId, count: audioTokens)
        if let prompt {
            tokens += tokenizer.encode(prompt)
            tokens.append(4)
        } else {
            tokens.append(4)
            if let language, !language.isEmpty {
                tokens += tokenizer.encode("lang:" + language.lowercased())
            }
            tokens.append(34)
        }
        return MLXArray(tokens.map(Int32.init))
    }

    func mergeEmbeddings(_ ids: MLXArray, audioFeatures: MLXArray) -> MLXArray {
        let isAudio = ids .== config.audioTokenId
        let text = languageModel.embedTokens(
            MLX.where(isAudio, MLXArray(Int32(0)), ids).expandedDimensions(axis: 0))
        let indices = clip(
            cumsum(isAudio.asType(.int32)) - 1, min: 0, max: audioFeatures.dim(0) - 1)
        let audio = audioFeatures.asType(text.dtype)[indices].expandedDimensions(axis: 0)
        return MLX.where(isAudio.expandedDimensions(axes: [0, 2]), audio, text)
    }

    func prepareInputs(_ audio: MLXArray, prompt: String?, language: String?) -> (
        MLXArray, MLXArray
    ) {
        let features = getAudioFeatures(extractFeatures(audio))
        let ids = buildPrompt(audioTokens: features.dim(0), language: language, prompt: prompt)
        let embeddings = mergeEmbeddings(ids, audioFeatures: features)
        eval(embeddings)
        return (ids, embeddings)
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        generateImpl(
            audio: audio, parameters: generationParameters, prompt: nil, prefillStepSize: 2048,
            minP: 0, emit: nil)
    }

    public func generate(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, prefillStepSize: Int = 2048, minP: Float = 0
    ) -> STTOutput {
        generateImpl(
            audio: audio, parameters: generationParameters ?? defaultGenerationParameters,
            prompt: prompt, prefillStepSize: prefillStepSize, minP: minP, emit: nil)
    }

    private func generateImpl(
        audio: MLXArray, parameters p: STTGenerateParameters, prompt: String?,
        prefillStepSize: Int, minP: Float, emit: ((String) -> Void)?
    ) -> STTOutput {
        guard let tokenizer else { preconditionFailure("Voxtral tokenizer is not loaded") }
        precondition(
            prefillStepSize > 0 && p.maxTokens >= 0 && p.temperature >= 0 && p.repetitionPenalty > 0
        )
        let start = Date()
        let (ids, embeddings) = prepareInputs(audio, prompt: prompt, language: p.language)
        var cache: [KVCache] = (0 ..< config.textConfig.numHiddenLayers).map { _ in KVCacheSimple()
        }
        let cacheConfiguration: KVCacheConfiguration?
        if let bits = p.kvBits {
            guard
                let affine = try? AffineKVCacheConfiguration(
                    bits: bits, groupSize: p.kvGroupSize, compressionStart: p.quantizedKVStart)
            else {
                preconditionFailure("Invalid Voxtral KV-cache quantization settings")
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
                    preconditionFailure("Unsupported Voxtral KV-cache configuration: \(error)")
                }
            }
        }
        var offset = 0
        while embeddings.dim(1) - offset > prefillStepSize {
            if Task.isCancelled { break }
            let hidden = embeddings[0..., offset ..< (offset + prefillStepSize), 0...]
            eval(languageModel(cache: cache, inputEmbeddings: hidden), cache)
            compressCache()
            offset += prefillStepSize
        }
        var logits = languageModel(cache: cache, inputEmbeddings: embeddings[0..., offset..., 0...])
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
            if Self.eosTokenIds.contains(token) { break }
            tokens.append(token)
            if emit != nil || p.verbose {
                let decoded = tokenizer.decode(tokenIds: tokens)
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
            logits = languageModel(inputs: next, cache: cache)
            eval(logits)
            compressCache()
        }
        let text = tokenizer.decode(tokenIds: tokens)
        if text.hasPrefix(emitted) {
            let tail = String(text.dropFirst(emitted.count))
            if !tail.isEmpty {
                emit?(tail)
                if p.verbose { print(tail, terminator: "") }
            }
        }
        if p.verbose { print() }
        let generationTime = Date().timeIntervalSince(generationStart)
        let duration = Double(audio.dim(0)) / Double(sampleRate)
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
        makeStream(
            audio: audio, generationParameters: generationParameters, prompt: nil,
            prefillStepSize: 2048, minP: 0)
    }

    public func generateStream(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        prompt: String? = nil, prefillStepSize: Int = 2048, minP: Float = 0
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        makeStream(
            audio: audio, generationParameters: generationParameters ?? defaultGenerationParameters,
            prompt: prompt, prefillStepSize: prefillStepSize, minP: minP)
    }

    private func makeStream(
        audio: MLXArray, generationParameters: STTGenerateParameters,
        prompt: String?, prefillStepSize: Int, minP: Float
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let model = UncheckedSendableBox(self)
        let input = UncheckedSendableBox(audio)
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                do {
                    try Task.checkCancellation()
                    let result = model.value.generateImpl(
                        audio: input.value, parameters: generationParameters,
                        prompt: prompt, prefillStepSize: prefillStepSize, minP: minP
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
            // MLX-VLM exports an unused top-level alias of the decoder embedding.
            if key.hasPrefix("embed_tokens.") || key.contains("rotary_emb.inv_freq") { continue }
            if ["audio_tower.conv1.weight", "audio_tower.conv2.weight"].contains(key),
                value.ndim == 3,
                value.dim(1) != 3 && value.dim(2) == 3
            {
                value = value.transposed(0, 2, 1)
            }
            result[canonicalVoxtralWeightPath(key)] = value
        }
        return result
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws
        -> VoxtralModel
    {
        guard let repo = Repo.ID(rawValue: modelRepo) else {
            throw STTModelError.invalidRepositoryID(modelRepo)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory)
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> VoxtralModel {
        let config = try JSONDecoder().decode(
            VoxtralConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try config.validate()
        let model = VoxtralModel(config)
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        let files: [URL]
        if FileManager.default.fileExists(atPath: index.path) {
            struct WeightIndex: Decodable { let weight_map: [String: String] }
            let map = try JSONDecoder().decode(WeightIndex.self, from: Data(contentsOf: index))
            let names = Set(map.weight_map.values).sorted()
            guard
                names.allSatisfy({
                    !$0.contains("/") && !$0.contains("\\") && $0.hasSuffix(".safetensors")
                })
            else {
                throw STTError.modelNotInitialized("Invalid Voxtral weight index")
            }
            files = names.map { directory.appendingPathComponent($0) }
        } else {
            files = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil
            )
            .filter { $0.pathExtension == "safetensors" }.sorted {
                $0.lastPathComponent < $1.lastPathComponent
            }
        }
        guard !files.isEmpty else {
            throw STTError.modelNotInitialized("No Voxtral safetensors found")
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
        model.tokenizer = try TekkenTokenizer.fromModelDirectory(directory)
        model.train(false)
        eval(model)
        return model
    }
}
