// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import HuggingFace
@preconcurrency import MLX
@preconcurrency import MLXAudioCore
import MLXNN
import Tokenizers

public struct VibeVoiceRealtimeConfig: Decodable {
    public var modelType = "vibevoice_streaming"
    public var acousticTokenizerConfig = VibeVoiceAcousticConfig()
    public var diffusionHeadConfig = VibeVoiceDiffusionConfig()
    public var decoderConfig = Qwen2Configuration(
        hiddenSize: 896, hiddenLayers: 24, intermediateSize: 4864, attentionHeads: 14,
        rmsNormEps: 1e-6,
        vocabularySize: 151936, kvHeads: 2)
    public var ttsBackboneNumHiddenLayers = 20
    public var acousticVaeDim = 64
    public var sampleRate = 24000
    public var quantization: BaseConfiguration.PerLayerQuantization?
    public init() {}
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case acousticTokenizerConfig = "acoustic_tokenizer_config"
        case diffusionHeadConfig = "diffusion_head_config"
        case decoderConfig = "decoder_config"
        case ttsBackboneNumHiddenLayers = "tts_backbone_num_hidden_layers"
        case acousticVaeDim = "acoustic_vae_dim"
        case sampleRate = "sample_rate"
        case samplingRate = "sampling_rate"
    }
    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        acousticTokenizerConfig =
            try c.decodeIfPresent(VibeVoiceAcousticConfig.self, forKey: .acousticTokenizerConfig)
            ?? acousticTokenizerConfig
        diffusionHeadConfig =
            try c.decodeIfPresent(VibeVoiceDiffusionConfig.self, forKey: .diffusionHeadConfig)
            ?? diffusionHeadConfig
        decoderConfig =
            try c.decodeIfPresent(Qwen2Configuration.self, forKey: .decoderConfig) ?? decoderConfig
        ttsBackboneNumHiddenLayers =
            try c.decodeIfPresent(Int.self, forKey: .ttsBackboneNumHiddenLayers)
            ?? ttsBackboneNumHiddenLayers
        acousticVaeDim =
            try c.decodeIfPresent(Int.self, forKey: .acousticVaeDim)
            ?? acousticTokenizerConfig.vaeDim
        sampleRate =
            try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? c.decodeIfPresent(
                Int.self, forKey: .samplingRate) ?? sampleRate
        quantization = (try? BaseConfiguration(from: decoder))?.perLayerQuantization
        if var q = quantization {
            q.perLayerQuantization = Dictionary(
                uniqueKeysWithValues: q.perLayerQuantization.map {
                    (canonicalVibeVoiceSynthesisPath($0.key), $0.value)
                })
            quantization = q
        }
    }
    func validate() throws {
        let a = acousticTokenizerConfig
        let d = decoderConfig
        let f = diffusionHeadConfig
        guard modelType == "vibevoice_streaming", a.causal, a.channels == 1,
            a.encoderRatios.allSatisfy({ $0 > 0 }), !a.encoderRatios.isEmpty,
            a.encoderDepths.count == a.encoderRatios.count + 1,
            a.encoderDepths.allSatisfy({ $0 >= 0 }),
            (a.decoderRatios ?? a.encoderRatios).allSatisfy({ $0 > 0 }),
            (a.decoderDepths ?? a.encoderDepths).count == (a.decoderRatios ?? a.encoderRatios).count
                + 1,
            (a.decoderDepths ?? a.encoderDepths).allSatisfy({ $0 >= 0 }),
            a.layernormEps.isFinite, a.layernormEps > 0, a.layerScaleInitValue.isFinite,
            a.vaeDim > 0, a.encoderNFilters > 0, a.decoderNFilters > 0,
            acousticVaeDim == a.vaeDim, f.latentSize == a.vaeDim, f.hiddenSize == d.hiddenSize,
            d.hiddenSize > 0, d.intermediateSize > 0, d.vocabularySize > 0, d.attentionHeads > 0,
            d.kvHeads > 0,
            d.hiddenSize % d.attentionHeads == 0, d.attentionHeads % d.kvHeads == 0,
            ttsBackboneNumHiddenLayers > 0, ttsBackboneNumHiddenLayers < d.hiddenLayers,
            f.headLayers >= 0, f.headFFNRatio.isFinite, f.headFFNRatio > 0,
            f.rmsNormEps.isFinite, f.rmsNormEps > 0,
            f.ddpmNumSteps > 0, f.ddpmNumInferenceSteps > 0,
            f.ddpmNumInferenceSteps <= f.ddpmNumSteps,
            ["epsilon", "v_prediction", "sample"].contains(f.predictionType),
            ["cosine", "scaled_linear", "squaredcos_cap_v2", "linear"].contains(f.ddpmBetaSchedule),
            sampleRate > 0
        else {
            throw AudioGenerationError.invalidInput("Unsupported VibeVoice Realtime configuration")
        }
    }
}

public struct VibeVoiceRealtimeOptions: Sendable {
    public var guidanceScale: Float
    public var diffusionSteps: Int?
    public init(guidanceScale: Float = 1.5, diffusionSteps: Int? = nil) {
        self.guidanceScale = guidanceScale
        self.diffusionSteps = diffusionSteps
    }
}

struct VibeVoiceRealtimeVoice {
    let tensors: [String: MLXArray]
    init(tensors: [String: MLXArray], config: VibeVoiceRealtimeConfig) throws {
        let textLayers = config.decoderConfig.hiddenLayers - config.ttsBackboneNumHiddenLayers
        let dim = config.decoderConfig.hiddenSize
        for name in ["lm_hidden", "tts_lm_hidden", "neg_tts_lm_hidden"] {
            guard let x = tensors[name], x.ndim == 3, x.dim(0) == 1, x.dim(1) > 0, x.dim(2) == dim
            else {
                throw AudioGenerationError.invalidInput("Invalid VibeVoice voice tensor: \(name)")
            }
        }
        for (prefix, layers) in [
            ("lm", textLayers), ("tts_lm", config.ttsBackboneNumHiddenLayers),
            ("neg_tts_lm", config.ttsBackboneNumHiddenLayers),
        ] {
            var sequenceLength: Int?
            for i in 0..<layers {
                guard let k = tensors["\(prefix)_key_\(i)"],
                    let v = tensors["\(prefix)_value_\(i)"],
                    k.ndim == 4, v.shape == k.shape, k.dim(0) == 1,
                    k.dim(1) == config.decoderConfig.kvHeads,
                    k.dim(3) == dim / config.decoderConfig.attentionHeads, k.dim(2) > 0,
                    sequenceLength == nil || sequenceLength == k.dim(2)
                else {
                    throw AudioGenerationError.invalidInput(
                        "Invalid VibeVoice \(prefix) voice cache layer \(i)")
                }
                sequenceLength = k.dim(2)
            }
        }
        self.tensors = tensors
    }
    func makeCache(_ prefix: String, layers: Int) -> [KVCache] {
        (0..<layers).map { i in
            let cache = KVCacheSimple()
            // Slice to create independent array contexts; indexed cache updates then preserve the voice snapshot.
            cache.state = [
                tensors["\(prefix)_key_\(i)"]![0..., 0..., 0..., 0...],
                tensors["\(prefix)_value_\(i)"]![0..., 0..., 0..., 0...],
            ]
            return cache
        }
    }
}

public final class VibeVoiceRealtimeModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: VibeVoiceRealtimeConfig
    public var options = VibeVoiceRealtimeOptions()
    @ModuleInfo(key: "language_model") var languageModel: Qwen2ModelInner
    @ModuleInfo(key: "tts_language_model") var ttsLanguageModel: Qwen2ModelInner
    @ModuleInfo(key: "tts_input_types") var inputTypes: Embedding
    @ModuleInfo(key: "acoustic_tokenizer") var acousticTokenizer: VibeVoiceAcousticTokenizer
    @ModuleInfo(key: "acoustic_connector") var acousticConnector: VibeVoiceSpeechConnector
    @ModuleInfo(key: "prediction_head") var predictionHead: VibeVoiceDiffusionHead
    @ModuleInfo(key: "tts_eos_classifier") var eosClassifier: VibeVoiceBinaryClassifier
    @ParameterInfo(key: "speech_scaling_factor") var speechScalingFactor: MLXArray
    @ParameterInfo(key: "speech_bias_factor") var speechBiasFactor: MLXArray
    var tokenizer: Tokenizer?
    var modelDirectory: URL?
    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 512)
    }
    public var availableVoices: [String] {
        guard let directory = modelDirectory?.appendingPathComponent("voices"),
            let files = try? FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
        else { return [] }
        return files.filter { $0.pathExtension == "safetensors" }.map {
            $0.deletingPathExtension().lastPathComponent
        }.sorted()
    }
    public init(_ config: VibeVoiceRealtimeConfig) {
        self.config = config
        var text = config.decoderConfig
        text.hiddenLayers -= config.ttsBackboneNumHiddenLayers
        var speech = config.decoderConfig
        speech.hiddenLayers = config.ttsBackboneNumHiddenLayers
        _languageModel.wrappedValue = Qwen2ModelInner(text, includeNorm: false)
        _ttsLanguageModel.wrappedValue = Qwen2ModelInner(speech)
        _inputTypes.wrappedValue = Embedding(embeddingCount: 2, dimensions: text.hiddenSize)
        _acousticTokenizer.wrappedValue = VibeVoiceAcousticTokenizer(config.acousticTokenizerConfig)
        _acousticConnector.wrappedValue = VibeVoiceSpeechConnector(
            input: config.acousticVaeDim, output: text.hiddenSize)
        _predictionHead.wrappedValue = VibeVoiceDiffusionHead(config.diffusionHeadConfig)
        _eosClassifier.wrappedValue = VibeVoiceBinaryClassifier(text.hiddenSize)
        _speechScalingFactor.wrappedValue = MLXArray(Float(1))
        _speechBiasFactor.wrappedValue = MLXArray(Float(0))
    }
    func loadVoice(_ name: String) throws -> VibeVoiceRealtimeVoice {
        guard !name.contains("/"), !name.contains("\\"), name != ".", name != "..",
            let directory = modelDirectory
        else {
            throw AudioGenerationError.invalidInput("Invalid voice name or missing model directory")
        }
        let url = directory.appendingPathComponent("voices/\(name).safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioGenerationError.invalidInput(
                "Voice cache not found: \(name). Available voices: \(availableVoices.joined(separator: ", "))"
            )
        }
        return try VibeVoiceRealtimeVoice(tensors: MLX.loadArrays(url: url), config: config)
    }
    func sampleSpeech(
        condition: MLXArray, negative: MLXArray, solver: VibeVoiceDPMSolver, guidance: Float,
        noise: MLXArray? = nil
    ) -> MLXArray {
        solver.reset()
        let condition = condition.asType(.float32)
        let negative = negative.asType(.float32)
        var speech = noise ?? MLXRandom.normal([condition.dim(0), config.acousticVaeDim])
        let useGuidance = guidance != 1
        let projected = predictionHead.projectCondition(
            useGuidance ? concatenated([condition, negative], axis: 0) : condition)
        for t in solver.timesteps {
            let c = projected + predictionHead.embedTimesteps(MLXArray([Float(t)]))
            var prediction = predictionHead.forward(
                useGuidance ? concatenated([speech, speech], axis: 0) : speech,
                conditionWithTimestep: c)
            if useGuidance {
                let parts = split(prediction, parts: 2, axis: 0)
                prediction = parts[1] + guidance * (parts[0] - parts[1])
            }
            speech = solver.step(prediction, sample: speech)
        }
        return speech
    }
    @discardableResult
    func run(
        text: String, voice: String?, parameters: GenerateParameters, interval: Double,
        emit: (MLXArray) -> Void
    ) throws -> (tokens: Int, prompt: Int) {
        try config.validate()
        guard let tokenizer else {
            throw AudioGenerationError.modelNotInitialized("VibeVoice tokenizer is unavailable")
        }
        guard interval.isFinite, options.guidanceScale.isFinite,
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw AudioGenerationError.invalidInput(
                "Text must be nonempty and synthesis settings must be finite")
        }
        let steps = options.diffusionSteps ?? config.diffusionHeadConfig.ddpmNumInferenceSteps
        guard steps > 0, steps <= config.diffusionHeadConfig.ddpmNumSteps,
            (parameters.maxTokens ?? 512) > 0
        else {
            throw AudioGenerationError.invalidInput(
                "VibeVoice token and diffusion limits must be positive")
        }
        if let seed = parameters.seed { MLXRandom.seed(seed) }
        let snapshot = try loadVoice(voice ?? "en-Carter_man")
        let textLayerCount = config.decoderConfig.hiddenLayers - config.ttsBackboneNumHiddenLayers
        let textCache = snapshot.makeCache("lm", layers: textLayerCount)
        let speechCache = snapshot.makeCache("tts_lm", layers: config.ttsBackboneNumHiddenLayers)
        let negativeCache = snapshot.makeCache(
            "neg_tts_lm", layers: config.ttsBackboneNumHiddenLayers)
        var positive = snapshot.tensors["tts_lm_hidden"]![0..., (-1)..., 0...]
        var negative = snapshot.tensors["neg_tts_lm_hidden"]![0..., (-1)..., 0...]
        let ids = tokenizer.encode(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n",
            addSpecialTokens: false)
        let solver = VibeVoiceDPMSolver(
            trainingSteps: config.diffusionHeadConfig.ddpmNumSteps,
            betaSchedule: config.diffusionHeadConfig.ddpmBetaSchedule,
            predictionType: config.diffusionHeadConfig.predictionType)
        solver.setTimesteps(steps)
        let convolutionCache = VibeVoiceConvolutionCache()
        let framesPerChunk =
            interval <= 0
            ? 1
            : max(
                1,
                Int(
                    min(
                        interval * Double(sampleRate)
                            / Double(config.acousticTokenizerConfig.hopLength),
                        Double(Int.max / 2))))
        var pending: [MLXArray] = []
        var textPosition = 0
        var count = 0
        var finished = false
        func flush() {
            guard !pending.isEmpty else { return }
            let latent = concatenated(pending, axis: 1)
            pending.removeAll(keepingCapacity: true)
            let samples = acousticTokenizer.decode(
                latent / speechScalingFactor - speechBiasFactor, cache: convolutionCache
            ).reshaped(-1)
            eval(samples)
            convolutionCache.evaluate()
            // Materialize a consumer-owned chunk before transferring it across tasks.
            let output = MLXArray(samples.asArray(Float.self))
            emit(output)
        }
        while !finished && count < (parameters.maxTokens ?? 512) {
            try Task.checkCancellation()
            if textPosition < ids.count {
                let end = min(ids.count, textPosition + 5)
                let tokens = MLXArray(Array(ids[textPosition..<end])).reshaped(1, -1)
                let textHidden = languageModel(tokens, cache: textCache)
                let types = inputTypes(.ones([1, end - textPosition], dtype: .int32))
                positive =
                    ttsLanguageModel(inputEmbeddings: textHidden + types, cache: speechCache)[
                        0..., (-1)..., 0...]
                textPosition = end
            }
            for _ in 0..<6 {
                try Task.checkCancellation()
                let latent = sampleSpeech(
                    condition: positive[0..., -1, 0...], negative: negative[0..., -1, 0...],
                    solver: solver,
                    guidance: options.guidanceScale
                ).expandedDimensions(axis: 1)
                pending.append(latent)
                count += 1
                let feedback =
                    acousticConnector(latent) + inputTypes(.zeros([1, 1], dtype: .int32))
                positive = ttsLanguageModel(inputEmbeddings: feedback, cache: speechCache)
                negative = ttsLanguageModel(inputEmbeddings: feedback, cache: negativeCache)
                let stop = sigmoid(eosClassifier(positive[0..., -1, 0...]))
                eval(stop, negative)
                finished = stop.item(Float.self) > 0.5 || count >= (parameters.maxTokens ?? 512)
                if pending.count >= framesPerChunk { flush() }
                if finished { break }
            }
        }
        flush()
        return (count, ids.count)
    }
    public func generate(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MaterializedArray {
        guard refAudio == nil else {
            throw AudioGenerationError.invalidInput(
                "VibeVoice Realtime uses published voice caches; reference audio is supported by full VibeVoice"
            )
        }
        var chunks: [MLXArray] = []
        try run(text: text, voice: voice, parameters: generationParameters, interval: 2) {
            chunks.append($0)
        }
        return (chunks.isEmpty ? MLXArray.zeros([0]) : concatenated(chunks)).materialized()
    }
    public func generateStream(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text, voice: voice, refAudio: refAudio, refText: refText, language: language,
            generationParameters: generationParameters, streamingInterval: 2)
    }
    public func generateStream(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
        language: String?,
        generationParameters: GenerateParameters, streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let reference = refAudio
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable [self] in
                do {
                    guard reference == nil else {
                        throw AudioGenerationError.invalidInput(
                            "VibeVoice Realtime requires a voice cache")
                    }
                    let start = Date()
                    let result = try run(
                        text: text, voice: voice, parameters: generationParameters,
                        interval: streamingInterval
                    ) { continuation.yield(.audio($0)) }
                    let elapsed = Date().timeIntervalSince(start)
                    continuation.yield(
                        .info(
                            AudioGenerationInfo(
                                promptTokenCount: result.prompt,
                                generationTokenCount: result.tokens, prefillTime: 0,
                                generateTime: elapsed,
                                tokensPerSecond: Double(result.tokens) / max(elapsed, 0.001),
                                peakMemoryUsage: Double(Memory.peakMemory) / 1e9)))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let shapes = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        var result: [String: MLXArray] = [:]
        for (source, tensor) in weights {
            let key = canonicalVibeVoiceSynthesisPath(source)
            if key.contains("rotary_emb.inv_freq") { continue }
            var value = tensor
            if let target = shapes[key], value.ndim == 3, value.shape != target {
                if key.contains("convtr"), value.transposed(1, 2, 0).shape == target {
                    value = value.transposed(1, 2, 0)
                } else if value.transposed(0, 2, 1).shape == target {
                    value = value.transposed(0, 2, 1)
                }
            }
            result[key] = value
        }
        return result
    }
    public static func fromPretrained(
        _ repository: String, hfToken: String? = nil, cache: HubCache = .default
    ) async throws -> VibeVoiceRealtimeModel {
        guard let repo = Repo.ID(rawValue: repository) else {
            throw TTSModelError.invalidRepositoryID(repository)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: "safetensors",
            additionalMatchingPatterns: ["voices/*.safetensors"], hfToken: hfToken, cache: cache)
        return try await fromModelDirectory(directory, hfToken: hfToken, cache: cache)
    }
    public static func fromModelDirectory(
        _ directory: URL, tokenizerDirectory: URL? = nil, hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> VibeVoiceRealtimeModel {
        let config = try JSONDecoder().decode(
            VibeVoiceRealtimeConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try config.validate()
        let model = VibeVoiceRealtimeModel(config)
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "safetensors" }.sorted {
            $0.lastPathComponent < $1.lastPathComponent
        }
        guard !files.isEmpty else {
            throw AudioGenerationError.modelNotInitialized("No VibeVoice weights found")
        }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        weights = model.sanitize(weights: weights)
        if let q = config.quantization {
            quantize(model: model) { path, _ in
                weights[path + ".scales"] != nil ? q.quantization(layer: path)?.asTuple : nil
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model)
        guard model.speechScalingFactor.size == 1, model.speechBiasFactor.size == 1,
            model.speechScalingFactor.item(Float.self).isFinite,
            model.speechScalingFactor.item(Float.self) != 0,
            model.speechBiasFactor.item(Float.self).isFinite
        else {
            throw AudioGenerationError.invalidInput(
                "VibeVoice speech scaling factors must be finite scalars with nonzero scale")
        }
        model.modelDirectory = directory
        if let tokenizerDirectory {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
        } else if FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("tokenizer.json").path)
        {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        } else {
            let processor =
                (try? JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: directory.appendingPathComponent("preprocessor_config.json"))))
                as? [String: Any]
            let name =
                processor?["language_model_pretrained_name"] as? String ?? "Qwen/Qwen2.5-0.5B"
            guard let repo = Repo.ID(rawValue: name) else {
                throw TTSModelError.invalidRepositoryID(name)
            }
            let client = HubClient(host: HubClient.defaultHost, bearerToken: hfToken, cache: cache)
            let tokenDirectory = try await client.downloadSnapshot(
                of: repo, kind: .model,
                matching: ["tokenizer*.json", "vocab.json", "merges.txt", "config.json"])
            model.tokenizer = try await AutoTokenizer.from(modelFolder: tokenDirectory)
        }
        return model
    }
}
