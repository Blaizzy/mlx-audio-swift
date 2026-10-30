// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
// KugelAudio: https://github.com/Kugelaudio/kugelaudio-open (MIT).
import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import MLXRandom
import Tokenizers

public final class KugelAudioModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let configuration: KugelAudioConfiguration
    public var generationOptions = KugelAudioGenerationOptions()
    public var sampleRate: Int { configuration.sampleRate }
    public var defaultGenerationParameters: GenerateParameters { .init(maxTokens: 2048, temperature: 0) }

    static let speechStartID = 151652
    static let speechEndID = 151653
    static let speechDiffusionID = 151654
    static let eosID = 151643
    static let validSpeechTokens = [speechStartID, speechEndID, speechDiffusionID, eosID]
    @ModuleInfo(key: "language_model") var languageModel: Qwen2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "acoustic_tokenizer") var acousticTokenizer: VibeVoiceAcousticTokenizer
    @ModuleInfo(key: "acoustic_connector") var acousticConnector: VibeVoiceSpeechConnector
    @ModuleInfo(key: "prediction_head") var predictionHead: VibeVoiceDiffusionHead
    @ParameterInfo(key: "speech_scaling_factor") var speechScalingFactor: MLXArray
    @ParameterInfo(key: "speech_bias_factor") var speechBiasFactor: MLXArray
    var tokenizer: (any Tokenizer)?

    public init(_ configuration: KugelAudioConfiguration) {
        self.configuration = configuration
        _languageModel.wrappedValue = Qwen2ModelInner(configuration.decoder)
        _lmHead.wrappedValue = Linear(configuration.decoder.hiddenSize, configuration.decoder.vocabularySize, bias: false)
        _acousticTokenizer.wrappedValue = VibeVoiceAcousticTokenizer(configuration.acousticTokenizer)
        _acousticConnector.wrappedValue = VibeVoiceSpeechConnector(input: configuration.acousticVAEDim,
            output: configuration.decoder.hiddenSize, eps: configuration.decoder.rmsNormEps)
        _predictionHead.wrappedValue = VibeVoiceDiffusionHead(configuration.diffusion)
        _speechScalingFactor.wrappedValue = MLXArray(Float(1))
        _speechBiasFactor.wrappedValue = MLXArray(Float(0))
    }

    static func prompt(_ text: String) -> String {
        var text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.hasPrefix("Speaker") { text = "Speaker 0: " + text }
        return " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n Text input:\n \(text)\n Speech output:\n"
    }
    func promptTokens(_ text: String) throws -> [Int] {
        guard let tokenizer else { throw AudioGenerationError.modelNotInitialized("KugelAudio tokenizer") }
        return tokenizer.encode(text: Self.prompt(text), addSpecialTokens: false) + [Self.speechStartID]
    }
    /// Select among the four speech controls, preserving full-vocabulary argmax tie ordering.
    static func speechDecision(_ scores: [Float], hasLatents: Bool) -> (token: Int, finalLatent: Bool) {
        let best = scores.indices.max { a, b in
            scores[a] == scores[b] ? validSpeechTokens[a] > validSpeechTokens[b] : scores[a] < scores[b]
        }!
        let token = validSpeechTokens[best]
        return (token, hasLatents && (token == speechEndID || token == eosID) && scores[2] > scores[best] - 5)
    }
    func sampleSpeech(condition: MLXArray, negative: MLXArray?, options: KugelAudioGenerationOptions,
                      solver: KugelAudioSDESolver, initialNoise: MLXArray? = nil,
                      stepNoise: [MLXArray]? = nil) throws -> MLXArray {
        let guided = options.guidanceScale > 1 && negative != nil
        let condition = condition.asType(.float32)
        let combined = guided ? concatenated([condition, negative!.asType(.float32)], axis: 0) : condition
        let projected = predictionHead.projectCondition(combined)
        var speech = initialNoise ?? MLXRandom.normal([condition.dim(0), configuration.acousticVAEDim])
        solver.reset()
        for (i, t) in solver.timesteps.enumerated() {
            try Task.checkCancellation()
            let noisy = guided ? concatenated([speech, speech], axis: 0) : speech
            // Original KugelAudio inference casts timesteps to the floating speech dtype.
            let time = MLXArray(Array(repeating: Float(t), count: noisy.dim(0)))
            var prediction = predictionHead.forward(noisy,
                conditionWithTimestep: projected + predictionHead.embedTimesteps(time)).asType(.float32)
            if guided {
                let n = condition.dim(0)
                let positive = prediction[0..<n], negative = prediction[n...]
                prediction = negative + options.guidanceScale * (positive - negative)
            }
            speech = solver.step(prediction, sample: speech, noise: stepNoise?[i])
            eval(speech)
        }
        return speech
    }

    private func synthesize(text: String, parameters: GenerateParameters,
                            options: KugelAudioGenerationOptions) throws -> MLXArray {
        try configuration.validate()
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenerationError.invalidInput("text must be non-empty")
        }
        let maxTokens = parameters.maxTokens ?? 2048
        let steps = options.diffusionSteps ?? configuration.diffusion.ddpmNumInferenceSteps
        guard maxTokens > 0, options.guidanceScale.isFinite, options.guidanceScale >= 0,
              steps > 0, steps <= configuration.diffusion.ddpmNumSteps else {
            throw AudioGenerationError.invalidInput("Invalid KugelAudio token budget or diffusion settings")
        }
        if let seed = parameters.seed { MLXRandom.seed(seed) }
        let prompt = try promptTokens(text)
        let cache: [KVCache] = (0..<configuration.decoder.hiddenLayers).map { _ in KVCacheSimple() }
        var hidden = languageModel(MLXArray(prompt.map(Int32.init)).reshaped(1, -1), cache: cache)
        eval(hidden)
        let negCache: [KVCache]? = options.guidanceScale > 1
            ? (0..<configuration.decoder.hiddenLayers).map { _ in KVCacheSimple() } : nil
        var negHidden: MLXArray?
        if let negCache {
            negHidden = languageModel(MLXArray([Int32(Self.speechStartID)]).reshaped(1, 1), cache: negCache)
            eval(negHidden!)
        }
        let solver = KugelAudioSDESolver(configuration.diffusion, steps: steps)
        let validIDs = MLXArray(Self.validSpeechTokens.map(Int32.init))
        var latents: [MLXArray] = []
        for _ in 0..<maxTokens {
            try Task.checkCancellation()
            let condition = hidden[0..., -1, 0...]
            let logits = lmHead(condition)[0].take(validIDs).asType(.float32).asArray(Float.self)
            let decision = Self.speechDecision(logits, hasLatents: !latents.isEmpty)
            if decision.token == Self.speechDiffusionID || decision.finalLatent {
                let speech = try sampleSpeech(condition: condition, negative: negHidden?[0..., -1, 0...],
                    options: options, solver: solver)
                latents.append(speech.expandedDimensions(axis: 1))
                if decision.finalLatent { break }
                let embedded = acousticConnector(speech.expandedDimensions(axis: 1))
                hidden = languageModel(inputEmbeddings: embedded, cache: cache)
                eval(hidden)
                if let negCache {
                    negHidden = languageModel(inputEmbeddings: embedded, cache: negCache)
                    eval(negHidden!)
                }
            } else if decision.token == Self.speechStartID {
                hidden = languageModel(MLXArray([Int32(Self.speechStartID)]).reshaped(1, 1), cache: cache)
                eval(hidden)
            } else { break }
        }
        guard !latents.isEmpty else { return .zeros([0]) }
        var latent = concatenated(latents, axis: 1)
        if !isNaN(speechScalingFactor).item(Bool.self) { latent = latent / speechScalingFactor - speechBiasFactor }
        var audio = acousticTokenizer.decode(latent).reshaped(-1).asType(.float32)
        let peak = abs(audio).max().item(Float.self)
        if peak > 1 { audio = audio * (0.95 / peak) }
        eval(audio)
        return audio
    }

    public func generate(text: String, options: KugelAudioGenerationOptions = .init(),
                         generationParameters: GenerateParameters = .init(maxTokens: 2048, temperature: 0)) async throws -> MaterializedArray {
        try synthesize(text: text, parameters: generationParameters, options: options).materialized()
    }
    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                         language: String?, generationParameters: GenerateParameters) async throws -> MaterializedArray {
        try validateReference(refAudio, refText: refText)
        return (try synthesize(text: text, parameters: generationParameters, options: generationOptions)).materialized()
    }
    private func validateReference(_ audio: MLXArray?, refText: String?) throws {
        if audio != nil || refText != nil {
            throw AudioGenerationError.invalidInput("KugelAudio's current synthesis path does not accept reference audio or transcripts")
        }
    }
    /// KugelAudio decodes all latents together and emits one complete waveform.
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        let reference = refAudio
        let options = generationOptions
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try self.validateReference(reference, refText: refText)
                    let audio = try self.synthesize(text: text, parameters: generationParameters, options: options)
                    continuation.yield(.audio(audio))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let shapes = Dictionary(uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        var result: [String: MLXArray] = [:]
        for (raw, value) in weights {
            let key = canonicalVibeVoiceSynthesisPath(raw)
            // The reference has no semantic feedback or reference-audio encoder path.
            if key.hasPrefix("semantic_tokenizer.") || key.hasPrefix("semantic_connector.")
                || key.hasPrefix("acoustic_tokenizer.encoder.") || key.hasSuffix("rotary_emb.inv_freq") { continue }
            var tensor = value
            if tensor.ndim == 3, tensor.shape != shapes[key] {
                tensor = key.contains("convtr") ? tensor.transposed(1, 2, 0) : tensor.swappedAxes(1, 2)
            }
            result[key] = tensor
        }
        return result
    }
    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> KugelAudioModel {
        guard let repo = Repo.ID(rawValue: modelRepo) else { throw TTSModelError.invalidRepositoryID(modelRepo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors", cache: cache)
        return try await fromModelDirectory(directory, cache: cache)
    }
    public static func fromModelDirectory(_ directory: URL, cache: HubCache = .default) async throws -> KugelAudioModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(KugelAudioConfiguration.self, from: data)
        try config.validate()
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        if let q = json?["quantization_config"] as? [String: Any], q["quant_method"] as? String == "bitsandbytes" {
            throw AudioGenerationError.invalidInput("KugelAudio requires original safetensors or MLX quantization; bitsandbytes weights are unsupported")
        }
        let model = KugelAudioModel(config)
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter({ $0.pathExtension == "safetensors" }).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            weights.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        let sanitized = model.sanitize(weights)
        if let q = config.quantization {
            quantize(model: model) { path, _ in
                sanitized["\(path).scales"] == nil ? nil : q.quantization(layer: path)?.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model)
        let tokenizerDir: URL
        if FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path) {
            tokenizerDir = directory
        } else if FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer/tokenizer.json").path) {
            tokenizerDir = directory.appendingPathComponent("tokenizer")
        } else {
            let repo = config.decoder.vocabularySize <= 151936 ? "Qwen/Qwen2.5-1.5B" : "Qwen/Qwen2.5-7B"
            let id = Repo.ID(rawValue: repo)!
            let client = HubClient(cache: cache)
            let info = try await client.getModel(id, revision: "main", filesMetadata: false)
            guard let revision = info.sha else {
                throw AudioGenerationError.modelNotInitialized("Could not resolve KugelAudio tokenizer revision")
            }
            tokenizerDir = cache.cacheDirectory.appendingPathComponent("mlx-audio/tokenizers/" + repo.replacingOccurrences(of: "/", with: "_"))
            _ = try await client.downloadSnapshot(of: id, kind: .model, to: tokenizerDir, revision: revision,
                matching: ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"])
        }
        model.tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
        return model
    }
}
