// Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN
import MLXRandom
import Tokenizers

final class VibeVoiceTTSSemanticTokenizer: Module {
    let encoder: VibeVoiceSynthesisEncoder
    init(_ config: VibeVoiceAcousticConfig) { encoder = VibeVoiceSynthesisEncoder(config) }
}

/// Full VibeVoice synthesis with reference voices and waveform semantic feedback.
public final class VibeVoiceTTSModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let configuration: VibeVoiceTTSConfiguration
    public var generationOptions = VibeVoiceTTSGenerationOptions()
    public var sampleRate: Int { configuration.sampleRate }
    public var defaultGenerationParameters: GenerateParameters { .init(temperature: 0.7) }
    static let speechStartID = 151652
    static let speechEndID = 151653
    static let speechDiffusionID = 151654
    static let eosID = 151643
    var eosTokenID = eosID
    var controlTokens = [speechStartID, speechEndID, speechDiffusionID, eosID]
    @ModuleInfo(key: "language_model") var languageModel: Qwen2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?
    @ModuleInfo(key: "acoustic_tokenizer") var acousticTokenizer: VibeVoiceAcousticTokenizer
    @ModuleInfo(key: "semantic_tokenizer") var semanticTokenizer: VibeVoiceTTSSemanticTokenizer
    @ModuleInfo(key: "acoustic_connector") var acousticConnector: VibeVoiceSpeechConnector
    @ModuleInfo(key: "semantic_connector") var semanticConnector: VibeVoiceSpeechConnector
    @ModuleInfo(key: "prediction_head") var predictionHead: VibeVoiceDiffusionHead
    @ParameterInfo(key: "speech_scaling_factor") var speechScalingFactor: MLXArray
    @ParameterInfo(key: "speech_bias_factor") var speechBiasFactor: MLXArray
    var tokenizer: (any Tokenizer)?

    public init(_ configuration: VibeVoiceTTSConfiguration) {
        self.configuration = configuration
        _languageModel.wrappedValue = Qwen2ModelInner(configuration.decoder)
        if !configuration.decoder.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(configuration.decoder.hiddenSize, configuration.decoder.vocabularySize, bias: false)
        }
        _acousticTokenizer.wrappedValue = VibeVoiceAcousticTokenizer(configuration.acousticTokenizer, includeEncoder: true)
        _semanticTokenizer.wrappedValue = VibeVoiceTTSSemanticTokenizer(configuration.semanticTokenizer)
        _acousticConnector.wrappedValue = VibeVoiceSpeechConnector(input: configuration.acousticVAEDim, output: configuration.decoder.hiddenSize)
        _semanticConnector.wrappedValue = VibeVoiceSpeechConnector(input: configuration.semanticVAEDim, output: configuration.decoder.hiddenSize)
        _predictionHead.wrappedValue = VibeVoiceDiffusionHead(configuration.diffusion)
        _speechScalingFactor.wrappedValue = MLXArray(Float(1))
        _speechBiasFactor.wrappedValue = MLXArray(Float(0))
    }
    func logits(_ hidden: MLXArray) -> MLXArray { lmHead?(hidden) ?? languageModel.projectToVocabulary(hidden) }
    static func resetNegativeCache(_ cache: [KVCache]?) {
        for var layer in cache ?? [] where layer.offset > 0 {
            let state = layer.state
            layer.state = state.map { $0[.ellipsis, ..<1, 0...] }
        }
    }
    func referenceEmbedding(_ input: MLXArray, sampleRate inputRate: Int,
                            samplingNoise: MLXArray? = nil, scaleNoise: MLXArray? = nil) throws -> MLXArray {
        guard (input.ndim == 1 || input.ndim == 2), input.size > 0,
              inputRate > 0, isFinite(input).all().item(Bool.self) else {
            throw AudioGenerationError.invalidInput("Reference audio must be finite, nonempty (samples,) or (samples, channels)")
        }
        var audio = (input.ndim == 2 ? mean(input, axis: -1) : input).asType(.float32)
        if inputRate != sampleRate { audio = try resampleAudio(audio, from: inputRate, to: sampleRate) }
        let rms = sqrt(mean(audio * audio) + Float(1e-6))
        audio = audio * (pow(Float(10), -25.0 / 20.0) / maximum(rms, Float(1e-6)))
        audio = audio / maximum(abs(audio).max() + Float(1e-6), 1)
        var latent = acousticTokenizer.encoder!(audio.reshaped(1, -1, 1))
        let std = configuration.acousticTokenizer.fixStd
        switch configuration.referenceDistribution {
        case "gaussian":
            let scale = (scaleNoise ?? MLXRandom.normal([1], dtype: latent.dtype)) * (std / 0.8)
            latent = latent + scale.reshaped(1, 1, 1) * (samplingNoise ?? MLXRandom.normal(latent.shape, dtype: latent.dtype))
        case "fix": latent = latent + std * (samplingNoise ?? MLXRandom.normal(latent.shape, dtype: latent.dtype))
        default: break
        }
        let result = acousticConnector((latent + speechBiasFactor) * speechScalingFactor)
        eval(result)
        return result
    }
    func sampleSpeech(condition: MLXArray, negative: MLXArray, options: VibeVoiceTTSGenerationOptions,
                      solver: VibeVoiceDPMSolver, initialNoise: MLXArray? = nil) throws -> MLXArray {
        let guided = options.guidanceScale != 1
        let combined = guided ? concatenated([condition, negative], axis: 0) : condition
        let projected = predictionHead.projectCondition(combined)
        var speech = initialNoise ?? MLXRandom.normal([condition.dim(0), configuration.acousticVAEDim], dtype: condition.dtype)
        solver.reset()
        for t in solver.timesteps {
            try Task.checkCancellation()
            let time = MLXArray([Float(t)]).asType(condition.dtype)
            let noisy = guided ? concatenated([speech, speech], axis: 0) : speech
            var prediction = predictionHead.forward(noisy, conditionWithTimestep: projected + predictionHead.embedTimesteps(time))
            if guided {
                let n = condition.dim(0)
                let positive = prediction[0..<n], negative = prediction[n...]
                prediction = negative + options.guidanceScale * (positive - negative)
            }
            speech = solver.step(prediction, sample: speech)
            eval(speech)
        }
        return speech
    }
    func selectToken(_ scores: MLXArray, options: VibeVoiceTTSGenerationOptions, parameters: GenerateParameters) -> Int {
        var scores = scores.take(MLXArray(controlTokens.map(Int32.init)), axis: -1).asType(.float32)
        if !options.sampleControlTokens { return controlTokens[argMax(scores, axis: -1).item(Int.self)] }
        scores = scores / max(parameters.temperature, 1e-5)
        // Filtering over at most five controls keeps reference top-k then top-p behavior explicit.
        var values = scores.reshaped(-1).asArray(Float.self)
        let sorted = values.indices.sorted { values[$0] > values[$1] }
        if parameters.topK > 0 && parameters.topK < values.count {
            for index in sorted.dropFirst(parameters.topK) { values[index] = -.infinity }
        }
        if parameters.topP > 0 && parameters.topP < 1 {
            let peak = values.max()!, total = values.reduce(Float(0)) { $0 + Foundation.exp($1 - peak) }
            var cumulative: Float = 0
            for index in sorted.reversed() {
                cumulative += Foundation.exp(values[index] - peak) / total
                if cumulative <= 1 - parameters.topP { values[index] = -.infinity }
            }
        }
        return controlTokens[MLXRandom.categorical(MLXArray(values)).item(Int.self)]
    }
    private func synthesize(text: String, references: [MLXArray], referenceSampleRate: Int,
                            parameters: GenerateParameters, options: VibeVoiceTTSGenerationOptions) throws -> MLXArray {
        try configuration.validate()
        guard let tokenizer else { throw AudioGenerationError.modelNotInitialized("VibeVoice tokenizer") }
        let turns = try VibeVoiceTTSPrompt.parse(text)
        let speakerCount = VibeVoiceTTSPrompt.speakerOrder(turns).count
        guard references.isEmpty || references.count == speakerCount else {
            throw AudioGenerationError.invalidInput("Provide one reference audio sample per speaker")
        }
        let steps = options.diffusionSteps ?? configuration.diffusion.ddpmNumInferenceSteps
        guard options.guidanceScale.isFinite, options.guidanceScale >= 0,
              steps > 0, steps <= configuration.diffusion.ddpmNumSteps,
              parameters.maxTokens == nil || parameters.maxTokens! > 0 else {
            throw AudioGenerationError.invalidInput("Invalid VibeVoice token budget or diffusion settings")
        }
        if let seed = parameters.seed { MLXRandom.seed(seed) }
        let embeddings = try references.map { try referenceEmbedding($0, sampleRate: referenceSampleRate) }
        let prompt = VibeVoiceTTSPrompt(turns: turns, referenceSteps: embeddings.isEmpty ? nil : embeddings.map { $0.dim(1) }, tokenizer: tokenizer)
        let contextBudget = configuration.maxPositionEmbeddings - prompt.tokens.count
        guard contextBudget > 0 else { throw AudioGenerationError.invalidInput("The speech prompt exceeds the model context length") }
        let maxSteps = min(contextBudget, parameters.maxTokens ?? 2 * prompt.tokens.count)
        let input = languageModel.embed(MLXArray(prompt.tokens.map(Int32.init)).reshaped(1, -1))
        for (range, embedding) in zip(prompt.referenceRanges, embeddings) { input[0..., range, 0...] = embedding.asType(input.dtype) }
        let cache: [KVCache] = (0..<configuration.decoder.hiddenLayers).map { _ in KVCacheSimple() }
        var hidden = languageModel(inputEmbeddings: input, cache: cache)
        eval(hidden)
        var negativeCache: [KVCache]?
        var previousEmbedding: MLXArray?
        let acousticCache = VibeVoiceConvolutionCache(), semanticCache = VibeVoiceConvolutionCache()
        let solver = VibeVoiceDPMSolver(trainingSteps: configuration.diffusion.ddpmNumSteps,
            betaSchedule: configuration.diffusion.ddpmBetaSchedule, predictionType: configuration.diffusion.predictionType)
        solver.setTimesteps(steps)
        var chunks: [MLXArray] = []
        for _ in 0..<maxSteps {
            try Task.checkCancellation()
            let condition = hidden[0..., -1, 0...]
            let token = selectToken(logits(condition), options: options, parameters: parameters)
            if token == eosTokenID { break }
            let next: MLXArray
            if token == Self.speechDiffusionID {
                var negative = condition
                if options.guidanceScale != 1 {
                    if negativeCache == nil { negativeCache = (0..<configuration.decoder.hiddenLayers).map { _ in KVCacheSimple() } }
                    let embedded = previousEmbedding ?? languageModel.embed(MLXArray([Int32(Self.speechStartID)]).reshaped(1, 1))
                    negative = languageModel(inputEmbeddings: embedded, cache: negativeCache)[0..., -1, 0...]
                    eval(negative)
                }
                let latent = try sampleSpeech(condition: condition, negative: negative, options: options, solver: solver).expandedDimensions(axis: 1)
                let decoded = acousticTokenizer.decode(latent / speechScalingFactor - speechBiasFactor, cache: acousticCache)
                chunks.append(decoded.reshaped(-1))
                let semantic = semanticTokenizer.encoder(decoded, cache: semanticCache)
                next = acousticConnector(latent) + semanticConnector(semantic)
                eval(next, decoded)
                acousticCache.evaluate(); semanticCache.evaluate()
            } else {
                next = languageModel.embed(MLXArray([Int32(token)]).reshaped(1, 1))
                if token == Self.speechStartID { Self.resetNegativeCache(negativeCache) }
                else if token == Self.speechEndID { acousticCache.state.removeAll(); semanticCache.state.removeAll() }
            }
            hidden = languageModel(inputEmbeddings: next, cache: cache)
            previousEmbedding = next
            eval(hidden)
        }
        let audio = chunks.isEmpty ? MLXArray.zeros([0]) : concatenated(chunks).asType(.float32)
        eval(audio)
        return audio
    }
    /// References follow first speaker appearance in the script. Samples may be mono or (samples, channels).
    public func generate(text: String, referenceAudio: sending [MLXArray] = [], referenceSampleRate: Int = 24000,
                         options: VibeVoiceTTSGenerationOptions = .init(),
                         generationParameters: GenerateParameters = .init(temperature: 0.7)) async throws -> sending MLXArray {
        try synthesize(text: text, references: referenceAudio, referenceSampleRate: referenceSampleRate,
            parameters: generationParameters, options: options)
    }
    public func generate(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?,
                         language: String?, generationParameters: GenerateParameters) async throws -> sending MLXArray {
        try synthesize(text: text, references: refAudio.map { [$0] } ?? [], referenceSampleRate: sampleRate,
            parameters: generationParameters, options: generationOptions)
    }
    /// The full-model reference returns one completed waveform; this adapter has the same behavior.
    public func generateStream(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let reference = SendingBox(refAudio), options = generationOptions
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let audio = try self.synthesize(text: text, references: reference.take().map { [$0] } ?? [],
                        referenceSampleRate: self.sampleRate, parameters: generationParameters, options: options)
                    continuation.yield(.audio(audio)); continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
