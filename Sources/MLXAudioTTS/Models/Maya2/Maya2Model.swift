import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXNN
import MLXAudioCore
import MLXAudioCodecs
import Tokenizers

private final class Maya2GenerationOwner: Sendable {}

/// Maya-2 Native multilingual speech synthesis. Low-level forward calls require serialized use.
public final class Maya2Model: Module, SpeechGenerationModel, @unchecked Sendable {
    public static let languages = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]
    public static let builtInVoices = ["Ahana", "Vaidehi", "Malini", "Charvi", "Jahnavi", "Anirban", "Nakul", "Vasant", "Siddhant", "Satvik", "Abhay", "Nivedita", "Ishwari", "Ritvik", "Vrinda", "Sreekar", "Kayal", "Shaunak", "Akshat", "Tanushree"]
    public var voices: [String] { Self.builtInVoices }
    public let configuration: Maya2Configuration
    let backbone: Maya2Backbone
    let depth: Maya2DepthDecoder
    @ModuleInfo(key: "audio_in") var audioIn: Maya2AudioInput
    @ModuleInfo(key: "speaker_anchor") var speakerAnchor: Maya2SpeakerAnchor?
    var _tokenizer: (any Tokenizer)?
    var _codec: Mimi?
    var _speakerEncoder: EcapaSpeakerEncoder?
    var _directory: URL?
    var _cache: HubCache = .default
    private let _generationCache = TTSGenerationCache()
    private let _generationOwner = Maya2GenerationOwner()

    public init(_ configuration: Maya2Configuration) {
        self.configuration = configuration
        backbone = Maya2Backbone(configuration); depth = Maya2DepthDecoder(configuration)
        _audioIn.wrappedValue = Maya2AudioInput(configuration)
        _speakerAnchor.wrappedValue = configuration.speakerAnchorEnabled ? Maya2SpeakerAnchor(configuration) : nil
    }
    public var sampleRate: Int { configuration.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 1500, temperature: 0.9, topP: 1, topK: 64)
    }
    func composeInputs(_ prompt: Maya2Prompt, references: MLXArray? = nil) -> MLXArray {
        let x = backbone.lm.model.embedTokens(MLXArray(prompt.tokens).expandedDimensions(axis: 0))
        let positions = prompt.kinds.indices.filter { prompt.kinds[$0] == 1 }
        if !positions.isEmpty {
            let codes = MLXArray(positions.flatMap { prompt.codes[$0] }).reshaped(positions.count, configuration.numCodebooks)
            x[0, MLXArray(positions), 0...] = audioIn.frames(codes).asType(x.dtype)
        }
        for pos in prompt.kinds.indices where prompt.kinds[pos] == 3 { x[0, pos, 0...] = audioIn.aeos().asType(x.dtype) }
        if let speakerAnchor {
            let anchor = speakerAnchor(references).asType(x.dtype)
            for pos in prompt.sentinels { x[0..., pos, 0...] = x[0..., pos, 0...] + anchor }
        }
        return x
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]? = nil) -> (MLXArray, MLXArray) {
        backbone(embeddings, cache: cache)
    }

    private func reference(_ audio: MLXArray, options: Maya2GenerationOptions) async throws -> (MLXArray?, MLXArray?) {
        guard audio.ndim == 1, audio.size >= sampleRate / 10, all(isFinite(audio)).item(Bool.self) else {
            throw AudioGenerationError.invalidInput("Reference must contain at least 0.1 seconds of finite mono 24 kHz audio")
        }
        var codes: MLXArray?, refs: MLXArray?
        if options.conditioning != .speaker {
            guard let codec = _codec else { throw AudioGenerationError.modelNotInitialized("Maya-2 Mimi codec is required") }
            codes = codec.encode(audio.asType(.float32).reshaped(1, 1, -1))[0]
            eval(codes!)
        }
        if options.conditioning != .inContext {
            if _speakerEncoder == nil {
                if let local = _directory?.appendingPathComponent("speaker_encoder/model.safetensors"), FileManager.default.fileExists(atPath: local.path) {
                    _speakerEncoder = try EcapaSpeakerEncoder.fromWeights(local)
                } else {
                    _speakerEncoder = try await EcapaSpeakerEncoder.fromPretrained(configuration.speakerEncoderRepo, cache: _cache)
                }
            }
            let samples = try resampleAudio(audio.asArray(Float.self), from: sampleRate, to: 16000)
            refs = try _speakerEncoder!.encode(MLXArray(samples)).reshaped(1, 1, -1)
            eval(refs!)
        }
        return (codes, refs)
    }

    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                         language: String?, generationParameters: GenerateParameters) async throws -> MaterializedArray {
        try await generate(text: text, voice: voice, refAudio: refAudio, refText: refText,
            language: language ?? "en", parameters: generationParameters)
    }
    public func generate(text: String, voice: String? = nil, refAudio: MaterializedArray? = nil,
                         refText: String? = nil, language: String = "en", options: Maya2GenerationOptions = .init(),
                         parameters: GenerateParameters? = nil) async throws -> MaterializedArray {
        try await _generationCache.withCache(model: _generationOwner, makeCache: { [] }) { _ in
            try await self.run(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language,
                options: options, parameters: parameters ?? self.defaultGenerationParameters, streamingInterval: nil, emit: { _ in })!
        }
    }
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText,
            language: language ?? "en", parameters: generationParameters)
    }
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters,
                               streamingInterval: Double) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText,
            language: language ?? "en", parameters: generationParameters, streamingInterval: streamingInterval)
    }
    public func generateStream(text: String, voice: String? = nil, refAudio: MaterializedArray? = nil,
                               refText: String? = nil, language: String = "en", options: Maya2GenerationOptions = .init(),
                               parameters: GenerateParameters? = nil, streamingInterval: Double = 0.32) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task.detached { @Sendable [self] in
            do {
                try await _generationCache.withCache(model: _generationOwner, makeCache: { [] }) { _ in
                    _ = try await self.run(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language,
                        options: options, parameters: parameters ?? self.defaultGenerationParameters,
                        streamingInterval: streamingInterval, emit: { continuation.yield($0) })
                }
                continuation.finish()
            } catch { continuation.finish(throwing: error) }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    static func filteredLogits(_ logits: MLXArray, history: [Int], parameters: GenerateParameters, options: Maya2GenerationOptions) -> MLXArray {
        var scaled = logits.asType(.float32) / (parameters.temperature == 0 ? 1 : parameters.temperature)
        if !history.isEmpty, options.presenceWindow > 0, options.presencePenalty != 0 {
            let ids = MLXArray(Array(Set(history.suffix(options.presenceWindow))).sorted())
            scaled[0..., ids] = scaled[0..., ids] - options.presencePenalty
        }
        if parameters.topK > 0 {
            let threshold = sorted(scaled, axis: -1)[0..., (-min(parameters.topK, scaled.dim(-1)))...][0..., ..<1]
            scaled = MLX.where(scaled .< threshold, -Float.infinity, scaled)
        }
        if parameters.topP < 1 {
            let indices = argSort(scaled, axis: -1)
            let probabilities = takeAlong(softmax(scaled, axis: -1), indices, axis: -1)
            let keep = cumsum(probabilities, axis: -1) .> (1 - parameters.topP)
            scaled = putAlong(scaled, indices, values: MLX.where(keep, takeAlong(scaled, indices, axis: -1), -Float.infinity), axis: -1)
        }
        return scaled
    }

    private func run(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String,
                     options: Maya2GenerationOptions, parameters: GenerateParameters, streamingInterval: Double?,
                     emit: (AudioGeneration) -> Void) async throws -> MaterializedArray? {
        guard let tokenizer = _tokenizer, let codec = _codec else {
            throw AudioGenerationError.modelNotInitialized("Load Maya-2 before generating")
        }
        let maxTokens = parameters.maxTokens ?? 1500
        guard maxTokens > 0, options.minTokens >= 0, parameters.temperature.isFinite, parameters.temperature >= 0,
              options.depthTemperature.isFinite, options.depthTemperature >= 0, parameters.topK >= 0, options.depthTopK >= 0,
              parameters.topP > 0, parameters.topP <= 1, options.presenceWindow >= 0, options.presencePenalty.isFinite,
              streamingInterval.map({ $0.isFinite && $0 > 0 && $0 * configuration.frameRate < Double(Int.max) }) ?? true else {
            throw AudioGenerationError.invalidInput("Invalid Maya-2 sampling or streaming parameters")
        }
        guard Self.languages.contains(language) else { throw AudioGenerationError.invalidInput("Unsupported Maya-2 language: \(language)") }
        var voice = voice, referenceCodes: MLXArray?, references: MLXArray?
        if let refAudio {
            guard voice == nil else { throw AudioGenerationError.invalidInput("Use either a named voice or refAudio") }
            if options.conditioning != .speaker, refText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true {
                throw AudioGenerationError.invalidInput("refText is required for in-context voice cloning")
            }
            (referenceCodes, references) = try await reference(refAudio, options: options)
        } else { voice = voice ?? "Ahana" }
        if let name = voice {
            guard let canonical = voices.first(where: { $0.caseInsensitiveCompare(name) == .orderedSame }) else {
                throw AudioGenerationError.invalidInput("Unknown Maya-2 voice: \(name)")
            }
            voice = canonical
        }
        let text = options.normalizeText ? try Maya2TextNormalizer.normalize(text, language: language) : text
        let prompt = try Maya2Prompt.build(text: text, language: language, voice: voice, referenceText: refText,
            referenceCodes: referenceCodes, referenceLanguage: options.referenceLanguage, numCodebooks: configuration.numCodebooks,
            encode: { tokenizer.encode(text: $0, addSpecialTokens: false) }, tokenID: { tokenizer.convertTokenToId($0) })
        guard maxTokens <= configuration.backbone.maxPositionEmbeddings - prompt.tokens.count else {
            throw AudioGenerationError.invalidInput("Prompt and maxTokens exceed the Maya-2 context length")
        }
        let start = Date()
        let cache: [KVCache] = (0..<configuration.backbone.hiddenLayers).map { _ in KVCacheSimple() }
        let memory = depth.speakerMemory(references)
        let decoder = MimiStreamingDecoder(codec)
        defer { decoder.reset(); codec.resetState() }
        let chunkFrames = max(1, Int(((streamingInterval ?? 0.32) * configuration.frameRate).rounded(.toNearestOrEven)))
        let sampler = GenerateParameters(temperature: parameters.temperature == 0 ? 0 : 1, seed: parameters.seed).sampler()
        let depthSampler = GenerateParameters(temperature: options.depthTemperature, topK: min(options.depthTopK, configuration.codebookSize), seed: parameters.seed.map { $0 &+ 1 }).sampler()
        var (hidden, logits) = self(composeInputs(prompt, references: references), cache: cache)
        eval(hidden, logits)
        let prefill = Date().timeIntervalSince(start)
        var history: [Int] = [], pending: [MLXArray] = []
        var hazard: Float = 0
        for step in 0..<maxTokens {
            try Task.checkCancellation()
            let raw = logits[0..., -1, 0...]
            var scaled = Self.filteredLogits(raw, history: history, parameters: parameters, options: options)
            let probabilities = softmax(scaled, axis: -1)
            let eos = configuration.codebookSize
            let eosProbability = probabilities[0, eos].item(Float.self)
            if step >= options.minTokens {
                hazard = 1 - (1 - hazard) * (1 - eosProbability)
                if argMax(probabilities, axis: -1).item(Int.self) == eos || eosProbability >= 0.35 || hazard >= 0.6 { break }
            }
            let columns = MLXArray(0...eos)
            scaled = MLX.where(columns .== eos, -Float.infinity, scaled)
            if !any(isFinite(scaled)).item(Bool.self) { scaled = MLX.where(columns .== eos, -Float.infinity, raw.asType(.float32)) }
            let cb0 = sampler.sample(logits: scaled)
            let codes = depth.generate(hidden: hidden[0..., -1, 0...], cb0: cb0, memory: memory,
                temperature: options.depthTemperature, topK: options.depthTopK, sampler: depthSampler)
            eval(codes)
            let token = cb0.item(Int.self)
            emit(.token(token)); history.append(token)
            let audio = decoder.decodeFrames(codes.expandedDimensions(axis: -1)).reshaped(-1).asType(.float32)
            eval(audio); pending.append(audio)
            if streamingInterval != nil, pending.count >= chunkFrames {
                emit(.audio(concatenated(pending).materialized())); pending.removeAll(keepingCapacity: true)
            }
            if step + 1 < maxTokens {
                (hidden, logits) = self(audioIn.frames(codes).expandedDimensions(axis: 1), cache: cache)
                eval(hidden, logits)
            }
        }
        guard !history.isEmpty else { throw AudioGenerationError.generationFailed("Maya-2 generated no audio frames") }
        let elapsed = Date().timeIntervalSince(start) - prefill
        emit(.info(AudioGenerationInfo(promptTokenCount: prompt.tokens.count, generationTokenCount: history.count,
            prefillTime: prefill, generateTime: elapsed, tokensPerSecond: Double(history.count) / max(elapsed, 0.001),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)))
        if streamingInterval != nil {
            if !pending.isEmpty { emit(.audio(concatenated(pending).materialized())) }
            return nil
        }
        return concatenated(pending).materialized()
    }
}
