import Foundation
import HuggingFace
@preconcurrency import MLX
@preconcurrency import MLXAudioCore
import MLXNN
import Tokenizers

public enum TADAError: Error, LocalizedError {
    case invalidInput(String)
    case missingComponent(String)
    public var errorDescription: String? {
        switch self { case .invalidInput(let message), .missingComponent(let message): message }
    }
}

public final class TADAModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: TADAConfiguration
    let model: TADALlama
    @ModuleInfo(key: "acoustic_proj") var acoustic: Linear
    @ModuleInfo(key: "acoustic_mask_emb") var maskEmbedding: Embedding
    @ModuleInfo(key: "time_start_embed") var timeStart: Embedding
    @ModuleInfo(key: "time_end_embed") var timeEnd: Embedding
    @ModuleInfo(key: "prediction_head") var head: TADADiffusionHead
    @ModuleInfo(key: "bottleneck_proj") var bottleneck: Linear?
    // Separate codec modules are deliberately outside the main checkpoint tree.
    private var _decoder: TADACodecDecoder?
    private var _encoder: TADACodecEncoder?
    private var _aligner: TADAAligner?
    private var _tokenizer: Tokenizers.Tokenizer?
    private var _directory: URL?
    private let generationLock = NSLock()
    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: GenerateParameters { .init(maxTokens: 1024, temperature: 0.6, topP: 0.9, repetitionPenalty: 1.1) }

    public init(_ config: TADAConfiguration) {
        self.config = config; model = TADALlama(config)
        _acoustic = ModuleInfo(wrappedValue: Linear(config.acousticDim, config.hiddenSize), key: "acoustic_proj")
        _maskEmbedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: 2, dimensions: config.hiddenSize), key: "acoustic_mask_emb")
        _timeStart = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.numTimeClasses, dimensions: config.hiddenSize), key: "time_start_embed")
        _timeEnd = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.numTimeClasses, dimensions: config.hiddenSize), key: "time_end_embed")
        _head = ModuleInfo(wrappedValue: TADADiffusionHead(config), key: "prediction_head")
        _bottleneck = ModuleInfo(wrappedValue: config.bottleneckDim.map { Linear(config.hiddenSize, $0) }, key: "bottleneck_proj")
    }
    func embeddings(ids: MLXArray, features: MLXArray, masks: MLXArray, before: MLXArray, after: MLXArray) -> MLXArray {
        model.embeddings(ids) + acoustic(features) + maskEmbedding(masks) + timeStart(before) + timeEnd(after)
    }
    func forward(ids: MLXArray, features: MLXArray, masks: MLXArray, before: MLXArray, after: MLXArray,
        cache: inout [(MLXArray, MLXArray)?]) -> MLXArray {
        model(embeddings(ids: ids, features: features, masks: masks, before: before, after: after), cache: &cache)
    }
    static func schedule(_ steps: Int, _ schedule: TADAGenerationOptions.Schedule) -> [Float] {
        let result: MLXArray
        switch schedule {
        case .cosine: result = 0.5 * (1 - cos(Float.pi * linspace(0, 1, count: steps + 1)))
        case .logsnr: result = sigmoid(-linspace(5, -5, count: steps + 1) / 2)
        default: result = linspace(0, 1, count: steps + 1)
        }
        var values = result.asArray(Float.self)
        values[0] = 0; values[steps] = 1
        return values
    }
    static func guidance(_ scale: Float, _ t: Float, _ schedule: TADAGenerationOptions.Schedule) -> Float {
        let scale = Double(scale), t = Double(t)
        switch schedule {
        case .linear: return Float(1 + (scale - 1) * (1 - t))
        case .cosine: return Float(1 + (scale - 1) * 0.5 * (1 + cos(.pi * t)))
        default: return Float(scale)
        }
    }
    func solve(_ noise: MLXArray, condition: MLXArray, negative: MLXArray, options: TADAGenerationOptions) throws -> MLXArray {
        let times = Self.schedule(options.flowSteps, options.timeSchedule)
        let cond = condition.reshaped(1, -1), neg = negative.reshaped(1, -1)
        var speech = noise
        for i in 0 ..< options.flowSteps {
            try Task.checkCancellation()
            let a = Self.guidance(options.acousticGuidance, times[i], options.guidanceSchedule)
            let d = Self.guidance(options.durationGuidance, times[i], options.guidanceSchedule)
            let velocity: MLXArray
            if a != 1 {
                var combined = concatenated([cond, neg], axis: 0)
                if let bottleneck { combined = bottleneck(combined) }
                let result = head(concatenated([speech, speech], axis: 0), time: MLXArray([times[i], times[i]]), condition: combined)
                let positive = result[0 ..< 1], negative = result[1 ..< 2]
                let scale = concatenated([MLXArray.ones([1, config.acousticDim]) * a, MLXArray.ones([1, 2 * config.timeBits]) * d], axis: 1)
                velocity = negative + scale * (positive - negative)
            } else {
                velocity = head(speech, time: MLXArray([times[i]]), condition: bottleneck?(cond) ?? cond)
            }
            speech = speech + (times[i + 1] - times[i]) * velocity
        }
        return speech
    }

    /// Audio may have any positive sample rate; tensors passed through the generic
    /// speech API are mono at the model's 24 kHz rate.
    public func encodeReference(audio: MLXArray, text: String, sampleRate: Int = 24000, sample: Bool = true, seed: UInt64? = nil) throws -> TADAReference {
        try acquireGeneration()
        defer { releaseGeneration() }
        return try encodeReferenceImpl(audio: audio, text: text, sampleRate: sampleRate, sample: sample, seed: seed)
    }
    private func encodeReferenceImpl(audio: MLXArray, text: String, sampleRate: Int = 24000, sample: Bool = true, seed: UInt64? = nil) throws -> TADAReference {
        try loadReferenceComponents()
        guard let encoder = _encoder, let aligner = _aligner, let tokenizer = _tokenizer else { throw TADAError.missingComponent("TADA reference components are unavailable") }
        let text = tadaNormalize(text), tokens = tokenizer.encode(text: text, addSpecialTokens: false)
        guard !tokens.isEmpty else { throw TADAError.invalidInput("Reference transcript must contain text") }
        guard audio.ndim == 1 || (audio.ndim == 2 && audio.dim(0) == 1), audio.size > 0 else {
            throw TADAError.invalidInput("Reference audio must be a nonempty mono waveform")
        }
        let resampled = try tadaResample(audio, from: sampleRate, to: self.sampleRate)
        let input16 = try tadaResample(resampled, from: self.sampleRate, to: 16000)
        guard input16.size >= 400 else { throw TADAError.invalidInput("Reference audio is too short for alignment") }
        let logits = aligner(input16)[0]
        let scores = take(logits, MLXArray(tokens), axis: 1).transposed(1, 0)
        let aligned = try tadaMonotonicAlign(scores: scores, audioLength: resampled.size, sampleRate: self.sampleRate)
        let masks = MLXArray(aligned.masks).reshaped(1, -1)
        let encoded = encoder.features(resampled, masks: masks)
        let positions = MLXArray(aligned.positions.map { Int32($0 - 1) })
        var features = take(encoded, positions, axis: 1)
        if sample {
            let random = seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()
            features = features + MLXRandom.normal(features.shape, key: random) * 0.5
        }
        features = (features - config.acousticMean) / config.acousticStd
        eval(features)
        return TADAReference(text: text, tokens: tokens, positions: aligned.positions, features: features)
    }

    private func validate(_ options: TADAGenerationOptions, maxTokens: Int) throws {
        guard options.flowSteps > 0, options.flowSteps <= 1000, maxTokens > config.shiftAcoustic,
              options.noiseTemperature.isFinite, options.noiseTemperature >= 0,
              options.acousticGuidance.isFinite, options.durationGuidance.isFinite,
              options.transitionSteps >= 0, options.speed.map({ $0.isFinite && $0 > 0 }) ?? true else {
            throw TADAError.invalidInput("Invalid TADA generation controls")
        }
    }
    private struct Prompt {
        var ids: [Int]
        var features: MLXArray?
        var masks: [Int32] = []
        var before: [Int] = []
        var after: [Int] = []
        var featureCount = 0
        var skip = 0
    }
    private func prompt(_ text: String, reference: TADAReference?, options: TADAGenerationOptions) throws -> Prompt {
        guard let reference else { throw TADAError.invalidInput("TADA requires reference audio and its matching transcript") }
        guard let tokenizer = _tokenizer else { throw TADAError.missingComponent("Tokenizer has not been loaded") }
        let normalized = tadaNormalize(text)
        guard !normalized.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw TADAError.invalidInput("Text must not be empty") }
        let prefix = [128000, 128006] + tokenizer.encode(text: "system", addSpecialTokens: false) + [128007, 128009, 128006] + tokenizer.encode(text: "assistant", addSpecialTokens: false) + [128007]
        let prefixCount = prefix.count - 1
        var p = Prompt(ids: prefix + reference.tokens + tokenizer.encode(text: " " + normalized, addSpecialTokens: false) + Array(repeating: 128009, count: config.shiftAcoustic))
        p.skip = max(0, prefixCount - options.transitionSteps) + options.transitionSteps - 1
        do {
            let count = prefixCount + reference.tokens.count
            let trim = options.transitionSteps > 0 && count > options.transitionSteps ? options.transitionSteps : 0
            p.featureCount = count - trim
            p.features = concatenated([MLXArray.zeros([1, prefixCount, config.acousticDim]), reference.features], axis: 1)[0..., 0 ..< p.featureCount, 0...]
            var gaps = [0], previous = 1
            for position in reference.positions { gaps.append(min(config.numTimeClasses - 1, max(0, position - previous))); previous = position }
            p.before = Array(repeating: 0, count: prefixCount) + Array(gaps.dropLast())
            p.after = Array(repeating: 0, count: prefixCount) + Array(gaps.dropFirst())
            p.before = Array(p.before.prefix(p.featureCount)); p.after = Array(p.after.prefix(p.featureCount))
            let mask = Array(repeating: Int32(0), count: prefixCount) + Array(repeating: Int32(1), count: reference.tokens.count)
            p.masks = Array(mask.prefix(p.featureCount).dropFirst()) + [1]
            var depth = 0
            for i in 0 ..< min(p.featureCount, p.ids.count) {
                let id = p.ids[i]
                if id == 128006 { depth += 1 }
                let structural = depth > 0 || [128000, 128001, 128006, 128007, 128009].contains(id)
                if id == 128007 { depth -= 1 }
                if !structural { p.ids[i] = 128004 }
            }
            p.skip = p.featureCount + options.transitionSteps - 1
        }
        return p
    }

    private func decode(_ features: [MLXArray], _ durations: [Int], skip: Int) throws -> MLXArray {
        guard let decoder = _decoder, !features.isEmpty else { throw TADAError.missingComponent("No acoustic frames generated") }
        let offset = skip > 0 && skip < features.count ? skip : 0
        var expanded: [MLXArray] = []
        var masks: [Int32] = []
        for i in offset ..< features.count {
            let zeros = max(0, durations[i] - 1)
            if zeros > 0 { expanded.append(.zeros([zeros, config.acousticDim])); masks += Array(repeating: 0, count: zeros) }
            expanded.append(features[i].reshaped(1, config.acousticDim) * config.acousticStd + config.acousticMean)
            masks.append(1)
        }
        let trailing = durations.last ?? 0
        if trailing > 0 { expanded.append(.zeros([trailing, config.acousticDim])); masks += Array(repeating: 0, count: trailing) }
        let audio = decoder(concatenated(expanded, axis: 0).expandedDimensions(axis: 0), masks: MLXArray(masks).reshaped(1, -1)).reshaped([-1])
        let leading = durations[offset] * (sampleRate / 50)
        let result = leading > 0 && leading < audio.size ? audio[leading...] : audio
        eval(result)
        return result
    }

    private func synthesize(text: String, reference: TADAReference?, parameters: GenerateParameters,
        options: TADAGenerationOptions, emit: ((AudioGeneration) -> Void)?, interval: Double) throws -> MaterializedArray {
        let maxTokens = parameters.maxTokens ?? 1024
        try validate(options, maxTokens: maxTokens)
        let random = parameters.seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()
        let p = try prompt(text, reference: reference, options: options)
        let count = min(p.ids.count, maxTokens), shift = config.shiftAcoustic
        var firstPass: [Int]?
        let passes = options.speed == nil ? 1 : 2
        var result = MLXArray.zeros([0])
        for pass in 0 ..< passes {
            var cache = Array<(MLXArray, MLXArray)?>(repeating: nil, count: config.numHiddenLayers)
            var features: [MLXArray] = [], durations: [Int] = []
            var acousticState = MLXArray.zeros([1, 1, config.acousticDim])
            var mask: Int32 = 0, before = 0, after = 0, emitted = 0, lastEmitFrames = 0
            var startStep = 0
            var promptBefore = p.before, promptAfter = p.after
            if let firstPass {
                promptBefore = [0] + firstPass
                promptAfter = firstPass + [1]
            }
            if let promptFeatures = p.features {
                let limit = min(p.ids.count - shift - 1, p.featureCount - 1,
                    promptBefore.count - 1, max(0, promptBefore.count - 2))
                if limit > 0 {
                    let length = min(count, shift + limit + 1)
                    if length > shift {
                        let n = min(length - shift - 1, p.featureCount)
                        let ac = concatenated([MLXArray.zeros([1, shift + 1, config.acousticDim]),
                            promptFeatures[0..., 0 ..< n, 0...]], axis: 1)
                        let ms = [Int32](repeating: 0, count: shift + 1) + Array(p.masks.prefix(n))
                        let tb = [Int](repeating: 0, count: shift + 1) + Array(promptBefore.dropFirst().prefix(n))
                        let ta = [Int](repeating: 0, count: shift + 1) + Array(promptAfter.dropFirst().prefix(n))
                        let embeds = embeddings(ids: MLXArray(Array(p.ids.prefix(length))).reshaped(1, length),
                            features: ac, masks: MLXArray(ms).reshaped(1, length), before: MLXArray(tb).reshaped(1, length), after: MLXArray(ta).reshaped(1, length))
                        let batch = options.acousticGuidance == 1 ? 1 : 2
                        let hidden = model(tiled(embeds, repetitions: [batch, 1, 1]), cache: &cache)
                        eval(hidden)
                        let frames = length - shift
                        for i in 0 ..< frames {
                            features.append(promptFeatures[0..., i ..< i + 1, 0...])
                            durations.append(promptBefore[i + 1])
                        }
                        acousticState = features.last!
                        mask = p.masks[frames - 1]
                        before = promptBefore[frames]; after = promptAfter[frames]
                        startStep = length
                    }
                }
            }
            for step in startStep ..< count {
                try Task.checkCancellation()
                let id = p.ids[step], negative = ![128006, 128007, 128009].contains(id) ? 128004 : id
                let batch = options.acousticGuidance == 1 ? 1 : 2
                let ids = MLXArray(batch == 2 ? [id, negative] : [id]).reshaped(batch, 1)
                let h = forward(ids: ids, features: tiled(acousticState, repetitions: [batch, 1, 1]),
                    masks: MLXArray(Array(repeating: mask, count: batch)).reshaped(batch, 1),
                    before: MLXArray(Array(repeating: before, count: batch)).reshaped(batch, 1),
                    after: MLXArray(Array(repeating: after, count: batch)).reshaped(batch, 1), cache: &cache)
                eval([h] + cache.flatMap { pair in pair.map { [$0.0, $0.1] } ?? [] })
                let speech = try solve(MLXRandom.normal([1, config.latentSize], key: random) * options.noiseTemperature,
                    condition: h[0 ..< 1], negative: batch == 2 ? h[1 ..< 2] : MLXArray.zeros([1, 1, config.hiddenSize]), options: options)
                eval(speech)
                if step == p.ids.count - 1 {
                    var logits = matmul(h[0 ..< 1, -1, 0...], model.embeddings.weight.T)
                    logits = MLX.where(MLXArray(0 ..< config.vocabSize) .== 128004, MLXArray(-Float.infinity), logits)
                    var processor = parameters.processor()
                    processor?.prompt(MLXArray(p.ids.map(Int32.init)))
                    logits = processor?.process(logits: logits) ?? logits
                    let next = withRandomState(random) { parameters.sampler().sample(logits: logits) }.item(Int.self)
                    if [128001, 128008, 128009].contains(next) { break }
                }
                if step >= shift {
                    let index = step - shift
                    if let promptFeatures = p.features, index < p.featureCount {
                        acousticState = promptFeatures[0..., index ..< index + 1, 0...]; mask = p.masks[index]
                    } else { acousticState = speech[0..., 0 ..< config.acousticDim].reshaped(1, 1, config.acousticDim); mask = 1 }
                    features.append(acousticState)
                    if let forced = firstPass, index < forced.count - 1 {
                        before = forced[index]; after = forced[index + 1]
                    } else if p.features != nil && index < promptBefore.count - 1 {
                        before = promptBefore[index + 1]; after = promptAfter[index + 1]
                    } else {
                        before = tadaGray(speech[0, config.acousticDim ..< config.acousticDim + config.timeBits])
                        after = tadaGray(speech[0, (config.acousticDim + config.timeBits) ..< config.latentSize])
                    }
                    durations.append(before)
                    // Keep 32 codec frames for the noncausal DAC receptive field.
                    // Re-decoding the accumulated prefix also closes attention segments.
                    let frames = durations.dropFirst(min(p.skip, durations.count)).reduce(0) { $0 + max(1, $1) }
                    if pass == passes - 1, let emit, frames - lastEmitFrames >= Int(interval * 50), frames > 32 {
                        let partial = try decode(features, durations + [before], skip: p.skip)
                        let safe = max(0, partial.size - (32 + before) * (sampleRate / 50))
                        if safe > emitted { emit(.audio(partial[emitted ..< safe])); emitted = safe }
                        lastEmitFrames = frames
                    }
                }
                emit?(.progress(Double(step + pass * count + 1) / Double(count * passes)))
            }
            guard !features.isEmpty else { throw TADAError.invalidInput("Token limit is too short to synthesize audio") }
            durations.append(durations.last ?? 0)
            if pass == 0, let speed = options.speed {
                firstPass = durations.map { min(config.numTimeClasses - 1, max(0, Int((Float($0) / speed).rounded(.toNearestOrEven)))) }
                continue
            }
            result = try decode(features, durations, skip: p.skip)
            if let emit, result.size > emitted { emit(.audio(result[emitted...])) }
        }
        return (emit == nil ? result : MLXArray.zeros([0])).materialized()
    }

    private func acquireGeneration() throws {
        guard generationLock.try() else { throw TADAError.invalidInput("TADA model is already generating") }
    }
    private func releaseGeneration() { generationLock.unlock() }

    private func generationInfo(text: String, reference: TADAReference?, parameters: GenerateParameters, elapsed: Double) -> AudioGenerationInfo {
        let promptCount = (reference?.tokens.count ?? 0) + 8
        let textCount = _tokenizer?.encode(text: " " + tadaNormalize(text), addSpecialTokens: false).count ?? 0
        let total = min(parameters.maxTokens ?? 1024, promptCount + textCount + config.shiftAcoustic)
        let generated = max(0, total - promptCount)
        return .init(promptTokenCount: promptCount, generationTokenCount: generated,
            prefillTime: 0, generateTime: elapsed, tokensPerSecond: Double(generated) / max(elapsed, 1e-9),
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    public func generate(text: String, reference: TADAReference? = nil, parameters: GenerateParameters = .init(maxTokens: 1024, temperature: 0.6, topP: 0.9, repetitionPenalty: 1.1), options: TADAGenerationOptions = .init()) async throws -> MaterializedArray {
        try acquireGeneration()
        defer { releaseGeneration() }
        return (try synthesize(text: text, reference: reference, parameters: parameters, options: options, emit: nil, interval: 2)).materialized()
    }
    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?, generationParameters: GenerateParameters) async throws -> MaterializedArray {
        guard refAudio != nil, refText != nil else { throw TADAError.invalidInput("Reference audio and a matching transcript are required for TADA") }
        try acquireGeneration()
        defer { releaseGeneration() }
        let reference = try refAudio.map { try encodeReferenceImpl(audio: $0, text: refText!, seed: generationParameters.seed) }
        return (try synthesize(text: text, reference: reference, parameters: generationParameters, options: .init(), emit: nil, interval: 2)).materialized()
    }
    public func generateStream(text: String, reference: TADAReference? = nil, parameters: GenerateParameters = .init(maxTokens: 1024, temperature: 0.6, topP: 0.9, repetitionPenalty: 1.1), options: TADAGenerationOptions = .init(), streamingInterval: Double = 2) -> AsyncThrowingStream<AudioGeneration, Error> {
        let prompt = reference
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [self] in
            do {
                try acquireGeneration()
                defer { releaseGeneration() }
                guard streamingInterval.isFinite, streamingInterval > 0 else { throw TADAError.invalidInput("Streaming interval must be positive") }
                let reference = prompt
                let start = Date()
                _ = try synthesize(text: text, reference: reference, parameters: parameters, options: options,
                    emit: { continuation.yield($0) }, interval: streamingInterval)
                let elapsed = Date().timeIntervalSince(start)
                continuation.yield(.info(generationInfo(text: text, reference: reference, parameters: parameters, elapsed: elapsed)))
                continuation.finish()
            } catch { continuation.finish(throwing: error) }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?, generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language, generationParameters: generationParameters, streamingInterval: 2)
    }
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?, generationParameters: GenerateParameters, streamingInterval: Double) -> AsyncThrowingStream<AudioGeneration, Error> {
        let referenceAudio = refAudio
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [self] in
            do {
                try acquireGeneration()
                defer { releaseGeneration() }
                let audio = referenceAudio
                guard audio != nil, refText != nil else { throw TADAError.invalidInput("Reference audio and a matching transcript are required for TADA") }
                guard streamingInterval.isFinite, streamingInterval > 0 else { throw TADAError.invalidInput("Streaming interval must be positive") }
                let reference = try audio.map { try encodeReferenceImpl(audio: $0, text: refText!, seed: generationParameters.seed) }
                let start = Date()
                _ = try synthesize(text: text, reference: reference, parameters: generationParameters, options: .init(), emit: { continuation.yield($0) }, interval: streamingInterval)
                let elapsed = Date().timeIntervalSince(start)
                continuation.yield(.info(generationInfo(text: text, reference: reference, parameters: generationParameters, elapsed: elapsed)))
                continuation.finish()
            } catch { continuation.finish(throwing: error) }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }
}

extension TADAModel {
    private static func hubClient(cache: HubCache, hfToken: String?) -> HubClient {
        if let hfToken, !hfToken.isEmpty {
            return HubClient(host: HubClient.defaultHost, bearerToken: hfToken, cache: cache)
        }
        return HubClient(cache: cache)
    }

    /// Loads the published nested package without converting or copying its weights.
    public static func fromPretrained(_ repository: String, cache: HubCache = .default, hfToken: String? = nil) async throws -> TADAModel {
        guard let id = Repo.ID(rawValue: repository) else { throw TTSModelError.invalidRepositoryID(repository) }
        let directory = cache.cacheDirectory.appendingPathComponent("mlx-audio/" + repository.replacingOccurrences(of: "/", with: "_"))
        let client = hubClient(cache: cache, hfToken: hfToken)
        let required = ["model/config.json", "model/weights.safetensors", "decoder/weights.safetensors", "encoder/weights.safetensors", "aligner/weights.safetensors"]
        if !required.allSatisfy({ FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }) {
            _ = try await client.downloadSnapshot(of: id, kind: .model, to: directory, matching: required)
        }
        return try await fromModelDirectory(directory, cache: cache, hfToken: hfToken)
    }
    public static func fromModelDirectory(_ directory: URL, cache: HubCache = .default, tokenizerDirectory: URL? = nil, hfToken: String? = nil) async throws -> TADAModel {
        let nested = directory.appendingPathComponent("model/config.json")
        let configURL = FileManager.default.fileExists(atPath: nested.path) ? nested : directory.appendingPathComponent("config.json")
        let config = try JSONDecoder().decode(TADAConfiguration.self, from: Data(contentsOf: configURL))
        let instance = TADAModel(config)
        let weights = try MLX.loadArrays(url: configURL.deletingLastPathComponent().appendingPathComponent("weights.safetensors"))
        try instance.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let decoder = TADACodecDecoder()
        try decoder.update(parameters: ModuleParameters.unflattened(try MLX.loadArrays(url: directory.appendingPathComponent("decoder/weights.safetensors"))), verify: .all)
        decoder.train(false); instance._decoder = decoder; instance._directory = directory
        let tokenizerFolder: URL
        if let tokenizerDirectory { tokenizerFolder = tokenizerDirectory }
        else if FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path) { tokenizerFolder = directory }
        else {
            guard let repo = Repo.ID(rawValue: config.tokenizerRepo) else { throw TADAError.invalidInput("Invalid tokenizer repository") }
            let client = hubClient(cache: cache, hfToken: hfToken)
            let file = try await client.downloadFile(at: "tokenizer.json", from: repo)
            _ = try await client.downloadFile(at: "tokenizer_config.json", from: repo)
            tokenizerFolder = file.deletingLastPathComponent()
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
        for (text, id) in [("<|begin_of_text|>", 128000), ("<|start_header_id|>", 128006), ("<|end_header_id|>", 128007), ("<|eot_id|>", 128009), ("<|finetune_right_pad_id|>", 128004)] {
            guard tokenizer.encode(text: text, addSpecialTokens: false) == [id] else { throw TADAError.invalidInput("TADA requires a compatible Llama 3 tokenizer: \(text)") }
        }
        instance._tokenizer = tokenizer
        instance.train(false); eval(instance, decoder)
        return instance
    }
    private func loadReferenceComponents() throws {
        guard _encoder == nil || _aligner == nil else { return }
        guard let directory = _directory else { throw TADAError.missingComponent("Reference codec directory is unavailable") }
        let encoder = TADACodecEncoder(), aligner = TADAAligner()
        try encoder.update(parameters: ModuleParameters.unflattened(try MLX.loadArrays(url: directory.appendingPathComponent("encoder/weights.safetensors"))), verify: .all)
        try aligner.update(parameters: ModuleParameters.unflattened(try MLX.loadArrays(url: directory.appendingPathComponent("aligner/weights.safetensors"))), verify: .all)
        encoder.train(false); aligner.train(false); eval(encoder, aligner)
        _encoder = encoder; _aligner = aligner
    }
}
