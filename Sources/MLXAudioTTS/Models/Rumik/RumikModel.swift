import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXNN
import MLXAudioCore
import MLXAudioCodecs
import Tokenizers

private final class RumikGenerationOwner: Sendable {}

/// Multilingual Indic and English synthesis with description and vocalization controls.
/// Like other speech models, direct forward calls require serialized use.
public final class RumikModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let configuration: RumikConfiguration
    let model: RumikCohereModel
    @ModuleInfo(key: "stop_predictor") var stopPredictor: MLXNN.Sequential
    private var _tokenizer: (any Tokenizer)?
    private var _mimi: Mimi?
    private let _generationCache = TTSGenerationCache()
    // The gate retains this token instead of retaining the model that owns it.
    private let _generationOwner = RumikGenerationOwner()

    public init(_ config: RumikConfiguration) {
        configuration = config
        model = RumikCohereModel(config)
        _stopPredictor.wrappedValue = MLXNN.Sequential {
            LayerNorm(dimensions: config.hiddenSize)
            Linear(config.hiddenSize, max(64, config.hiddenSize / 4))
            GELU()
            Linear(max(64, config.hiddenSize / 4), 1)
        }
    }

    public var sampleRate: Int { configuration.sampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(maxTokens: 2048, temperature: 0.8, topP: 1, topK: 30)
    }
    public var voices: [String] { configuration.speakers }

    public func makeCache() -> [KVCache] {
        (0..<configuration.hiddenLayers).map { i in
            (i + 1) % configuration.slidingWindowPattern == 0
                ? KVCacheSimple() : RotatingKVCache(maxSize: configuration.slidingWindow, keep: 0)
        }
    }
    public func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        model.embedding.asLinear(model(tokens, cache: cache)) * configuration.logitScale
    }
    public func stopProbability(_ hidden: MLXArray) -> MLXArray {
        sigmoid(stopPredictor(hidden))
    }

    public func buildPrompt(text: String, voice: String? = nil, description: String? = nil) throws -> String {
        let speaker: String
        if let voice {
            guard let match = voices.first(where: {
                $0.caseInsensitiveCompare(voice.trimmingCharacters(in: .whitespacesAndNewlines)) == .orderedSame
            }) else {
                throw AudioGenerationError.invalidInput("Unknown voice \(voice). Available: \(voices.joined(separator: ", "))")
            }
            speaker = match
        } else {
            guard let first = voices.first else { throw AudioGenerationError.invalidInput("No Rumik voices") }
            speaker = first
        }
        var text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let description, !description.isEmpty, !text.contains("<description=") {
            text = "<description=\"\(description.trimmingCharacters(in: .whitespacesAndNewlines))\"> \(text)"
        }
        return "<text>\(speaker): \(text)<audio>"
    }

    /// Feed the code-major/quantizer-minor vocabulary into complete Mimi frames.
    /// Invalid round-robin order discards the partial frame and resynchronizes at q=0.
    public func pushAudioToken(_ token: Int, frame: inout [Int32]) -> [Int32]? {
        let c = configuration
        guard (c.firstUnitID...c.lastUnitID).contains(token) else { frame = []; return nil }
        let offset = token - c.firstUnitID
        let (code, q) = (offset / c.numQuantizers, offset % c.numQuantizers)
        if q == frame.count {
            frame.append(Int32(code))
            if frame.count == c.numQuantizers {
                let complete = frame; frame = []; return complete
            }
        } else {
            frame = q == 0 ? [Int32(code)] : []
        }
        return nil
    }

    public func audioTokensToCodes(_ tokens: [Int]) throws -> MLXArray {
        var frame: [Int32] = []; var frames: [[Int32]] = []
        for token in tokens {
            if token == configuration.audioEndTokenID { break }
            if let complete = pushAudioToken(token, frame: &frame) { frames.append(complete) }
        }
        guard !frames.isEmpty else { throw AudioGenerationError.generationFailed("No complete Rumik audio frame") }
        return framesToCodes(frames)
    }
    private func framesToCodes(_ frames: [[Int32]]) -> MLXArray {
        MLXArray(frames.flatMap { $0 }).reshaped(frames.count, configuration.numQuantizers)
            .transposed().expandedDimensions(axis: 0)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") || key == "lm_head.weight" { continue }
            var key = key
            for i in [0, 1, 3] {
                if key.hasPrefix("stop_predictor.\(i).") {
                    key = key.replacingOccurrences(of: "stop_predictor.\(i).", with: "stop_predictor.layers.\(i).")
                }
            }
            result[key] = value
        }
        return result
    }

    public static func fromPretrained(_ repo: String, cache: HubCache = .default) async throws -> RumikModel {
        guard let id = Repo.ID(rawValue: repo) else { throw TTSModelError.invalidRepositoryID(repo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: id, requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory, cache: cache)
    }

    public static func fromModelDirectory(_ directory: URL, cache: HubCache = .default) async throws -> RumikModel {
        let config = try JSONDecoder().decode(RumikConfiguration.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try config.validate()
        let model = RumikModel(config)
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path }
        guard !files.isEmpty else { throw AudioGenerationError.modelNotInitialized("No Rumik safetensors") }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try loadArrays(url: file)) { _, new in new } }
        weights = model.sanitize(weights: weights)
        if let q = config.quantization {
            quantize(model: model) { path, _ in
                weights["\(path).scales"] == nil ? nil : q.asTuple
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model)
        model._tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let localCodec = directory.appendingPathComponent("mimi")
        if FileManager.default.fileExists(atPath: localCodec.path) {
            model._mimi = try Mimi.fromModelDirectory(localCodec)
        } else {
            model._mimi = try await Mimi.fromPretrained(repoId: config.mimiRepo, cache: cache, progressHandler: { _ in })
        }
        return model
    }

    public func generate(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MaterializedArray {
        guard refAudio == nil, refText == nil else {
            throw AudioGenerationError.invalidInput("Rumik does not support reference-audio voice cloning")
        }
        return (try await generate(text: text, voice: voice, parameters: generationParameters)).materialized()
    }

    public func generate(
        text: String, voice: String? = nil, options: RumikGenerationOptions = .init(),
        parameters: GenerateParameters? = nil
    ) async throws -> MaterializedArray {
        let result = try await _generationCache.withCache(model: _generationOwner, makeCache: { [] }) { _ in
            let audio = try self.run(text: text, voice: voice, options: options,
                                     parameters: parameters ?? self.defaultGenerationParameters,
                                     cache: self.makeCache(), streamingInterval: nil, emit: { _ in })
            return audio!.materialized()
        }
        return result
    }

    public func generateStream(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, voice: voice, refAudio: refAudio, refText: refText, language: language,
                       generationParameters: generationParameters, streamingInterval: 0.5)
    }

    public func generateStream(
        text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
        generationParameters: GenerateParameters, streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        guard refAudio == nil, refText == nil else {
            return AsyncThrowingStream { $0.finish(throwing: AudioGenerationError.invalidInput("Rumik does not support reference-audio voice cloning")) }
        }
        return generateStream(text: text, voice: voice, parameters: generationParameters, streamingInterval: streamingInterval)
    }

    public func generateStream(
        text: String, voice: String? = nil, options: RumikGenerationOptions = .init(),
        parameters: GenerateParameters? = nil, streamingInterval: Double = 0.5
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [self] in
            do {
                try await _generationCache.withCache(model: _generationOwner, makeCache: { [] }) { _ in
                    _ = try self.run(text: text, voice: voice, options: options,
                                     parameters: parameters ?? self.defaultGenerationParameters,
                                     cache: self.makeCache(), streamingInterval: streamingInterval,
                                     emit: { continuation.yield($0) })
                }
                continuation.finish()
            } catch { continuation.finish(throwing: error) }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    private func run(
        text: String, voice: String?, options: RumikGenerationOptions, parameters: GenerateParameters,
        cache: [KVCache], streamingInterval: Double?, emit: (AudioGeneration) -> Void
    ) throws -> MLXArray? {
        guard let tokenizer = _tokenizer, let mimi = _mimi else {
            throw AudioGenerationError.modelNotInitialized("Rumik tokenizer and Mimi are required")
        }
        guard options.minTokens >= 0, (parameters.maxTokens ?? 2048) > 0,
              streamingInterval == nil || (streamingInterval!.isFinite && streamingInterval! > 0
                  && streamingInterval! * configuration.frameRate < Double(Int.max)) else {
            throw AudioGenerationError.invalidInput("Invalid Rumik token or streaming limits")
        }
        let description = options.description.flatMap { $0.isEmpty ? nil : $0 } ?? options.instruct
        let prompt = try buildPrompt(text: text, voice: voice, description: description)
        let ids = tokenizer.encode(text: prompt)
        let head = RumikAudioHead(embedding: model.embedding, config: configuration)
        let sampler = parameters.sampler()
        var processor = parameters.processor()
        let start = Date()
        var hidden = model(MLXArray(ids.map(Int32.init)).expandedDimensions(axis: 0), cache: cache)[0..., -1, 0...]
        eval(hidden)
        let prefillTime = Date().timeIntervalSince(start)
        let decoder = MimiStreamingDecoder(mimi)
        defer { decoder.reset(); mimi.resetState() }
        let chunkFrames = max(1, Int(((streamingInterval ?? 0.5) * configuration.frameRate).rounded(.toNearestOrEven)))
        var frames: [[Int32]] = []; var frame: [Int32] = []
        var tokenCount = 0; var completeCount = 0
        let unitCount = configuration.numQuantizers * configuration.codebookSize
        let cols = MLXArray(0...unitCount)
        for step in 0..<(parameters.maxTokens ?? 2048) {
            try Task.checkCancellation()
            let token: Int = autoreleasepool {
                var logits = head(hidden) * configuration.logitScale
                if step < options.minTokens {
                    logits = MLX.where(cols .< unitCount, logits, -Float.infinity)
                } else {
                    let stop = stopProbability(hidden) .> 0.5
                    let forced = MLX.where(cols .== unitCount, MLXArray(0).asType(logits.dtype), MLXArray(-Float.infinity).asType(logits.dtype))
                    logits = MLX.where(stop, forced, logits)
                }
                // Processor history uses audio-head columns; generated feedback uses vocabulary ids.
                logits = processor?.process(logits: logits) ?? logits
                let next = sampler.sample(logits: logits)
                processor?.didSample(token: next)
                let col = next.item(Int.self)
                return col == unitCount ? configuration.audioEndTokenID : configuration.firstUnitID + col
            }
            tokenCount += 1
            emit(.token(token))
            if token == configuration.audioEndTokenID { break }
            if let complete = pushAudioToken(token, frame: &frame) {
                frames.append(complete); completeCount += 1
            }
            if streamingInterval != nil && frames.count >= chunkFrames {
                let audio = decoder.decodeFrames(framesToCodes(frames)).reshaped(-1)
                eval(audio)
                frames.removeAll(keepingCapacity: true)
                emit(.audio(audio))
            }
            if step + 1 < (parameters.maxTokens ?? 2048) {
                hidden = model(MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)[0..., -1, 0...]
                eval(hidden)
            }
        }
        guard completeCount > 0 else { throw AudioGenerationError.generationFailed("No complete Rumik audio frame") }
        let elapsed = Date().timeIntervalSince(start) - prefillTime
        emit(.info(AudioGenerationInfo(promptTokenCount: ids.count, generationTokenCount: tokenCount,
                                       prefillTime: prefillTime, generateTime: elapsed,
                                       tokensPerSecond: Double(tokenCount) / max(elapsed, 0.001),
                                       peakMemoryUsage: Double(Memory.peakMemory) / 1e9)))
        if streamingInterval != nil {
            if !frames.isEmpty {
                let audio = decoder.decodeFrames(framesToCodes(frames)).reshaped(-1)
                emit(.audio(audio))
            }
            return nil
        }
        let audio = mimi.decode(framesToCodes(frames)).reshaped(-1)
        eval(audio)
        return audio
    }
}
