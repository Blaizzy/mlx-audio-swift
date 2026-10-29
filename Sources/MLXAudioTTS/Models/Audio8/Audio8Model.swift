import Foundation
import HuggingFace
@preconcurrency import MLX
@preconcurrency import MLXAudioCore
import MLXNN
import Tokenizers

/// Controls specific to Audio8. Temperature, top-p, top-k, token budget, and seed
/// are supplied through `GenerateParameters`.
public struct Audio8GenerationOptions: Sendable {
    public var doSample: Bool
    public init(doSample: Bool = true) { self.doSample = doSample }
}

func audio8FilteredScores(_ logits: MLXArray, topK: Int, topP: Float, temperature: Float) -> MLXArray {
    let count = logits.shape.last!
    let indices = argSort(-logits, axis: -1)
    let sorted = takeAlong(logits, indices, axis: -1)
    let ranks = MLX.arange(count).reshaped([1, count])
    let cumulative = cumsum(softmax(sorted, axis: -1), axis: -1)
    // Legacy Audio8 nucleus filtering removes the crossing token, preserving rank zero.
    let remove = ((cumulative .> topP) .|| (ranks .>= max(1, min(topK, count)))) .&& (ranks .> 0)
    let mask = putAlong(MLXArray.zeros(remove.shape, dtype: .bool), indices, values: remove, axis: -1)
    return which(mask, MLXArray(-Float.infinity).asType(logits.dtype), logits) / max(temperature, 1e-5)
}

func audio8RaceSample(_ scores: MLXArray, uniform: MLXArray) -> MLXArray {
    argMax(softmax(scores, axis: -1) / -log(uniform), axis: -1).asType(.int32)
}

public final class Audio8Model: Module, SpeechGenerationModel, @unchecked Sendable {
    public static let defaultRepositoryID = "mlx-community/Audio8-TTS-Preview-0.6b-bf16"
    public let config: Audio8Config
    let model: Audio8Backbone
    let codec: Audio8Codec
    private var tokenizer: (any Tokenizer)?
    public var sampleRate: Int { config.codecSampleRate }
    public var defaultGenerationParameters: GenerateParameters {
        .init(maxTokens: 512, temperature: 0.7, topP: 0.9, topK: 50)
    }
    init(config: Audio8Config) {
        self.config = config; model = Audio8Backbone(config); codec = Audio8Codec(config)
    }

    static func cleanText(_ text: String) -> String { text.split(whereSeparator: \.isWhitespace).joined(separator: " ") }
    static func referenceText(_ text: String) -> String {
        let cleaned = cleanText(text)
        return cleaned.range(of: #"<\|speaker:\d+\|>"#, options: .regularExpression) != nil
            ? cleaned : "<|speaker:0|>" + cleaned
    }
    func promptSegments(text: String, referenceText: String?, hasReference: Bool) throws -> ([Int], [Int]) {
        guard let tokenizer else { throw AudioGenerationError.modelNotInitialized("Audio8 tokenizer is unavailable") }
        let target = Self.cleanText(text)
        guard !target.isEmpty else { throw AudioGenerationError.invalidInput("Text must not be empty") }
        func encode(_ parts: [String]) -> [Int] { parts.flatMap { tokenizer.encode(text: $0, addSpecialTokens: false) } }
        let suffix = ["<|im_end|>\n", "<|im_start|>user\n", target,
                      "<|im_end|>\n", "<|im_start|>assistant\n<|voice|>"]
        if !hasReference {
            return (encode(["<|im_start|>system\n", "convert the provided text to speech"] + suffix), [])
        }
        guard let referenceText, !Self.cleanText(referenceText).isEmpty else {
            throw AudioGenerationError.invalidInput("refText is required with Audio8 reference audio")
        }
        return (encode(["<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            Self.referenceText(referenceText), "\n\nSpeech:\n"]), encode(suffix))
    }

    func preparePrompt(prefix: [Int], suffix: [Int], referenceCodes: MLXArray?) -> MLXArray {
        let frames = referenceCodes?.shape[2] ?? 0
        let length = prefix.count + frames + suffix.count
        var rows = Array(repeating: Int32(0), count: (config.numCodebooks + 1) * length)
        for (i, token) in prefix.enumerated() { rows[i] = Int32(token) }
        for (i, token) in suffix.enumerated() { rows[prefix.count + frames + i] = Int32(token) }
        if let referenceCodes {
            let codes = referenceCodes.asArray(Int32.self)
            for i in 0..<frames {
                rows[prefix.count + i] = codes[i] + Int32(config.semanticBeginID)
                for q in 0..<config.numCodebooks { rows[(q + 1) * length + prefix.count + i] = codes[q * frames + i] }
            }
        }
        return MLXArray(rows).reshaped([1, config.numCodebooks + 1, length])
    }

    private func semanticScores(_ logits: MLXArray) -> MLXArray {
        if config.usesFalconSlow { return logits }
        let ids = MLX.arange(config.vocabSize).reshaped([1, -1])
        let valid = ((ids .>= config.semanticBeginID) .&& (ids .<= config.semanticEndID)) .|| (ids .== config.eosTokenID)
        return which(valid, logits, MLXArray(-Float.infinity).asType(logits.dtype))
    }
    private func sample(_ logits: MLXArray, parameters p: GenerateParameters, key: MLXArray,
                        options: Audio8GenerationOptions, semantic: Bool = false,
                        previous: [Int32] = [], rescueKey: MLXArray? = nil) -> MLXArray {
        let logits = semantic ? semanticScores(logits) : logits
        let scores = audio8FilteredScores(logits, topK: p.topK, topP: p.topP, temperature: p.temperature)
        guard options.doSample else { return argMax(scores, axis: -1).asType(.int32) }
        let normal = audio8RaceSample(scores, uniform: MLXRandom.uniform(low: 0.0, high: 1.0, scores.shape, key: key))
        guard semantic, let rescueKey else { return normal }
        let high = audio8FilteredScores(logits, topK: p.topK, topP: config.rasTopP, temperature: config.rasTemperature)
        let rescue = audio8RaceSample(high, uniform: MLXRandom.uniform(low: 0.0, high: 1.0, high.shape, key: rescueKey))
        // Evaluate both draws even before the history is initialized, as in the reference.
        eval(normal, rescue)
        let token = normal.item(Int32.self)
        let isSemantic = config.usesFalconSlow ? (0..<config.codebookSize).contains(Int(token))
            : (config.semanticBeginID...config.semanticEndID).contains(Int(token))
        return previous.contains(token) && isSemantic ? rescue : normal
    }

    func generateCodes(prompt: MLXArray, parameters p: GenerateParameters, options: Audio8GenerationOptions) throws -> MLXArray {
        let width = prompt.shape[2]
        guard width < config.maxSeqLen else { throw AudioGenerationError.invalidInput("Audio8 prompt exceeds its context window") }
        guard p.topK > 0, p.topP > 0, p.topP <= 1, p.temperature.isFinite, p.temperature >= 0,
              (p.maxTokens ?? 512) > 0 else { throw AudioGenerationError.invalidInput("Invalid Audio8 sampling parameters; topK must be positive") }
        let cache = model.makeSlowCache()
        var key = MLXRandom.key(p.seed ?? UInt64.random(in: 0...UInt64.max))
        var (logits, hidden) = model.slow(prompt, cache: cache)
        logits = logits[0..., -1, 0...]; hidden = hidden[0..., (hidden.shape[1] - 1)..<hidden.shape[1], 0...]
        var previous: [Int32] = [], frames: [MLXArray] = []
        for step in 0..<min(p.maxTokens ?? 512, config.maxSeqLen - width) {
            try Task.checkCancellation()
            let keys = MLXRandom.split(key: key, into: 4); key = keys[3]
            let semantic = sample(logits, parameters: p, key: keys[0], options: options,
                                  semantic: true, previous: previous, rescueKey: keys[1])
            let semanticID = semantic.item(Int32.self)
            if semanticID == (config.usesFalconSlow ? config.codebookSize : config.eosTokenID) { break }
            let fastCache: [KVCache] = model.fastLayers.map { _ in KVCacheSimple() }
            let projected = model.fastProjectIn?(hidden) ?? hidden
            eval(model.fast(projected, position: 0, cache: fastCache))
            var current = clip(config.usesFalconSlow ? semantic : semantic - Int32(config.semanticBeginID), min: 0, max: config.codebookSize - 1)
            var codes = [current], fastHidden = model.fastEmbeddings(current).expandedDimensions(axis: 1)
            var fastKey = keys[2]
            for position in 1..<config.numCodebooks {
                let scores = model.fast(fastHidden, position: position, cache: fastCache)
                let (nextKey, subkey) = MLXRandom.split(key: fastKey); fastKey = nextKey
                current = sample(scores, parameters: p, key: subkey, options: options)
                codes.append(current); fastHidden = model.fastEmbeddings(current).expandedDimensions(axis: 1)
            }
            let frame = stacked(codes, axis: 1)
            eval(frame); frames.append(frame)
            // The first reference step initializes zeros; it only appends semantic
            // tokens from the second step onward.
            if step == 0 { previous = Array(repeating: 0, count: config.rasWindowSize) }
            else { previous.removeFirst(); previous.append(semanticID) }
            if step + 1 < min(p.maxTokens ?? 512, config.maxSeqLen - width) {
                let fullSemantic = config.usesFalconSlow ? semantic + Int32(config.semanticBeginID) : semantic
                let column = concatenated([fullSemantic.reshaped([1, 1]), frame], axis: 1).expandedDimensions(axis: 2)
                (logits, hidden) = model.slow(column, offset: width + step, cache: cache)
                logits = logits[0..., -1, 0...]
            }
        }
        guard !frames.isEmpty else { throw AudioGenerationError.generationFailed("Audio8 generated no audio frames") }
        return stacked(frames, axis: 2)
    }

    private func run(text: String, reference: MLXArray?, transcript: String?, parameters: GenerateParameters,
                     options: Audio8GenerationOptions) throws -> (MLXArray, Int, Int) {
        let (prefix, suffix) = try promptSegments(text: text, referenceText: transcript, hasReference: reference != nil)
        var referenceCodes: MLXArray?
        if let reference {
            guard reference.size > 0,
                  reference.ndim == 1 || (reference.ndim == 2 && reference.shape.contains(1))
                    || (reference.ndim == 3 && reference.shape[0] == 1 && reference.shape[2] == 1),
                  all(isFinite(reference)).item(Bool.self) else {
                throw AudioGenerationError.invalidInput("Audio8 reference must be a finite, nonempty mono waveform at 44,100 Hz")
            }
            referenceCodes = codec.encode(reference)
            eval(referenceCodes!)
        }
        let prompt = preparePrompt(prefix: prefix, suffix: suffix, referenceCodes: referenceCodes)
        let codes = try generateCodes(prompt: prompt, parameters: parameters, options: options)
        let audio = codec.decode(codes)
        eval(audio)
        return (audio, prompt.shape[2], codes.shape[2])
    }

    public func generate(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
                         generationParameters: GenerateParameters) async throws -> sending MLXArray {
        try await generate(text: text, refAudio: refAudio, refText: refText,
                           generationParameters: generationParameters, options: .init())
    }
    public func generate(text: String, refAudio: sending MLXArray? = nil, refText: String? = nil,
                         generationParameters: GenerateParameters? = nil,
                         options: Audio8GenerationOptions = .init()) async throws -> sending MLXArray {
        let (audio, _, _) = try run(text: text, reference: refAudio, transcript: refText,
                         parameters: generationParameters ?? defaultGenerationParameters, options: options)
        nonisolated(unsafe) let result = audio
        return result
    }
    /// Audio8's reference generation emits one complete waveform per request.
    public func generateStream(text: String, voice: String?, refAudio: sending MLXArray?, refText: String?, language: String?,
                               generationParameters: GenerateParameters) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let reference = SendingBox(refAudio)
        return AsyncThrowingStream { continuation in
            let task = Task { @Sendable [self] in
                do {
                    let start = Date()
                    let (audio, prompt, frames) = try run(text: text, reference: reference.take(), transcript: refText,
                                                         parameters: generationParameters, options: .init())
                    let elapsed = Date().timeIntervalSince(start)
                    let info = AudioGenerationInfo(promptTokenCount: prompt, generationTokenCount: frames,
                        prefillTime: 0, generateTime: elapsed, tokensPerSecond: Double(frames) / max(elapsed, 1e-6),
                        peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
                    nonisolated(unsafe) let output = audio
                    if case .terminated = continuation.yield(.audio(output)) { throw CancellationError() }
                    continuation.yield(.info(info)); continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: weights.map { key, value in
            var parts = key.split(separator: ".").map(String.init)
            if parts.count > 4, parts[0] == "codec", parts[1] == "quantizer",
               ["downsample", "upsample"].contains(parts[2]) { parts.insert("layers", at: 4) }
            let name = parts.joined(separator: ".")
            let tensor = name.hasPrefix("model.slow.") && name.hasSuffix("conv1d.weight")
                && value.ndim == 3 && value.dim(1) == 1 && value.dim(2) > 1
                ? value.transposed(0, 2, 1) : value
            return (name, tensor)
        })
    }
    public static func fromPretrained(_ repository: String = defaultRepositoryID, cache: HubCache = .default) async throws -> Audio8Model {
        guard let id = Repo.ID(rawValue: repository) else { throw TTSModelError.invalidRepositoryID(repository) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: id, requiredExtension: ".safetensors", cache: cache)
        return try await fromModelDirectory(directory)
    }
    public static func fromModelDirectory(_ directory: URL) async throws -> Audio8Model {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Audio8Config.self, from: data)
        try config.validate()
        guard config.codecFilename == URL(fileURLWithPath: config.codecFilename).lastPathComponent,
              config.codecFilename.hasSuffix(".safetensors") else {
            throw AudioGenerationError.invalidInput("Audio8 requires its published safetensors codec")
        }
        let model = Audio8Model(config: config)
        var weights = try loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        for (key, value) in try loadArrays(url: directory.appendingPathComponent(config.codecFilename)) {
            guard weights[key] == nil else { throw AudioGenerationError.invalidInput("Duplicate Audio8 weight: \(key)") }
            weights[key] = value
        }
        weights = sanitize(weights)
        if let q = config.quantization {
            quantize(model: model) { path, _ in weights["\(path).scales"] == nil ? nil : q.asTuple }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        for (token, expected) in [("<|im_end|>", config.eosTokenID), ("<|semantic:0|>", config.semanticBeginID),
                                  ("<|semantic:\(config.codebookSize - 1)|>", config.semanticEndID)] {
            guard model.tokenizer?.convertTokenToId(token) == expected else {
                throw AudioGenerationError.invalidInput("Audio8 tokenizer has an incompatible \(token) ID")
            }
        }
        return model
    }
}
