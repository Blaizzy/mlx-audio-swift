import Foundation
import HuggingFace
@preconcurrency import MLX
@preconcurrency import MLXAudioCore
import MLXNN
import MLXRandom

public enum LongCatGuidance: String, Sendable { case cfg, apg }

public struct LongCatGenerationOptions: Sendable {
    public var steps: Int
    public var guidanceScale: Float
    public var guidance: LongCatGuidance
    public var seed: UInt64
    public var chunkSeconds: Double
    public var overlapSeconds: Double
    public init(steps: Int = 16, guidanceScale: Float = 4, guidance: LongCatGuidance = .cfg,
                seed: UInt64 = 1024, chunkSeconds: Double = 2, overlapSeconds: Double = 0.5) {
        self.steps = steps; self.guidanceScale = guidanceScale; self.guidance = guidance
        self.seed = seed; self.chunkSeconds = chunkSeconds; self.overlapSeconds = overlapSeconds
    }
}

/// Native LongCat conditional-flow synthesis with UMT5 text conditioning and a 24 kHz WAV-VAE.
public final class LongCatModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: LongCatConfiguration
    public var sampleRate: Int { config.samplingRate }
    public let defaultGenerationParameters = GenerateParameters(temperature: 0, seed: 1024)
    @ModuleInfo(key: "text_encoder") var textEncoder: LongCatUMT5Encoder
    @ModuleInfo var transformer: LongCatDiT
    @ModuleInfo var vae: LongCatVAE
    var tokenizer: SentencePieceTokenizer?

    init(_ config: LongCatConfiguration, tokenizer: SentencePieceTokenizer? = nil) {
        self.config = config; self.tokenizer = tokenizer
        _textEncoder.wrappedValue = LongCatUMT5Encoder(config.textEncoderConfig)
        _transformer.wrappedValue = LongCatDiT(config)
        _vae.wrappedValue = LongCatVAE(config.vaeConfig)
    }

    static func normalizedText(_ text: String) -> String {
        text.lowercased().replacingOccurrences(of: "[\"‘’]", with: " ", options: .regularExpression)
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func approximateDuration(_ text: String, maximum: Double) -> Double {
        var chinese = 0, alphabetic = 0, other = 0
        for scalar in text.unicodeScalars where !CharacterSet.whitespacesAndNewlines.contains(scalar) {
            if (0x4e00...0x9fff).contains(scalar.value) { chinese += 1 }
            else if CharacterSet.letters.contains(scalar) { alphabetic += 1 }
            else { other += 1 }
        }
        if chinese > alphabetic { chinese += other } else { alphabetic += other }
        return min(maximum, Double(chinese) * 0.21 + Double(alphabetic) * 0.082)
    }

    func tokenize(_ text: String) throws -> [Int] {
        guard let tokenizer else { throw AudioGenerationError.modelNotInitialized("LongCat UMT5 tokenizer is unavailable") }
        // HF recognizes added tokens before applying the unigram model. Each
        // ordinary segment gets its own SentencePiece word-boundary prefix.
        let pattern = "<(?:pad|s|unk)>|</s>|<extra_id_[0-9]+>"
        let expression = try NSRegularExpression(pattern: pattern)
        let source = text as NSString
        var result: [Int] = [], cursor = 0
        for match in expression.matches(in: text, range: NSRange(location: 0, length: source.length)) {
            let special = source.substring(with: match.range)
            guard let id = tokenizer.tokenID(for: special) else { continue }
            let ordinary = source.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !ordinary.isEmpty { result += tokenizer.encodeWithByteFallback(ordinary) }
            result.append(id)
            cursor = NSMaxRange(match.range)
        }
        let ordinary = source.substring(from: cursor).trimmingCharacters(in: .whitespacesAndNewlines)
        if !ordinary.isEmpty { result += tokenizer.encodeWithByteFallback(ordinary) }
        return result + [1]
    }

    static func normalizeTextFeatures(_ x: MLXArray) -> MLXArray {
        // Preserve BF16 operation rounding while accumulating reductions in
        // Float32. Older MLX CPU sums otherwise accumulate BF16 sequentially,
        // which differs from the text encoder's reference reduction contract.
        if x.dtype == .bfloat16 || x.dtype == .float16 {
            let reciprocal = 1 / Float(x.dim(-1))
            let mu = sum(x.asType(.float32), axis: -1, keepDims: true).asType(x.dtype) * reciprocal
            let centered = x - mu
            let v = sum(centered.square().asType(.float32), axis: -1, keepDims: true).asType(x.dtype) * reciprocal
            return centered * rsqrt(v + 1e-6)
        }
        return (x - mean(x, axis: -1, keepDims: true)) * rsqrt(variance(x, axis: -1, keepDims: true) + 1e-6)
    }

    func encodeText(_ ids: MLXArray, mask: MLXArray) -> MLXArray {
        var (hidden, embedding) = textEncoder(ids, mask: mask)
        if config.textNormFeat { hidden = Self.normalizeTextFeatures(hidden) }
        if config.textAddEmbed {
            if config.textNormFeat { embedding = Self.normalizeTextFeatures(embedding) }
            hidden = hidden + embedding
        }
        return hidden.asType(.float32)
    }

    func encodePromptAudio(_ input: MLXArray, noise: MLXArray? = nil) throws -> MLXArray {
        var wav: MLXArray
        switch input.ndim {
        case 1: wav = input.reshaped(1, -1, 1)
        case 2:
            if input.dim(0) == 1 { wav = input.reshaped(1, -1, 1) }
            else if input.dim(1) == 1 { wav = input.expandedDimensions(axis: 0) }
            else { throw AudioGenerationError.invalidInput("LongCat reference audio must be mono") }
        case 3:
            guard input.dim(0) == 1, input.dim(1) == 1 || input.dim(2) == 1 else {
                throw AudioGenerationError.invalidInput("LongCat reference audio must have one mono waveform")
            }
            wav = input.dim(2) == 1 ? input : input.transposed(0, 2, 1)
        default: throw AudioGenerationError.invalidInput("LongCat reference audio has invalid rank")
        }
        guard wav.dim(1) > 0 else { throw AudioGenerationError.invalidInput("LongCat reference audio is empty") }
        let hop = config.latentHop
        let frames = (wav.dim(1) + hop - 1) / hop
        let maxFrames = Int(config.maxWavDuration * Float(sampleRate) / Float(hop))
        guard frames < maxFrames else { throw AudioGenerationError.invalidInput("LongCat reference audio must be shorter than the maximum synthesis duration") }
        let padding = frames * hop - wav.dim(1) + hop * 3
        wav = padded(wav, widths: [.init(0), .init((0, padding)), .init(0)])
        let latent = vae.encode(wav, noise: noise)
        return latent[0..., 0..<(latent.dim(1) - 3), 0...]
    }

    static func projectedGuidance(_ conditional: MLXArray, _ unconditional: MLXArray,
                                  scale: Float, momentum: inout MLXArray?) -> MLXArray {
        let difference = conditional - unconditional
        let update = momentum.map { difference - 0.3 * $0 } ?? difference
        momentum = update
        let f = update.asType(.float32), c = conditional.asType(.float32)
        let direction = c / (sqrt(sum(c.square(), axes: [-1, -2], keepDims: true)) + 1e-8)
        let parallel = sum(f * direction, axes: [-1, -2], keepDims: true) * direction
        return conditional + scale * (f - 0.5 * parallel)
    }

    func sampleLatents(text: String, refAudio: MLXArray?, refText: String?, options: LongCatGenerationOptions,
                       initialNoise: MLXArray? = nil) throws -> MLXArray {
        guard options.steps >= 2, options.guidanceScale.isFinite, options.guidanceScale >= 0 else {
            throw AudioGenerationError.invalidInput("LongCat requires at least two time points and a finite nonnegative guidance scale")
        }
        let normalized = Self.normalizedText(text)
        guard !normalized.isEmpty else { throw AudioGenerationError.invalidInput("LongCat synthesis text is empty") }
        MLXRandom.seed(options.seed)
        let fullText = refAudio != nil && refText != nil ? Self.normalizedText(refText!) + " " + normalized : normalized
        let ids = try tokenize(fullText)
        let attentionMask = MLXArray.ones([1, ids.count])
        let condition = encodeText(MLXArray(ids).reshaped(1, -1), mask: attentionMask)
        let conditionLength = MLXArray([Float(ids.count)])
        let prompt = try refAudio.map { try encodePromptAudio($0) }
        let promptFrames = prompt?.dim(1) ?? 0
        let promptSeconds = Double(promptFrames * config.latentHop) / Double(sampleRate)
        var duration = Self.approximateDuration(normalized, maximum: Double(config.maxWavDuration) - promptSeconds)
        if prompt != nil, let refText {
            let estimate = Self.approximateDuration(refText, maximum: Double(config.maxWavDuration))
            duration *= min(1.5, max(1, promptSeconds / max(estimate, 1e-6)))
        }
        let maxFrames = Int(config.maxWavDuration * Float(sampleRate) / Float(config.latentHop))
        let totalFrames = min(maxFrames, Int(duration * Double(sampleRate) / Double(config.latentHop)) + promptFrames)
        guard totalFrames > promptFrames else { throw AudioGenerationError.invalidInput("LongCat text is too short for one audio frame") }
        let latentCondition: MLXArray
        if let prompt { latentCondition = padded(prompt, widths: [.init(0), .init((0, totalFrames - promptFrames)), .init(0)]) }
        else { latentCondition = .zeros([1, totalFrames, config.latentDim]) }
        let emptyLatent = MLXArray.zeros(latentCondition.shape)
        let negativeText = MLXArray.zeros(condition.shape)
        let mask = MLXArray.ones([1, totalFrames], dtype: .bool)
        let textMask = attentionMask.asType(.bool)
        var y = initialNoise ?? MLXRandom.normal([1, totalFrames, config.latentDim])
        guard y.shape == [1, totalFrames, config.latentDim] else { throw AudioGenerationError.invalidInput("LongCat initial noise has the wrong shape") }
        let promptNoise = y[0..., 0..<promptFrames, 0...]
        var momentum: MLXArray?
        for i in 0..<(options.steps - 1) {
            try Task.checkCancellation()
            let time = Float(i) / Float(options.steps - 1), dt = Float(1) / Float(options.steps - 1)
            if promptFrames > 0 {
                let prefix = promptNoise * (1 - time) + latentCondition[0..., 0..<promptFrames, 0...] * time
                y = concatenated([prefix, y[0..., promptFrames..., 0...]], axis: 1)
            }
            var prediction = transformer(y, text: condition, textLength: conditionLength, time: MLXArray([time]),
                mask: mask, textMask: textMask, latentCondition: latentCondition, returnLayer: config.repaDitLayer).0
            if options.guidanceScale >= 1e-5 {
                let nullInput = promptFrames > 0 ? concatenated([MLXArray.zeros([1, promptFrames, config.latentDim]), y[0..., promptFrames..., 0...]], axis: 1) : y
                let nullPrediction = transformer(nullInput, text: negativeText, textLength: conditionLength, time: MLXArray([time]),
                    mask: mask, textMask: textMask, latentCondition: emptyLatent, returnLayer: config.repaDitLayer).0
                if options.guidance == .cfg { prediction = prediction + (prediction - nullPrediction) * options.guidanceScale }
                else {
                    let sample = y[0..., promptFrames..., 0...]
                    let conditional = sample + (1 - time) * prediction[0..., promptFrames..., 0...]
                    let unconditional = sample + (1 - time) * nullPrediction[0..., promptFrames..., 0...]
                    let guided = Self.projectedGuidance(conditional, unconditional, scale: options.guidanceScale, momentum: &momentum)
                    prediction = padded((guided - sample) / (1 - time), widths: [.init(0), .init((promptFrames, 0)), .init(0)])
                }
            }
            y = y + prediction * dt
            eval(y)
        }
        return y[0..., promptFrames..., 0...]
    }

    public func generate(text: String, refAudio: MaterializedArray? = nil, refText: String? = nil,
                         options: LongCatGenerationOptions = .init()) async throws -> MaterializedArray {
        let latent = try sampleLatents(text: text, refAudio: refAudio, refText: refText, options: options)
        let audio = vae.decode(latent).flattened()
        eval(audio)
        return audio.materialized()
    }

    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
                         generationParameters: GenerateParameters) async throws -> MaterializedArray {
        try await generate(text: text, refAudio: refAudio, refText: refText,
            options: .init(seed: generationParameters.seed ?? 1024))
    }

    func decodeChunks(_ latent: MLXArray, options: LongCatGenerationOptions, emit: (MaterializedArray) throws -> Void) throws {
        guard options.chunkSeconds.isFinite, options.chunkSeconds > 0,
              options.overlapSeconds.isFinite, options.overlapSeconds >= 0,
              options.overlapSeconds < options.chunkSeconds else {
            throw AudioGenerationError.invalidInput("LongCat overlap must be nonnegative and shorter than the decoding chunk")
        }
        let ratio = config.vaeConfig.downsamplingRatio
        let chunk = max(1, Int(min(Double(latent.dim(1)), options.chunkSeconds * Double(sampleRate) / Double(ratio))))
        let overlap = max(0, Int(min(Double(latent.dim(1)), options.overlapSeconds * Double(sampleRate) / Double(ratio))))
        let hop = max(1, chunk - overlap), overlapSamples = overlap * ratio
        var tail: MLXArray?
        var start = 0
        while start < latent.dim(1) {
            try Task.checkCancellation()
            let end = min(start + chunk, latent.dim(1)), last = end == latent.dim(1)
            let left = min(overlap, start), right = min(overlap, latent.dim(1) - end)
            let decoded = vae.decode(latent[0..., (start - left)..<(end + right), 0...]).flattened()
            let audio = decoded[(left * ratio)..<(decoded.size - right * ratio)]
            var output: MLXArray
            if let previous = tail, overlapSamples > 0 {
                let count = min(overlapSamples, previous.size, audio.size)
                let phase = MLXArray.linspace(0, 1, count: count)
                let fade = 0.5 * (1 - cos(Float.pi * phase))
                let blended = previous[0..<count] * (1 - fade) + audio[0..<count] * fade
                output = concatenated([blended, audio[count...]])
            } else { output = audio }
            if !last, overlapSamples > 0 {
                let committed = output.size - overlapSamples
                tail = output[committed...]
                output = output[0..<committed]
                eval(tail!)
            }
            try emit(output.materialized())
            if last { break }
            start += hop
        }
    }

    public func generateStream(text: String, refAudio: MaterializedArray? = nil, refText: String? = nil,
                               options: LongCatGenerationOptions = .init()) -> AsyncThrowingStream<AudioGeneration, Error> {
        let reference = refAudio
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else { continuation.finish(throwing: AudioGenerationError.modelNotInitialized("LongCat model was released")); return }
            do {
                let start = Date()
                let latents = try self.sampleLatents(text: text, refAudio: reference, refText: refText, options: options)
                var samples = 0
                try self.decodeChunks(latents, options: options) { audio in
                    samples += audio.size
                    continuation.yield(.audio(audio))
                }
                let elapsed = Date().timeIntervalSince(start)
                continuation.yield(.info(AudioGenerationInfo(promptTokenCount: 0, generationTokenCount: samples,
                    prefillTime: 0, generateTime: elapsed, tokensPerSecond: Double(samples) / max(elapsed, 0.001),
                    peakMemoryUsage: Double(Memory.peakMemory) / 1e9)))
                continuation.finish()
            } catch { continuation.finish(throwing: error) }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
                               generationParameters: GenerateParameters) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, refAudio: refAudio, refText: refText, options: .init(seed: generationParameters.seed ?? 1024))
    }
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?, language: String?,
                               generationParameters: GenerateParameters, streamingInterval: Double) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(text: text, refAudio: refAudio, refText: refText,
            options: .init(seed: generationParameters.seed ?? 1024, chunkSeconds: streamingInterval,
                overlapSeconds: min(0.5, max(0, streamingInterval / 4))))
    }

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> LongCatModel {
        guard let repo = Repo.ID(rawValue: modelRepo) else { throw TTSModelError.invalidRepositoryID(modelRepo) }
        let directory = try await ModelUtils.resolveOrDownloadModel(repoID: repo, requiredExtension: "safetensors", cache: cache)
        return try await fromModelDirectory(directory, cache: cache)
    }

    public static func fromModelDirectory(_ directory: URL, cache: HubCache = .default,
                                           tokenizerRevision: String = "main") async throws -> LongCatModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(LongCatConfiguration.self, from: data)
        try validate(config)
        let tokenizerURL: URL
        if let local = [directory.appendingPathComponent("tokenizer/tokenizer.json"), directory.appendingPathComponent("tokenizer.json")]
            .first(where: { FileManager.default.fileExists(atPath: $0.path) }) { tokenizerURL = local }
        else {
            guard let repo = Repo.ID(rawValue: config.textEncoderModel) else { throw TTSModelError.invalidRepositoryID(config.textEncoderModel) }
            tokenizerURL = try await HubClient(cache: cache).downloadFile(at: "tokenizer.json", from: repo, revision: tokenizerRevision)
        }
        let tokenizer = try SentencePieceTokenizer.from(tokenizerJSONURL: tokenizerURL)
        guard tokenizer.vocabularySize <= config.textEncoderConfig.vocabSize,
              tokenizer.tokenID(for: "</s>") == 1 else {
            throw AudioGenerationError.invalidInput("LongCat requires a compatible UMT5 vocabulary with EOS token 1")
        }
        let model = LongCatModel(config, tokenizer: tokenizer)
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw AudioGenerationError.modelNotInitialized("LongCat safetensors weights are missing") }
        var weights: [String: MLXArray] = [:]
        for file in files { weights.merge(try MLX.loadArrays(url: file)) { _, new in new } }
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        if config.quantization != nil || base.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard !path.hasPrefix("vae."), weights[path + ".scales"] != nil else { return nil }
                return base.perLayerQuantization?.quantization(layer: path)?.asTuple ?? config.quantization?.asTuple
            }
        }
        try model.update(parameters: .unflattened(weights), verify: .all)
        model.train(false)
        eval(model)
        return model
    }

    static func validate(_ c: LongCatConfiguration) throws {
        let v = c.vaeConfig, t = c.textEncoderConfig
        guard c.ditDim > 0, c.ditHeads > 0, c.ditDim % c.ditHeads == 0, c.ditDim / c.ditHeads % 2 == 0,
              c.ditDepth > 0, c.ditFfMult.isFinite, c.ditFfMult > 0, c.ditEps.isFinite, c.ditEps > 0,
              c.latentDim > 0, c.latentHop > 0, c.samplingRate > 0, c.maxWavDuration.isFinite, c.maxWavDuration > 0,
              ["global", "local"].contains(c.ditAdalnType), t.dModel == c.ditTextDim,
              t.vocabSize > 1, t.dModel > 0, t.dFf > 0, t.numHeads > 0, t.dKv > 0, t.numLayers > 0,
              t.relativeAttentionNumBuckets >= 4, t.relativeAttentionNumBuckets % 4 == 0,
              t.relativeAttentionMaxDistance > t.relativeAttentionNumBuckets / 4,
              t.layerNormEpsilon.isFinite, t.layerNormEpsilon > 0,
              v.channels > 0, !v.strides.isEmpty, v.strides.count == v.cMults.count,
              v.strides.allSatisfy({ $0 > 0 && $0 % 2 == 0 }), v.cMults.allSatisfy({ $0 > 0 }),
              v.latentDim == c.latentDim, v.encoderLatentDim == c.latentDim * 2,
              v.strides.reduce(1, *) == v.downsamplingRatio, v.downsamplingRatio == c.latentHop,
              v.sampleRate == c.samplingRate, v.inChannels == 1, v.scale.isFinite, v.scale > 0 else {
            throw AudioGenerationError.invalidInput("Unsupported or inconsistent LongCat architecture configuration")
        }
        let channels = ([1] + v.cMults).map { $0 * v.channels }
        for i in v.strides.indices {
            let shortcutChannels = channels[i] * v.strides[i]
            guard (v.downsampleShortcut != "averaging" || shortcutChannels % channels[i + 1] == 0),
                  (v.upsampleShortcut != "duplicating" || shortcutChannels % channels[i + 1] == 0) else {
                throw AudioGenerationError.invalidInput("LongCat VAE shortcut channel ratios must be integral")
            }
        }
        guard (v.outShortcut != "averaging" || channels.last! % v.encoderLatentDim == 0),
              (v.inShortcut != "duplicating" || channels.last! % v.latentDim == 0) else {
            throw AudioGenerationError.invalidInput("LongCat VAE latent shortcut ratios must be integral")
        }
    }
}
