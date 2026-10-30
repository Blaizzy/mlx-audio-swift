import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
import MLXAudioCore
import MLXNN
import Tokenizers

/// A reusable Bark continuation, using semantic tokens and Encodec codebooks.
public struct BarkHistory {
    public let semantic: MLXArray
    public let coarse: MLXArray
    public let fine: MLXArray

    public init(semantic: MLXArray, coarse: MLXArray, fine: MLXArray) throws {
        guard semantic.ndim == 1, semantic.size > 0, coarse.ndim == 2,
              coarse.dim(0) == 2, coarse.dim(1) > 0, fine.ndim == 2,
              fine.dim(0) == 8, fine.dim(1) > 0,
              semantic.min().item(Int.self) >= 0, semantic.max().item(Int.self) < 10000,
              coarse.min().item(Int.self) >= 0, coarse.max().item(Int.self) < 1024,
              fine.min().item(Int.self) >= 0, fine.max().item(Int.self) < 1024 else {
            throw AudioGenerationError.invalidInput("Invalid Bark history token shapes or ranges")
        }
        self.semantic = semantic.asType(.int32)
        self.coarse = coarse.asType(.int32)
        self.fine = fine.asType(.int32)
    }
}

/// Bark's semantic, coarse and noncausal fine transformers with the 24 kHz Encodec decoder.
public final class BarkModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: BarkConfig
    @ModuleInfo var semantic: BarkGPT
    @ModuleInfo(key: "coarse_acoustics") var coarse: BarkGPT
    @ModuleInfo(key: "fine_acoustics") var fine: BarkFineGPT
    private var tokenizer: (any Tokenizer)?
    private var codec: Encodec?
    private var folder: URL?
    private var hubCache: HubCache = .default
    public var options = BarkGenerationOptions()
    public var sampleRate: Int { config.sampleRate }
    public var defaultGenerationParameters: GenerateParameters { .init(maxTokens: 768, temperature: 0.7) }

    public init(_ config: BarkConfig) {
        self.config = config
        _semantic.wrappedValue = BarkGPT(config.semantic)
        _coarse.wrappedValue = BarkGPT(config.coarse)
        _fine.wrappedValue = BarkFineGPT(config.fine)
    }

    public static func fromPretrained(_ repo: String, cache: HubCache = .default) async throws -> BarkModel {
        guard let id = Repo.ID(rawValue: repo) else { throw TTSModelError.invalidRepositoryID(repo) }
        let folder = try await ModelUtils.resolveOrDownloadModel(repoID: id, requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.npy", "speaker_embeddings/**/*.npy"], cache: cache)
        return try await fromModelDirectory(folder, cache: cache)
    }

    public static func fromModelDirectory(_ folder: URL, cache: HubCache = .default) async throws -> BarkModel {
        let config = try JSONDecoder().decode(BarkConfig.self,
            from: Data(contentsOf: folder.appendingPathComponent("config.json")))
        guard config.sampleRate == 24000, config.fine.codebooks == 8,
              config.fine.givenCodebooks == 1,
              [config.semantic, config.coarse, config.fine].allSatisfy({
                  $0.blockSize == 1024 && $0.width > 0 && $0.heads > 0
                      && $0.width % $0.heads == 0 && $0.layers >= 0
              }), config.semantic.inputVocabSize > 129599, config.semantic.outputVocabSize > 10000,
              config.coarse.inputVocabSize > 12050, config.coarse.outputVocabSize >= 12048,
              config.fine.inputVocabSize > 1024, config.fine.outputVocabSize >= 1024 else {
            throw AudioGenerationError.invalidInput("Bark requires eight codebooks and 24 kHz audio")
        }
        let model = BarkModel(config)
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
            where file.pathExtension == "safetensors" {
            weights.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        weights = model.sanitize(weights)
        if let q = config.quantization {
            quantize(model: model) { path, _ in weights["\(path).scales"] == nil ? nil : q.asTuple }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        model.tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        let localCodec = folder.appendingPathComponent(config.codecPath)
        if FileManager.default.fileExists(atPath: localCodec.appendingPathComponent("config.json").path) {
            model.codec = try Encodec.fromModelDirectory(localCodec)
        } else {
            model.codec = try await Encodec.fromPretrained(config.codecPath, cache: cache)
        }
        guard model.codec?.samplingRate == 24000, model.codec?.channels == 1 else {
            throw AudioGenerationError.invalidInput("Bark's codec must run at 24 kHz")
        }
        model.folder = folder; model.hubCache = cache
        return model
    }

    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (name, value) in weights where !name.hasPrefix("codec.") {
            // Published causal masks are static Boolean buffers, rebuilt from the token positions.
            if name.hasSuffix(".attn.bias") { continue }
            let key = name.replacingOccurrences(of: "_orig_mod.transformer.", with: "")
                .replacingOccurrences(of: "_orig_mod.", with: "")
                .replacingOccurrences(of: ".h.", with: ".layers.")
            result[key] = value
        }
        return result
    }

    /// Load published individual NumPy presets, including nested `v2/` names.
    public func history(for voice: String) async throws -> BarkHistory {
        guard let folder else { throw AudioGenerationError.modelNotInitialized("Bark preset folder") }
        guard !voice.hasPrefix("/"), !voice.split(separator: "/").contains(".."), !voice.contains("\\") else {
            throw AudioGenerationError.invalidInput("Invalid Bark preset name")
        }
        let manifestURL = folder.appendingPathComponent("speaker_embeddings_path.json")
        let manifest = (try? JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL))) as? [String: Any]
        let entry = manifest?[voice] as? [String: String]
        var arrays: [String: MLXArray] = [:]
        for kind in ["semantic_prompt", "coarse_prompt", "fine_prompt"] {
            let path = entry?[kind] ?? "speaker_embeddings/\(voice)_\(kind).npy"
            guard !path.hasPrefix("/"), !path.split(separator: "/").contains(".."), !path.contains("\\") else {
                throw AudioGenerationError.invalidInput("Invalid Bark preset path")
            }
            var file = folder.appendingPathComponent(path)
            if !FileManager.default.fileExists(atPath: file.path) {
                // The published MLX manifest also names presets hosted by its upstream processor repo.
                guard entry != nil, let source = manifest?["repo_or_path"] as? String,
                      let repo = Repo.ID(rawValue: source) else {
                    throw AudioGenerationError.invalidInput("Bark voice '\(voice)' is unavailable")
                }
                file = try await HubClient(cache: hubCache).downloadFile(at: path, from: repo)
            }
            arrays[kind] = try MLX.loadArray(url: file)
        }
        return try BarkHistory(semantic: arrays["semantic_prompt"]!, coarse: arrays["coarse_prompt"]!,
            fine: arrays["fine_prompt"]!)
    }

    public func generate(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                         language: String?, generationParameters: GenerateParameters) async throws -> MaterializedArray {
        guard refAudio == nil else {
            throw AudioGenerationError.invalidInput("Bark accepts token histories or speaker presets for conditioning")
        }
        let history: BarkHistory?
        if let voice { history = try await self.history(for: voice) } else { history = nil }
        return try (generate(text: text, history: history, parameters: generationParameters)).materialized()
    }

    /// Generate with an explicit token history. Language is inferred from the multilingual text.
    public func generate(text: String, history: BarkHistory? = nil,
                         parameters: GenerateParameters? = nil) throws -> MaterializedArray {
        guard let tokenizer, let codec else {
            throw AudioGenerationError.modelNotInitialized("Bark tokenizer and Encodec")
        }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AudioGenerationError.invalidInput("Bark needs nonempty text")
        }
        guard (60...630).contains(options.maxCoarseHistory), options.coarseWindow > 0,
              options.coarseWindow <= 768 - options.maxCoarseHistory else {
            throw AudioGenerationError.invalidInput("Bark coarse history/window exceeds its context")
        }
        let p = parameters ?? defaultGenerationParameters
        let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
        let semanticTokens = try generateSemantic(tokens, history: history, parameters: p)
        guard !semanticTokens.isEmpty else {
            throw AudioGenerationError.generationFailed("Bark stopped before generating semantic tokens")
        }
        let coarseTokens = try generateCoarse(semanticTokens, history: history, parameters: p)
        let fineTokens = try generateFine(coarseTokens, history: history, parameters: p)
        try Task.checkCancellation()
        let codes = fineTokens.reshaped([1, 1, 8, fineTokens.dim(1)])
        let audio = codec.decode(codes, [nil]).reshaped([-1]).asType(.float32)
        eval(audio)
        try Task.checkCancellation()
        return audio.materialized()
    }

    func semanticInput(_ tokens: [Int], history: BarkHistory?) -> [Int32] {
        var text = tokens.prefix(256).map { Int32($0 + 10048) }
        text += [Int32](repeating: 129595, count: 256 - text.count)
        var past = history?.semantic.asArray(Int32.self).suffix(256).map { $0 } ?? []
        past += [Int32](repeating: 10000, count: 256 - past.count)
        return text + past + [129599]
    }

    func generateSemantic(_ tokens: [Int], history: BarkHistory?, parameters p: GenerateParameters) throws -> [Int32] {
        let cache = semantic.makeCache()
        var input = MLXArray(semanticInput(tokens, history: history)).reshaped([1, -1])
        let sampler = p.sampler()
        var output: [Int32] = []
        for _ in 0..<max(0, min(p.maxTokens ?? 768, config.semantic.blockSize - 256)) {
            try Task.checkCancellation()
            let logits = semantic(input, mergeContext: true, cache: cache)[0, 0]
            let relevant = options.allowEarlyStop
                ? logits[0..<10001] : logits[0..<10000]
            let token = sampler.sample(logits: relevant).item(Int32.self)
            if token == 10000 { break }
            output.append(token)
            input = MLXArray([token]).reshaped([1, 1])
        }
        return output
    }

    static func coarseSteps(semanticCount: Int) -> Int {
        Int(floor(Double(semanticCount) * 75 / 49.9)) * 2
    }

    func generateCoarse(_ semanticTokens: [Int32], history: BarkHistory?, parameters p: GenerateParameters) throws -> MLXArray {
        let ratio = 75.0 / 49.9 * 2
        let maxSemanticHistory = Int(floor(Double(options.maxCoarseHistory) / ratio))
        var pastSemantic: [Int32] = []; var pastCoarse: [Int32] = []
        if let history {
            let sem = history.semantic.asArray(Int32.self)
            let co = history.coarse.asArray(Int32.self)
            let frames = history.coarse.dim(1)
            let n = min(maxSemanticHistory, sem.count - sem.count % 2, Int(floor(Double(2 * frames) / ratio)))
            pastSemantic = Array(sem.suffix(n))
            let flat = (0..<frames).flatMap { [co[$0] + 10000, co[frames + $0] + 11024] }
            let count = Int((Double(n) * ratio).rounded(.toNearestOrEven))
            pastCoarse = Array(flat.suffix(count).dropLast(2))
        }
        let semanticAll = pastSemantic + semanticTokens
        var coarseAll = pastCoarse
        let steps = Self.coarseSteps(semanticCount: semanticTokens.count)
        guard steps > 0 else { throw AudioGenerationError.generationFailed("Bark generated too few semantic tokens") }
        let sampler = p.sampler()
        var step = 0
        while step < steps {
            let semanticIndex = pastSemantic.count + Int((Double(step) / ratio).rounded(.toNearestOrEven))
            var sem = Array(semanticAll.dropFirst(max(0, semanticIndex - maxSemanticHistory)).prefix(256))
            sem += [Int32](repeating: 12048, count: 256 - sem.count)
            let context = sem + [12050] + Array(coarseAll.suffix(options.maxCoarseHistory))
            var input = MLXArray(context).reshaped([1, -1])
            let cache = coarse.makeCache()
            for _ in 0..<min(options.coarseWindow, steps - step) {
                try Task.checkCancellation()
                let start = 10000 + (step % 2) * 1024
                let logits = coarse(input, cache: cache)[0, 0, start..<(start + 1024)]
                let token = sampler.sample(logits: logits).item(Int32.self) + Int32(start)
                coarseAll.append(token)
                input = MLXArray([token]).reshaped([1, 1])
                step += 1
            }
        }
        let out = Array(coarseAll.dropFirst(pastCoarse.count))
        let frames = out.count / 2
        let values: [Int32] = (0..<frames).map { out[$0 * 2] - 10000 }
            + (0..<frames).map { out[$0 * 2 + 1] - 11024 }
        return MLXArray(values).reshaped([2, frames])
    }

    func generateFine(_ coarseTokens: MLXArray, history: BarkHistory?, parameters p: GenerateParameters) throws -> MLXArray {
        let frames = coarseTokens.dim(1)
        let historyCount = options.useFineHistory ? min(history?.fine.dim(1) ?? 0, 512) : 0
        let length = max(1024, historyCount + frames)
        let data = MLXArray.full([length, 8], values: MLXArray(Int32(1024)))
        if let history, historyCount > 0 {
            data[0..<historyCount, 0...] = history.fine[0..., (history.fine.dim(1) - historyCount)...].T
        }
        data[historyCount..<(historyCount + frames), 0..<2] = coarseTokens.T
        let loops = max(0, Int(ceil(Double(frames - (1024 - historyCount)) / 512))) + 1
        var fineParameters = p
        fineParameters.temperature = options.fineTemperature ?? p.temperature
        let sampler = fineParameters.sampler()
        for loop in 0..<loops {
            let start = min(loop * 512, length - 1024)
            let fill = min(historyCount + loop * 512, length - 512)
            let relative = fill - start
            let buffer = data[start..<(start + 1024), 0...].expandedDimensions(axis: 0)
            for codebook in 2..<8 {
                try Task.checkCancellation()
                let logits = fine(buffer, codebook: codebook)[0, relative..., 0..<1024]
                let predictions = sampler.sample(logits: logits).asType(.int32).reshaped([-1])
                buffer[0, relative..., codebook] = predictions
                eval(buffer)
            }
            data[fill..<(start + 1024), 2...] = buffer[0, relative..., 2...]
            eval(data)
        }
        return data[historyCount..<(historyCount + frames), 0...].T
    }

    /// Bark emits its complete utterance after the fine stage and codec finish.
    public func generateStream(text: String, voice: String?, refAudio: MaterializedArray?, refText: String?,
                               language: String?, generationParameters: GenerateParameters)
        -> AsyncThrowingStream<AudioGeneration, Error> {
        let reference = refAudio
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else { continuation.finish(); return }
            do {
                let audio = try await self.generate(text: text, voice: voice, refAudio: reference,
                    refText: refText, language: language, generationParameters: generationParameters)
                continuation.yield(.audio(audio)); continuation.finish()
            } catch { continuation.finish(throwing: error) }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }
}
