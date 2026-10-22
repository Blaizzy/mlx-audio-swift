import Foundation
import HuggingFace
import MLX
import MLXAudioCore
import MLXNN

public enum MegaASRRoutingMode: String, Sendable {
    case automatic = "auto"
    case base
    case robust
}

public struct MegaASRConfig: Decodable {
    public let asr: Qwen3ASRConfig
    public let routerWeights: String
    public let loraWeights: String
    public let routerHeads: Int
    enum CodingKeys: String, CodingKey {
        case routerWeights = "router_weights"
        case loraWeights = "lora_weights"
        case routerConfig = "router_config"
        case nhead
    }
    private struct RouterConfig: Decodable { let nhead: Int? }
    public init(from decoder: Decoder) throws {
        asr = try Qwen3ASRConfig(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        routerWeights =
            try c.decodeIfPresent(String.self, forKey: .routerWeights)
            ?? "extras/router.safetensors"
        loraWeights =
            try c.decodeIfPresent(String.self, forKey: .loraWeights) ?? "extras/lora.safetensors"
        routerHeads =
            try c.decodeIfPresent(RouterConfig.self, forKey: .routerConfig)?.nhead
            ?? c.decodeIfPresent(Int.self, forKey: .nhead) ?? 4
    }
}

/// Immutable original/adapted snapshots avoid accumulated BF16 add/subtract drift.
/// Array views hold independent MLX contexts because Module.update mutates its array handles.
final class MegaASRLoRAState {
    private let model: Module
    private let originals: [String: MLXArray]
    private let adapted: [String: MLXArray]
    private(set) var active = false

    init(model: Module, factors: [String: MLXArray]) throws {
        self.model = model
        var a: [String: MLXArray] = [:]
        var b: [String: MLXArray] = [:]
        for (key, value) in factors {
            if key.hasSuffix(".lora_A") {
                a[String(key.dropLast(7))] = value
            } else if key.hasSuffix(".lora_B") {
                b[String(key.dropLast(7))] = value
            } else {
                throw STTError.invalidInput("Unexpected prepared LoRA tensor: \(key)")
            }
        }
        guard !a.isEmpty, Set(a.keys) == Set(b.keys) else {
            throw STTError.invalidInput("LoRA requires matching nonempty A/B factors")
        }
        let leaves = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var original: [String: MLXArray] = [:]
        var robust: [String: MLXArray] = [:]
        for name in a.keys.sorted() {
            // Python exposes root.layers as an alias of model.layers. Published
            // sidecars may supply distinct contributions for both paths.
            let target = name.hasPrefix("layers.") ? "model." + name : name
            guard let leaf = leaves[target], leaf is Linear,
                let weight = parameters[target + ".weight"], weight.ndim == 2,
                let left = a[name], let right = b[name], left.ndim == 2, right.ndim == 2,
                left.dim(0) > 0, left.dim(0) == right.dim(1),
                left.dim(1) == weight.dim(1), right.dim(0) == weight.dim(0)
            else {
                throw STTError.invalidInput(
                    "LoRA target is missing, quantized, or has incompatible factors: \(name)")
            }
            let key = target + ".weight"
            let base = weight.reshaped(weight.shape)
            if original[key] == nil { original[key] = base }
            // Prepared factors include their scaling. Compose alias contributions
            // in sorted order, retaining Python's per-update rounding.
            robust[key] =
                (robust[key] ?? base)
                + matmul(right.asType(.float32), left.asType(.float32)).asType(base.dtype)
            eval(robust[key]!)
        }
        originals = original
        adapted = robust
    }

    func setActive(_ want: Bool) {
        guard want != active else { return }
        model.update(parameters: .unflattened(want ? adapted : originals))
        active = want
    }
}

/// Routed Qwen3-ASR with a native quality classifier and dense LoRA sidecars.
/// Inference requests, including streaming requests, are serialized per instance.
public final class MegaASRModel: STTGenerationModel {
    public let config: MegaASRConfig
    public let router: MegaASRAudioQualityRouter
    let backbone: Qwen3ASRModel
    private let adapters: MegaASRLoRAState
    private let inferenceGate = DispatchSemaphore(value: 1)
    public var sampleRate: Int { backbone.sampleRate }
    public var defaultGenerationParameters: STTGenerateParameters {
        backbone.defaultGenerationParameters
    }

    private init(
        config: MegaASRConfig, backbone: Qwen3ASRModel, router: MegaASRAudioQualityRouter,
        factors: [String: MLXArray]
    ) throws {
        self.config = config
        self.backbone = backbone
        self.router = router
        adapters = try MegaASRLoRAState(model: backbone, factors: factors)
    }

    private func prepareAudio(_ audio: MLXArray) -> MLXArray {
        precondition(audio.ndim == 1 || audio.ndim == 2, "Expected mono or time-major stereo PCM")
        return (audio.ndim == 2 ? audio.mean(axis: -1) : audio).asType(.float32)
    }

    private func selectRoute(audio: MLXArray, mode: MegaASRRoutingMode) {
        let useLoRA: Bool
        switch mode {
        case .automatic: useLoRA = router.route(audio: audio).usesLoRA
        case .base: useLoRA = false
        case .robust: useLoRA = true
        }
        adapters.setActive(useLoRA)
    }

    public func transcribe(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        context: String = "", routing: MegaASRRoutingMode = .automatic
    ) -> STTOutput {
        let p = generationParameters ?? defaultGenerationParameters
        let audio = prepareAudio(audio)
        inferenceGate.wait()
        defer { inferenceGate.signal() }
        selectRoute(audio: audio, mode: routing)
        return performTranscription(audio: audio, parameters: p, context: context, emit: nil)
    }

    public func generate(audio: MLXArray, generationParameters: STTGenerateParameters) -> STTOutput
    {
        transcribe(audio: audio, generationParameters: generationParameters)
    }

    private func acquireGateCancellable() throws {
        while inferenceGate.wait(timeout: .now() + .milliseconds(50)) != .success {
            try Task.checkCancellation()
        }
    }

    public func generateStream(audio: MLXArray, generationParameters: STTGenerateParameters)
        -> AsyncThrowingStream<STTGeneration, Error>
    {
        transcribeStream(audio: audio, generationParameters: generationParameters)
    }

    public func transcribeStream(
        audio: MLXArray, generationParameters: STTGenerateParameters? = nil,
        context: String = "", routing: MegaASRRoutingMode = .automatic
    ) -> AsyncThrowingStream<STTGeneration, Error> {
        let modelBox = UncheckedSendableBox(self)
        let audioBox = UncheckedSendableBox(audio)
        let p = generationParameters ?? defaultGenerationParameters
        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                let model = modelBox.value
                do {
                    let audio = model.prepareAudio(audioBox.value)
                    try model.acquireGateCancellable()
                    defer { model.inferenceGate.signal() }
                    try Task.checkCancellation()
                    model.selectRoute(audio: audio, mode: routing)
                    let result = model.performTranscription(
                        audio: audio, parameters: p, context: context
                    ) {
                        continuation.yield(.token($0))
                    }
                    try Task.checkCancellation()
                    continuation.yield(.result(result))
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    // Both APIs execute this loop under the same gate. No child decoder task can
    // outlive cancellation and race with a subsequent adapter switch.
    private func performTranscription(
        audio: MLXArray, parameters p: STTGenerateParameters,
        context: String, emit: ((String) -> Void)?
    ) -> STTOutput {
        precondition(p.kvBits == nil, "Mega-ASR does not currently support quantized KV caches")
        guard let tokenizer = backbone.tokenizer else {
            preconditionFailure("Mega-ASR tokenizer is not loaded")
        }
        let started = Date()
        let language = backbone.normalizeLanguageName(p.language)
        var texts: [String] = []
        var languages: [String?] = []
        var segments: [[String: Any]] = []
        var promptCount = 0
        var generatedCount = 0
        var generationSeconds: Double = 0
        let sampler: any LogitSampler =
            p.temperature == 0
            ? ArgMaxSampler()
            : TopPSampler(temperature: p.temperature, topP: p.topP, topK: p.topK)
        chunks: for (chunk, offset) in splitAudioIntoChunks(
            audio, sampleRate: sampleRate,
            chunkDuration: p.chunkDuration, minChunkDuration: p.minChunkDuration)
        {
            if Task.isCancelled { break }
            let budget = p.maxTokens - generatedCount
            if budget <= 0 { break }
            let (features, mask, audioTokens) = backbone.preprocessAudio(chunk)
            let ids = backbone.buildPrompt(
                numAudioTokens: audioTokens, context: context, language: language)
            let encoded = backbone.getAudioFeatures(features, featureAttentionMask: mask)
            eval(encoded)
            let tokenEmbeddings = backbone.model.embedTokens(ids)
            let embeddings = backbone.mergeAudioFeatures(
                inputsEmbeds: tokenEmbeddings,
                audioFeatures: encoded.asType(tokenEmbeddings.dtype), inputIds: ids)
            let cache = backbone.makeCache()
            promptCount += ids.dim(1)
            var consumed = 0
            while ids.dim(1) - consumed > 1 {
                if Task.isCancelled { break chunks }
                let n = min(2048, ids.dim(1) - consumed - 1)
                eval(
                    backbone.model(
                        inputsEmbeds: embeddings[0..., consumed..<(consumed + n), 0...],
                        cache: cache))
                consumed += n
            }
            let hidden = backbone.model(
                inputsEmbeds: embeddings[0..., consumed..., 0...], cache: cache)
            var logits = backbone.lmHead?(hidden) ?? backbone.model.embedTokens.asLinear(hidden)
            let generationStart = Date()
            var tokens: [Int] = []
            var previous = ""
            for _ in 0..<budget {
                if Task.isCancelled { break chunks }
                var scores = logits[0, -1, 0...]
                if p.repetitionPenalty != 1, !tokens.isEmpty {
                    let recent = MLXArray(
                        Array(Set(tokens.suffix(max(1, p.repetitionContextSize)))).map(Int32.init))
                    let values = scores[recent]
                    scores[recent] = MLX.where(
                        values .> 0, values / p.repetitionPenalty, values * p.repetitionPenalty)
                }
                let token = sampler.sample(logits: scores).item(Int.self)
                if token == 151645 || token == 151643 { break }
                tokens.append(token)
                if let emit {
                    let raw = tokenizer.decode(tokens: tokens)
                    let lower = raw.lowercased()
                    let pendingHeader =
                        language == nil && !raw.contains("<asr_text>")
                        && ("language".hasPrefix(lower) || lower.hasPrefix("language"))
                    let partial = backbone.parseGeneratedChunk(raw, forcedLanguage: language).text
                    if !pendingHeader, partial.hasPrefix(previous), !partial.hasSuffix("\u{FFFD}") {
                        if previous.isEmpty, !texts.isEmpty { emit(" ") }
                        emit(String(partial.dropFirst(previous.count)))
                        previous = partial
                    }
                }
                if p.repetitionPenalty == 1, tokens.count >= 24, Set(tokens.suffix(24)).count <= 3 {
                    break
                }
                if tokens.count < budget {
                    logits = backbone(
                        inputIds: MLXArray([Int32(token)]).reshaped(1, 1), cache: cache)
                }
            }
            generationSeconds += Date().timeIntervalSince(generationStart)
            let parsed = backbone.parseGeneratedChunk(
                tokenizer.decode(tokens: tokens), forcedLanguage: language)
            if let emit, parsed.text.hasPrefix(previous), parsed.text.count > previous.count {
                if previous.isEmpty, !texts.isEmpty { emit(" ") }
                emit(String(parsed.text.dropFirst(previous.count)))
            }
            texts.append(parsed.text)
            languages.append(parsed.language)
            generatedCount += tokens.count
            segments.append([
                "text": parsed.text, "start": Double(offset),
                "end": Double(offset) + Double(chunk.dim(0)) / Double(sampleRate),
            ])
        }
        let elapsed = Date().timeIntervalSince(started)
        return STTOutput(
            text: texts.joined(separator: " "), segments: segments,
            language: Qwen3ASRModel.mergeLanguages(languages), promptTokens: promptCount,
            generationTokens: generatedCount, totalTokens: promptCount + generatedCount,
            promptTps: Double(promptCount) / max(elapsed - generationSeconds, 1e-9),
            generationTps: Double(generatedCount) / max(generationSeconds, 1e-9),
            totalTime: elapsed,
            peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
    }

    private static func sidecar(_ relative: String, in directory: URL) throws -> URL {
        guard !relative.hasPrefix("/"), !relative.split(separator: "/").contains(".."),
            !relative.isEmpty
        else {
            throw STTError.invalidInput("Sidecar paths must stay inside the checkpoint directory")
        }
        let path = directory.appendingPathComponent(relative)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw STTError.modelNotInitialized("Missing Mega-ASR sidecar: \(relative)")
        }
        return path
    }

    public static func fromModelDirectory(_ directory: URL) async throws -> MegaASRModel {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(MegaASRConfig.self, from: data)
        guard config.asr.perLayerQuantization == nil else {
            throw STTError.invalidInput("Mega-ASR requires dense base weights")
        }
        let routerPath = try sidecar(config.routerWeights, in: directory)
        let loraPath = try sidecar(config.loraWeights, in: directory)
        let router = try MegaASRAudioQualityRouter.fromWeights(
            MLX.loadArrays(url: routerPath), heads: config.routerHeads)
        let factors = try MLX.loadArrays(url: loraPath)
        let backbone = try await Qwen3ASRModel.fromModelDirectory(directory)
        let model = try MegaASRModel(
            config: config, backbone: backbone, router: router, factors: factors)
        backbone.train(false)
        return model
    }

    public static func fromPretrained(_ modelPath: String, cache: HubCache = .default) async throws
        -> MegaASRModel
    {
        let local = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath)
        if FileManager.default.fileExists(atPath: local.appendingPathComponent("config.json").path)
        {
            return try await fromModelDirectory(local)
        }
        guard let repo = Repo.ID(rawValue: modelPath) else {
            throw STTModelError.invalidRepositoryID(modelPath)
        }
        let directory = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo, requiredExtension: "safetensors",
            additionalMatchingPatterns: ["extras/*.safetensors"],
            hfToken: ProcessInfo.processInfo.environment["HF_TOKEN"], cache: cache)
        return try await fromModelDirectory(directory)
    }
}
