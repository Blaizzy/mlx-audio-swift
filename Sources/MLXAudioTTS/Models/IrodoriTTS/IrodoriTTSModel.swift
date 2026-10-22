import CryptoKit
import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCodecs
@preconcurrency import MLXAudioCore
import MLXNN
import Tokenizers

/// Irodori-TTS — Japanese flow-matching TTS (Echo-TTS family). Rectified-Flow
/// DiT over Semantic-DACVAE-Japanese-32dim latents (48 kHz), with v3 automatic
/// duration prediction and VoiceDesign (caption) conditioning.
/// Port of mlx_audio/tts/models/irodori_tts/irodori_tts.py (the `Model` class).
public final class IrodoriTTSModel: Module, @unchecked Sendable {
    public let config: IrodoriTTSConfig
    public let sampleRate: Int
    public let defaultGenerationParameters: GenerateParameters

    @ModuleInfo(key: "model") var model: IrodoriDiT

    var dacvae: DACVAE?
    var tokenizer: Tokenizers.Tokenizer?
    var captionTokenizer: Tokenizers.Tokenizer?

    /// Wall-clock phase timings of the most recent generation.
    public private(set) var lastTimings: IrodoriTTSTimings?

    /// Default VoiceDesign caption used when no `voice` is supplied.
    static let defaultCaption = "落ち着いた自然な声で、はっきりと読み上げてください。"

    init(config: IrodoriTTSConfig) throws {
        self.config = config
        self.sampleRate = config.sampleRate
        self.defaultGenerationParameters = GenerateParameters(
            maxTokens: config.sampler.sequenceLength, temperature: 0, topP: 1)
        self._model = ModuleInfo(wrappedValue: try IrodoriDiT(cfg: config.dit))
    }

    // MARK: - Weight sanitisation (mirror irodori_tts.py `sanitize`)

    /// Remap PyTorch/MLX-community checkpoint keys (snake_case) to the Swift module
    /// tree's keys. The DiT uses camelCase @ModuleInfo properties whose `key:` is
    /// dropped by the `self._x = ModuleInfo(wrappedValue:)` init pattern, so each
    /// path component must be camelCased to match (the same approach as EchoTTSModel).
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // snake_case → camelCase, with the adaLN containers matching their property names.
        func normalizedComponent(_ component: String) -> String {
            switch component {
            case "attention_adaln": return "attentionAdaLN"
            case "mlp_adaln": return "mlpAdaLN"
            default:
                guard component.contains("_") else { return component }
                let parts = component.split(separator: "_")
                guard let head = parts.first else { return component }
                return String(head) + parts.dropFirst().map {
                    $0.prefix(1).uppercased() + $0.dropFirst()
                }.joined()
            }
        }

        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        for (rawKey, v) in weights {
            var bareKey = rawKey.hasPrefix("model.") ? String(rawKey.dropFirst("model.".count)) : rawKey
            // PyTorch Sequential integer keys → MLX nn.Sequential "layers.N"
            if bareKey.hasPrefix("cond_module.") || bareKey.hasPrefix("delta_cond_module.") {
                var parts = bareKey.split(separator: ".").map(String.init)
                if parts.count > 1, Int(parts[1]) != nil {
                    parts.insert("layers", at: 1)
                    bareKey = parts.joined(separator: ".")
                }
            }
            // camelCase every non-numeric path component
            let normalized = bareKey.split(separator: ".").map { part -> String in
                let c = String(part)
                return Int(c) == nil ? normalizedComponent(c) : c
            }.joined(separator: ".")
            out["model.\(normalized)"] = v
        }
        return out
    }

    // MARK: - Text / caption preparation

    /// v4 checkpoints use the upstream normaliser; v1–v3 keep the original Swift
    /// port's normaliser so their output stays unchanged.
    func normalizeText(_ text: String) -> String {
        config.dit.usePretrainedTextEncoder
            ? irodoriNormalizeTextUpstream(text).irodoriPythonStripped()
            : irodoriNormalizeText(text)
    }

    func prepareText(_ text: String) throws -> (ids: MLXArray, mask: MLXArray) {
        guard let tokenizer else { throw IrodoriTTSError.tokenizer("Text tokenizer not loaded") }
        let encoded = try irodoriEncodeText(
            normalizeText(text), tokenizer: tokenizer,
            maxLength: config.maxTextLength, addBos: config.dit.textAddBos)
        return trimForPretrainedEncoder(encoded)
    }

    func prepareCaption(_ caption: String) throws -> (ids: MLXArray, mask: MLXArray) {
        let tok = captionTokenizer ?? tokenizer
        guard let tok else { throw IrodoriTTSError.tokenizer("Caption tokenizer not loaded") }
        var encoded = try irodoriEncodeText(
            caption, tokenizer: tok,
            maxLength: config.maxCaptionLength, addBos: config.dit.captionAddBosResolved)
        if caption.isEmpty {
            // An empty caption still tokenizes to a BOS token. Leaving it unmasked would
            // present it as a real caption to the DiT and the duration predictor.
            encoded.mask = MLXArray.zeros(like: encoded.mask)
        }
        return trimForPretrainedEncoder(encoded)
    }

    /// v4: drop the right padding. Every consumer masks padded keys out (attention,
    /// duration token sum, caption mean), so the result is unchanged while the
    /// ModernBERT backbone and the DiT's context attention run on the real tokens only.
    private func trimForPretrainedEncoder(_ encoded: (ids: MLXArray, mask: MLXArray)) -> (ids: MLXArray, mask: MLXArray) {
        guard config.dit.usePretrainedTextEncoder else { return encoded }
        // An all-masked (empty) caption keeps its BOS slot so the sequence is non-empty.
        let length = max(1, encoded.mask.asType(.int32).sum().item(Int.self))
        return (encoded.ids[0..., 0 ..< length], encoded.mask[0..., 0 ..< length])
    }

    // MARK: - Reference-audio encoding (optional voice clone)

    func encodeRefAudio(_ audioIn: MLXArray) throws -> (latent: MLXArray, mask: MLXArray) {
        guard let dacvae else { throw IrodoriTTSError.generation("DACVAE not loaded") }
        var audio = audioIn
        if audio.ndim == 1 { audio = audio.expandedDimensions(axis: 0) }
        else if audio.ndim == 2 && audio.shape[0] > 1 { audio = mean(audio, axis: 0, keepDims: true) }

        var maxSteps = config.maxSpeakerLatentLength
        if let refMaxSeconds = config.refMaxSeconds, refMaxSeconds > 0 {
            let steps = Int(refMaxSeconds * Float(config.sampleRate) / Float(config.audioDownsampleFactor))
            maxSteps = min(maxSteps, max(1, steps))
        }
        let maxSamples = maxSteps * config.audioDownsampleFactor
        if audio.shape[1] > maxSamples { audio = audio[0..., 0..<maxSamples] }

        // DACVAE.encode expects (B, L, 1) → returns (B, 128, T) channels-first
        let latentCF = dacvae.encode(audio.expandedDimensions(axis: 2))
        var latent = latentCF.transposed(0, 2, 1)  // (B, T, C)

        var actualT = audio.shape[1] / config.audioDownsampleFactor
        actualT = min(actualT, latent.shape[1])
        latent = latent[0..., 0..<actualT]
        var maskLen = actualT

        let p = config.dit.speakerPatchSize
        if p > 1 && actualT % p != 0 {
            let trim = (actualT / p) * p
            latent = latent[0..., 0..<trim]
            maskLen = trim
        }
        let mask = MLXArray.ones([1, maskLen], dtype: .bool)
        return (latent, mask)
    }

    /// Placeholder reference latent used when no reference audio is given (masked out).
    /// It spans one full speaker patch, otherwise patching yields an empty sequence
    /// (v4 uses speaker_patch_size = 4).
    func zeroReferenceLatent() -> MLXArray {
        MLXArray.zeros([1, config.dit.speakerPatchSize, config.dit.latentDim])
    }

    // MARK: - Duration → latent step count

    private func computeLatentSteps(
        text: String,
        textMask: MLXArray,
        refMask: MLXArray?,
        captionMask: MLXArray?,
        encoded enc: IrodoriEncodedConditions,
        secondsOverride: Float?
    ) throws -> Int {
        let dsr = Float(config.sampleRate) / Float(config.audioDownsampleFactor)

        if let secs = secondsOverride {
            let clamped = min(config.sampler.maxSeconds, max(config.sampler.minSeconds, secs))
            let targetSamples = Int(clamped * Float(config.sampleRate))
            return Int(ceil(Double(targetSamples) / Double(config.audioDownsampleFactor)))
        }

        guard config.dit.useDurationPredictor else {
            return config.sampler.sequenceLength  // fallback
        }

        let normalized = normalizeText(text)
        let tokenCount = textMask.sum().item(Int.self)
        let hasSpeaker = refMask.map { $0.any().item(Bool.self) } ?? false

        let features = irodoriBuildDurationFeatures(
            texts: [normalized], tokenCounts: [tokenCount],
            maxTextLen: config.maxTextLength, hasSpeaker: [hasSpeaker])

        let hasCaption = (captionMask?.any().item(Bool.self)) ?? false
        let predLog = try model.predictDurationLogFrames(
            textState: enc.textState, textMask: enc.textMask,
            speakerState: enc.speakerState,
            durationFeatures: features,
            hasSpeaker: MLXArray([hasSpeaker]),
            captionState: enc.captionState, captionMask: enc.captionMask,
            hasCaption: MLXArray([hasCaption]))

        let predFrames = expm1(predLog[0]).item(Float.self)
        let scaled = predFrames * config.sampler.durationScale
        let minFrames = max(1, Int(ceil(Double(config.sampler.minSeconds * dsr))))
        let maxFrames = max(1, Int(floor(Double(config.sampler.maxSeconds * dsr))))
        return max(minFrames, min(maxFrames, Int((scaled).rounded())))
    }

    // MARK: - Generate (latent → waveform)

    func generateWaveform(
        text: String,
        caption: String?,
        refAudio: sending MLXArray?,
        rngSeed: Int,
        secondsOverride: Float?
    ) throws -> MLXArray {
        guard let dacvae else {
            throw AudioGenerationError.modelNotInitialized("Irodori-TTS requires DACVAE to be loaded")
        }

        var timings = IrodoriTTSTimings()
        var clock = CFAbsoluteTimeGetCurrent()
        func lap() -> Double {
            let now = CFAbsoluteTimeGetCurrent()
            defer { clock = now }
            return now - clock
        }

        let (textIDs, textMask) = try prepareText(text)

        var captionIDs: MLXArray?
        var captionMask: MLXArray?
        if config.dit.useCaptionCondition {
            // Upstream strips the caption but does not normalise it.
            let cap = (caption ?? Self.defaultCaption).trimmingCharacters(in: .whitespacesAndNewlines)
            let c = try prepareCaption(cap)
            captionIDs = c.ids
            captionMask = c.mask
        }

        var refLatent: MLXArray?
        var refMask: MLXArray?
        if let refAudio {
            let r = try encodeRefAudio(refAudio)
            refLatent = r.latent
            refMask = r.mask
        }
        // Zero speaker context when needed (matches Python defaults)
        if config.dit.useSpeakerConditionResolved || !config.dit.useCaptionCondition {
            if refLatent == nil { refLatent = zeroReferenceLatent() }
            if refMask == nil { refMask = MLXArray.zeros([1, refLatent!.shape[1]], dtype: .bool) }
        }

        // Encode once: duration prediction and the sampler share these states.
        let encoded = model.encodeConditionsFull(
            textInputIDs: textIDs, textMask: textMask,
            refLatent: refLatent, refMask: refMask,
            captionInputIDs: captionIDs, captionMask: captionMask)
        eval([encoded.textState, encoded.speakerState, encoded.captionState].compactMap { $0 })
        timings.encode = lap()

        let latentSteps = try computeLatentSteps(
            text: text, textMask: textMask,
            refMask: refMask, captionMask: captionMask,
            encoded: encoded,
            secondsOverride: secondsOverride)
        timings.duration = lap()

        let patchedSteps = Int(ceil(Double(latentSteps) / Double(config.dit.latentPatchSize)))

        var params = IrodoriSamplerParams()
        let s = config.sampler
        params.numSteps = s.numSteps
        params.cfgScaleText = s.cfgScaleText
        params.cfgScaleSpeaker = s.cfgScaleSpeaker
        params.cfgScaleCaption = s.cfgScaleCaption
        params.cfgGuidanceMode = s.cfgGuidanceMode
        params.cfgMinT = s.cfgMinT
        params.cfgMaxT = s.cfgMaxT
        params.truncationFactor = s.truncationFactor
        params.rescaleK = s.rescaleK
        params.rescaleSigma = s.rescaleSigma
        params.contextKvCache = s.contextKvCache
        params.speakerKvScale = s.speakerKvScale
        params.speakerKvMinT = s.speakerKvMinT
        params.speakerKvMaxLayers = s.speakerKvMaxLayers
        params.tScheduleMode = s.tScheduleMode
        params.swayCoeff = s.swayCoeff

        let latentOut: MLXArray
        if config.dit.useMeanflow {
            // MeanFlow bakes CFG into distillation: RF-only knobs above do not apply.
            latentOut = try irodoriSampleEulerMeanFlow(
                model: model, encoded: encoded,
                latentDim: config.dit.patchedLatentDim,
                sequenceLength: patchedSteps,
                rngSeed: rngSeed, numSteps: s.numSteps)
        } else {
            latentOut = try irodoriSampleEulerCFG(
                model: model,
                textInputIDs: textIDs, textMask: textMask,
                refLatent: refLatent, refMask: refMask,
                captionInputIDs: captionIDs, captionMask: captionMask,
                latentDim: config.dit.patchedLatentDim,
                sequenceLength: patchedSteps,
                rngSeed: rngSeed, params: params,
                encoded: encoded)
        }
        timings.sampler = lap()

        // Decode latent → waveform. (1, T, latentDim) → (1, latentDim, T)
        let latentForDecode = latentOut.transposed(0, 2, 1)
        // Chunked decode keeps ConvTranspose intermediates small on-device.
        var audioOut = dacvae.decode(latentForDecode, chunkSize: 50)  // (1, L, 1)
        audioOut = audioOut[0..., 0..., 0]  // (1, L)
        eval(audioOut)

        // Trim trailing silence, clamped to the predicted/manual duration.
        let silenceT = irodoriFindSilencePoint(latentOut[0])
        var trimSamples = silenceT * config.audioDownsampleFactor
        let targetSamples = latentSteps * config.audioDownsampleFactor
        trimSamples = min(trimSamples, targetSamples)
        trimSamples = min(trimSamples, audioOut.shape[1])
        if trimSamples > 0 { audioOut = audioOut[0..., 0..<trimSamples] }

        let waveform = audioOut[0]
        eval(waveform)
        timings.decode = lap()
        timings.latentFrames = latentSteps
        lastTimings = timings
        return waveform
    }

    // MARK: - Loading

    public static func fromPretrained(
        _ modelRepo: String,
        cache: HubCache = .default
    ) async throws -> IrodoriTTSModel {
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }
        // Pull the model + the dacvae/ subdirectory.
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: ".safetensors",
            additionalMatchingPatterns: ["dacvae/*", "tokenizer/*"],
            cache: cache)
        return try await fromModelDirectory(modelDir, cache: cache)
    }

    public static func fromModelDirectory(
        _ modelDir: URL,
        cache: HubCache = .default
    ) async throws -> IrodoriTTSModel {
        let configURL = modelDir.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(IrodoriTTSConfig.self, from: configData)
        let model = try IrodoriTTSModel(config: config)

        // Load DiT weights (top-level safetensors only — exclude dacvae/)
        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        let weightFiles = files
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !weightFiles.isEmpty else {
            throw AudioGenerationError.modelNotInitialized("No model safetensors in \(modelDir.path)")
        }
        var weights = [String: MLXArray]()
        for file in weightFiles {
            for (k, v) in try loadArrays(url: file) { weights[k] = v }
        }
        let sanitized = model.sanitize(weights: weights)

        // Quantize matching Linear layers before loading packed weights (8-bit checkpoints).
        if let q = decodeQuantization(configURL: configURL) {
            quantize(model: model) { path, _ in
                sanitized["\(path).scales"] != nil ? (q.groupSize, q.bits, .affine) : nil
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys)
        eval(model.parameters())

        // DACVAE codec from the local dacvae/ subdir (fp weights — not quantized).
        let dacvaeDir = modelDir.appendingPathComponent("dacvae")
        if FileManager.default.fileExists(atPath: dacvaeDir.appendingPathComponent("config.json").path) {
            model.dacvae = try DACVAE.fromModelDirectory(dacvaeDir)
        } else {
            model.dacvae = try await DACVAE.fromPretrained(config.dacvaeRepo, cache: cache)
        }

        // Prefer a tokenizer bundled with the converted weights (v4 ships one) so
        // inference does not depend on the upstream repo staying available.
        let tokRepo = config.dit.textTokenizerRepo
        let bundledTokenizer = modelDir.appendingPathComponent("tokenizer")
        if FileManager.default.fileExists(atPath: bundledTokenizer.appendingPathComponent("tokenizer_config.json").path) {
            let tokDir = try irodoriSanitizedTokenizerCopy(of: bundledTokenizer)
            model.tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)
            if config.dit.captionTokenizerRepoResolved == tokRepo {
                model.captionTokenizer = model.tokenizer
            }
        }
        if model.tokenizer == nil, let tokRepoID = Repo.ID(rawValue: tokRepo) {
            let tokDir = try await ModelUtils.resolveOrDownloadModel(
                repoID: tokRepoID, requiredExtension: ".json", cache: cache)
            // swift-transformers' tokenizer loader crashes on llm-jp's config; strip the
            // pieces we don't need (we add BOS manually, like the Python's
            // add_special_tokens=False) and cap the giant model_max_length sentinel.
            irodoriSanitizeTokenizerFiles(in: tokDir)
            model.tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)
            let capRepo = config.dit.captionTokenizerRepoResolved
            if capRepo == tokRepo {
                model.captionTokenizer = model.tokenizer
            } else if let capRepoID = Repo.ID(rawValue: capRepo) {
                let capDir = try await ModelUtils.resolveOrDownloadModel(
                    repoID: capRepoID, requiredExtension: ".json", cache: cache)
                model.captionTokenizer = try await AutoTokenizer.from(modelFolder: capDir)
            }
        }
        guard model.tokenizer != nil else {
            throw IrodoriTTSError.tokenizer("Could not load tokenizer from \(tokRepo)")
        }

        return model
    }
}

// MARK: - SpeechGenerationModel conformance

extension IrodoriTTSModel: SpeechGenerationModel {
    public func generate(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> sending MLXArray {
        _ = refText
        _ = language
        // `voice` carries the VoiceDesign caption.
        return try generateWaveform(
            text: text, caption: voice, refAudio: refAudio,
            rngSeed: 0, secondsOverride: nil)
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let refAudio = SendingBox(refAudio)
        let task = Task { @Sendable [weak self] in
            let refAudio = refAudio.take()
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let started = CFAbsoluteTimeGetCurrent()
                let waveform = try self.generateWaveform(
                    text: text, caption: voice, refAudio: refAudio, rngSeed: 0, secondsOverride: nil)
                let elapsed = max(CFAbsoluteTimeGetCurrent() - started, 1e-6)
                let info = AudioGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: waveform.shape.last ?? 0,
                    prefillTime: 0, generateTime: elapsed,
                    tokensPerSecond: 0,
                    peakMemoryUsage: Double(Memory.peakMemory) / 1e9)
                continuation.yield(.info(info))
                continuation.yield(.audio(waveform))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }
}

// MARK: - Helpers

/// Detect trailing silence in a generated latent (T, D). Mirrors `_find_silence_point`.
func irodoriFindSilencePoint(_ latent: MLXArray, windowSize: Int = 20, stdThreshold: Float = 0.05) -> Int {
    let t = latent.shape[0]
    let d = latent.shape[1]
    // Pull to host once; windows are small and T is bounded (~300).
    let flat = latent.asArray(Float.self)  // row-major (T, D)
    func windowStats(_ start: Int) -> (std: Float, mean: Float) {
        var sum: Float = 0, sumSq: Float = 0
        let count = windowSize * d
        for r in 0..<windowSize {
            let row = start + r
            for c in 0..<d {
                let v = (row < t) ? flat[row * d + c] : 0  // zero-padding past the end
                sum += v
                sumSq += v * v
            }
        }
        let m = sum / Float(count)
        let variance = max(0, sumSq / Float(count) - m * m)
        return (sqrt(variance), m)
    }
    for i in 0..<t {
        let (std, m) = windowStats(i)
        if std < stdThreshold && abs(m) < 0.1 { return i }
    }
    return t
}

/// Make the llm-jp (SentencePiece **Unigram**) tokenizer loadable by
/// swift-transformers 1.1.x. That version resolves the model class from
/// `tokenizer_class` alone (ignoring `model.type`): `PreTrainedTokenizerFast` maps
/// to BPETokenizer, which then fatal-errors ("requires merges") on a Unigram model.
/// We rewrite `tokenizer_class` to `XLMRobertaTokenizer` — which its registry maps
/// to `UnigramTokenizer` — and also cap the giant `model_max_length` sentinel and
/// strip the unused `post_processor`/`decoder` (we add BOS manually, encode-only,
/// matching the Python's `add_special_tokens=False`). Best-effort; no-op on failure.
func irodoriSanitizeTokenizerFiles(in dir: URL) {
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        let url = dir.appendingPathComponent(name)
        guard let data = try? Data(contentsOf: url),
              let sanitized = try? irodoriSanitizedTokenizerData(data, name: name)
        else { continue }
        try? sanitized.write(to: url, options: .atomic)
    }
}

private func irodoriSanitizedTokenizerData(_ data: Data, name: String) throws -> Data {
    guard var json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw IrodoriTTSError.tokenizer("Expected a JSON object in \(name)")
    }
    if name == "tokenizer.json" {
        json["post_processor"] = NSNull()
        json["decoder"] = NSNull()
    } else {
        // Force swift-transformers to instantiate UnigramTokenizer (not BPE).
        json["tokenizer_class"] = "XLMRobertaTokenizer"
        if let mml = json["model_max_length"] as? NSNumber, mml.doubleValue > 1_000_000 {
            json["model_max_length"] = 4096
        }
    }
    return try JSONSerialization.data(withJSONObject: json)
}

/// Write sanitized copies of a bundled tokenizer's JSON files into the user
/// cache, leaving the downloaded files (often symlinks into a shared HF blob
/// store) untouched.
/// Each content version is immutable and published only after both files are
/// sanitized, so concurrent callers (including other processes) cannot see a
/// partially prepared tokenizer or modify one another's copy.
func irodoriSanitizedTokenizerCopy(of source: URL, cacheDirectory: URL? = nil) throws -> URL {
    let fm = FileManager.default
    let names = ["tokenizer.json", "tokenizer_config.json"]
    let files = try names.map { name in
        (name: name, data: try Data(contentsOf: source.appendingPathComponent(name)))
    }
    var hash = SHA256()
    for file in files {
        hash.update(data: Data(SHA256.hash(data: file.data)))
    }
    let key = hash.finalize().map { String(format: "%02x", $0) }.joined()
    let caches = try cacheDirectory ?? fm.url(
        for: .cachesDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    let root = caches.appendingPathComponent("mlx-audio/irodori-tokenizer", isDirectory: true)
    // Version the sanitization scheme as well as the source contents.
    let dest = root.appendingPathComponent("v2-\(key)", isDirectory: true)
    if fm.fileExists(atPath: dest.path) { return dest }

    try fm.createDirectory(at: root, withIntermediateDirectories: true)
    let staging = root.appendingPathComponent(".\(UUID().uuidString)", isDirectory: true)
    try fm.createDirectory(at: staging, withIntermediateDirectories: false)
    defer { try? fm.removeItem(at: staging) }
    for file in files {
        let data = try irodoriSanitizedTokenizerData(file.data, name: file.name)
        try data.write(to: staging.appendingPathComponent(file.name), options: .atomic)
    }
    do {
        // A rename within the same directory publishes both files atomically.
        try fm.moveItem(at: staging, to: dest)
    } catch {
        // Another caller may have published this exact content while we prepared it.
        guard fm.fileExists(atPath: dest.path) else { throw error }
    }
    return dest
}

/// Wall-clock phase timings of one Irodori-TTS generation, in seconds.
public struct IrodoriTTSTimings: Sendable {
    /// Text + caption (+ reference) condition encoding, run once per utterance.
    public var encode: Double = 0
    public var duration: Double = 0
    public var sampler: Double = 0
    /// DACVAE decode plus silence trimming.
    public var decode: Double = 0
    /// Latent frames requested from the sampler (predicted or manual duration).
    public var latentFrames: Int = 0
}

/// Minimal quantization spec read straight from config.json.
private struct IrodoriQuantSpec { let groupSize: Int; let bits: Int }

private func decodeQuantization(configURL: URL) -> IrodoriQuantSpec? {
    guard let data = try? Data(contentsOf: configURL),
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
    let q = (obj["quantization"] as? [String: Any]) ?? (obj["quantization_config"] as? [String: Any])
    guard let q, let g = q["group_size"] as? Int, let b = q["bits"] as? Int else { return nil }
    return IrodoriQuantSpec(groupSize: g, bits: b)
}
