//
//  ChatterboxModel.swift
//  MLXAudio
//
//  Top-level Chatterbox TTS model.
//  Two-stage pipeline: T3 (text→speech tokens) + S3Gen (speech tokens→audio).
//  Supports both Regular (LLaMA backbone) and Turbo (GPT-2 backbone) variants.
//  Ported from mlx-audio Python: chatterbox/chatterbox.py
//

import AVFoundation
import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXNN
@preconcurrency import MLXLMCommon
import Tokenizers

// MARK: - Default Voice Conditioning

/// Pre-computed voice conditioning loaded from conds.safetensors (Turbo default voice).
public struct DefaultConditioning {
    /// T3 conditioning
    public var speakerEmb: MLXArray           // (1, 256)
    public var condPromptSpeechTokens: MLXArray // (1, T)
    public var emotionAdv: MLXArray           // (1, 1, 1)

    /// S3Gen conditioning
    public var xVector: MLXArray              // (1, 192)
    public var promptToken: MLXArray          // (1, T)
    public var promptTokenLen: MLXArray       // (1,)
    public var promptFeat: MLXArray           // (1, T, 80)
    public var promptFeatLen: MLXArray        // not stored — derived from promptFeat
}

// MARK: - Chatterbox Model

/// Chatterbox TTS: two-stage speech synthesis.
///
/// Stage 1 (T3): LLaMA or GPT-2 backbone converts text tokens → speech tokens,
/// conditioned on speaker embedding + optional prompt + emotion scalar.
///
/// Stage 2 (S3Gen): Flow matching decoder (Euler ODE) + HiFi-GAN vocoder converts
/// speech tokens → mel spectrogram → waveform at 24kHz.
///
/// Two variants:
/// - **Regular** (`Chatterbox-TTS-fp16`): LLaMA 520M, 500M params, 23 languages, emotion control
/// - **Turbo** (`chatterbox-turbo-fp16`): GPT-2 Medium, 350M params, English only, faster
public final class ChatterboxModel: Module, SpeechGenerationModel, @unchecked Sendable {

    // MARK: - Configuration

    public let config: ChatterboxConfiguration

    // MARK: - Sub-models

    /// Voice encoder: extracts 256-dim speaker embedding from reference audio.
    @ModuleInfo(key: "ve") var voiceEncoder: VoiceEncoder

    /// T3: text-to-speech-token model.
    /// Either T3Model (LLaMA) or T3GPT2Model (GPT-2), stored as Module for @ModuleInfo compatibility.
    @ModuleInfo(key: "t3") var t3: Module

    /// S3Gen: speech-token-to-audio model (Conformer + flow matching + HiFi-GAN).
    @ModuleInfo(key: "s3gen") var s3gen: CausalMaskedDiffWithXvec

    // MARK: - State

    /// Text tokenizer loaded from tokenizer.json.
    public var tokenizer: Tokenizer?

    /// Pre-computed default voice conditioning (from conds.safetensors).
    public var defaultConditioning: DefaultConditioning?

    /// Model directory (for loading auxiliary files).
    private var modelDir: URL?

    // MARK: - Protocol conformance

    public var sampleRate: Int { config.s3genSr }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: 0.8)
    }

    // MARK: - Special tokens

    private var sotToken: Int { config.t3Config.startTextToken }
    private var eotToken: Int { config.t3Config.stopTextToken }
    private var sosToken: Int { config.t3Config.startSpeechToken }
    private var eosToken: Int { config.t3Config.stopSpeechToken }
    private var speechVocabSize: Int { config.t3Config.speechTokensDictSize }

    // MARK: - Initialization

    public init(_ config: ChatterboxConfiguration = .default) {
        self.config = config

        self._voiceEncoder.wrappedValue = VoiceEncoder()

        // Create the appropriate T3 model based on config
        if config.isTurbo {
            let gpt2Config = config.gpt2Config ?? .medium
            self._t3.wrappedValue = T3GPT2Model(config.t3Config, gpt2Config: gpt2Config)
        } else {
            self._t3.wrappedValue = T3Model(config.t3Config)
        }

        self._s3gen.wrappedValue = CausalMaskedDiffWithXvec(
            decoderInChannels: config.decoderInChannels,
            meanflow: config.meanflow
        )
    }

    // MARK: - Weight Sanitization

    /// Route weights by prefix to the correct sub-model.
    ///
    /// Handles differences between Regular and Turbo weight key formats:
    /// - Regular S3Gen: `s3gen.flow.{decoder,encoder,...}` → strip `flow.` prefix
    /// - Both models: `s3gen.speaker_encoder.*` → dropped (loaded separately or unused)
    /// - Both models: `s3gen.tokenizer.*` → dropped (Turbo bakes it in, we don't use it)
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var veWeights = [String: MLXArray]()
        var t3Weights = [String: MLXArray]()
        var s3genWeights = [String: MLXArray]()
        var result = [String: MLXArray]()

        for (key, value) in weights {
            if key.hasPrefix("ve.") {
                let subKey = String(key.dropFirst("ve.".count))
                veWeights[subKey] = value
            } else if key.hasPrefix("t3.") {
                let subKey = String(key.dropFirst("t3.".count))
                t3Weights[subKey] = value
            } else if key.hasPrefix("s3gen.") {
                var subKey = String(key.dropFirst("s3gen.".count))

                // Drop speaker_encoder weights — not loaded into module hierarchy
                if subKey.hasPrefix("speaker_encoder.") { continue }
                // Drop tokenizer weights — baked-in S3 tokenizer not used in Swift
                if subKey.hasPrefix("tokenizer.") { continue }

                // Regular model: strip `flow.` prefix so keys match module structure
                // e.g. s3gen.flow.decoder.* → s3gen.decoder.*
                if subKey.hasPrefix("flow.") {
                    subKey = String(subKey.dropFirst("flow.".count))
                }

                s3genWeights[subKey] = value
            }
            // Drop campplus.* (neither model has top-level campplus weights)
            // Drop any other unknown prefixes
        }

        // Sub-model sanitization
        let sanitizedVE = voiceEncoder.sanitize(weights: veWeights)
        let sanitizedT3: [String: MLXArray]
        if let t3llama = t3 as? T3Model {
            sanitizedT3 = t3llama.sanitize(weights: t3Weights)
        } else if let t3gpt2 = t3 as? T3GPT2Model {
            sanitizedT3 = t3gpt2.sanitize(weights: t3Weights)
        } else {
            sanitizedT3 = t3Weights
        }

        // Reconstruct with prefixes
        for (key, value) in sanitizedVE {
            result["ve.\(key)"] = value
        }
        for (key, value) in sanitizedT3 {
            result["t3.\(key)"] = value
        }
        for (key, value) in s3genWeights {
            result["s3gen.\(key)"] = value
        }

        return result
    }

    // MARK: - Text Tokenization

    /// Tokenize text into token IDs for T3.
    func tokenizeText(_ text: String) throws -> MLXArray {
        guard let tokenizer = tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let encoded = tokenizer.encode(text: text)

        if config.meanflow {
            // Turbo: GPT-2 tokenizer — use raw token IDs, no SOT/EOT wrapping
            let ids = encoded.map { Int32($0) }
            return MLXArray(ids).reshaped([1, -1])
        } else {
            // Regular (LLaMA): custom EnTokenizer — wrap with SOT/EOT
            var ids = [Int32(sotToken)]
            ids.append(contentsOf: encoded.map { Int32($0) })
            ids.append(Int32(eotToken))
            return MLXArray(ids)
        }
    }

    // MARK: - Reference Audio Processing

    /// Process reference audio to extract speaker embedding and prompt tokens.
    func prepareConditionals(
        refAudio: MLXArray,
        refAudioSR: Int = 24000
    ) throws -> (T3Cond, MLXArray, MLXArray, MLXArray, MLXArray) {
        var audio = refAudio
        if audio.ndim > 1 {
            audio = audio.mean(axis: 0)
        }

        // Resample to 16kHz for VoiceEncoder
        let s3Audio: MLXArray
        if refAudioSR != ChatterboxConstants.s3SampleRate {
            s3Audio = resampleAudio(audio, fromSR: refAudioSR, toSR: ChatterboxConstants.s3SampleRate)
        } else {
            s3Audio = audio
        }

        // Resample to 24kHz for S3Gen decoder conditioning
        let s3genAudio: MLXArray
        if refAudioSR != ChatterboxConstants.s3genSampleRate {
            s3genAudio = resampleAudio(audio, fromSR: refAudioSR, toSR: ChatterboxConstants.s3genSampleRate)
        } else {
            s3genAudio = audio
        }

        // Truncate to conditioning lengths
        let encCondLen = config.encCondLen
        let decCondLen = config.decCondLen

        let s3AudioTrunc = s3Audio.dim(0) > encCondLen ? s3Audio[..<encCondLen] : s3Audio
        let s3genAudioTrunc = s3genAudio.dim(0) > decCondLen ? s3genAudio[..<decCondLen] : s3genAudio

        // 1. Speaker embedding from VoiceEncoder
        let veMels = voiceEncoderMelSpectrogram(s3AudioTrunc)
        let veMelsTransposed = veMels.transposed().expandedDimensions(axis: 0)
        let speakerEmb = voiceEncoder.inference(
            mels: veMelsTransposed,
            melLens: [veMelsTransposed.dim(1)]
        )
        eval(speakerEmb)

        // 2. Prompt speech tokens (empty for now — S3TokenizerV2 not yet implemented)
        let promptSpeechTokens = MLXArray.zeros([1, 0]).asType(.int32)

        // 3. X-vector for S3Gen — compute from mel spectrogram
        // Note: speaker_encoder weights are not loaded into module; for now use zeros
        let xVector = MLXArray.zeros([1, 192])

        // 4. S3Gen prompt features
        let promptFeat = s3genMelSpectrogram(
            y: s3genAudioTrunc.expandedDimensions(axis: 0),
            samplingRate: ChatterboxConstants.s3genSampleRate
        )
        let promptFeatLen = MLXArray([Int32(promptFeat.dim(2))])

        // Build T3Cond
        let t3Cond = T3Cond(
            speakerEmb: speakerEmb,
            condPromptSpeechTokens: promptSpeechTokens.dim(1) > 0 ? promptSpeechTokens : nil,
            condPromptSpeechEmb: nil,
            emotionAdv: MLXArray(Float(0.5))
        )

        return (t3Cond, xVector, promptSpeechTokens, promptFeat, promptFeatLen)
    }

    // MARK: - Speech Token Post-processing

    /// Drop invalid speech tokens (out of vocab range).
    /// Matches Python: `mask = np.where(np.array(speech_tokens) < 6561)[0]`
    /// Only keeps actual speech tokens (< startSpeechToken), not control tokens.
    func dropInvalidTokens(_ tokens: MLXArray) -> MLXArray {
        let flat = tokens.reshaped([-1])
        let count = flat.dim(0)
        // Python filters with `< 6561` (SPEECH_VOCAB_SIZE), which excludes control tokens
        // (start=6561, stop=6562). Use the startSpeechToken as the threshold.
        let threshold = config.t3Config.startSpeechToken
        var validIds = [Int32]()

        for i in 0 ..< count {
            let id = flat[i].item(Int.self)
            if id >= 0 && id < threshold {
                validIds.append(Int32(id))
            }
        }

        if validIds.isEmpty {
            return MLXArray([Int32(0)])
        }
        return MLXArray(validIds)
    }

    // MARK: - Generation

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        // Use reference audio, or fall back to default conditioning
        let t3Cond: T3Cond
        let xVector: MLXArray
        let promptTokens: MLXArray
        let promptFeat: MLXArray

        if let refAudio = refAudio {
            let (cond, xv, pt, pf, _) = try prepareConditionals(refAudio: refAudio)
            t3Cond = cond
            xVector = xv
            promptTokens = pt
            promptFeat = pf
        } else if let defaults = defaultConditioning {
            // Use pre-computed default voice
            t3Cond = T3Cond(
                speakerEmb: defaults.speakerEmb,
                condPromptSpeechTokens: defaults.condPromptSpeechTokens,
                condPromptSpeechEmb: nil,
                emotionAdv: defaults.emotionAdv
            )
            xVector = defaults.xVector
            promptTokens = defaults.promptToken
            promptFeat = defaults.promptFeat
        } else {
            throw AudioGenerationError.invalidInput(
                "Chatterbox requires reference audio for voice cloning. Pass refAudio parameter."
            )
        }

        // Tokenize text
        let textTokens = try tokenizeText(text)

        let temperature = generationParameters.temperature
        let topP = generationParameters.topP ?? 0.95

        // Stage 1: T3 — generate speech tokens
        var t3CondMut = t3Cond
        let speechTokens: MLXArray

        if let t3gpt2 = t3 as? T3GPT2Model {
            // Turbo: GPT-2 inference (no CFG)
            speechTokens = t3gpt2.inference(
                t3Cond: &t3CondMut,
                textTokens: textTokens,
                maxNewTokens: config.t3Config.maxSpeechTokens,
                temperature: temperature,
                topK: 1000,
                topP: topP,
                repetitionPenalty: 1.2
            )
        } else if let t3llama = t3 as? T3Model {
            // Regular: LLaMA inference (with CFG)
            speechTokens = t3llama.inference(
                t3Cond: &t3CondMut,
                textTokens: textTokens,
                maxNewTokens: config.t3Config.maxSpeechTokens,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: 1.2,
                cfgWeight: 0.5
            )
        } else {
            throw AudioGenerationError.modelNotInitialized("Unknown T3 model type")
        }
        eval(speechTokens)

        // Post-process: remove OOV tokens and append silence
        let cleanTokens = dropInvalidTokens(speechTokens)
        // Append 3 silence tokens (S3GEN_SIL = 4299) to reduce artifacts at end
        let silenceTokens = MLXArray([Int32(4299), Int32(4299), Int32(4299)])
        let finalTokens = MLX.concatenated([cleanTokens, silenceTokens])

        // Stage 2: S3Gen — speech tokens → mel → waveform
        let tokenArr = finalTokens.reshaped([1, -1])
        let tokenLen = MLXArray([Int32(tokenArr.dim(1))])
        let promptTokenLen = MLXArray([Int32(promptTokens.dim(1))])

        // Prepare prompt mel features for S3Gen conditioning
        // s3genMelSpectrogram returns (B, 80, T') — transpose to (B, T', 80) for inference
        let promptFeatTransposed: MLXArray
        if promptFeat.dim(1) == 80 && promptFeat.ndim == 3 {
            // Shape is (B, 80, T') — transpose to (B, T', 80)
            promptFeatTransposed = promptFeat.transposed(0, 2, 1)
        } else {
            // Already (B, T', 80) or some other shape
            promptFeatTransposed = promptFeat
        }

        // Run flow matching inference with raw int32 token IDs
        // Meanflow (turbo distilled): 2 steps. Non-meanflow (regular): 10 steps.
        let nTimesteps = config.meanflow ? 2 : 10
        let mel = s3gen.inference(
            token: tokenArr,
            tokenLen: tokenLen,
            promptToken: promptTokens,
            promptTokenLen: promptTokenLen,
            promptFeat: promptFeatTransposed,
            embedding: xVector,
            nTimesteps: nTimesteps
        )
        eval(mel)

        // Vocoder: mel → waveform via HiFi-GAN
        var (waveform, _) = s3gen.vocoder(mel)

        // Apply fade-in window to reduce vocoder spillover artifacts at start (matches Python trim_fade)
        // 480 samples silence + 480 samples cosine ramp = 960 samples (40ms at 24kHz)
        let nTrim = sampleRate / 50  // 480
        let fadeLen = nTrim * 2      // 960
        if waveform.dim(waveform.ndim - 1) >= fadeLen {
            // Build fade: [zeros(480), cosine_ramp(480)] — matches Python trim_fade
            let zeros = MLXArray.zeros([nTrim])
            let cosValues = (0..<nTrim).map { i -> Float in
                let t = Float.pi * (1.0 - Float(i) / Float(nTrim - 1))
                return (cos(t) + 1) / 2
            }
            let cosineRamp = MLXArray(cosValues)
            let trimFade = MLX.concatenated([zeros, cosineRamp])

            // Multiply first fadeLen samples of waveform
            let fadePart = waveform[0..., ..<fadeLen] * trimFade
            let restPart = waveform[0..., fadeLen...]
            waveform = MLX.concatenated([fadePart, restPart], axis: -1)
        }

        return waveform
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()

        Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let startTime = Date()
                let audio = try await self.generate(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    generationParameters: generationParameters
                )
                let generateTime = Date().timeIntervalSince(startTime)

                continuation.yield(.audio(audio))

                let info = AudioGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: audio.dim(audio.ndim - 1),
                    prefillTime: 0,
                    generateTime: generateTime,
                    tokensPerSecond: Double(audio.dim(audio.ndim - 1)) / max(generateTime, 0.001),
                    peakMemoryUsage: 0
                )
                continuation.yield(.info(info))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }

        return stream
    }

    // MARK: - Factory

    /// Load Chatterbox model from HuggingFace repository.
    ///
    /// Supports both Regular (`Chatterbox-TTS-fp16`) and Turbo (`chatterbox-turbo-fp16`).
    /// Automatically detects model variant from config.json.
    public static func fromPretrained(_ modelRepo: String) async throws -> ChatterboxModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken
        )

        // Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        let config: ChatterboxConfiguration
        if FileManager.default.fileExists(atPath: configURL.path) {
            let configData = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(ChatterboxConfiguration.self, from: configData)
        } else {
            config = .default
        }

        // Create model (T3 variant selected by config)
        let model = ChatterboxModel(config)
        model.modelDir = modelDir

        // Load main weights
        let weights = try loadChatterboxWeights(modelDir: modelDir)

        // Sanitize weights
        let sanitizedWeights = model.sanitize(weights: weights)

        // Quantization
        if config.quantization != nil || config.perLayerQuantization != nil {
            quantize(model: model) { path, _ in
                guard sanitizedWeights["\(path).scales"] != nil else { return nil }
                if let perLayerQuant = config.perLayerQuantization,
                   let layerQuant = perLayerQuant.quantization(layer: path)
                {
                    return layerQuant.asTuple
                }
                return config.quantization?.asTuple
            }
        }

        // Update model parameters — allow unused keys since we drop speaker_encoder/tokenizer
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [])

        eval(model)

        // Ensure tokenizer files exist before loading
        // Turbo: ships with slow tokenizer only (vocab.json + merges.txt) — generate fast tokenizer.json
        let tokenizerJsonPath = modelDir.appendingPathComponent("tokenizer.json")
        if !FileManager.default.fileExists(atPath: tokenizerJsonPath.path) {
            let vocabPath = modelDir.appendingPathComponent("vocab.json")
            let mergesPath = modelDir.appendingPathComponent("merges.txt")
            if FileManager.default.fileExists(atPath: vocabPath.path),
               FileManager.default.fileExists(atPath: mergesPath.path)
            {
                try generateTokenizerJson(
                    vocabPath: vocabPath,
                    mergesPath: mergesPath,
                    tokenizerConfigPath: modelDir.appendingPathComponent("tokenizer_config.json"),
                    outputPath: tokenizerJsonPath
                )
            }
        }

        // Regular: has tokenizer.json but missing tokenizer_config.json — generate minimal config
        let tokenizerConfigPath = modelDir.appendingPathComponent("tokenizer_config.json")
        if !FileManager.default.fileExists(atPath: tokenizerConfigPath.path) {
            let minimalConfig: [String: Any] = ["tokenizer_class": "GPT2Tokenizer"]
            let data = try JSONSerialization.data(withJSONObject: minimalConfig)
            try data.write(to: tokenizerConfigPath)
        }

        // Load tokenizer
        do {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            print("Warning: Could not load tokenizer from model folder: \(error)")
        }

        // Load default voice conditioning (conds.safetensors)
        let condsURL = modelDir.appendingPathComponent("conds.safetensors")
        if FileManager.default.fileExists(atPath: condsURL.path) {
            do {
                let condsWeights = try MLX.loadArrays(url: condsURL)
                model.defaultConditioning = DefaultConditioning(
                    speakerEmb: condsWeights["t3.speaker_emb"] ?? MLXArray.zeros([1, 256]),
                    condPromptSpeechTokens: condsWeights["t3.cond_prompt_speech_tokens"] ?? MLXArray.zeros([1, 0]).asType(.int32),
                    emotionAdv: condsWeights["t3.emotion_adv"] ?? MLXArray(Float(0.5)).reshaped([1, 1, 1]),
                    xVector: condsWeights["gen.embedding"] ?? MLXArray.zeros([1, 192]),
                    promptToken: condsWeights["gen.prompt_token"] ?? MLXArray.zeros([1, 0]).asType(.int32),
                    promptTokenLen: condsWeights["gen.prompt_token_len"] ?? MLXArray([Int32(0)]),
                    promptFeat: condsWeights["gen.prompt_feat"] ?? MLXArray.zeros([1, 0, 80]),
                    promptFeatLen: MLXArray([Int32(0)])
                )
                eval(model.defaultConditioning!.speakerEmb)
                eval(model.defaultConditioning!.xVector)
            } catch {
                print("Warning: Could not load default conditioning from conds.safetensors: \(error)")
            }
        }

        return model
    }
}

// MARK: - Weight Loading

/// Load safetensors weights for Chatterbox.
///
/// Handles both single `model.safetensors` and sharded patterns.
/// Excludes `conds.safetensors` (loaded separately for default voice).
private func loadChatterboxWeights(modelDir: URL) throws -> [String: MLXArray] {
    let singleWeightsURL = modelDir.appendingPathComponent("model.safetensors")
    if FileManager.default.fileExists(atPath: singleWeightsURL.path) {
        return try MLX.loadArrays(url: singleWeightsURL)
    }

    let fm = FileManager.default
    let files = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
    let safetensorFiles = files
        .filter {
            $0.pathExtension == "safetensors"
                && $0.lastPathComponent != "conds.safetensors"
        }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !safetensorFiles.isEmpty else {
        throw AudioGenerationError.modelNotInitialized("No .safetensors files found in \(modelDir.path)")
    }

    var allWeights = [String: MLXArray]()
    for file in safetensorFiles {
        let shardWeights = try MLX.loadArrays(url: file)
        for (key, value) in shardWeights {
            allWeights[key] = value
        }
    }

    return allWeights
}

// MARK: - Audio Resampling

/// Linear interpolation audio resampling.
private func resampleAudio(_ audio: MLXArray, fromSR: Int, toSR: Int) -> MLXArray {
    guard fromSR != toSR else { return audio }

    let inputLength = audio.dim(0)
    let outputLength = Int(Double(inputLength) * Double(toSR) / Double(fromSR))

    let inputIndices = MLXArray(0 ..< outputLength).asType(.float32) * Float(fromSR) / Float(toSR)
    let floorIndices = MLX.floor(inputIndices).asType(.int32)
    let ceilIndices = MLX.minimum(floorIndices + 1, MLXArray(Int32(inputLength - 1)))
    let fractions = inputIndices - floorIndices.asType(.float32)

    let floorValues = audio[floorIndices]
    let ceilValues = audio[ceilIndices]

    return floorValues * (1.0 - fractions) + ceilValues * fractions
}

// MARK: - Tokenizer Generation

/// Generate `tokenizer.json` (fast tokenizer format) from `vocab.json` + `merges.txt`.
///
/// Chatterbox Turbo ships with a slow tokenizer (vocab.json + merges.txt) but
/// swift-transformers requires tokenizer.json. This builds the fast tokenizer JSON
/// from the available files, using GPT-2 style BPE with ByteLevel pre-tokenizer.
///
/// Pattern reused from Qwen3TTS which has the same requirement.
private func generateTokenizerJson(
    vocabPath: URL,
    mergesPath: URL,
    tokenizerConfigPath: URL,
    outputPath: URL
) throws {
    // Read vocab
    let vocabData = try Data(contentsOf: vocabPath)
    let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] ?? [:]

    // Read merges (skip header line "#version: ...")
    let mergesText = try String(contentsOf: mergesPath, encoding: .utf8)
    let mergeLines = mergesText.components(separatedBy: .newlines)
        .filter { !$0.isEmpty && !$0.hasPrefix("#") }

    // Read added_tokens from tokenizer_config.json (if available)
    var addedTokens = [[String: Any]]()
    if let configData = try? Data(contentsOf: tokenizerConfigPath),
       let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
       let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: [String: Any]]
    {
        for (idStr, tokenInfo) in addedTokensDecoder {
            guard let tokenId = Int(idStr),
                  let content = tokenInfo["content"] as? String else { continue }
            let entry: [String: Any] = [
                "id": tokenId,
                "content": content,
                "single_word": tokenInfo["single_word"] as? Bool ?? false,
                "lstrip": tokenInfo["lstrip"] as? Bool ?? false,
                "rstrip": tokenInfo["rstrip"] as? Bool ?? false,
                "normalized": tokenInfo["normalized"] as? Bool ?? false,
                "special": tokenInfo["special"] as? Bool ?? true,
            ]
            addedTokens.append(entry)
        }
        addedTokens.sort { ($0["id"] as? Int ?? 0) < ($1["id"] as? Int ?? 0) }
    }

    // Build tokenizer.json — GPT-2 style BPE with ByteLevel pre-tokenizer
    let tokenizerJson: [String: Any] = [
        "version": "1.0",
        "truncation": NSNull(),
        "padding": NSNull(),
        "added_tokens": addedTokens,
        "normalizer": NSNull(),
        "pre_tokenizer": [
            "type": "Sequence",
            "pretokenizers": [
                [
                    "type": "Split",
                    "pattern": [
                        "Regex":
                            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    ],
                    "behavior": "Isolated",
                    "invert": false,
                ] as [String: Any],
                [
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": true,
                    "use_regex": false,
                ] as [String: Any],
            ] as [[String: Any]],
        ] as [String: Any],
        "post_processor": NSNull(),
        "decoder": [
            "type": "ByteLevel",
            "add_prefix_space": true,
            "trim_offsets": true,
            "use_regex": true,
        ] as [String: Any],
        "model": [
            "type": "BPE",
            "dropout": NSNull(),
            "unk_token": NSNull(),
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocabDict,
            "merges": mergeLines,
        ] as [String: Any],
    ]

    let jsonData = try JSONSerialization.data(withJSONObject: tokenizerJson, options: [.sortedKeys])
    try jsonData.write(to: outputPath)
}
