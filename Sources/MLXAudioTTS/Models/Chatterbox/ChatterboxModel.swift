//
//  ChatterboxModel.swift
//  MLXAudio
//
//  Top-level Chatterbox TTS model.
//  Two-stage pipeline: T3 (text→speech tokens) + S3Gen (speech tokens→audio).
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

// MARK: - Chatterbox Model

/// Chatterbox TTS: two-stage speech synthesis.
///
/// Stage 1 (T3): LLaMA backbone converts text tokens → speech tokens (6561 vocab),
/// conditioned on speaker embedding + optional prompt + emotion scalar.
///
/// Stage 2 (S3Gen): Flow matching decoder (Euler ODE) + HiFi-GAN vocoder converts
/// speech tokens → mel spectrogram → waveform at 24kHz.
public final class ChatterboxModel: Module, SpeechGenerationModel, @unchecked Sendable {

    // MARK: - Configuration

    public let config: ChatterboxConfiguration

    // MARK: - Sub-models

    /// Voice encoder: extracts 256-dim speaker embedding from reference audio.
    @ModuleInfo(key: "ve") var voiceEncoder: VoiceEncoder

    /// T3: text-to-speech-token model (LLaMA backbone).
    @ModuleInfo(key: "t3") var t3: T3Model

    /// S3Gen: speech-token-to-audio model (Conformer + flow matching + HiFi-GAN).
    @ModuleInfo(key: "s3gen") var s3gen: CausalMaskedDiffWithXvec

    /// CAMPPlus: x-vector speaker encoder for S3Gen conditioning.
    @ModuleInfo(key: "campplus") var campplus: CAMPPlus

    // MARK: - Tokenizer

    /// Text tokenizer loaded from tokenizer.json.
    public var tokenizer: Tokenizer?

    /// Model directory (for loading auxiliary files).
    private var modelDir: URL?

    // MARK: - Protocol conformance

    public var sampleRate: Int { config.s3genSr }

    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(temperature: 0.8)
    }

    // MARK: - Special tokens

    /// Start-of-text token.
    private var sotToken: Int { config.t3Config.startTextToken }
    /// End-of-text token.
    private var eotToken: Int { config.t3Config.stopTextToken }
    /// Start-of-speech token.
    private var sosToken: Int { config.t3Config.startSpeechToken }
    /// End-of-speech token.
    private var eosToken: Int { config.t3Config.stopSpeechToken }
    /// Speech vocab size (without special tokens).
    private var speechVocabSize: Int { ChatterboxConstants.speechVocabSize }

    // MARK: - Initialization

    public init(_ config: ChatterboxConfiguration = .default) {
        self.config = config

        self._voiceEncoder.wrappedValue = VoiceEncoder()
        self._t3.wrappedValue = T3Model(config.t3Config)
        self._s3gen.wrappedValue = CausalMaskedDiffWithXvec()
        self._campplus.wrappedValue = CAMPPlus()
    }

    // MARK: - Weight Sanitization

    /// Route weights by prefix to the correct sub-model.
    ///
    /// Python weight keys are prefixed: `ve.*`, `t3.*`, `s3gen.*`, `campplus.*`.
    /// Each sub-model then does its own internal sanitization.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var veWeights = [String: MLXArray]()
        var t3Weights = [String: MLXArray]()
        var s3genWeights = [String: MLXArray]()
        var campplusWeights = [String: MLXArray]()
        var otherWeights = [String: MLXArray]()

        for (key, value) in weights {
            if key.hasPrefix("ve.") {
                let subKey = String(key.dropFirst("ve.".count))
                veWeights[subKey] = value
            } else if key.hasPrefix("t3.") {
                let subKey = String(key.dropFirst("t3.".count))
                t3Weights[subKey] = value
            } else if key.hasPrefix("s3gen.") {
                let subKey = String(key.dropFirst("s3gen.".count))
                s3genWeights[subKey] = value
            } else if key.hasPrefix("campplus.") {
                let subKey = String(key.dropFirst("campplus.".count))
                campplusWeights[subKey] = value
            } else {
                otherWeights[key] = value
            }
        }

        // Sub-model sanitization
        let sanitizedVE = voiceEncoder.sanitize(weights: veWeights)
        let sanitizedT3 = t3.sanitize(weights: t3Weights)

        // Reconstruct with prefixes
        var result = [String: MLXArray]()

        for (key, value) in sanitizedVE {
            result["ve.\(key)"] = value
        }
        for (key, value) in sanitizedT3 {
            result["t3.\(key)"] = value
        }
        for (key, value) in s3genWeights {
            result["s3gen.\(key)"] = value
        }
        for (key, value) in campplusWeights {
            result["campplus.\(key)"] = value
        }
        for (key, value) in otherWeights {
            result[key] = value
        }

        return result
    }

    // MARK: - Text Tokenization

    /// Tokenize text into token IDs for T3.
    ///
    /// The tokenizer maps text to IDs. We wrap with [SOT, ..., EOT].
    func tokenizeText(_ text: String) throws -> MLXArray {
        guard let tokenizer = tokenizer else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }

        let encoded = tokenizer.encode(text: text)
        var ids = [Int32(sotToken)]
        ids.append(contentsOf: encoded.map { Int32($0) })
        ids.append(Int32(eotToken))

        return MLXArray(ids)
    }

    // MARK: - Reference Audio Processing

    /// Process reference audio to extract speaker embedding and prompt tokens.
    ///
    /// Returns T3Cond for T3 conditioning and gen dict for S3Gen conditioning.
    func prepareConditionals(
        refAudio: MLXArray,
        refAudioSR: Int = 24000
    ) throws -> (T3Cond, MLXArray, MLXArray, MLXArray, MLXArray) {
        // Ensure mono
        var audio = refAudio
        if audio.ndim > 1 {
            audio = audio.mean(axis: 0)
        }

        // Resample to 16kHz for VoiceEncoder + S3Tokenizer
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

        let s3AudioTrunc = s3Audio.dim(0) > encCondLen
            ? s3Audio[..<encCondLen]
            : s3Audio

        let s3genAudioTrunc = s3genAudio.dim(0) > decCondLen
            ? s3genAudio[..<decCondLen]
            : s3genAudio

        // 1. Speaker embedding from VoiceEncoder
        let veMels = voiceEncoderMelSpectrogram(s3AudioTrunc)
        // veMels: (M, T') → need (1, T', M) for VoiceEncoder
        let veMelsTransposed = veMels.transposed().expandedDimensions(axis: 0)
        let speakerEmb = voiceEncoder.inference(
            mels: veMelsTransposed,
            melLens: [veMelsTransposed.dim(1)]
        ) // (1, 256)
        eval(speakerEmb)

        // 2. S3 Tokenizer for prompt speech tokens
        // Note: S3TokenizerV2 is loaded separately — for now use empty prompt if not available
        // In a full implementation, we'd load S3TokenizerV2 and tokenize s3AudioTrunc
        let promptSpeechTokens = MLXArray.zeros([1, 0]).asType(.int32)
        let promptSpeechTokenLen = MLXArray([Int32(0)])

        // 3. CAMPPlus x-vector for S3Gen conditioning
        let campplusMels = s3genMelSpectrogram(
            y: s3genAudioTrunc.expandedDimensions(axis: 0),
            samplingRate: ChatterboxConstants.s3genSampleRate
        )
        // campplusMels: (1, 80, T') → need (1, T', 80) for CAMPPlus
        let campplusMelsT = campplusMels.transposed(0, 2, 1)
        let xVector = campplus(campplusMelsT) // (1, 192)
        eval(xVector)

        // 4. S3Gen prompt features (mel of decoder conditioning audio)
        let promptFeat = s3genMelSpectrogram(
            y: s3genAudioTrunc.expandedDimensions(axis: 0),
            samplingRate: ChatterboxConstants.s3genSampleRate
        )
        // promptFeat: (1, 80, T') — already in (B, M, T') format for decoder
        let promptFeatT = promptFeat
        let promptFeatLen = MLXArray([Int32(promptFeat.dim(2))])

        // Build T3Cond
        let t3Cond = T3Cond(
            speakerEmb: speakerEmb,
            condPromptSpeechTokens: promptSpeechTokens.dim(1) > 0 ? promptSpeechTokens : nil,
            condPromptSpeechEmb: nil,
            emotionAdv: MLXArray(Float(0.5))
        )

        return (t3Cond, xVector, promptSpeechTokens, promptFeatT, promptFeatLen)
    }

    // MARK: - Speech Token Post-processing

    /// Drop invalid speech tokens (out of vocab range).
    /// Matches Python: `drop_invalid_tokens`.
    func dropInvalidTokens(_ tokens: MLXArray) -> MLXArray {
        let flat = tokens.reshaped([-1])
        let count = flat.dim(0)
        var validIds = [Int32]()

        for i in 0 ..< count {
            let id = flat[i].item(Int.self)
            if id >= 0 && id < speechVocabSize {
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
        guard let refAudio = refAudio else {
            throw AudioGenerationError.invalidInput(
                "Chatterbox requires reference audio for voice cloning. Pass refAudio parameter."
            )
        }

        // Tokenize text
        let textTokens = try tokenizeText(text)

        // Prepare conditionals from reference audio
        var (t3Cond, xVector, promptTokens, promptFeat, promptFeatLen) = try prepareConditionals(
            refAudio: refAudio
        )

        // Stage 1: T3 — generate speech tokens
        let temperature = generationParameters.temperature
        let topP = generationParameters.topP ?? 0.95

        let speechTokens = t3.inference(
            t3Cond: &t3Cond,
            textTokens: textTokens,
            maxNewTokens: config.t3Config.maxSpeechTokens,
            temperature: temperature,
            topP: topP,
            repetitionPenalty: 1.2,
            cfgWeight: 0.5
        )
        eval(speechTokens)

        // Post-process: remove special tokens
        let cleanTokens = dropInvalidTokens(speechTokens)

        // Stage 2: S3Gen — speech tokens → mel → waveform
        let tokenArr = cleanTokens.reshaped([1, -1])
        let tokenLen = MLXArray([Int32(tokenArr.dim(1))])
        let promptTokenLen = MLXArray([Int32(promptTokens.dim(1))])

        // Embed prompt tokens through S3Gen encoder for conditioning
        let promptTokenEmb: MLXArray
        let promptEmbLen: MLXArray
        if promptTokens.dim(1) > 0 {
            // Use S3Gen encoder to embed prompt tokens
            let (emb, embLen) = s3gen.embedRef(
                speechTokens: promptTokens.asType(.float32),
                speechTokenLens: promptTokenLen)
            promptTokenEmb = emb
            promptEmbLen = embLen
        } else {
            // No prompt — use empty tensors
            promptTokenEmb = MLXArray.zeros([1, 0, 512])
            promptEmbLen = MLXArray([Int32(0)])
        }

        // Run flow matching inference
        let mel = s3gen.inference(
            token: tokenArr.asType(.float32),
            tokenLen: tokenLen,
            prompt: promptTokenEmb.dim(1) > 0
                ? promptTokenEmb
                : MLXArray.zeros([1, 0, 512]),
            promptLen: promptEmbLen,
            xVector: xVector,
            nTimesteps: 10
        )
        eval(mel)

        // Vocoder: mel → waveform via HiFi-GAN
        // The S3Gen model contains a HiFTGenerator vocoder accessed through its decoder property.
        // Use the vocoder to convert mel spectrogram to waveform.
        let (waveform, _) = s3gen.vocoder(mel)

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
    /// Downloads model weights, config, and tokenizer. Supports quantized variants.
    ///
    /// - Parameter modelRepo: HuggingFace repository ID (e.g., "mlx-community/chatterbox-turbo-fp16").
    /// - Returns: Loaded and ready-to-use ChatterboxModel.
    public static func fromPretrained(_ modelRepo: String) async throws -> ChatterboxModel {
        // 1. Get HF token
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String

        // 2. Download/resolve model directory
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }

        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken
        )

        // 3. Load config
        let configURL = modelDir.appendingPathComponent("config.json")
        let config: ChatterboxConfiguration
        if FileManager.default.fileExists(atPath: configURL.path) {
            let configData = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(ChatterboxConfiguration.self, from: configData)
        } else {
            config = .default
        }

        // 4. Create model
        let model = ChatterboxModel(config)
        model.modelDir = modelDir

        // 5. Load weights
        let weights = try loadChatterboxWeights(modelDir: modelDir)

        // 6. Sanitize weights
        let sanitizedWeights = model.sanitize(weights: weights)

        // 7. Quantization (if configured)
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

        // 8. Update model parameters
        try model.update(parameters: ModuleParameters.unflattened(sanitizedWeights), verify: [.noUnusedKeys])

        // 9. Evaluate
        eval(model)

        // 10. Load tokenizer
        do {
            model.tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        } catch {
            print("Warning: Could not load tokenizer from model folder: \(error)")
            // Chatterbox uses a custom tokenizer — may need special handling
        }

        return model
    }
}

// MARK: - Weight Loading

/// Load safetensors weights for Chatterbox.
///
/// Handles both single `model.safetensors` and sharded `model-00001-of-00002.safetensors` patterns.
private func loadChatterboxWeights(modelDir: URL) throws -> [String: MLXArray] {
    // Check for single weights file
    let singleWeightsURL = modelDir.appendingPathComponent("model.safetensors")
    if FileManager.default.fileExists(atPath: singleWeightsURL.path) {
        return try MLX.loadArrays(url: singleWeightsURL)
    }

    // Check for sharded weights
    let fm = FileManager.default
    let files = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
    let safetensorFiles = files
        .filter { $0.pathExtension == "safetensors" }
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

/// Simple nearest-neighbor audio resampling.
///
/// For production use, a proper sinc-interpolation resampler would be better,
/// but this matches the basic functionality needed for conditioning audio.
private func resampleAudio(_ audio: MLXArray, fromSR: Int, toSR: Int) -> MLXArray {
    guard fromSR != toSR else { return audio }

    let inputLength = audio.dim(0)
    let outputLength = Int(Double(inputLength) * Double(toSR) / Double(fromSR))

    // Linear interpolation resampling
    let inputIndices = MLXArray(0 ..< outputLength).asType(.float32) * Float(fromSR) / Float(toSR)
    let floorIndices = MLX.floor(inputIndices).asType(.int32)
    let ceilIndices = MLX.minimum(floorIndices + 1, MLXArray(Int32(inputLength - 1)))
    let fractions = inputIndices - floorIndices.asType(.float32)

    let floorValues = audio[floorIndices]
    let ceilValues = audio[ceilIndices]

    return floorValues * (1.0 - fractions) + ceilValues * fractions
}
