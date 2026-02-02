//
//  Qwen3TTSModel.swift
//  MLXAudio
//
//  Top-level model wrapper that orchestrates:
//  - Text tokenizer (swift-transformers)
//  - Talker model (text-to-codes)
//  - Speech tokenizer (codes-to-audio)
//
//  Ported from mlx_audio/tts/models/qwen3_tts/qwen3_tts.py
//

import Foundation
import HuggingFace
import MLX
import MLXNN
import Tokenizers

// MARK: - Qwen3TTSModel

/// Top-level Qwen3-TTS model for text-to-speech synthesis.
///
/// Usage:
/// ```swift
/// let model = try await Qwen3TTSModel.load(from: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
/// let audio = try model.generate(text: "Hello, world!")
/// // audio is [samples] at 24kHz
///
/// // Voice cloning with reference audio:
/// let refAudio = ... // MLXArray of audio samples at 24kHz
/// let audio = try model.generate(text: "Hello!", refAudio: refAudio)
/// ```
public class Qwen3TTSModel {
    public let config: Qwen3TTSModelConfig
    public let talker: Qwen3TTSTalkerForConditionalGeneration
    public let speechTokenizer: Qwen3TTSSpeechTokenizer
    public var textTokenizer: Tokenizer?

    /// Speaker encoder for voice cloning (only available for "base" model type).
    public var speakerEncoder: Qwen3TTSSpeakerEncoder?

    /// Sample rate of generated audio (24kHz).
    public var sampleRate: Int { config.sampleRate }

    /// Whether voice cloning is supported (requires speaker encoder).
    public var supportsVoiceCloning: Bool { speakerEncoder != nil }

    /// Initialize with config and sub-models.
    public init(
        config: Qwen3TTSModelConfig,
        talker: Qwen3TTSTalkerForConditionalGeneration,
        speechTokenizer: Qwen3TTSSpeechTokenizer,
        speakerEncoder: Qwen3TTSSpeakerEncoder? = nil
    ) {
        self.config = config
        self.talker = talker
        self.speechTokenizer = speechTokenizer
        self.speakerEncoder = speakerEncoder
    }

    // MARK: - Loading

    /// Load model from a HuggingFace repository.
    ///
    /// - Parameters:
    ///   - repo: HuggingFace repo ID (e.g., "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
    /// - Returns: Loaded model ready for generation
    public static func load(from repo: String) async throws -> Qwen3TTSModel {
        let client = HubClient.default
        let cache = client.cache ?? HubCache.default

        guard let repoID = Repo.ID(rawValue: repo) else {
            throw Qwen3TTSError.invalidRepoID(repo)
        }

        // Use a persistent cache directory based on repo ID
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDirectory = cache.cacheDirectory
            .appendingPathComponent(modelSubdir)

        // Download model files using HubClient
        try await client.downloadSnapshot(
            of: repoID,
            kind: .model,
            to: modelDirectory,
            revision: "main",
            matching: ["*.safetensors", "*.json", "*.txt"]
        )

        return try await load(from: modelDirectory)
    }

    /// Load model from a local directory.
    ///
    /// - Parameter directory: URL to the model directory containing config.json, model.safetensors, etc.
    /// - Returns: Loaded model ready for generation
    public static func load(from directory: URL) async throws -> Qwen3TTSModel {
        // Load config
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw Qwen3TTSError.configNotFound(configURL)
        }
        let config = try Qwen3TTSModelConfig.load(from: configURL)

        // Create talker model
        guard let talkerConfig = config.talkerConfig else {
            throw Qwen3TTSError.invalidConfig("Missing talker_config")
        }
        let talker = Qwen3TTSTalkerForConditionalGeneration(config: talkerConfig)

        // Load speech tokenizer config
        // First try main config, then try speech_tokenizer/config.json
        let speechTokenizerConfig: Qwen3TTSTokenizerConfig
        if let tokenizerConfig = config.tokenizerConfig {
            speechTokenizerConfig = tokenizerConfig
        } else {
            // Try loading from speech_tokenizer subdirectory
            let speechTokenizerConfigURL = directory
                .appendingPathComponent("speech_tokenizer")
                .appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: speechTokenizerConfigURL.path) {
                let data = try Data(contentsOf: speechTokenizerConfigURL)
                speechTokenizerConfig = try JSONDecoder().decode(Qwen3TTSTokenizerConfig.self, from: data)
            } else {
                // Use default config
                speechTokenizerConfig = Qwen3TTSTokenizerConfig()
            }
        }
        let speechTokenizer = Qwen3TTSSpeechTokenizer(config: speechTokenizerConfig)

        // Load talker weights with quantization support
        try talker.loadWeights(from: directory, quantization: config.effectiveQuantization)

        // Load speech tokenizer weights from speech_tokenizer subdirectory
        let speechTokenizerWeightsURL = directory
            .appendingPathComponent("speech_tokenizer")
            .appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: speechTokenizerWeightsURL.path) {
            try speechTokenizer.loadWeights(from: speechTokenizerWeightsURL)
        } else {
            // Fallback to flat structure
            let flatURL = directory.appendingPathComponent("speech_tokenizer.safetensors")
            if FileManager.default.fileExists(atPath: flatURL.path) {
                try speechTokenizer.loadWeights(from: flatURL)
            }
        }

        // Create speaker encoder for base models (voice cloning support)
        var speakerEncoder: Qwen3TTSSpeakerEncoder? = nil
        if config.ttsModelType == "base",
           let speakerEncoderConfig = config.speakerEncoderConfig {
            speakerEncoder = Qwen3TTSSpeakerEncoder(config: speakerEncoderConfig)

            // Load speaker encoder weights from main model.safetensors
            let modelWeightsURL = directory.appendingPathComponent("model.safetensors")
            if FileManager.default.fileExists(atPath: modelWeightsURL.path),
               let encoder = speakerEncoder {
                try loadSpeakerEncoderWeights(
                    speakerEncoder: encoder,
                    from: modelWeightsURL
                )
            }
        }

        let model = Qwen3TTSModel(
            config: config,
            talker: talker,
            speechTokenizer: speechTokenizer,
            speakerEncoder: speakerEncoder
        )

        // Load text tokenizer
        // Try tokenizer.json first, then fall back to tokenizer_config.json + vocab.json
        let tokenizerJsonURL = directory.appendingPathComponent("tokenizer.json")
        let tokenizerConfigURL = directory.appendingPathComponent("tokenizer_config.json")

        if FileManager.default.fileExists(atPath: tokenizerJsonURL.path) ||
           FileManager.default.fileExists(atPath: tokenizerConfigURL.path) {
            do {
                model.textTokenizer = try await AutoTokenizer.from(modelFolder: directory)
            } catch {
                print("Warning: Failed to load tokenizer: \(error)")
            }
        } else {
            print("Warning: No tokenizer files found in model directory")
        }

        return model
    }

    /// Load speaker encoder weights from safetensors file.
    private static func loadSpeakerEncoderWeights(
        speakerEncoder: Qwen3TTSSpeakerEncoder,
        from url: URL
    ) throws {
        let weights = try loadArrays(url: url)

        // Sanitize weights for speaker encoder
        let sanitizedWeights = Qwen3TTSSpeakerEncoder.sanitize(weights: weights)

        if !sanitizedWeights.isEmpty {
            try speakerEncoder.update(parameters: ModuleParameters.unflattened(sanitizedWeights))
            eval(speakerEncoder)
        }
    }

    // MARK: - Speaker Embedding Extraction

    /// Extract speaker embedding from reference audio for voice cloning.
    ///
    /// - Parameters:
    ///   - audio: Audio waveform [samples] at 24kHz
    ///   - sampleRate: Sample rate of the audio (must be 24000)
    /// - Returns: Speaker embedding [1, enc_dim] (typically [1, 1024])
    /// - Throws: Error if speaker encoder is not available or sample rate is incorrect
    public func extractSpeakerEmbedding(
        audio: MLXArray,
        sampleRate: Int = 24000
    ) throws -> MLXArray {
        guard let encoder = speakerEncoder else {
            throw Qwen3TTSError.speakerEncoderNotAvailable
        }

        guard sampleRate == 24000 else {
            throw Qwen3TTSError.invalidSampleRate(expected: 24000, got: sampleRate)
        }

        // Normalize audio amplitude for consistent mel spectrogram
        let audioMax = abs(audio).max()
        let normalizedAudio = audio / (audioMax + MLXArray(Float(1e-8)))
        eval(normalizedAudio)

        // Compute mel spectrogram from normalized audio
        let mels = melSpectrogram(
            audio: normalizedAudio,
            nFFT: 1024,
            numMels: 128,
            sampleRate: 24000,
            hopSize: 256,
            winSize: 1024,
            fMin: 0,
            fMax: 12000
        )  // [batch, frames, mels]
        eval(mels)

        // Extract speaker embedding
        let embedding = encoder(mels)  // [batch, enc_dim]
        eval(embedding)

        return embedding
    }

    // MARK: - Generation

    /// Generate speech audio from text.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - temperature: Sampling temperature (default: 1.0)
    ///   - topK: Top-k sampling (default: 50)
    ///   - topP: Top-p (nucleus) sampling (default: 0.95)
    ///   - maxTokens: Maximum tokens to generate (default: 2000)
    ///   - repetitionPenalty: Repetition penalty (default: 1.0)
    ///   - speaker: Speaker name for voice selection (CustomVoice models)
    ///   - refAudio: Reference audio for voice cloning [samples] at 24kHz (Base models only)
    ///   - instruct: Voice description or style instruction
    /// - Returns: Audio waveform as MLXArray [samples] at 24kHz
    public func generate(
        text: String,
        temperature: Float = 1.0,
        topK: Int = 50,
        topP: Float = 0.95,
        maxTokens: Int = 2000,
        repetitionPenalty: Float = 1.0,
        speaker: String? = nil,
        refAudio: MLXArray? = nil,
        instruct: String? = nil
    ) throws -> MLXArray {
        // Validate input parameters
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Qwen3TTSError.invalidInput("Text cannot be empty")
        }
        guard temperature > 0 else {
            throw Qwen3TTSError.invalidParameter("temperature", "must be positive, got \(temperature)")
        }
        guard topP >= 0 && topP <= 1 else {
            throw Qwen3TTSError.invalidParameter("topP", "must be in [0, 1], got \(topP)")
        }
        guard topK > 0 else {
            throw Qwen3TTSError.invalidParameter("topK", "must be positive, got \(topK)")
        }
        guard maxTokens > 0 else {
            throw Qwen3TTSError.invalidParameter("maxTokens", "must be positive, got \(maxTokens)")
        }

        guard let tokenizer = textTokenizer else {
            throw Qwen3TTSError.tokenizerNotLoaded
        }

        // Format text with chat template
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let tokens = tokenizer.encode(text: chatText)
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])

        // Cap max_tokens based on target text length to prevent runaway generation
        // when EOS logit doesn't become dominant (seen especially with 0.6B model).
        // At 12.5 Hz codec rate, ~3-5 codec tokens per text token is typical speech.
        // Factor of 6 gives ~50% margin for slow speech / pauses.
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Extract speaker embedding from reference audio if provided
        var speakerEmbedding: MLXArray? = nil
        if let audio = refAudio {
            speakerEmbedding = try extractSpeakerEmbedding(audio: audio)
        }

        // Tokenize instruct if provided (for VoiceDesign/CustomVoice models)
        var instructIds: MLXArray? = nil
        if let instructText = instruct {
            let instructFormatted = "<|im_start|>user\n\(instructText)<|im_end|>\n"
            let instructTokens = tokenizer.encode(text: instructFormatted)
            instructIds = MLXArray(instructTokens.map { Int32($0) }).reshaped([1, instructTokens.count])
        }

        // Generate audio codes
        let codes = talker.generate(
            inputIds: inputIds,
            maxTokens: effectiveMaxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            speaker: speaker,
            speakerEmbedding: speakerEmbedding,
            instructIds: instructIds
        )

        // Check if any codes were generated
        if codes.shape[1] == 0 {
            throw Qwen3TTSError.noCodesGenerated
        }


        // Decode to audio
        let (audio, audioLengths) = speechTokenizer.decode(codes)
        eval(audio, audioLengths)

        // Return first batch item, trimmed to valid length
        let validLength = audioLengths[0].item(Int.self)
        if validLength > 0 && validLength < audio.shape[1] {
            return audio[0, 0..<validLength]
        }
        return audio[0]
    }

    /// Generate speech audio with streaming callback.
    ///
    /// - Parameters:
    ///   - text: Input text to synthesize
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k sampling
    ///   - topP: Top-p sampling
    ///   - maxTokens: Maximum tokens to generate
    ///   - speaker: Optional speaker name for voice selection (CustomVoice models)
    ///   - refAudio: Reference audio for voice cloning [samples] at 24kHz (Base models only)
    ///   - instruct: Optional voice description or style instruction
    ///   - onProgress: Callback called with (currentStep, generatedCodes)
    /// - Returns: Audio waveform as MLXArray [samples] at 24kHz
    public func generate(
        text: String,
        temperature: Float = 1.0,
        topK: Int = 50,
        topP: Float = 0.95,
        maxTokens: Int = 2000,
        repetitionPenalty: Float = 1.0,
        speaker: String? = nil,
        refAudio: MLXArray? = nil,
        instruct: String? = nil,
        onProgress: @escaping (Int, Int) -> Void
    ) throws -> MLXArray {
        // Validate input parameters
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Qwen3TTSError.invalidInput("Text cannot be empty")
        }
        guard temperature > 0 else {
            throw Qwen3TTSError.invalidParameter("temperature", "must be positive, got \(temperature)")
        }
        guard topP >= 0 && topP <= 1 else {
            throw Qwen3TTSError.invalidParameter("topP", "must be in [0, 1], got \(topP)")
        }
        guard topK > 0 else {
            throw Qwen3TTSError.invalidParameter("topK", "must be positive, got \(topK)")
        }
        guard maxTokens > 0 else {
            throw Qwen3TTSError.invalidParameter("maxTokens", "must be positive, got \(maxTokens)")
        }

        guard let tokenizer = textTokenizer else {
            throw Qwen3TTSError.tokenizerNotLoaded
        }

        // Format text with chat template
        let chatText = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let tokens = tokenizer.encode(text: chatText)
        let inputIds = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])

        // Cap max_tokens based on target text length to prevent runaway generation
        // when EOS logit doesn't become dominant (seen especially with 0.6B model).
        // At 12.5 Hz codec rate, ~3-5 codec tokens per text token is typical speech.
        // Factor of 6 gives ~50% margin for slow speech / pauses.
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Extract speaker embedding from reference audio if provided
        var speakerEmbedding: MLXArray? = nil
        if let audio = refAudio {
            speakerEmbedding = try extractSpeakerEmbedding(audio: audio)
        }

        // Tokenize instruct if provided
        var instructIds: MLXArray? = nil
        if let instructText = instruct {
            let instructFormatted = "<|im_start|>user\n\(instructText)<|im_end|>\n"
            let instructTokens = tokenizer.encode(text: instructFormatted)
            instructIds = MLXArray(instructTokens.map { Int32($0) }).reshaped([1, instructTokens.count])
        }

        // Prepare generation inputs
        let (initialEmbeds, trailingTextHidden, ttsPadEmbed) = talker.prepareGenerationInputs(inputIds: inputIds, speaker: speaker, speakerEmbedding: speakerEmbedding, instructIds: instructIds)

        // Initialize generation state
        let talkerCache = talker.makeCache()
        var inputEmbeds = initialEmbeds
        var trailingIdx = 0
        var generatedCodes: [MLXArray] = []
        var generatedTokens: [Int] = []

        let eosTokenId = talker.config.codecEosTokenId

        // Generation loop with progress callback
        for step in 0..<effectiveMaxTokens {
            // Forward pass
            let (logits, hiddenStates) = talker(inputEmbeds, cache: talkerCache)
            eval(logits)
            eval(hiddenStates)

            // Sample first codebook token
            let nextToken = talker.sampleToken(
                logits: logits,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedTokens,
                eosTokenId: eosTokenId
            )
            eval(nextToken)

            let tokenValue = nextToken[0, 0].item(Int32.self)

            if tokenValue == Int32(eosTokenId) {
                break
            }

            generatedTokens.append(Int(tokenValue))

            // Generate remaining codebook tokens
            var codeTokens: [MLXArray] = [nextToken]
            let hiddenSeqLen = hiddenStates.shape[1]
            let codeHidden = hiddenStates[0..., (hiddenSeqLen - 1)..<hiddenSeqLen, 0...]
            let codePredictorCache = talker.codePredictor.makeCache()

            for codeIdx in 0..<(talker.config.numCodeGroups - 1) {
                let codeInput: MLXArray
                if codeIdx == 0 {
                    let code0Embed = talker.model.codecEmbedding(nextToken.asType(.int32))
                    codeInput = concatenated([codeHidden, code0Embed], axis: 1)
                } else {
                    let prevCode = codeTokens[codeTokens.count - 1]
                    codeInput = talker.codePredictor.codecEmbedding[codeIdx - 1](prevCode.asType(.int32))
                }

                let (codeLogits, _, _) = talker.codePredictor(codeInput, cache: codePredictorCache, generationStep: codeIdx)
                eval(codeLogits)

                let nextCode = talker.sampleToken(
                    logits: codeLogits,
                    temperature: temperature,
                    topK: topK,
                    topP: topP
                )
                eval(nextCode)
                codeTokens.append(nextCode)
            }

            let allCodes = concatenated(codeTokens, axis: 1)
            generatedCodes.append(allCodes)

            // Progress callback
            onProgress(step + 1, generatedCodes.count)

            // Prepare next input
            let textEmbed: MLXArray
            if trailingIdx < trailingTextHidden.shape[1] {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            var codecEmbed = talker.model.codecEmbedding(nextToken.asType(.int32))
            for (i, code) in codeTokens.dropFirst().enumerated() {
                let embedding = talker.codePredictor.codecEmbedding[i](code.asType(.int32))
                codecEmbed = codecEmbed + embedding
            }

            inputEmbeds = textEmbed + codecEmbed
            eval(inputEmbeds)
        }

        if generatedCodes.isEmpty {
            throw Qwen3TTSError.noCodesGenerated
        }

        // Stack codes and decode
        let codes = stacked(generatedCodes, axis: 1).asType(.int32)
        let (audio, audioLengths) = speechTokenizer.decode(codes)
        eval(audio, audioLengths)

        let validLength = audioLengths[0].item(Int.self)
        if validLength > 0 && validLength < audio.shape[1] {
            return audio[0, 0..<validLength]
        }
        return audio[0]
    }
}

// MARK: - Errors

/// Errors that can occur during Qwen3-TTS operations.
public enum Qwen3TTSError: Error, LocalizedError {
    case configNotFound(URL)
    case invalidConfig(String)
    case weightsNotFound(URL)
    case tokenizerNotLoaded
    case noCodesGenerated
    case audioSaveFailed(String)
    case speakerEncoderNotAvailable
    case invalidSampleRate(expected: Int, got: Int)
    case invalidInput(String)
    case invalidParameter(String, String)  // (paramName, reason)
    case invalidRepoID(String)

    public var errorDescription: String? {
        switch self {
        case .configNotFound(let url):
            return "Config file not found at: \(url.path)"
        case .invalidConfig(let message):
            return "Invalid config: \(message)"
        case .weightsNotFound(let url):
            return "Weights file not found at: \(url.path)"
        case .tokenizerNotLoaded:
            return "Text tokenizer not loaded. Ensure tokenizer.json exists in the model directory."
        case .noCodesGenerated:
            return "No audio codes were generated. The model may have hit EOS immediately."
        case .audioSaveFailed(let message):
            return "Failed to save audio: \(message)"
        case .speakerEncoderNotAvailable:
            return "Speaker encoder not available. Voice cloning requires a 'base' model type with speaker encoder."
        case .invalidSampleRate(let expected, let got):
            return "Invalid sample rate: expected \(expected)Hz, got \(got)Hz. Reference audio must be at 24kHz."
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .invalidParameter(let param, let reason):
            return "Invalid parameter '\(param)': \(reason)"
        case .invalidRepoID(let repo):
            return "Invalid HuggingFace repository ID: \(repo)"
        }
    }
}

// MARK: - Audio File Writing

extension Qwen3TTSModel {
    /// Save audio waveform to a WAV file.
    ///
    /// - Parameters:
    ///   - audio: Audio waveform [samples]
    ///   - url: Output file URL (should end in .wav)
    public static func saveWAV(audio: MLXArray, to url: URL) throws {
        let samples = audio.asArray(Float.self)
        try writeWAV(samples: samples, sampleRate: 24000, to: url)
    }
}

/// Write audio samples to a WAV file.
///
/// - Parameters:
///   - samples: Audio samples as Float array (normalized -1 to 1)
///   - sampleRate: Sample rate in Hz
///   - url: Output file URL
private func writeWAV(samples: [Float], sampleRate: Int, to url: URL) throws {
    var data = Data()

    // Convert to 16-bit PCM
    let int16Samples = samples.map { sample -> Int16 in
        let clamped = max(-1.0, min(1.0, sample))
        return Int16(clamped * Float(Int16.max))
    }

    let numSamples = UInt32(int16Samples.count)
    let numChannels: UInt16 = 1
    let bitsPerSample: UInt16 = 16
    let byteRate = UInt32(sampleRate) * UInt32(numChannels) * UInt32(bitsPerSample) / 8
    let blockAlign = numChannels * bitsPerSample / 8
    let dataSize = numSamples * UInt32(blockAlign)
    let fileSize = 36 + dataSize

    // RIFF header
    data.append(contentsOf: "RIFF".utf8)
    data.append(contentsOf: withUnsafeBytes(of: fileSize.littleEndian) { Array($0) })
    data.append(contentsOf: "WAVE".utf8)

    // fmt chunk
    data.append(contentsOf: "fmt ".utf8)
    data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })  // chunk size
    data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })   // PCM format
    data.append(contentsOf: withUnsafeBytes(of: numChannels.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: byteRate.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
    data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian) { Array($0) })

    // data chunk
    data.append(contentsOf: "data".utf8)
    data.append(contentsOf: withUnsafeBytes(of: dataSize.littleEndian) { Array($0) })

    // Audio samples
    for sample in int16Samples {
        data.append(contentsOf: withUnsafeBytes(of: sample.littleEndian) { Array($0) })
    }

    try data.write(to: url)
}
