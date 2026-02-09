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
import MLXAudioCore
import MLXLMCommon
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
public class Qwen3TTSModel: @unchecked Sendable {
    public let config: Qwen3TTSModelConfig
    public let talker: Qwen3TTSTalkerForConditionalGeneration
    public let speechTokenizer: Qwen3TTSSpeechTokenizer
    public var textTokenizer: Tokenizer?

    /// Speaker encoder for voice cloning (only available for "base" model type).
    public var speakerEncoder: Qwen3TTSSpeakerEncoder?

    /// Sample rate of generated audio (24kHz).
    public var sampleRate: Int { config.sampleRate }

    /// Whether voice cloning is supported (via ICL encoder or speaker encoder).
    public var supportsVoiceCloning: Bool { speechTokenizer.hasEncoder || speakerEncoder != nil }

    /// Whether ICL (In-Context Learning) voice cloning is available (higher quality).
    public var supportsICLCloning: Bool { speechTokenizer.hasEncoder }

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

        // Check if model is already cached (has config.json, safetensors, and tokenizer files)
        let configExists = FileManager.default.fileExists(atPath: modelDirectory.appendingPathComponent("config.json").path)
        let hasSafetensors = (try? FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil))?.contains { $0.pathExtension == "safetensors" } ?? false
        let hasTokenizer = FileManager.default.fileExists(atPath: modelDirectory.appendingPathComponent("tokenizer.json").path)
            || FileManager.default.fileExists(atPath: modelDirectory.appendingPathComponent("vocab.json").path)

        if configExists && hasSafetensors && hasTokenizer {
            print("Using cached model at: \(modelDirectory.path)")
        } else {
            // Download model files using HubClient
            print("Downloading model to: \(modelDirectory.path)")
            try await client.downloadSnapshot(
                of: repoID,
                kind: .model,
                to: modelDirectory,
                revision: "main",
                matching: ["*.safetensors", "*.json", "*.txt"]
            )
        }

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
        // Generate tokenizer.json if missing (Qwen3-TTS ships without it;
        // swift-transformers requires the fast tokenizer format)
        let tokenizerJsonURL = directory.appendingPathComponent("tokenizer.json")
        if !FileManager.default.fileExists(atPath: tokenizerJsonURL.path) {
            let vocabURL = directory.appendingPathComponent("vocab.json")
            let mergesURL = directory.appendingPathComponent("merges.txt")
            if FileManager.default.fileExists(atPath: vocabURL.path)
                && FileManager.default.fileExists(atPath: mergesURL.path) {
                do {
                    try generateTokenizerJson(
                        vocabPath: vocabURL,
                        mergesPath: mergesURL,
                        tokenizerConfigPath: directory.appendingPathComponent("tokenizer_config.json"),
                        outputPath: tokenizerJsonURL
                    )
                    print("Generated tokenizer.json from vocab.json + merges.txt")
                } catch {
                    print("Warning: Failed to generate tokenizer.json: \(error)")
                }
            }
        }

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
    ///   - language: Language code (e.g. "en", "zh") for language ID injection
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
        language: String? = nil
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
            instructIds: instructIds,
            language: language
        )

        // Check if any codes were generated
        if codes.shape[1] == 0 {
            throw Qwen3TTSError.noCodesGenerated
        }

        // Decode to audio
        let (audio, audioLengths) = speechTokenizer.decode(codes)
        eval(audio, audioLengths)

        // Return first batch item, trimmed to valid length
        let totalSamples = audio.shape[1]
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
    ///   - language: Optional language code (e.g. "en", "zh") for language ID injection
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
        language: String? = nil,
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
        let (initialEmbeds, trailingTextHidden, ttsPadEmbed) = talker.prepareGenerationInputs(inputIds: inputIds, speaker: speaker, speakerEmbedding: speakerEmbedding, instructIds: instructIds, language: language)
        eval(initialEmbeds)
        eval(trailingTextHidden)
        eval(ttsPadEmbed)

        // Run shared generation loop
        let generatedCodes = try runGenerationLoop(
            initialEmbeds: initialEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            maxTokens: effectiveMaxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            onProgress: onProgress
        )

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

    // MARK: - Shared Generation Loop

    /// Run the token generation loop shared by both `generate()` and `generateICL()`.
    ///
    /// Initializes the talker cache, runs the forward-pass/sample/codebook loop until
    /// EOS or `maxTokens`, and returns the generated code arrays.
    private func runGenerationLoop(
        initialEmbeds: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbed: MLXArray,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        onProgress: ((Int, Int) -> Void)?
    ) throws -> [MLXArray] {
        let talkerCache = talker.makeCache()
        var inputEmbeds = initialEmbeds
        var generatedCodes: [MLXArray] = []
        var generatedTokens: [Int] = []
        var trailingIdx = 0

        let eosTokenId = talker.config.codecEosTokenId
        let vocabSize = talker.config.vocabSize
        let suppressTokens = Array((vocabSize - 1024)..<vocabSize).filter { $0 != eosTokenId }

        for step in 0..<maxTokens {
            let (logits, hiddenStates) = talker(inputEmbeds, cache: talkerCache)
            eval(logits)
            eval(hiddenStates)

            let nextToken = talker.sampleToken(
                logits: logits,
                temperature: temperature,
                topK: topK,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                generatedTokens: generatedTokens,
                suppressTokens: suppressTokens,
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
            eval(allCodes)
            generatedCodes.append(allCodes)

            onProgress?(step + 1, generatedCodes.count)

            // NOTE: periodic Memory.clearCache() removed — it corrupts generation
            // on iOS by interfering with Metal buffer reuse around step 25.
            // With eval() at each step, memory growth is bounded without clearing.

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
                codecEmbed = codecEmbed + talker.codePredictor.codecEmbedding[i](code.asType(.int32))
            }

            inputEmbeds = textEmbed + codecEmbed
            eval(inputEmbeds)
        }

        Memory.clearCache()

        if generatedCodes.isEmpty {
            throw Qwen3TTSError.noCodesGenerated
        }

        return generatedCodes
    }

    // MARK: - ICL Voice Cloning

    /// Prepare inputs for ICL (In-Context Learning) voice cloning.
    ///
    /// Encodes reference audio through the speech tokenizer encoder, builds combined
    /// text+codec embeddings for both reference and target, assembles the full prefill.
    ///
    /// - Parameters:
    ///   - text: Target text to synthesize
    ///   - refAudio: Reference audio waveform [samples] at 24kHz
    ///   - refText: Transcript of the reference audio
    ///   - language: Language code (e.g. "en", "zh")
    /// - Returns: (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodes)
    private func prepareICLGenerationInputs(
        text: String,
        refAudio: MLXArray,
        refText: String,
        language: String? = nil
    ) throws -> (inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray, refCodes: MLXArray) {
        guard let tokenizer = textTokenizer else {
            throw Qwen3TTSError.tokenizerNotLoaded
        }

        let talkerConfig = talker.config

        // 1. Encode reference audio -> ref_codes [1, 16, ref_time]
        var audioForEncoder = refAudio
        if refAudio.ndim == 1 {
            audioForEncoder = refAudio.reshaped([1, 1, -1])  // [1, 1, samples]
        } else if refAudio.ndim == 2 {
            audioForEncoder = refAudio.reshaped([1, refAudio.shape[0], refAudio.shape[1]])
        }
        let refCodes = try speechTokenizer.encode(audioForEncoder)  // [1, 16, ref_time]
        eval(refCodes)

        // 2. Tokenize ref_text and target_text separately
        let refChat = "<|im_start|>assistant\n\(refText)<|im_end|>\n"
        let refIds = MLXArray(tokenizer.encode(text: refChat).map { Int32($0) }).reshaped([1, -1])
        // Pure ref text tokens: skip first 3 (role) and last 2 (<|im_end|>\n)
        let refTextIds = refIds[0..., 3..<(refIds.shape[1] - 2)]

        let targetChat = "<|im_start|>assistant\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        let targetIds = MLXArray(tokenizer.encode(text: targetChat).map { Int32($0) }).reshaped([1, -1])
        // Pure target text tokens: skip first 3 (role) and last 5 (trailing template)
        let textIds = targetIds[0..., 3..<(targetIds.shape[1] - 5)]

        // 3. TTS special tokens
        let ttsTokens = MLXArray([
            Int32(talkerConfig.ttsBosTokenId),
            Int32(talkerConfig.ttsEosTokenId),
            Int32(talkerConfig.ttsPadTokenId),
        ]).reshaped([1, 3])
        let ttsEmbeds = talker.textProjection(talker.model.textEmbedding(ttsTokens))
        let ttsBosEmbed = ttsEmbeds[0..., 0..<1, 0...]
        let ttsEosEmbed = ttsEmbeds[0..., 1..<2, 0...]
        let ttsPadEmbed = ttsEmbeds[0..., 2..<3, 0...]

        // 4. Build text_embed: text_projection(text_embeddings(ref_tokens + target_tokens)) + eos
        let combinedTextIds = concatenated([refTextIds, textIds], axis: 1)
        var textEmbed = talker.textProjection(talker.model.textEmbedding(combinedTextIds))
        textEmbed = concatenated([textEmbed, ttsEosEmbed], axis: 1)
        let textLens = textEmbed.shape[1]

        // 5. Build codec_embed: codec_bos + sum_of_all_codebook_embeddings(ref_codes)
        let firstCbCodes = refCodes[0..., 0, 0...]  // [1, ref_time]
        var refCodecEmbed = talker.model.codecEmbedding(firstCbCodes.asType(.int32))
        for i in 0..<(talkerConfig.numCodeGroups - 1) {
            let cbCodes = refCodes[0..., i + 1, 0...]  // [1, ref_time]
            refCodecEmbed = refCodecEmbed + talker.codePredictor.codecEmbedding[i](cbCodes.asType(.int32))
        }

        // Prepend codec_bos
        let codecBosEmbed = talker.model.codecEmbedding(
            MLXArray([Int32(talkerConfig.codecBosId)]).reshaped([1, 1])
        )
        let codecEmbedICL = concatenated([codecBosEmbed, refCodecEmbed], axis: 1)
        let codecLens = codecEmbedICL.shape[1]

        // 6. Non-streaming mode overlay
        // All text first (overlaid with codec_pad), then all codec (overlaid with tts_pad)
        let codecPadEmbed = talker.model.codecEmbedding(
            MLXArray([Int32(talkerConfig.codecPadId)]).reshaped([1, 1])
        )
        let hiddenDim = textEmbed.shape[2]
        let textWithCodecPad = textEmbed + broadcast(codecPadEmbed, to: [1, textLens, hiddenDim])
        let codecWithTextPad = codecEmbedICL + broadcast(ttsPadEmbed, to: [1, codecLens, hiddenDim])
        let iclInputEmbed = concatenated([textWithCodecPad, codecWithTextPad], axis: 1)

        // In non-streaming mode, trailing_text_hidden is just tts_pad_embed
        let trailingTextHidden = ttsPadEmbed

        // 7. Language ID
        var languageId: Int? = nil
        if let lang = language?.lowercased(),
           let langMap = talkerConfig.codecLanguageId,
           let langId = langMap[lang] {
            languageId = langId
        }

        // 8. Speaker embedding (ICL still uses x-vector for additional conditioning)
        var speakerEmbed: MLXArray? = nil
        if speakerEncoder != nil {
            speakerEmbed = try extractSpeakerEmbedding(audio: refAudio)
        }

        // 9. Build codec prefix (think/nothink + speaker + pad + bos)
        var codecPrefill: [Int32]
        if let langId = languageId {
            codecPrefill = [
                Int32(talkerConfig.codecThinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(langId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        } else {
            codecPrefill = [
                Int32(talkerConfig.codecNothinkId),
                Int32(talkerConfig.codecThinkBosId),
                Int32(talkerConfig.codecThinkEosId),
            ]
        }

        var codecPrefixEmbed = talker.model.codecEmbedding(
            MLXArray(codecPrefill).reshaped([1, codecPrefill.count])
        )
        let codecPrefixSuffix = talker.model.codecEmbedding(
            MLXArray([Int32(talkerConfig.codecPadId), Int32(talkerConfig.codecBosId)]).reshaped([1, 2])
        )

        if let spkEmbed = speakerEmbed {
            codecPrefixEmbed = concatenated([
                codecPrefixEmbed,
                spkEmbed.reshaped([1, 1, -1]),
                codecPrefixSuffix,
            ], axis: 1)
        } else {
            codecPrefixEmbed = concatenated([codecPrefixEmbed, codecPrefixSuffix], axis: 1)
        }

        // 10. Role embedding (first 3 tokens: <|im_start|>assistant\n)
        let roleEmbed = talker.textProjection(
            talker.model.textEmbedding(targetIds[0..., 0..<3])
        )

        // 11. Build pad/bos prefix (text side overlaid with codec prefix[:-1])
        let padCount = codecPrefixEmbed.shape[1] - 2
        let padEmbeds = broadcast(ttsPadEmbed, to: [1, padCount, hiddenDim])
        var combinedPrefix = concatenated([padEmbeds, ttsBosEmbed], axis: 1)
        combinedPrefix = combinedPrefix + codecPrefixEmbed[0..., 0..<(codecPrefixEmbed.shape[1] - 1), 0...]

        // 12. Full input_embeds: role + codec_prefix + icl_embed
        let inputEmbeds = concatenated([roleEmbed, combinedPrefix, iclInputEmbed], axis: 1)

        return (inputEmbeds, trailingTextHidden, ttsPadEmbed, refCodes)
    }

    /// Generate speech using ICL (In-Context Learning) voice cloning.
    ///
    /// Encodes reference audio through the speech tokenizer encoder, generates codes
    /// with reference context, then prepends ref codes for decoding and trims.
    ///
    /// - Parameters:
    ///   - text: Target text to synthesize
    ///   - refAudio: Reference audio [samples] at 24kHz
    ///   - refText: Transcript of the reference audio
    ///   - language: Language code
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k sampling
    ///   - topP: Top-p sampling
    ///   - maxTokens: Maximum tokens to generate
    ///   - repetitionPenalty: Repetition penalty
    ///   - onProgress: Optional progress callback (step, totalCodes)
    /// - Returns: Audio waveform [samples] at 24kHz
    public func generateICL(
        text: String,
        refAudio: MLXArray,
        refText: String,
        language: String? = nil,
        temperature: Float = 0.9,
        topK: Int = 50,
        topP: Float = 1.0,
        maxTokens: Int = 4096,
        repetitionPenalty: Float = 1.5,
        onProgress: ((Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        guard let tokenizer = textTokenizer else {
            throw Qwen3TTSError.tokenizerNotLoaded
        }

        // Prepare ICL inputs
        let (initialEmbeds, trailingTextHidden, ttsPadEmbed, refCodes) =
            try prepareICLGenerationInputs(
                text: text,
                refAudio: refAudio,
                refText: refText,
                language: language
            )

        // Cap max_tokens based on TARGET text length only (matching Python mlx-audio).
        // With a tight cap, the model focuses on generating target text codes
        // rather than regenerating the full ref text, making simple proportional
        // trimming sufficient.
        let targetTokenCount = tokenizer.encode(text: text).count
        let effectiveMaxTokens = min(maxTokens, max(75, targetTokenCount * 6))

        // Materialize inputs before generation loop (prevents lazy graph buildup on iOS)
        eval(initialEmbeds)
        eval(trailingTextHidden)
        eval(ttsPadEmbed)

        // Run shared generation loop
        let generatedCodes = try runGenerationLoop(
            initialEmbeds: initialEmbeds,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbed: ttsPadEmbed,
            maxTokens: effectiveMaxTokens,
            temperature: temperature,
            topK: topK,
            topP: topP,
            repetitionPenalty: repetitionPenalty,
            onProgress: onProgress
        )

        // Stack generated codes: [1, gen_len, num_code_groups]
        let genCodes = stacked(generatedCodes, axis: 1).asType(.int32)

        // Prepend reference codes to generated codes for decoding.
        // The neural decoder uses context across the full sequence, producing
        // a more natural transition at the boundary (matches official implementation).
        // refCodes: [1, 16, ref_time] -> [1, ref_time, 16]
        let refCodesT = refCodes.transposed(0, 2, 1)
        let refLen = refCodes.shape[2]
        let fullCodes = concatenated([refCodesT, genCodes], axis: 1)
        let totalLen = fullCodes.shape[1]

        let (audio, audioLengths) = speechTokenizer.decode(fullCodes)
        eval(audio, audioLengths)

        var result = audio[0]  // Remove batch dim

        // Trim to valid length
        let validLen = audioLengths[0].item(Int.self)
        if validLen > 0 && validLen < result.shape[0] {
            result = result[0..<validLen]
        }

        // Remove the prepended reference audio using proportional trimming
        // (matches Python mlx-audio / official implementation).
        // With target-only max_tokens cap, the model generates predominantly
        // target text codes, so simple proportional trim is sufficient.
        let totalSamples = result.shape[0]
        let cut = Int(Double(refLen) / Double(max(totalLen, 1)) * Double(totalSamples))
        if cut > 0 && cut < totalSamples {
            result = result[cut..<totalSamples]
        }

        eval(result)
        return result
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

// MARK: - Tokenizer Generation

/// Generate a fast tokenizer.json from vocab.json + merges.txt.
///
/// Qwen3-TTS repos ship with a slow tokenizer (vocab.json + merges.txt) but
/// swift-transformers requires tokenizer.json (fast tokenizer format).
private func generateTokenizerJson(
    vocabPath: URL,
    mergesPath: URL,
    tokenizerConfigPath: URL,
    outputPath: URL
) throws {
    // Read vocab
    let vocabData = try Data(contentsOf: vocabPath)
    let vocabDict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] ?? [:]

    // Read merges (skip header lines starting with #)
    let mergesText = try String(contentsOf: mergesPath, encoding: .utf8)
    let mergeLines = mergesText.components(separatedBy: .newlines)
        .filter { !$0.isEmpty && !$0.hasPrefix("#") }

    // Read added_tokens from tokenizer_config.json
    var addedTokens = [[String: Any]]()
    if let configData = try? Data(contentsOf: tokenizerConfigPath),
       let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
       let addedTokensDecoder = configDict["added_tokens_decoder"] as? [String: [String: Any]] {
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

    // Build tokenizer.json
    // Qwen2 uses ByteLevel BPE with a GPT-2-style regex pre-tokenizer
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
                        "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
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

// MARK: - SpeechGenerationModel Conformance

extension Qwen3TTSModel: SpeechGenerationModel {

    /// Alias for load(from:) to match other models' API
    public static func fromPretrained(_ modelRepo: String) async throws -> Qwen3TTSModel {
        try await load(from: modelRepo)
    }

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        instruct: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        // ICL path: encoder available + ref audio + ref text → high-quality voice cloning
        if supportsICLCloning, let audio = refAudio, let rText = refText, !rText.isEmpty {
            return try generateICL(
                text: text,
                refAudio: audio,
                refText: rText,
                language: language,
                temperature: generationParameters.temperature,
                topK: 50,
                topP: generationParameters.topP,
                maxTokens: generationParameters.maxTokens ?? 4096,
                repetitionPenalty: generationParameters.repetitionPenalty ?? 1.5
            )
        }

        // Fallback: speaker-encoder path or no voice cloning
        // For voice cloning (Base model), refText is the reference audio transcription.
        // It goes into the user turn of the prompt, same position as instruct.
        let effectiveInstruct: String? = {
            switch (instruct, refText) {
            case let (i?, r?): return "\(r)\n\(i)"
            case let (i?, nil): return i
            case let (nil, r?): return r
            case (nil, nil): return nil
            }
        }()

        return try generate(
            text: text,
            temperature: generationParameters.temperature,
            topK: 50,
            topP: generationParameters.topP,
            maxTokens: generationParameters.maxTokens ?? 2000,
            repetitionPenalty: generationParameters.repetitionPenalty ?? 1.0,
            speaker: voice,
            refAudio: refAudio,
            instruct: effectiveInstruct,
            language: language
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        instruct: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        // Capture all parameters as local constants for Sendable compliance
        let capturedText = text
        let capturedVoice = voice
        let capturedRefAudio = refAudio
        let capturedRefText = refText
        let capturedLanguage = language
        let capturedInstruct = instruct
        let capturedTemp = generationParameters.temperature
        let capturedTopP = generationParameters.topP
        let capturedMaxTokens = generationParameters.maxTokens ?? 2000
        let capturedRepPenalty = generationParameters.repetitionPenalty ?? 1.0
        let useICL = supportsICLCloning && refAudio != nil && refText != nil && !(refText?.isEmpty ?? true)

        return AsyncThrowingStream { continuation in
            let model = self
            continuation.onTermination = { _ in }

            let startTime = Date()
            var tokenCount = 0

            do {
                let audio: MLXArray

                if useICL, let rAudio = capturedRefAudio, let rText = capturedRefText {
                    // ICL path: high-quality voice cloning
                    audio = try model.generateICL(
                        text: capturedText,
                        refAudio: rAudio,
                        refText: rText,
                        language: capturedLanguage,
                        temperature: capturedTemp,
                        topK: 50,
                        topP: capturedTopP,
                        maxTokens: capturedMaxTokens,
                        repetitionPenalty: capturedRepPenalty
                    ) { step, totalCodes in
                        tokenCount = step
                        continuation.yield(.token(step))
                    }
                } else {
                    // Fallback: speaker-encoder path
                    let effectiveInstruct: String? = {
                        switch (capturedInstruct, capturedRefText) {
                        case let (i?, r?): return "\(r)\n\(i)"
                        case let (i?, nil): return i
                        case let (nil, r?): return r
                        case (nil, nil): return nil
                        }
                    }()

                    audio = try model.generate(
                        text: capturedText,
                        temperature: capturedTemp,
                        topK: 50,
                        topP: capturedTopP,
                        maxTokens: capturedMaxTokens,
                        repetitionPenalty: capturedRepPenalty,
                        speaker: capturedVoice,
                        refAudio: capturedRefAudio,
                        instruct: effectiveInstruct,
                        language: capturedLanguage
                    ) { step, totalCodes in
                        tokenCount = step
                        continuation.yield(.token(step))
                    }
                }

                let elapsed = Date().timeIntervalSince(startTime)
                let tps = elapsed > 0 ? Double(tokenCount) / elapsed : 0

                continuation.yield(.info(AudioGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: tokenCount,
                    prefillTime: 0,
                    generateTime: elapsed,
                    tokensPerSecond: tps,
                    peakMemoryUsage: 0
                )))
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }
}
