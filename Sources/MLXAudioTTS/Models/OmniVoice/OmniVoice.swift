import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - OmniVoice Model

/// OmniVoice: A massively multilingual zero-shot TTS model supporting over 600 languages.
///
/// Built on a novel diffusion language model architecture with a Qwen3 LLM backbone,
/// OmniVoice supports:
/// - **Voice Cloning**: Clone any voice from a reference audio + transcript
/// - **Voice Design**: Create custom voices via text instructions
/// - **Auto Voice**: Default voice when no voice specification is provided
///
/// The model uses 8 audio codebooks with hierarchical weighting and produces 24kHz audio.
public final class OmniVoiceModel: Module, SpeechGenerationModel, @unchecked Sendable {
    // MARK: - Properties

    let config: OmniVoiceConfig
    var tokenizer: Tokenizer?

    /// Audio tokenizer for encoding/decoding audio tokens
    var audioTokenizer: OmniVoiceAudioTokenizer?

    public var sampleRate: Int { config.sampleRate }

    /// Default generation parameters for OmniVoice
    public var defaultGenerationParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: 4096,
            temperature: 1.0,
            topP: 0.95,
            repetitionPenalty: 1.05
        )
    }

    // MARK: - OmniVoice-specific parameters

    /// Number of diffusion steps (default: 32)
    private var numStep: Int = 32

    /// Classifier-free guidance scale (default: 2.0)
    private var guidanceScale: Float = 2.0

    /// Speech speed factor (default: 1.0)
    private var speed: Float = 1.0

    /// Fixed output duration in seconds (optional, overrides duration estimation)
    private var duration: Float?

    /// Time shift parameter for diffusion (default: 0.1)
    private var tShift: Float = 0.1

    /// Whether to denoise output (default: true)
    private var denoise: Bool = true

    /// Whether to postprocess output audio (default: true)
    private var postprocessOutput: Bool = true

    /// Layer penalty factor for diffusion (default: 5.0)
    private var layerPenaltyFactor: Float = 5.0

    /// Position temperature for codebook sampling (default: 5.0)
    private var positionTemperature: Float = 5.0

    /// Class temperature for codebook sampling (default: 0.0)
    private var classTemperature: Float = 0.0

    // MARK: - Initialization

    init(config: OmniVoiceConfig) {
        self.config = config
    }

    // MARK: - SpeechGenerationModel protocol

    public func generate(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        guard tokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
        }
        guard audioTokenizer != nil else {
            throw AudioGenerationError.modelNotInitialized("Audio tokenizer not loaded")
        }

        return try await generateAudio(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        generateStream(
            text: text,
            voice: voice,
            refAudio: refAudio,
            refText: refText,
            language: language,
            generationParameters: generationParameters,
            streamingInterval: 2.0
        )
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters,
        streamingInterval: Double
    ) -> AsyncThrowingStream<AudioGeneration, Error> {
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        Task { @Sendable [weak self] in
            guard let self else { return }
            do {
                guard tokenizer != nil else {
                    throw AudioGenerationError.modelNotInitialized("Tokenizer not loaded")
                }
                guard audioTokenizer != nil else {
                    throw AudioGenerationError.modelNotInitialized("Audio tokenizer not loaded")
                }

                let audio = try await generateAudio(
                    text: text,
                    voice: voice,
                    refAudio: refAudio,
                    refText: refText,
                    language: language,
                    generationParameters: generationParameters
                )

                // Emit info
                let info = AudioGenerationInfo(
                    promptTokenCount: 0,
                    generationTokenCount: 0,
                    prefillTime: 0,
                    generateTime: 0,
                    tokensPerSecond: 0,
                    peakMemoryUsage: Double(Memory.peakMemory) / 1e9
                )
                continuation.yield(.info(info))
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        return stream
    }

    // MARK: - Generation

    /// Set OmniVoice-specific generation parameters
    public func setGenerationConfig(
        numStep: Int = 32,
        guidanceScale: Float = 2.0,
        speed: Float = 1.0,
        duration: Float? = nil,
        tShift: Float = 0.1,
        denoise: Bool = true,
        postprocessOutput: Bool = true,
        layerPenaltyFactor: Float = 5.0,
        positionTemperature: Float = 5.0,
        classTemperature: Float = 0.0
    ) {
        self.numStep = numStep
        self.guidanceScale = guidanceScale
        self.speed = speed
        self.duration = duration
        self.tShift = tShift
        self.denoise = denoise
        self.postprocessOutput = postprocessOutput
        self.layerPenaltyFactor = layerPenaltyFactor
        self.positionTemperature = positionTemperature
        self.classTemperature = classTemperature
    }

    private func generateAudio(
        text: String,
        voice: String?,
        refAudio: MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> MLXArray {
        // TODO: Implement full diffusion-based generation
        // For now, return placeholder audio to verify the pipeline works
        print("OmniVoice generateAudio called with:")
        print("  text: \(text)")
        print("  voice: \(voice ?? "nil")")
        print("  refAudio: \(refAudio != nil ? "yes" : "no")")
        print("  refText: \(refText ?? "nil")")
        print("  language: \(language ?? "nil")")
        print("  numStep: \(self.numStep)")
        print("  guidanceScale: \(self.guidanceScale)")
        print("  speed: \(self.speed)")

        // Generate placeholder audio (2 seconds at 24kHz)
        let sampleCount = sampleRate * 2
        let audio = MLXRandom.normal([sampleCount]) * 0.3
        eval(audio)
        return audio
    }

    // MARK: - Model Loading

    /// Load OmniVoice model from pretrained weights
    public static func fromPretrained(
        _ repoID: String,
        cache: HubCache = .default
    ) async throws -> OmniVoiceModel {
        guard let repo = Repo.ID(rawValue: repoID) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(repoID)")
        }

        // Download and parse config
        let configURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: "json",
            additionalMatchingPatterns: ["config.json"]
        ).appendingPathComponent("config.json")

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(OmniVoiceConfig.self, from: configData)

        let model = OmniVoiceModel(config: config)

        // TODO: Load tokenizer
        // model.tokenizer = try await loadTokenizer(repoID: repoID, cache: cache)

        // TODO: Load audio tokenizer
        // model.audioTokenizer = try await loadAudioTokenizer(repoID: repoID, cache: cache)

        return model
    }
}

// MARK: - OmniVoice Audio Tokenizer

/// Audio tokenizer for OmniVoice - handles encoding/decoding of audio tokens
public final class OmniVoiceAudioTokenizer: Module {
    let config: OmniVoiceAudioTokenizerConfig

    // Acoustic model (DAC-based)
    var acousticModel: OmniVoiceDACModel?

    // Semantic model (Hubert-based)
    var semanticModel: OmniVoiceHubertModel?

    init(config: OmniVoiceAudioTokenizerConfig) {
        self.config = config
    }

    /// Encode audio waveform to discrete tokens
    public func encode(_ audio: MLXArray) throws -> MLXArray {
        // TODO: Implement encoding
        return MLXArray.zeros([1, 100])
    }

    /// Decode discrete tokens back to audio waveform
    public func decode(_ tokens: MLXArray) throws -> MLXArray {
        // TODO: Implement decoding
        return MLXArray.zeros([tokens.dim(0) * 960])
    }

    public static func fromPretrained(
        repoID: String,
        cache: HubCache = .default
    ) async throws -> OmniVoiceAudioTokenizer {
        guard let repo = Repo.ID(rawValue: repoID) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(repoID)")
        }

        // Load config
        let configURL = try await ModelUtils.resolveOrDownloadModel(
            repoID: repo,
            requiredExtension: "json",
            additionalMatchingPatterns: ["audio_tokenizer/config.json"]
        ).appendingPathComponent("audio_tokenizer/config.json")

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(OmniVoiceAudioTokenizerConfig.self, from: configData)

        let tokenizer = OmniVoiceAudioTokenizer(config: config)
        // TODO: Load acoustic and semantic models
        return tokenizer
    }
}

// MARK: - OmniVoice DAC Model (Acoustic)

/// DAC-based acoustic model for audio tokenization
public final class OmniVoiceDACModel: Module {
    let config: OmniVoiceAudioTokenizerConfig

    init(config: OmniVoiceAudioTokenizerConfig) {
        self.config = config
    }

    static func fromPretrained(
        repoID: String,
        config: OmniVoiceAudioTokenizerConfig,
        cache: HubCache
    ) async throws -> OmniVoiceDACModel {
        return OmniVoiceDACModel(config: config)
    }
}

// MARK: - OmniVoice Hubert Model (Semantic)

/// Hubert-based semantic model for audio tokenization
public final class OmniVoiceHubertModel: Module {
    let config: OmniVoiceAudioTokenizerConfig

    init(config: OmniVoiceAudioTokenizerConfig) {
        self.config = config
    }

    static func fromPretrained(
        repoID: String,
        config: OmniVoiceAudioTokenizerConfig,
        cache: HubCache
    ) async throws -> OmniVoiceHubertModel {
        return OmniVoiceHubertModel(config: config)
    }
}
