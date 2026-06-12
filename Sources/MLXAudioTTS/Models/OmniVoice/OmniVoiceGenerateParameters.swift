import Foundation

/// Generation parameters specific to OmniVoice diffusion-based TTS.
///
/// These parameters control the diffusion process, voice characteristics,
/// and output quality. They supplement the standard `GenerateParameters`
/// (temperature, topP, maxTokens) from MLXLMCommon.
public struct OmniVoiceGenerateParameters: Sendable {
    // MARK: - Diffusion Parameters

    /// Number of diffusion steps (default: 32).
    /// Higher values produce better quality but slower generation.
    /// Range: 8-64, typical: 16-32
    public var numStep: Int

    /// Classifier-free guidance scale (default: 2.0).
    /// Controls how strongly the model follows the text prompt.
    /// Higher values = more faithful to prompt but potentially less natural.
    /// Range: 1.0-5.0, typical: 1.5-3.0
    public var guidanceScale: Float

    // MARK: - Speech Characteristics

    /// Speech speed factor (default: 1.0).
    /// Values > 1.0 produce faster speech, < 1.0 produces slower speech.
    /// Range: 0.5-2.0, typical: 0.8-1.2
    public var speed: Float

    /// Fixed output duration in seconds (optional).
    /// If set, overrides the model's duration estimation.
    /// The speed factor is automatically adjusted to match while preserving language-aware pacing.
    public var duration: Float?

    /// Time shift parameter for diffusion (default: 0.1).
    /// Controls the temporal dynamics of the diffusion process.
    /// Lower values = smoother transitions, higher values = more variation.
    public var tShift: Float

    // MARK: - Output Processing

    /// Whether to denoise output audio (default: true).
    /// Applies normalization and noise reduction to the generated audio.
    public var denoise: Bool

    /// Whether to postprocess output audio (default: true).
    /// Applies final quality improvements to the generated waveform.
    public var postprocessOutput: Bool

    // MARK: - Advanced Parameters

    /// Layer penalty factor for diffusion (default: 5.0).
    /// Controls the influence of different network layers on the output.
    /// Higher values emphasize deeper layer representations.
    public var layerPenaltyFactor: Float

    /// Position temperature for codebook sampling (default: 5.0).
    /// Controls the randomness of positional aspects in generation.
    /// Higher values = more variation in timing/prosody.
    public var positionTemperature: Float

    /// Class temperature for codebook sampling (default: 0.0).
    /// Controls the randomness of phoneme/class selection.
    /// Lower values = more deterministic, higher values = more variation.
    public var classTemperature: Float

    // MARK: - Initialization

    public init(
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

    /// Default parameters optimized for fast generation.
    public static var fast: OmniVoiceGenerateParameters {
        OmniVoiceGenerateParameters(
            numStep: 16,
            guidanceScale: 1.5,
            speed: 1.0,
            tShift: 0.1,
            denoise: true,
            postprocessOutput: true,
            layerPenaltyFactor: 5.0,
            positionTemperature: 5.0,
            classTemperature: 0.0
        )
    }

    /// Default parameters optimized for high quality generation.
    public static var highQuality: OmniVoiceGenerateParameters {
        OmniVoiceGenerateParameters(
            numStep: 64,
            guidanceScale: 2.5,
            speed: 1.0,
            tShift: 0.1,
            denoise: true,
            postprocessOutput: true,
            layerPenaltyFactor: 5.0,
            positionTemperature: 5.0,
            classTemperature: 0.0
        )
    }
}
