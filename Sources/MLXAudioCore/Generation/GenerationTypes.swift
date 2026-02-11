//
//  GenerationTypes.swift
//  MLXAudioCore
//
//  Common types for audio generation shared across TTS, STT, and STS models.
//

import Foundation
@preconcurrency import MLX

// MARK: - Generation Info

/// Information about the audio generation process.
public struct AudioGenerationInfo: Sendable {
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let prefillTime: TimeInterval
    public let generateTime: TimeInterval
    public let tokensPerSecond: Double
    public let peakMemoryUsage: Double

    public init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        prefillTime: TimeInterval,
        generateTime: TimeInterval,
        tokensPerSecond: Double,
        peakMemoryUsage: Double
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.prefillTime = prefillTime
        self.generateTime = generateTime
        self.tokensPerSecond = tokensPerSecond
        self.peakMemoryUsage = peakMemoryUsage
    }

    public var summary: String {
        """
        Prompt:     \(promptTokenCount) tokens, \(String(format: "%.2f", Double(promptTokenCount) / max(prefillTime, 0.001))) tokens/s, \(String(format: "%.3f", prefillTime))s
        Generation: \(generationTokenCount) tokens, \(String(format: "%.2f", tokensPerSecond)) tokens/s, \(String(format: "%.3f", generateTime))s
        Peak Memory Usage: \(peakMemoryUsage) GB
        """
    }
}

// MARK: - Generation Events

/// Events emitted during audio generation.
public enum AudioGeneration: Sendable {
    /// A generated token ID
    case token(Int)
    /// Generation statistics
    case info(AudioGenerationInfo)
    /// Final generated audio
    case audio(MLXArray)
}

// MARK: - Generation Errors

/// Errors that can occur during audio generation.
public enum AudioGenerationError: Error, LocalizedError {
    case modelNotInitialized(String)
    case generationFailed(String)
    case invalidInput(String)
    case audioDecodingFailed(String)
    case audioEncodingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotInitialized(let message):
            return "Model not initialized: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .audioDecodingFailed(let message):
            return "Audio decoding failed: \(message)"
        case .audioEncodingFailed(let message):
            return "Audio encoding failed: \(message)"
        }
    }
}

// MARK: - Token Configuration Protocol

/// Protocol for model-specific token configuration.
public protocol AudioTokenConfiguration {
    /// Token ID for start of speech
    var startOfSpeech: Int { get }
    /// Token ID for end of speech
    var endOfSpeech: Int { get }
    /// Token ID for end of text
    var endOfText: Int { get }
    /// Offset added to audio token indices
    var audioTokensStart: Int { get }
    /// Pad token ID
    var padTokenId: Int { get }
}

// MARK: - Generation Parameters

/// Parameters for controlling audio generation.
public struct AudioGenerateParameters: Sendable {
    public let maxTokens: Int
    public let temperature: Float
    public let topP: Float
    public let repetitionPenalty: Float
    public let repetitionContextSize: Int

    public init(
        maxTokens: Int = 1200,
        temperature: Float = 0.6,
        topP: Float = 0.8,
        repetitionPenalty: Float = 1.3,
        repetitionContextSize: Int = 20
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
    }
}
