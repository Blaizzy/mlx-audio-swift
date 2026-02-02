//
//  STTTypes.swift
//  MLXAudioSTT
//
//  Common types for Speech-to-Text module.
//  API design aligned with MLX-Audio Swift SDK v1.
//

import Foundation

// MARK: - STT Generation Events

/// Events emitted during speech-to-text streaming generation.
public enum STTGeneration: Sendable {
    /// A generated text token during transcription
    case token(String)
    /// Generation statistics
    case info(STTGenerationInfo)
    /// Final transcription result
    case result(STTOutput)
}

/// Information about the STT generation process.
public struct STTGenerationInfo: Sendable {
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
        Peak Memory Usage: \(String(format: "%.2f", peakMemoryUsage)) GB
        """
    }
}

// MARK: - STT Output

/// Output from speech-to-text transcription.
public struct STTOutput: Sendable {
    /// The transcribed text.
    public let text: String

    /// Transcription segments with timing information.
    public let segments: [STTSegment]

    /// Detected language.
    public let language: String?

    /// Number of tokens in the prompt (audio encoding).
    public let promptTokens: Int

    /// Number of tokens generated.
    public let generationTokens: Int

    /// Total number of tokens processed.
    public let totalTokens: Int

    /// Prompt processing tokens per second.
    public let promptTps: Double

    /// Generation tokens per second.
    public let generationTps: Double

    /// Total processing time in seconds.
    public let totalTime: Double

    /// Peak memory usage in GB.
    public let peakMemoryUsage: Double

    public init(
        text: String,
        segments: [STTSegment] = [],
        language: String? = nil,
        promptTokens: Int = 0,
        generationTokens: Int = 0,
        totalTokens: Int = 0,
        promptTps: Double = 0.0,
        generationTps: Double = 0.0,
        totalTime: Double = 0.0,
        peakMemoryUsage: Double = 0.0
    ) {
        self.text = text
        self.segments = segments
        self.language = language
        self.promptTokens = promptTokens
        self.generationTokens = generationTokens
        self.totalTokens = totalTokens
        self.promptTps = promptTps
        self.generationTps = generationTps
        self.totalTime = totalTime
        self.peakMemoryUsage = peakMemoryUsage
    }
}

extension STTOutput: CustomStringConvertible {
    public var description: String {
        var result = "STTOutput:\n"
        result += "  text: \(text)\n"
        if let language = language {
            result += "  language: \(language)\n"
        }
        if !segments.isEmpty {
            result += "  segments: \(segments.count)\n"
        }
        result += "  prompt_tokens: \(promptTokens)\n"
        result += "  generation_tokens: \(generationTokens)\n"
        result += "  total_tokens: \(totalTokens)\n"
        result += "  prompt_tps: \(String(format: "%.2f", promptTps))\n"
        result += "  generation_tps: \(String(format: "%.2f", generationTps))\n"
        result += "  total_time: \(String(format: "%.2f", totalTime))s\n"
        result += "  peak_memory_usage: \(String(format: "%.2f", peakMemoryUsage)) GB"
        return result
    }
}

// MARK: - STT Segment

/// A transcription segment with timing information.
public struct STTSegment: Sendable, Codable {
    /// Segment text.
    public let text: String

    /// Start time in seconds.
    public let start: Double

    /// End time in seconds.
    public let end: Double

    /// Confidence score (0-1).
    public let confidence: Double?

    public init(text: String, start: Double, end: Double, confidence: Double? = nil) {
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
    }

    /// Time range as ClosedRange.
    public var timeRange: ClosedRange<TimeInterval> {
        start...end
    }
}

// MARK: - STT Error

/// Errors that can occur during STT generation.
public enum STTError: Error, LocalizedError, Sendable {
    case modelNotInitialized(String)
    case generationFailed(String)
    case invalidInput(String)
    case audioProcessingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotInitialized(let message):
            return "Model not initialized: \(message)"
        case .generationFailed(let message):
            return "Generation failed: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .audioProcessingFailed(let message):
            return "Audio processing failed: \(message)"
        }
    }
}
