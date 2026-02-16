//
//  StreamingTypes.swift
//  MLXAudioSTT
//
//  Created by Prince Canuma on 07/02/2026.
//

import Foundation

// MARK: - Delay Presets

/// Controls the tradeoff between latency and accuracy for streaming transcription.
public enum DelayPreset: Sendable {
    /// ~200ms delay — fastest feedback, may have more provisional corrections
    case realtime
    /// ~480ms delay — balanced for voice agent use cases
    case agent
    /// ~2400ms delay — higher accuracy, suitable for subtitles
    case subtitle
    /// Custom delay in milliseconds
    case custom(ms: Int)

    public var delayMs: Int {
        switch self {
        case .realtime: return 200
        case .agent: return 480
        case .subtitle: return 2400
        case .custom(let ms): return ms
        }
    }
}

// MARK: - Streaming Configuration

/// Configuration for a streaming inference session.
public struct StreamingConfig: Sendable {
    /// How often to run decode passes (seconds)
    public var decodeIntervalSeconds: Double
    /// Faster decode interval used briefly after an 8s window boundary
    public var boundaryDecodeIntervalSeconds: Double
    /// Duration to keep boundary fast cadence active (seconds)
    public var boundaryBoostSeconds: Double
    /// Overlap duration between consecutive 8s encoder windows
    public var encoderWindowOverlapSeconds: Double
    /// Maximum number of cached encoder windows (~8s each)
    public var maxCachedWindows: Int
    /// Delay preset controlling provisional → confirmed promotion
    public var delayPreset: DelayPreset
    /// Language for transcription
    public var language: String
    /// Sampling temperature (0 = greedy)
    public var temperature: Float
    /// Maximum tokens per decode pass
    public var maxTokensPerPass: Int
    /// Minimum consecutive matching passes before provisional tokens can promote
    public var minAgreementPasses: Int
    /// Stronger agreement threshold while boundary boost is active
    public var boundaryMinAgreementPasses: Int
    /// Maximum encoder windows visible to the decoder per pass (~8s each)
    public var maxDecodeWindows: Int
    /// Whether to run a one-shot decode on each completed 8s window for accuracy
    public var finalizeCompletedWindows: Bool

    public init(
        decodeIntervalSeconds: Double = 1.0,
        boundaryDecodeIntervalSeconds: Double = 0.2,
        boundaryBoostSeconds: Double = 1.0,
        encoderWindowOverlapSeconds: Double = 1.0,
        maxCachedWindows: Int = 60,
        delayPreset: DelayPreset = .agent,
        language: String = "English",
        temperature: Float = 0.0,
        maxTokensPerPass: Int = 512,
        minAgreementPasses: Int = 2,
        boundaryMinAgreementPasses: Int = 3,
        maxDecodeWindows: Int = 1,
        finalizeCompletedWindows: Bool = true
    ) {
        self.decodeIntervalSeconds = decodeIntervalSeconds
        self.boundaryDecodeIntervalSeconds = boundaryDecodeIntervalSeconds
        self.boundaryBoostSeconds = boundaryBoostSeconds
        self.encoderWindowOverlapSeconds = encoderWindowOverlapSeconds
        self.maxCachedWindows = maxCachedWindows
        self.delayPreset = delayPreset
        self.language = language
        self.temperature = temperature
        self.maxTokensPerPass = maxTokensPerPass
        self.minAgreementPasses = minAgreementPasses
        self.boundaryMinAgreementPasses = boundaryMinAgreementPasses
        self.maxDecodeWindows = maxDecodeWindows
        self.finalizeCompletedWindows = finalizeCompletedWindows
    }
}

// MARK: - Transcription Events

/// Events emitted by a streaming inference session.
public enum TranscriptionEvent: Sendable {
    /// Provisional text that may still change
    case provisional(text: String)
    /// Text that has been confirmed and will not change
    case confirmed(text: String)
    /// Combined display update with both confirmed and provisional text
    case displayUpdate(confirmedText: String, provisionalText: String)
    /// Performance statistics
    case stats(StreamingStats)
    /// Session has ended with final text
    case ended(fullText: String)
}

// MARK: - Streaming Stats

/// Performance statistics for a streaming session.
public struct StreamingStats: Sendable {
    /// Number of encoder windows processed
    public var encodedWindowCount: Int
    /// Total audio duration processed so far (seconds)
    public var totalAudioSeconds: Double
    /// Tokens generated per second
    public var tokensPerSecond: Double
    /// Real-time factor (< 1.0 means faster than real-time)
    public var realTimeFactor: Double
    /// Peak memory usage in GB
    public var peakMemoryGB: Double

    public init(
        encodedWindowCount: Int = 0,
        totalAudioSeconds: Double = 0,
        tokensPerSecond: Double = 0,
        realTimeFactor: Double = 0,
        peakMemoryGB: Double = 0
    ) {
        self.encodedWindowCount = encodedWindowCount
        self.totalAudioSeconds = totalAudioSeconds
        self.tokensPerSecond = tokensPerSecond
        self.realTimeFactor = realTimeFactor
        self.peakMemoryGB = peakMemoryGB
    }
}
