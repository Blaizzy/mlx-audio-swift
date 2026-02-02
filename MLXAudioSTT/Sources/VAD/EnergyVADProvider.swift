import Foundation
import MLX

/// Energy-based Voice Activity Detection
/// Uses RMS (Root Mean Square) energy to detect speech segments
public final class EnergyVADProvider: VADProvider, @unchecked Sendable {
    private let config: EnergyVADConfig
    private let segmentConfig: VADSegmentConfig

    public struct EnergyVADConfig: Sendable {
        /// RMS threshold for speech detection (0.0-1.0)
        public var speechThreshold: Float
        /// Window size in seconds for RMS calculation
        public var windowDuration: TimeInterval
        /// Hop size in seconds between windows
        public var hopDuration: TimeInterval
        /// Smoothing factor for energy (0=none, 1=max)
        public var smoothingFactor: Float

        public init(
            speechThreshold: Float = 0.02,
            windowDuration: TimeInterval = 0.025,
            hopDuration: TimeInterval = 0.010,
            smoothingFactor: Float = 0.9
        ) {
            self.speechThreshold = speechThreshold
            self.windowDuration = windowDuration
            self.hopDuration = hopDuration
            self.smoothingFactor = smoothingFactor
        }

        public static let `default` = EnergyVADConfig()
    }

    public init(config: EnergyVADConfig = .default, segmentConfig: VADSegmentConfig = .default) {
        self.config = config
        self.segmentConfig = segmentConfig
    }

    public func detectSpeech(in audio: MLXArray, sampleRate: Int) async throws -> [SpeechSegment] {
        let probabilities = try await speechProbabilities(in: audio, sampleRate: sampleRate)
        // Probabilities are already normalized to threshold (1.0 = at threshold level)
        return segmentFromProbabilities(probabilities, threshold: 1.0, config: segmentConfig)
    }

    public func speechProbabilities(in audio: MLXArray, sampleRate: Int) async throws -> [(time: TimeInterval, probability: Float)] {
        let windowSamples = Int(config.windowDuration * Double(sampleRate))
        let hopSamples = Int(config.hopDuration * Double(sampleRate))

        guard windowSamples > 0, hopSamples > 0 else {
            return []
        }

        let audioData = audio.asArray(Float.self)
        let numSamples = audioData.count

        guard numSamples >= windowSamples else {
            return []
        }

        var probabilities: [(time: TimeInterval, probability: Float)] = []
        var smoothedEnergy: Float = 0

        var windowStart = 0
        while windowStart + windowSamples <= numSamples {
            let windowEnd = windowStart + windowSamples
            let window = Array(audioData[windowStart..<windowEnd])

            let rms = calculateRMS(window)

            if config.smoothingFactor > 0 && !probabilities.isEmpty {
                smoothedEnergy = config.smoothingFactor * smoothedEnergy + (1 - config.smoothingFactor) * rms
            } else {
                smoothedEnergy = rms
            }

            let normalizedEnergy = min(smoothedEnergy / config.speechThreshold, 1.0)

            let time = Double(windowStart) / Double(sampleRate)
            probabilities.append((time: time, probability: normalizedEnergy))

            windowStart += hopSamples
        }

        return probabilities
    }

    public func reset() async {
        // Energy-based VAD is stateless, nothing to reset
    }

    private func calculateRMS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        let sumOfSquares = samples.reduce(0) { $0 + $1 * $1 }
        return sqrt(sumOfSquares / Float(samples.count))
    }
}
