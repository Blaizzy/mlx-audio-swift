import Foundation
import MLX
import MLXNN

/// Stateful audio processor for streaming VAD.
/// NOT thread-safe - use one per stream/task.
public final class VADIterator {
    private let model: SileroVADModel
    private let config: VADConfig
    private var state: VADState
    private var chunksProcessed: Int = 0

    /// Enable range validation (min/max check) on each chunk.
    /// Disable for performance if you trust your audio pipeline.
    public var validateRange: Bool = true

    public init(model: SileroVADModel, config: VADConfig = .default) {
        self.model = model
        self.config = config
        self.state = .initial()
    }

    public var currentTimestamp: TimeInterval {
        Double(chunksProcessed) * VADAudioFormat.chunkDuration
    }

    /// Process a single audio chunk.
    /// - Parameter audio: Audio samples, shape [512] or [1, 512], 16kHz Float32 in [-1, 1]
    /// - Returns: VADResult with probability, isSpeech flag, and timestamp
    /// - Throws: VADError for invalid input
    public func process(_ audio: MLXArray) throws -> VADResult {
        let validatedAudio = try validateAndNormalize(audio)

        let (probArray, newState) = model(validatedAudio, state: state)
        eval(probArray)
        let probability = probArray.item(Float.self)
        state = newState

        let timestamp = currentTimestamp
        chunksProcessed += 1

        return VADResult(
            probability: probability,
            isSpeech: probability >= config.threshold,
            timestamp: timestamp
        )
    }

    public func reset() {
        state.reset()
        chunksProcessed = 0
    }

    private func validateAndNormalize(_ audio: MLXArray) throws -> MLXArray {
        guard audio.dtype == .float32 else {
            throw VADError.invalidDtype(expected: .float32, got: audio.dtype)
        }

        var normalized = audio

        if normalized.ndim == 1 {
            guard normalized.shape[0] == VADAudioFormat.chunkSamples else {
                throw VADError.invalidChunkSize(
                    expected: VADAudioFormat.chunkSamples,
                    got: normalized.shape[0]
                )
            }
            normalized = normalized.expandedDimensions(axis: 0)
        } else if normalized.ndim == 2 {
            guard normalized.shape[0] == 1 else {
                throw VADError.invalidAudioShape(
                    expected: "[512] or [1, 512]",
                    got: normalized.shape
                )
            }
            guard normalized.shape[1] == VADAudioFormat.chunkSamples else {
                throw VADError.invalidChunkSize(
                    expected: VADAudioFormat.chunkSamples,
                    got: normalized.shape[1]
                )
            }
        } else {
            throw VADError.invalidAudioShape(
                expected: "[512] or [1, 512]",
                got: normalized.shape
            )
        }

        if validateRange {
            eval(normalized)
            let minVal = normalized.min().item(Float.self)
            let maxVal = normalized.max().item(Float.self)

            if minVal < VADAudioFormat.valueRange.lowerBound ||
               maxVal > VADAudioFormat.valueRange.upperBound {
                throw VADError.audioOutOfRange(min: minVal, max: maxVal)
            }
        }

        return normalized
    }
}
