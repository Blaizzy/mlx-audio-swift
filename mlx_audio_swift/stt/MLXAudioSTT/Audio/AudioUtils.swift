import Foundation
import MLX
import MLXRandom

/// Audio utility functions for Whisper STT
public enum AudioUtils {

    /// Padding strategy for short audio
    public enum PaddingStrategy {
        /// Pad with zeros (silence) - can cause Whisper to output "..." for short chunks
        case zero
        /// Repeat audio content to fill the gap - prevents silence detection issues
        /// Uses decay for very short audio to avoid excessive repetition confusing Whisper
        case `repeat`
        /// Pad with low-level noise - alternative to silence that doesn't confuse Whisper
        case noise(amplitude: Float)
    }

    /// Maximum number of repetitions before switching to noise fill
    /// Whisper's loop detection triggers at ~4+ repetitions, causing output suppression
    /// 3 repetitions provides good content ratio without triggering suppression
    private static let maxRepetitions = 3

    /// Minimum audio ratio before switching to zero padding
    /// For audio <50% of target length, repetition causes transcription of repeated content
    /// Research recommendation: "Don't force transcription of audio where >50% is padding"
    private static let minContentRatioForRepeat: Double = 0.50

    /// Pad or trim audio to a specific length
    /// - Parameters:
    ///   - audio: Input audio as MLXArray, shape [nSamples]
    ///   - length: Target length in samples (default: 480000 for 30s at 16kHz)
    ///   - strategy: Padding strategy for short audio (default: .repeat to avoid silence issues)
    /// - Returns: Audio padded or trimmed to exact length
    ///
    /// - Note: For very short audio (<33% of target length), zero padding is used automatically
    ///   regardless of the specified strategy to prevent repetition artifacts. This is because
    ///   Whisper's silence detection ("...") is preferable to transcribing repeated content.
    public static func padOrTrim(
        _ audio: MLXArray,
        length: Int = AudioConstants.nSamples,
        strategy: PaddingStrategy = .repeat
    ) -> MLXArray {
        let currentLength = audio.shape[0]

        if currentLength == length {
            return audio
        } else if currentLength > length {
            return audio[0..<length]
        } else {
            // For very short audio, use zero padding to avoid repetition artifacts
            let contentRatio = Double(currentLength) / Double(length)
            let effectiveStrategy = contentRatio < minContentRatioForRepeat ? .zero : strategy

            switch effectiveStrategy {
            case .zero:
                let padding = MLXArray.zeros([length - currentLength])
                return MLX.concatenated([audio, padding], axis: 0)
            case .repeat:
                return repeatPadWithNoiseFill(audio, toLength: length)
            case .noise(let amplitude):
                let noise = MLXRandom.uniform(
                    low: MLXArray(-amplitude),
                    high: MLXArray(amplitude),
                    [length - currentLength]
                )
                return MLX.concatenated([audio, noise], axis: 0)
            }
        }
    }

    /// Repeat audio content with noise fill for very short audio
    /// For audio requiring >maxRepetitions, fills remaining space with signal-scaled noise
    /// to avoid Whisper detecting and suppressing repetitive/looped content
    private static func repeatPadWithNoiseFill(_ audio: MLXArray, toLength length: Int) -> MLXArray {
        let currentLength = audio.shape[0]
        guard currentLength > 0 else {
            return MLXArray.zeros([length])
        }

        let repetitionsNeeded = (length + currentLength - 1) / currentLength

        if repetitionsNeeded <= maxRepetitions {
            // Simple repeat - safe number of repetitions
            return simpleRepeatPad(audio, toLength: length)
        } else {
            // Repeat up to maxRepetitions, then fill with noise
            return repeatThenNoiseFill(audio, toLength: length)
        }
    }

    /// Simple repeat without decay
    private static func simpleRepeatPad(_ audio: MLXArray, toLength length: Int) -> MLXArray {
        let currentLength = audio.shape[0]
        var result = audio

        while result.shape[0] < length {
            let remaining = length - result.shape[0]
            if remaining >= currentLength {
                result = MLX.concatenated([result, audio], axis: 0)
            } else {
                result = MLX.concatenated([result, audio[0..<remaining]], axis: 0)
            }
        }

        return result[0..<length]
    }

    /// Repeat up to maxRepetitions, then fill remaining space with signal-scaled noise
    /// This avoids Whisper's loop detection while maintaining some signal presence
    private static func repeatThenNoiseFill(_ audio: MLXArray, toLength length: Int) -> MLXArray {
        let currentLength = audio.shape[0]

        // Repeat audio up to maxRepetitions
        let repeatedLength = min(currentLength * maxRepetitions, length)
        var result = simpleRepeatPad(audio, toLength: repeatedLength)

        // Fill remaining space with very low-level noise
        // Using extremely low amplitude (0.0001) - just enough to avoid pure silence
        // while not being interpreted as speech by Whisper
        let remaining = length - result.shape[0]
        if remaining > 0 {
            let noiseAmplitude: Float = 0.0001  // Near-silence, ~-80dB

            let noise = MLXRandom.uniform(
                low: MLXArray(-noiseAmplitude),
                high: MLXArray(noiseAmplitude),
                [remaining]
            )
            result = MLX.concatenated([result, noise], axis: 0)
        }

        return result
    }
}
