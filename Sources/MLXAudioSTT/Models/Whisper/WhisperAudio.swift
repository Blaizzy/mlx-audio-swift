import Foundation
import MLX
import MLXAudioCore

enum WhisperAudio {
    /// Pad or trim a 1D waveform to the 30 s window Whisper expects.
    static func padOrTrimToWindow(_ audio: MLXArray) -> MLXArray {
        let target = WhisperAudioConfig.chunkLengthSamples
        let n = audio.dim(0)
        if n == target { return audio }
        if n > target { return audio[0..<target] }
        return MLX.padded(audio, widths: [.init((0, target - n))])
    }

    // Whisper only uses 80 or 128 mel bins; cache the filterbank per bin count.
    private static let filterCacheLock = NSLock()
    nonisolated(unsafe) private static var filterCache: [Int: MLXArray] = [:]

    static func melFilters(nMels: Int) -> MLXArray {
        filterCacheLock.lock()
        defer { filterCacheLock.unlock() }
        if let cached = filterCache[nMels] { return cached }
        let filters = MLXAudioCore.melFilters(
            sampleRate: WhisperAudioConfig.sampleRate,
            nFft: WhisperAudioConfig.nFft,
            nMels: nMels,
            fMin: 0,
            fMax: Float(WhisperAudioConfig.sampleRate) / 2.0,
            norm: "slaney",
            melScale: .slaney
        )
        filterCache[nMels] = filters
        return filters
    }

    /// Compute the log-mel spectrogram for a 1D, 16 kHz, mono waveform; output
    /// shape is `[nMels, nFrames]`.
    static func logMelSpectrogram(_ audio: MLXArray, nMels: Int) -> MLXArray {
        let nFft = WhisperAudioConfig.nFft
        let hop = WhisperAudioConfig.hopLength

        let nIdx = MLXArray(0..<nFft).asType(.float32)
        let window = 0.5 * (1.0 - cos((2.0 * Float.pi * nIdx) / Float(nFft)))

        let mono: MLXArray = audio.ndim > 1 ? audio.reshaped([-1]) : audio
        let padded = reflectPad(mono.asType(.float32), pad: nFft / 2)

        let nSamples = padded.dim(0)
        let nFrames = nSamples >= nFft ? 1 + (nSamples - nFft) / hop : 0
        if nFrames <= 0 {
            return MLXArray.zeros([nMels, 0], type: Float.self)
        }

        let framesView = asStrided(
            padded,
            [nFrames, nFft],
            strides: [hop, 1],
            offset: 0
        )
        let windowed = framesView * window.expandedDimensions(axis: 0)
        let spectrum = MLXFFT.rfft(windowed, axis: -1)
        var magnitudes = MLX.abs(spectrum).square()

        // Drop the last frame to match torch.stft(center=True).
        if magnitudes.shape[0] > 0 {
            magnitudes = magnitudes[0..<(magnitudes.shape[0] - 1), 0...]
        }
        magnitudes = magnitudes.transposed(1, 0)

        let filters = melFilters(nMels: nMels)
        var melSpec = MLX.matmul(filters.transposed(1, 0), magnitudes)

        melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
        var logSpec = MLX.log10(melSpec)
        let dynamicFloor = logSpec.max() - MLXArray(Float(8.0))
        logSpec = MLX.maximum(logSpec, dynamicFloor)
        logSpec = (logSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))
        return logSpec
    }

    /// Pad/trim to 30 s and return the encoder's input features
    /// `[1, nFrames, nMels]`.
    static func encoderFeatures(audio: MLXArray, nMels: Int) -> MLXArray {
        let window = padOrTrimToWindow(audio)
        let mel = logMelSpectrogram(window, nMels: nMels)
        return mel.transposed(1, 0).expandedDimensions(axis: 0)
    }

    private static func reflectPad(_ audio: MLXArray, pad: Int) -> MLXArray {
        if pad <= 0 { return audio }
        let n = audio.dim(0)
        if n <= 1 {
            return MLX.padded(audio, widths: [.init((pad, pad))])
        }
        let leftCount = min(pad, n - 1)
        let rightCount = min(pad, n - 1)
        let leftSrc = audio[1...leftCount]
        let rightSrc = audio[(n - 1 - rightCount)..<(n - 1)]
        let left = reverseAlongFirstAxis(leftSrc)
        let right = reverseAlongFirstAxis(rightSrc)
        var pieces: [MLXArray] = []
        if leftCount < pad {
            pieces.append(MLXArray.zeros([pad - leftCount], type: Float.self))
        }
        pieces.append(left)
        pieces.append(audio)
        pieces.append(right)
        if rightCount < pad {
            pieces.append(MLXArray.zeros([pad - rightCount], type: Float.self))
        }
        return MLX.concatenated(pieces, axis: 0)
    }

    private static func reverseAlongFirstAxis(_ array: MLXArray) -> MLXArray {
        let n = array.dim(0)
        if n <= 1 { return array }
        let idx = MLXArray((0..<n).reversed().map { Int32($0) })
        return array[idx]
    }
}
