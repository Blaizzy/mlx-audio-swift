import Foundation
import MLX
import MLXAudioCore

enum ParakeetAudio {
    static func logMelSpectrogram(
        _ audio: MLXArray,
        config: ParakeetPreprocessConfig
    ) -> MLXArray {
        let originalDType = audio.dtype
        var x = audio

        if config.padTo > 0 && x.shape[0] < config.padTo {
            let padLength = config.padTo - x.shape[0]
            let paddedTail = MLXArray(Array(repeating: config.padValue, count: padLength))
            x = MLX.concatenated([x, paddedTail], axis: 0)
        }

        if config.preemph > 0 && x.shape[0] > 1 {
            let first = x[0..<1]
            let rest = x[1...] - Float(config.preemph) * x[..<(x.shape[0] - 1)]
            x = MLX.concatenated([first, rest], axis: 0)
        }

        let window = makeWindow(name: config.window, winLength: config.winLength, fftLength: config.nFft)
        let stftOutput = stft(
            audio: x,
            window: window,
            nFft: config.nFft,
            hopLength: config.hopLength,
            padMode: .reflect
        )

        let power = MLX.abs(stftOutput).square().asType(originalDType)
        let filters = melFilters(
            sampleRate: config.sampleRate,
            nFft: config.nFft,
            nMels: config.features,
            norm: config.normalize,
            melScale: .slaney
        )

        var mel = MLX.matmul(power, filters.asType(power.dtype))
        mel = MLX.log(mel + MLXArray(1e-5, dtype: mel.dtype))

        let normalized: MLXArray
        if config.normalize == "per_feature" {
            let mean = MLX.mean(mel, axis: 0, keepDims: true)
            let std = MLX.std(mel, axis: 0, keepDims: true)
            normalized = (mel - mean) / (std + MLXArray(1e-5, dtype: mel.dtype))
        } else {
            let mean = MLX.mean(mel)
            let std = MLX.std(mel)
            normalized = (mel - mean) / (std + MLXArray(1e-5, dtype: mel.dtype))
        }

        return normalized.expandedDimensions(axis: 0).asType(originalDType)
    }

    private static func makeWindow(name: String, winLength: Int, fftLength: Int) -> MLXArray {
        let base: MLXArray
        switch name.lowercased() {
        case "hann", "hanning":
            base = hanningWindow(size: winLength)
        case "hamming":
            base = hammingWindow(size: winLength)
        case "blackman":
            base = blackmanWindow(size: winLength)
        case "bartlett":
            base = bartlettWindow(size: winLength)
        default:
            base = hanningWindow(size: winLength)
        }

        if winLength >= fftLength {
            return base[0..<fftLength]
        }

        let rightPad = fftLength - winLength
        let right = MLXArray(Array(repeating: Float(0), count: rightPad))
        return MLX.concatenated([base, right], axis: 0)
    }

    private static func hammingWindow(size: Int) -> MLXArray {
        if size <= 1 {
            return MLXArray(Array(repeating: Float(1), count: max(size, 1)))
        }
        let denom = Float(size - 1)
        let values = (0..<size).map { n in
            Float(0.54) - Float(0.46) * cos(2 * Float.pi * Float(n) / denom)
        }
        return MLXArray(values)
    }

    private static func blackmanWindow(size: Int) -> MLXArray {
        if size <= 1 {
            return MLXArray(Array(repeating: Float(1), count: max(size, 1)))
        }
        let denom = Float(size - 1)
        let values = (0..<size).map { n in
            let k = 2 * Float.pi * Float(n) / denom
            return Float(0.42) - Float(0.5) * cos(k) + Float(0.08) * cos(2 * k)
        }
        return MLXArray(values)
    }

    private static func bartlettWindow(size: Int) -> MLXArray {
        if size <= 1 {
            return MLXArray(Array(repeating: Float(1), count: max(size, 1)))
        }
        let mid = Float(size - 1) / 2
        let values = (0..<size).map { n in
            Float(1) - abs((Float(n) - mid) / mid)
        }
        return MLXArray(values)
    }
}
