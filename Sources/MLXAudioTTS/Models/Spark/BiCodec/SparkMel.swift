import Foundation
import MLX
import MLXAudioCore
import MLXFFT

enum SparkMel {
    static func melSpectrogram(
        _ wav: MLXArray,
        sampleRate: Int = 16_000,
        nMels: Int = 128,
        nFft: Int = 1024,
        winLength: Int = 640,
        hopLength: Int = 320,
        fMin: Float = 10,
        fMax: Float? = nil
    ) -> MLXArray {
        let mono: MLXArray = wav.ndim > 1 ? wav.reshaped([-1]) : wav

        let winIdx = MLXArray(0 ..< winLength).asType(.float32)
        var window = 0.5 * (1.0 - cos((2.0 * Float.pi * winIdx) / Float(winLength)))
        if winLength < nFft {
            window = MLX.concatenated([window, MLXArray.zeros([nFft - winLength], type: Float.self)], axis: 0)
        }

        let padded = reflectPad(mono.asType(.float32), pad: nFft / 2)
        let nSamples = padded.dim(0)
        let nFrames = nSamples >= nFft ? 1 + (nSamples - nFft) / hopLength : 0
        if nFrames <= 0 {
            return MLXArray.zeros([1, 0, nMels], type: Float.self)
        }

        let frames = asStrided(padded, [nFrames, nFft], strides: [hopLength, 1], offset: 0)
        let spectrum = MLXFFT.rfft(frames * window.expandedDimensions(axis: 0), axis: -1)
        let magnitudes = MLX.abs(spectrum)

        let filters = melFilters(
            sampleRate: sampleRate, nFft: nFft, nMels: nMels,
            fMin: fMin, fMax: fMax, norm: "slaney", melScale: .slaney)

        let mel = MLX.matmul(magnitudes, filters)
        return mel.expandedDimensions(axis: 0)
    }

    private static func reflectPad(_ audio: MLXArray, pad: Int) -> MLXArray {
        if pad <= 0 { return audio }
        let n = audio.dim(0)
        if n <= 1 { return MLX.padded(audio, widths: [.init((pad, pad))]) }
        let count = min(pad, n - 1)
        let left = reverseAlongFirstAxis(audio[1 ... count])
        let right = reverseAlongFirstAxis(audio[(n - 1 - count) ..< (n - 1)])
        var pieces: [MLXArray] = []
        if count < pad { pieces.append(MLXArray.zeros([pad - count], type: Float.self)) }
        pieces.append(left)
        pieces.append(audio)
        pieces.append(right)
        if count < pad { pieces.append(MLXArray.zeros([pad - count], type: Float.self)) }
        return MLX.concatenated(pieces, axis: 0)
    }

    private static func reverseAlongFirstAxis(_ array: MLXArray) -> MLXArray {
        let n = array.dim(0)
        if n <= 1 { return array }
        return array[MLXArray((0 ..< n).reversed().map { Int32($0) })]
    }
}
