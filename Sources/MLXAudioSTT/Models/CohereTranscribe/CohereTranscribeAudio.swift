import Foundation
import MLX
import MLXAudioCore

enum CohereTranscribeAudio {
    static func computeMelFilters(
        sampleRate: Int = 16000,
        nFft: Int = 512,
        numMels: Int = 128
    ) -> MLXArray {
        melFilters(
            sampleRate: sampleRate,
            nFft: nFft,
            nMels: numMels,
            fMin: 0,
            fMax: Float(sampleRate) / 2.0,
            norm: "slaney",
            melScale: .slaney
        )
    }

    static func computeFeatures(
        audio: MLXArray,
        melFilters: MLXArray,
        nFft: Int = 512,
        winLength: Int = 400,
        hopLength: Int = 160
    ) -> MLXArray {
        let audio1D = audio.ndim > 1 ? audio.reshaped([-1]).asType(.float32) : audio.asType(.float32)
        let emphasized = preEmphasis(audio1D)
        let window = centeredWindow(nFft: nFft, winLength: winLength)
        let spectrum = stft(audio: emphasized, window: window, nFft: nFft, hopLength: hopLength, padMode: .constant)
        let nFrames = spectrum.shape[0]

        if nFrames <= 0 {
            return MLXArray.zeros([1, melFilters.shape[1], 0], type: Float.self)
        }

        let powerSpectrum = MLX.abs(spectrum).square()
        var melSpec = MLX.matmul(powerSpectrum, melFilters)
        melSpec = MLX.log(melSpec + MLXArray(pow(Float(2.0), -24.0)))
        melSpec = melSpec.transposed(1, 0).expandedDimensions(axis: 0)

        let mean = melSpec.mean(axes: [2], keepDims: true)
        let variance = melSpec.variance(axes: [2], keepDims: true)
        let std = MLX.sqrt(variance) + MLXArray(Float(1e-5))

        return (melSpec - mean) / std
    }

    private static func preEmphasis(_ audio: MLXArray, factor: Float = 0.97) -> MLXArray {
        guard audio.shape[0] > 1 else { return audio }

        let coeff = MLXArray(factor)
        let first = audio[0..<1]
        let rest = audio[1..<audio.shape[0]] - coeff * audio[0..<(audio.shape[0] - 1)]
        return MLX.concatenated([first, rest], axis: 0)
    }

    private static func centeredWindow(nFft: Int, winLength: Int) -> MLXArray {
        guard nFft >= winLength else { return hanningWindow(size: nFft) }
        let leftPad = (nFft - winLength) / 2
        let rightPad = nFft - winLength - leftPad
        let left = MLXArray.zeros([leftPad], type: Float.self)
        let middle = hanningWindow(size: winLength)
        let right = MLXArray.zeros([rightPad], type: Float.self)
        return MLX.concatenated([left, middle, right], axis: 0)
    }
}
