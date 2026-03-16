import MLX
import MLXAudioCore

enum FireRedASR2Audio {
    static let sampleRate = 16000
    static let frameLength = 400
    static let hopLength = 160
    static let numMels = 80

    static func extractFbank(_ audio: MLXArray) -> MLXArray {
        var waveform = audio
        if waveform.ndim > 1 {
            waveform = waveform.mean(axis: -1)
        }
        waveform = waveform.reshaped([-1]).asType(.float32)

        if waveform.shape[0] == 0 {
            return MLXArray.zeros([0, numMels], type: Float.self)
        }

        let amplitude = abs(waveform).max().item(Float.self)
        if amplitude <= 1.0 {
            waveform = waveform * MLXArray(Float(32768.0))
        }

        return computeKaldiFbank(
            waveform,
            sampleRate: sampleRate,
            winLength: frameLength,
            hopLength: hopLength,
            numMels: numMels
        )
    }

    static func applyCMVN(
        _ features: MLXArray,
        means: MLXArray,
        istd: MLXArray
    ) -> MLXArray {
        guard features.shape.last == means.shape.last, means.shape == istd.shape else {
            return features
        }
        return (features - means) * istd
    }

    private static func computeKaldiFbank(
        _ audio: MLXArray,
        sampleRate: Int,
        winLength: Int,
        hopLength: Int,
        numMels: Int
    ) -> MLXArray {
        guard sampleRate > 0, winLength > 0, hopLength > 0, numMels > 0 else {
            return MLXArray.zeros([0, max(numMels, 0)], type: Float.self)
        }

        let signalLength = audio.shape[0]
        guard signalLength >= winLength else {
            return MLXArray.zeros([0, numMels], type: Float.self)
        }

        let numFrames = 1 + (signalLength - winLength) / hopLength
        guard numFrames > 0 else {
            return MLXArray.zeros([0, numMels], type: Float.self)
        }

        var frames = asStrided(audio, [numFrames, winLength], strides: [hopLength, 1])
        frames = frames - MLX.mean(frames, axis: 1, keepDims: true)

        let preemph = MLXArray(Float(0.97))
        let first = frames[0..., 0..<1] - preemph * frames[0..., 0..<1]
        let rest = frames[0..., 1..<winLength] - preemph * frames[0..., 0..<(winLength - 1)]
        frames = MLX.concatenated([first, rest], axis: 1)

        let window = hammingWindow(size: winLength, periodic: false)
        frames = frames * window

        let fftLength = nextPowerOfTwo(winLength)
        if fftLength > winLength {
            let rightPad = MLXArray.zeros([numFrames, fftLength - winLength], type: Float.self)
            frames = MLX.concatenated([frames, rightPad], axis: 1)
        }

        let powerSpectrum = MLX.abs(MLXFFT.rfft(frames, axis: 1)).square()
        let melBank = melFilters(
            sampleRate: sampleRate,
            nFft: fftLength,
            nMels: numMels,
            fMin: 20.0,
            norm: nil,
            melScale: .htk
        )
        let fbanks = MLX.matmul(powerSpectrum, melBank)
        return MLX.log(MLX.maximum(fbanks, MLXArray(Float(1e-10))))
    }

    private static func nextPowerOfTwo(_ value: Int) -> Int {
        guard value > 1 else { return max(value, 1) }
        var n = 1
        while n < value {
            n <<= 1
        }
        return n
    }
}
