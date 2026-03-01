import Foundation
import MLX
import MLXAudioCore

enum VoxtralRealtimeAudio {
    static func computeMelFilters(
        numMelBins: Int = 128,
        windowSize: Int = 400,
        sampleRate: Int = 16000
    ) -> MLXArray {
        melFilters(
            sampleRate: sampleRate,
            nFft: windowSize,
            nMels: numMelBins,
            fMin: 0,
            fMax: 8000,
            norm: "slaney",
            melScale: .slaney
        )
    }

    static func computeMelSpectrogram(
        audio: MLXArray,
        melFilters: MLXArray,
        windowSize: Int = 400,
        hopLength: Int = 160,
        globalLogMelMax: Float = 1.5
    ) -> MLXArray {
        // Periodic Hann window uses N denominator, not N-1.
        let n = MLXArray(0..<windowSize).asType(.float32)
        let window = 0.5 * (1.0 - cos((2.0 * Float.pi * n) / Float(windowSize)))

        let audio1D: MLXArray
        if audio.ndim > 1 {
            audio1D = audio.reshaped([-1])
        } else {
            audio1D = audio
        }

        let paddedAudio = reflectPadCenter(audio1D, pad: windowSize / 2)
        let nSamples = paddedAudio.shape[0]
        let nFrames = 1 + max(0, (nSamples - windowSize) / hopLength)

        if nFrames <= 0 {
            return MLXArray.zeros([melFilters.shape[1], 0], type: Float.self)
        }

        let frameIdx = asStrided(
            paddedAudio,
            [nFrames, windowSize],
            strides: [hopLength, 1],
            offset: 0
        )

        let frames = frameIdx * window.expandedDimensions(axis: 0)
        let spectrum = MLXFFT.rfft(frames, axis: -1)
        var magnitudes = MLX.abs(spectrum).square()

        // Match reference: drop last frame, then transpose to [freq, frames].
        if magnitudes.shape[0] > 0 {
            magnitudes = magnitudes[0..<(magnitudes.shape[0] - 1), 0...]
        }
        magnitudes = magnitudes.transposed(1, 0)

        var melSpec = MLX.matmul(melFilters.transposed(1, 0), magnitudes)
        melSpec = MLX.maximum(melSpec, MLXArray(Float(1e-10)))
        var logSpec = MLX.log10(melSpec)

        let minVal = globalLogMelMax - 8.0
        logSpec = MLX.maximum(logSpec, MLXArray(minVal))
        logSpec = (logSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))
        return logSpec
    }

    private static func reflectPadCenter(_ audio: MLXArray, pad: Int) -> MLXArray {
        guard pad > 0 else { return audio.asType(.float32) }

        let samples = audio.asType(.float32).asArray(Float.self)
        guard !samples.isEmpty else {
            return MLXArray.zeros([2 * pad], type: Float.self)
        }

        func reflectIndex(_ idx: Int, count: Int) -> Int {
            if count <= 1 { return 0 }
            var i = idx
            while i < 0 || i >= count {
                if i < 0 {
                    i = -i
                } else {
                    i = 2 * count - i - 2
                }
            }
            return i
        }

        var out = [Float](repeating: 0, count: samples.count + 2 * pad)
        for i in 0..<out.count {
            let src = i - pad
            out[i] = samples[reflectIndex(src, count: samples.count)]
        }

        return MLXArray(out)
    }
}
