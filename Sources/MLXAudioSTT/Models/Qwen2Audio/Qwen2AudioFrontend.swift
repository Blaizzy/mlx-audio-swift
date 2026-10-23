import Foundation
import MLX
import MLXAudioCore

/// Qwen2-Audio's Python frontend uses HTK filters with edges rounded to FFT bins.
struct Qwen2AudioFrontend {
    let maxSamples: Int
    let audioTokens: Int
    private let window: MLXArray
    private let filters: MLXArray

    init(_ config: Qwen2AudioEncoderConfig) {
        maxSamples = config.maxSourcePositions * 2 * 160
        audioTokens = (config.maxSourcePositions + 1) / 2
        window = MLXArray(
            (0 ..< 400).map { Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / 400))) })
        let melMaximum = 2595 * log10(1 + 8000.0 / 700)
        let edges = (0 ..< (config.numMelBins + 2)).map {
            Int(
                floor(
                    401
                        * (700
                            * (pow(
                                10, Double($0) * melMaximum / Double(config.numMelBins + 1) / 2595)
                                - 1))
                        / 16000))
        }
        var values = [Float](repeating: 0, count: 201 * config.numMelBins)
        for m in 0 ..< config.numMelBins {
            let lower = edges[m]
            let center = edges[m + 1]
            let upper = edges[m + 2]
            for f in max(0, lower) ..< min(201, upper) {
                if f < center, center > lower {
                    values[f * config.numMelBins + m] = Float(f - lower) / Float(center - lower)
                } else if upper > center {
                    values[f * config.numMelBins + m] = Float(upper - f) / Float(upper - center)
                }
            }
        }
        filters = MLXArray(values).reshaped(201, config.numMelBins)
    }

    func callAsFunction(_ audio: MLXArray, dtype: DType) -> MLXArray {
        precondition(
            (audio.ndim == 1 || audio.ndim == 2) && audio.size > 0,
            "Expected nonempty mono or time-major audio at 16 kHz")
        var waveform = (audio.ndim == 2 ? audio.mean(axis: -1) : audio).asType(.float32)
        if waveform.size < maxSamples {
            waveform = padded(waveform, widths: [.init((0, maxSamples - waveform.size))])
        } else {
            waveform = waveform[..<maxSamples]
        }
        let spectrum = stft(
            audio: waveform, window: window, nFft: 400, hopLength: 160, padMode: .reflect)
        let mel = abs(spectrum).square().matmul(filters)
        var logmel = log10(maximum(mel, 1e-10))
        logmel = (maximum(logmel, logmel.max() - 8) + 4) / 4
        return logmel.transposed(1, 0).expandedDimensions(axis: 0).asType(dtype)
    }
}
