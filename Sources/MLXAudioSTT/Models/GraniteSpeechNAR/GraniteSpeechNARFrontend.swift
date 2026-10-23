import Foundation
import MLX
import MLXAudioCore

/// NAR truncates to complete 20 ms pairs before log-mel normalization.
enum GraniteSpeechNARFrontend {
    private static var window: MLXArray {
        let values = (0 ..< 400).map { Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / 400))) }
        return concatenated([MLXArray.zeros([56]), MLXArray(values), MLXArray.zeros([56])])
    }
    // Compute the HTK triangle edges in double precision, then store float32 weights.
    private static var filters: MLXArray {
        let melMax = 2595 * log10(1 + 8000.0 / 700)
        let edges = (0 ..< 82).map { 700 * (pow(10, melMax * Double($0) / 81 / 2595) - 1) }
        var weights = [Float](repeating: 0, count: 257 * 80)
        for f in 0 ..< 257 {
            let hz = Double(f) * 16000 / 512
            for m in 0 ..< 80 {
                weights[f * 80 + m] = Float(
                    max(
                        0,
                        min(
                            (hz - edges[m]) / (edges[m + 1] - edges[m]),
                            (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]))))
            }
        }
        return MLXArray(weights).reshaped(257, 80)
    }

    static func features(_ audio: MLXArray) -> MLXArray {
        let waveform = audio.asType(.float32)
        let frames = 2 * (waveform.size / 320)
        let spectrum = stft(
            audio: waveform, window: window, nFft: 512, hopLength: 160, padMode: .reflect)
        let mel = abs(spectrum).square().matmul(filters)[..<frames]
        var logmel = log10(maximum(mel, 1e-10))
        logmel = maximum(logmel, logmel.max() - 8) / 4 + 1
        return logmel.reshaped(frames / 2, 160)
    }
}
