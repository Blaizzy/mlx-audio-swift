// Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
import Foundation
import MLX
import MLXAudioCore
import MLXFFT

struct Confucius4Features {
    /// Kaiser-best polyphase FIR, matching scipy.signal.resample_poly with
    /// edge extension and the reference's 64-zero, high-attenuation filter.
    static func resample(_ samples: [Float], from source: Int, to target: Int) -> [Float] {
        guard source != target, !samples.isEmpty else { return samples }
        func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
        func i0(_ value: Double) -> Double {
            var term = 1.0, sum = 1.0
            for i in 1...80 {
                term *= value * value / (4 * Double(i * i)); sum += term
                if term < sum * 1e-16 { break }
            }
            return sum
        }
        let divisor = gcd(source, target), up = target / divisor, down = source / divisor
        let half = 64 * max(up, down), cutoff = 0.9475937167399596 / Double(max(up, down)), beta = 14.769656459379492
        let denominator = i0(beta)
        var filter = (0...(2 * half)).map { i -> Double in
            let offset = Double(i - half), argument = cutoff * offset
            let sinc = argument == 0 ? 1 : sin(Double.pi * argument) / (Double.pi * argument)
            let radius = offset / Double(half)
            return cutoff * sinc * i0(beta * sqrt(max(0, 1 - radius * radius))) / denominator
        }
        let scale = Double(up) / filter.reduce(0, +)
        for i in filter.indices { filter[i] *= scale }
        let count = (samples.count * up + down - 1) / down
        var output = [Float](repeating: 0, count: count)
        for n in 0..<count {
            let center = n * down + half
            let first = Int(ceil(Double(center - 2 * half) / Double(up)))
            let last = center / up
            var value = 0.0
            for k in first...last {
                let tap = center - k * up
                value += Double(samples[min(max(k, 0), samples.count - 1)]) * filter[tap]
            }
            output[n] = Float(value)
        }
        return output
    }

    static func fbank(_ audio: MLXArray, mel: MLXArray, window: MLXArray) -> MLXArray {
        let count = 1 + (audio.size - 400) / 160
        let indices = MLX.arange(count)[0..., .newAxis] * 160 + MLX.arange(400)[.newAxis]
        var frames = take(audio.asType(.float32) * 32768, indices, axis: 0)
        frames = frames - frames.mean(axis: 1, keepDims: true)
        let emphasized = concatenated([frames[0..., 0..<1] * 0.03,
            frames[0..., 1...] - 0.97 * frames[0..., 0..<399]], axis: 1) * window
        let padded = concatenated([emphasized, MLXArray.zeros([count, 112])], axis: 1)
        let spectrum = abs(MLXFFT.rfft(padded, n: 512, axis: 1))
        var output = log(maximum(matmul(spectrum * spectrum, mel), 1.192092955078125e-7))
        let delta = output - output.mean(axis: 0, keepDims: true)
        let variance = (delta * delta).sum(axis: 0, keepDims: true) / Float(count - 1)
        output = delta / sqrt(variance + 1e-7)
        let even = count - count % 2
        return output[0..<even].reshaped([1, even / 2, 160])
    }

    /// Librosa Slaney triangles computed in Double, as in the published frontend.
    static func referenceFilters() -> MLXArray {
        let spacing = 200.0 / 3, minMel = 1000 / spacing, logStep = log(6.4) / 27
        let top = minMel + log(11025 / 1000.0) / logStep
        let frequencies = (0..<82).map { i -> Double in
            let m = top * Double(i) / 81
            return m < minMel ? spacing * m : 1000 * exp(logStep * (m - minMel))
        }
        var values: [Float] = []; values.reserveCapacity(513 * 80)
        for f in 0..<513 {
            let hz = Double(f) * 22050 / 1024
            for m in 0..<80 {
                let lower = (hz - frequencies[m]) / (frequencies[m + 1] - frequencies[m])
                let upper = (frequencies[m + 2] - hz) / (frequencies[m + 2] - frequencies[m + 1])
                values.append(Float(max(0, min(lower, upper)) * 2 / (frequencies[m + 2] - frequencies[m])))
            }
        }
        return MLXArray(values).reshaped([513, 80])
    }
    static func referenceMel(_ audio16k: MLXArray) throws -> MLXArray {
        let samples = resample(audio16k.asArray(Float.self), from: 16000, to: 22050)
        let audio = MLXArray(samples).reshaped([1, -1, 1])
        let padded = Confucius4Weights.reflect(audio, 384).reshaped([-1])
        let count = 1 + (padded.size - 1024) / 256
        let frames = asStrided(padded, [count, 1024], strides: [256, 1])
        let window = MLXArray((0..<1024).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / 1024)) })
        let magnitude = abs(MLXFFT.rfft(frames * window, n: 1024, axis: 1))
        let spectrum = sqrt(magnitude * magnitude + 1e-9)
        return log(maximum(matmul(spectrum, referenceFilters()), 1e-5))[.newAxis]
    }
}
