import Foundation
@preconcurrency import MLX
import MLXFFT
import MLXAudioCore

enum OuteTTSAudio {
    /// BS.1770 mono normalization used by the OuteTTS reference codec interface.
    static func normalize(_ audio: MLXArray, sampleRate: Int = 24_000) throws -> sending MLXArray {
        let samples = audio.asType(.float32).asArray(Float.self)
        guard !samples.isEmpty, samples.allSatisfy(\.isFinite) else {
            throw AudioGenerationError.audioDecodingFailed("OuteTTS audio is empty or nonfinite")
        }
        let block = Int((0.4 * Double(sampleRate)).rounded())
        let hop = block / 4
        var weighted = samples.map(Double.init)
        if weighted.count < block { weighted += Array(repeating: 0, count: block - weighted.count) }

        func filter(_ input: [Double], frequency: Double, q: Double, gain: Double?) -> [Double] {
            let k = tan(.pi * frequency / Double(sampleRate))
            let d = 1 + k / q + k * k
            let a1 = 2 * (k * k - 1) / d
            let a2 = (1 - k / q + k * k) / d
            var b = [1.0, -2.0, 1.0]
            if let gain {
                let g = pow(10, gain / 20)
                let m = pow(g, 0.4996667741545416)
                b = [(g + m * k / q + k * k) / d, 2 * (k * k - g) / d, (g - m * k / q + k * k) / d]
            }
            var s1 = 0.0, s2 = 0.0
            return input.map { x in
                let y = b[0] * x + s1
                s1 = s2 + b[1] * x - a1 * y
                s2 = b[2] * x - a2 * y
                return y
            }
        }
        weighted = filter(weighted, frequency: 1681.974450955533, q: 0.7071752369554196, gain: 3.999843853973347)
        weighted = filter(weighted, frequency: 38.13547087602444, q: 0.5003270373238773, gain: nil)
        var prefix = [Double](repeating: 0, count: weighted.count + 1)
        for i in weighted.indices { prefix[i + 1] = prefix[i] + weighted[i] * weighted[i] }
        let energies = stride(from: 0, through: weighted.count - block, by: hop).map {
            (prefix[$0 + block] - prefix[$0]) / Double(block)
        }
        let absolute = energies.filter { -0.691 + 10 * log10($0) > -70 }
        guard !absolute.isEmpty else { return MLXArray(samples) }
        let threshold = absolute.reduce(0, +) / Double(absolute.count) * 0.1
        let gated = absolute.filter { $0 > threshold }
        guard !gated.isEmpty else { return MLXArray(samples) }
        let loudness = -0.691 + 10 * log10(gated.reduce(0, +) / Double(gated.count))
        let gain = pow(10, (-18 - loudness) / 20)
        let peak = samples.reduce(0.0) { max($0, abs(Double($1))) }
        let scale = min(gain, peak > 0 ? pow(10, -1.0 / 20) / peak : gain)
        return MLXArray(samples.map { Float(Double($0) * scale) })
    }

    static func features(_ audio: MLXArray, sampleRate: Int = 24_000) -> OuteTTSFeatures {
        guard audio.size > 0 else { return .init() }
        let audio = audio.flattened().asType(.float32)
        let magnitude = abs(MLXFFT.rfft(audio))
        let frequencies = MLXArray(0..<magnitude.size).asType(.float32) / Float(max(magnitude.size - 1, 1))
        let centroid = sum(magnitude * frequencies) / (sum(magnitude) + 1e-10)
        let energy = sqrt(mean(audio * audio))

        let frameLength = 400, hop = 160
        let pad = (frameLength - audio.size % hop) % hop
        let padded = MLX.padded(audio, widths: [IntOrPair((0, pad))])
        let frameCount = (padded.size - frameLength) / hop + 1
        var pitch: Float = 0
        if padded.size >= frameLength && frameCount > 0 {
            let frames = asStrided(padded, [frameCount, frameLength], strides: [hop, 1])
            let windowed = frames * hanningWindow(size: frameLength)
            let transformed = MLXFFT.rfft(windowed, n: frameLength * 2, axis: -1)
            let autocorrelation = MLXFFT.irfft(abs(transformed) * abs(transformed), axis: -1)[0..., 0..<frameLength]
            let low = max(1, sampleRate / 600), high = min(frameLength, sampleRate / 75)
            let indices = argMax(autocorrelation[0..., low..<high], axis: -1).asType(.int32) + Int32(low)
            let column = indices.expandedDimensions(axis: -1)
            let peak = takeAlong(autocorrelation, column, axis: -1).squeezed(axis: -1)
            let clipped = clip(indices, min: 1, max: frameLength - 2).expandedDimensions(axis: -1)
            let alpha = takeAlong(autocorrelation, clipped - 1, axis: -1).squeezed(axis: -1)
            let beta = takeAlong(autocorrelation, clipped, axis: -1).squeezed(axis: -1)
            let gamma = takeAlong(autocorrelation, clipped + 1, axis: -1).squeezed(axis: -1)
            let delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma + 1e-8)
            let periods = (indices.asType(.float32) + delta) / Float(sampleRate)
            let voiced = peak / (autocorrelation[0..., 0] + 1e-8) .> 0.3
            let pitches = clip(MLX.where(voiced, 1 / periods, MLXArray(Float(0))), min: 75, max: 600)
            pitch = max(0, min(1, (mean(pitches).item(Float.self) - 75) / 525))
        }
        eval(energy, centroid)
        return OuteTTSFeatures(
            energy: Int((energy.item(Float.self) * 100).rounded(.toNearestOrEven)),
            spectralCentroid: Int((centroid.item(Float.self) * 100).rounded(.toNearestOrEven)),
            pitch: Int((pitch * 100).rounded(.toNearestOrEven)))
    }
}
