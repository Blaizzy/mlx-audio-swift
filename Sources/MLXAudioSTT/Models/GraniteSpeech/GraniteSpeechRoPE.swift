import Foundation
import MLX

struct GraniteSpeechRotaryEmbedding {
    let dimensions: Int
    let base: Float
    let magnitude: Float
    let frequencies: MLXArray?

    init(dimensions: Int, config: GraniteSpeechTextConfig) {
        self.dimensions = dimensions
        let base = config.ropeTheta
        self.base = base
        guard let scaling = config.ropeScaling, scaling.ropeType == "yarn" else {
            magnitude = 1
            frequencies = nil
            return
        }
        let factor = scaling.factor
        precondition(factor > 0 && scaling.originalMaxPositionEmbeddings > 0)
        func correction(_ rotations: Float) -> Float {
            Float(dimensions)
                * log(Float(scaling.originalMaxPositionEmbeddings) / (rotations * 2 * .pi))
                / (2 * log(base))
        }
        let low = max(floor(correction(scaling.betaFast)), 0)
        var high = min(ceil(correction(scaling.betaSlow)), Float(dimensions - 1))
        if low == high { high += 0.001 }
        let ramp = clip(
            (arange(dimensions / 2).asType(.float32) - low) / (high - low), min: 0, max: 1)
        let mask = 1 - ramp
        let extra = pow(base, arange(0, dimensions, step: 2).asType(.float32) / Float(dimensions))
        let interpolated = factor * extra
        frequencies = (interpolated * extra) / (interpolated * mask + extra * (1 - mask))
        func mscale(_ value: Float) -> Float { factor <= 1 ? 1 : 0.1 * value * log(factor) + 1 }
        magnitude = mscale(scaling.mscale) / mscale(scaling.mscaleAllDim)
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            magnitude == 1 ? x : x * magnitude, dimensions: dimensions,
            traditional: false, base: frequencies == nil ? base : nil,
            scale: 1, offset: offset, freqs: frequencies)
    }
}
