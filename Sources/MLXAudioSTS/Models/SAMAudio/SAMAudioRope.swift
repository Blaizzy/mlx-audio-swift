import Foundation
import MLX
import MLXNN

/// Rotary position embeddings for SAMAudio attention blocks.
public final class RotaryEmbedding: Module {
    let theta: Float
    let headDim: Int
    let maxSequenceLength: Int
    let scaleFactor: Int
    let lowFreqFactor: Int
    let highFreqFactor: Int
    let oldContextLength: Int

    private var lowFreqWavelength: Float?
    private var highFreqWavelength: Float?
    private var freqsCis: MLXArray = MLXArray([])

    public init(
        theta: Float,
        headDim: Int,
        maxSequenceLength: Int = 1024,
        scaleFactor: Int = 1,
        lowFreqFactor: Int = 1,
        highFreqFactor: Int = 32,
        oldContextLength: Int = 8192
    ) {
        self.theta = theta
        self.headDim = headDim
        self.maxSequenceLength = maxSequenceLength
        self.scaleFactor = scaleFactor
        self.lowFreqFactor = lowFreqFactor
        self.highFreqFactor = highFreqFactor
        self.oldContextLength = oldContextLength

        if scaleFactor != 1 {
            self.lowFreqWavelength = Float(oldContextLength) / Float(lowFreqFactor)
            self.highFreqWavelength = Float(oldContextLength) / Float(highFreqFactor)
        }

        super.init()
        resetParameters()
    }

    public func resetParameters() {
        freqsCis = precomputeFreqsCis(dim: headDim, end: maxSequenceLength, theta: theta)
    }

    private func applyScaling(_ freqs: MLXArray) -> MLXArray {
        guard scaleFactor != 1,
              let lowFreqWavelength,
              let highFreqWavelength else {
            return freqs
        }

        let rawFreqs = freqs.asArray(Float.self)
        var scaled = [Float]()
        scaled.reserveCapacity(rawFreqs.count)

        for freq in rawFreqs {
            let wavelength = 2 * Float.pi / freq
            if wavelength < highFreqWavelength {
                scaled.append(freq)
            } else if wavelength > lowFreqWavelength {
                scaled.append(freq / Float(scaleFactor))
            } else {
                let smooth = (
                    Float(oldContextLength) / wavelength - Float(lowFreqFactor)
                ) / Float(highFreqFactor - lowFreqFactor)
                let blended = (1 - smooth) * (freq / Float(scaleFactor)) + smooth * freq
                scaled.append(blended)
            }
        }

        return MLXArray(scaled).asType(freqs.dtype)
    }

    private func precomputeFreqsCis(dim: Int, end: Int, theta: Float) -> MLXArray {
        let halfDim = dim / 2
        let indices = MLXArray(Array(0..<halfDim).map { Float($0) }).asType(.float32)
        var freqs = 1.0 / MLX.pow(MLXArray(theta), indices / Float(halfDim * 2) * 2.0)
        freqs = applyScaling(freqs)

        let t = MLXArray(Array(0..<end).map { Float($0) }).asType(.float32)
        let outer = t.reshaped([end, 1]) * freqs.reshaped([1, halfDim])

        let cosVals = MLX.cos(outer)
        let sinVals = MLX.sin(outer)

        var packed = MLX.stacked([cosVals, -sinVals, sinVals, cosVals], axis: -1)
        packed = packed.reshaped([end, halfDim, 2, 2])
        packed = packed.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        return packed
    }

    public func callAsFunction(_ x: MLXArray, bhle: Bool = false, offset: Int = 0) -> MLXArray {
        var input = x
        if bhle {
            input = input.transposed(0, 2, 1, 3)
        }

        let batchSize = input.shape[0]
        let sequenceLength = input.shape[1]
        let numHeads = input.shape[2]
        let dim = input.shape[3]

        precondition(offset + sequenceLength <= maxSequenceLength, "RoPE offset exceeds precomputed length")

        let reshaped = input.reshaped([batchSize, sequenceLength, numHeads, dim / 2, 1, 2])
        let freqsSlice = freqsCis[0..., offset..<(offset + sequenceLength), 0..., 0..., 0..., 0...]
        var rotated = (reshaped * freqsSlice).sum(axis: -1)
        rotated = rotated.reshaped([batchSize, sequenceLength, numHeads, dim])

        if bhle {
            rotated = rotated.transposed(0, 2, 1, 3)
        }
        return rotated.asType(x.dtype)
    }
}

/// Applies rotary embeddings to query and key tensors.
public func applyRotaryEmb(
    _ xq: MLXArray,
    _ xk: MLXArray,
    freqsCos: MLXArray,
    freqsSin: MLXArray
) -> (MLXArray, MLXArray) {
    let cos = freqsCos.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    let sin = freqsSin.expandedDimensions(axis: 0).expandedDimensions(axis: 2)

    let headDim = xq.shape[xq.ndim - 1]
    let half = headDim / 2

    let xq1 = xq[0..., 0..., 0..., 0..<half]
    let xq2 = xq[0..., 0..., 0..., half..<headDim]
    let xk1 = xk[0..., 0..., 0..., 0..<half]
    let xk2 = xk[0..., 0..., 0..., half..<headDim]

    let xqOut = MLX.concatenated([xq1 * cos - xq2 * sin, xq1 * sin + xq2 * cos], axis: -1)
    let xkOut = MLX.concatenated([xk1 * cos - xk2 * sin, xk1 * sin + xk2 * cos], axis: -1)

    return (xqOut.asType(xq.dtype), xkOut.asType(xk.dtype))
}
