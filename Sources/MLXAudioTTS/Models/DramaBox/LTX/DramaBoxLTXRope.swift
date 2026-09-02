import Foundation
@preconcurrency import MLX

/// Inverse-frequency grid in Double, matching the upstream NumPy fp64 path.
func dramaBoxGenerateFreqGrid(theta: Double, nPosDims: Int, innerDim: Int) -> MLXArray {
    let nElem = 2 * nPosDims
    let count = innerDim / nElem
    let logTheta = log(theta)
    let start = log(1.0) / logTheta
    let end = log(theta) / logTheta
    var values = [Float](repeating: 0, count: count)
    if count == 1 {
        values[0] = Float(pow(theta, start) * Double.pi / 2.0)
    } else {
        for i in 0..<count {
            let t = start + (end - start) * Double(i) / Double(count - 1)
            values[i] = Float(pow(theta, t) * Double.pi / 2.0)
        }
    }
    return MLXArray(values)
}

func dramaBoxPrecomputeSplitFreqs1D(
    seqLen: Int,
    innerDim: Int,
    numHeads: Int,
    theta: Float,
    maxPos: Int,
    outDtype: DType
) -> (MLXArray, MLXArray) {
    let inv = dramaBoxGenerateFreqGrid(theta: Double(theta), nPosDims: 1, innerDim: innerDim)
    let positions = MLXArray(Array(0..<seqLen).map { Float($0) })
    let scaled = (positions / Float(maxPos)) * 2.0 - 1.0
    var freqs = scaled.reshaped([seqLen, 1]) * inv.reshaped([1, inv.dim(0)])
    freqs = freqs.reshaped([1, seqLen, freqs.dim(1)])

    let expected = innerDim / 2
    var cos = MLX.cos(freqs)
    var sin = MLX.sin(freqs)
    let padSize = expected - cos.dim(-1)
    if padSize > 0 {
        let cosPad = MLXArray.ones([1, seqLen, padSize], dtype: cos.dtype)
        let sinPad = MLXArray.zeros([1, seqLen, padSize], dtype: sin.dtype)
        cos = concatenated([cosPad, cos], axis: -1)
        sin = concatenated([sinPad, sin], axis: -1)
    }
    cos = cos.reshaped(1, seqLen, numHeads, -1).transposed(0, 2, 1, 3)
    sin = sin.reshaped(1, seqLen, numHeads, -1).transposed(0, 2, 1, 3)
    return (cos.asType(outDtype), sin.asType(outDtype))
}

func dramaBoxPrecomputeSplitFreqsFromPositions(
    positions: MLXArray,
    innerDim: Int,
    numHeads: Int,
    theta: Float,
    maxPos: Float,
    outDtype: DType
) -> (MLXArray, MLXArray) {
    precondition(positions.ndim == 4 && positions.dim(-1) == 2)
    precondition(positions.dim(1) == 1, "audio-only 1D positions")
    let inv = dramaBoxGenerateFreqGrid(theta: Double(theta), nPosDims: 1, innerDim: innerDim)
    let start = positions[.ellipsis, 0]
    let end = positions[.ellipsis, 1]
    let middle = ((start + end) / 2.0)[0..., 0, 0...]
    let scaled = (middle / maxPos) * 2.0 - 1.0
    let freqs = scaled.expandedDimensions(axis: -1) * inv.reshaped([1, 1, inv.dim(0)])

    var cos = MLX.cos(freqs)
    var sin = MLX.sin(freqs)
    let expected = innerDim / 2
    let padSize = expected - cos.dim(-1)
    let B = cos.dim(0)
    let T = cos.dim(1)
    if padSize > 0 {
        cos = concatenated([MLXArray.ones([B, T, padSize], dtype: cos.dtype), cos], axis: -1)
        sin = concatenated([MLXArray.zeros([B, T, padSize], dtype: sin.dtype), sin], axis: -1)
    }
    cos = cos.reshaped(B, T, numHeads, -1).transposed(0, 2, 1, 3)
    sin = sin.reshaped(B, T, numHeads, -1).transposed(0, 2, 1, 3)
    return (cos.asType(outDtype), sin.asType(outDtype))
}

func dramaBoxApplySplitRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let B = x.dim(0)
    let T = x.dim(1)
    let H = cos.dim(1)
    let D = cos.dim(-1) * 2
    var work = x.reshaped(B, T, H, D).transposed(0, 2, 1, 3)
    let half = D / 2
    let first = work[.ellipsis, 0..<half]
    let second = work[.ellipsis, half...]
    let newFirst = first * cos - second * sin
    let newSecond = second * cos + first * sin
    work = concatenated([newFirst, newSecond], axis: -1)
    return work.transposed(0, 2, 1, 3).reshaped(B, T, H * D)
}
