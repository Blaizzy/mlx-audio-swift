import Foundation
@preconcurrency import MLX

// MARK: - 1D Linear Interpolation

public func interpolate1d(_ input: MLXArray, size: Int) -> MLXArray {
    let inWidth = input.shape[2]
    if inWidth == size { return input }
    if inWidth < 1 || size < 1 { return input }

    let scale = Float(inWidth) / Float(size)
    var xCoords = MLXArray(Array((0..<size).map { Float($0) * scale + 0.5 * scale - 0.5 }))
    xCoords = MLX.clip(xCoords, min: 0, max: Float(inWidth - 1))

    let xLow = MLX.floor(xCoords).asType(.int32)
    let xHigh = MLX.minimum(xLow + 1, MLXArray(Int32(inWidth - 1)))
    let xFrac = xCoords - xLow.asType(.float32)

    let yLow = input[0..., 0..., xLow]
    let yHigh = input[0..., 0..., xHigh]
    let fracExpanded = xFrac.reshaped([1, 1, size])
    return yLow * (1 - fracExpanded) + yHigh * fracExpanded
}
