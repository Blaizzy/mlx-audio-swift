import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Instance Normalization

public class InstanceNorm1d: Module {
    public let numFeatures: Int
    public let eps: Float

    public init(numFeatures: Int, eps: Float = 1e-5) {
        self.numFeatures = numFeatures
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.variance(x, axis: -1, keepDims: true)
        return (x - mean) / MLX.sqrt(variance + eps)
    }
}

// MARK: - Adaptive Instance Normalization

public class AdaIN1d: Module {
    @ModuleInfo public var norm: InstanceNorm1d
    @ModuleInfo public var fc: Linear

    public init(styleDim: Int, numFeatures: Int) {
        _norm = ModuleInfo(wrappedValue: InstanceNorm1d(numFeatures: numFeatures))
        _fc = ModuleInfo(wrappedValue: Linear(styleDim, numFeatures * 2))
    }

    public func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var h = fc(s)
        h = h.expandedDimensions(axis: 2)
        let parts = MLX.split(h, parts: 2, axis: 1)
        let gamma = parts[0]
        let beta = parts[1]
        return (1 + gamma) * norm(x) + beta
    }
}

// MARK: - Adaptive Layer Normalization

public class AdaLayerNorm: Module {
    public let channels: Int
    public let eps: Float
    @ModuleInfo public var fc: Linear

    public init(styleDim: Int, channels: Int, eps: Float = 1e-5) {
        self.channels = channels
        self.eps = eps
        _fc = ModuleInfo(wrappedValue: Linear(styleDim, channels * 2))
    }

    public func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var h = fc(s)
        h = h.reshaped([h.shape[0], h.shape[1], 1])
        let parts = MLX.split(h, parts: 2, axis: 1)
        let gamma = parts[0].transposed(2, 0, 1)
        let beta = parts[1].transposed(2, 0, 1)

        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let variance = MLX.variance(x, axis: -1, keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)

        return (1 + gamma) * normalized + beta
    }
}
