// Copyright © 2023-2024 Apple Inc. Adapted from mlx-lm (MIT).

import Foundation
import MLX
import MLXNN

/// A bank of expert projections selected without materializing unused weights.
public class ExpertLinear: Module, Quantizable {
    public let weight: MLXArray
    public init(input: Int, output: Int, experts: Int) {
        weight = MLXRandom.normal([experts, output, input]) * sqrt(1 / Float(input))
    }
    public init(weight: MLXArray) { self.weight = weight }
    public func callAsFunction(_ x: MLXArray, indices: MLXArray, sorted: Bool = false) -> MLXArray {
        gatherMM(x, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
    }
    public func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedExpertLinear(weight: weight, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public final class QuantizedExpertLinear: ExpertLinear, Quantized {
    public let scales: MLXArray
    public let biases: MLXArray?
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public init(weight: MLXArray, groupSize: Int, bits: Int, mode: QuantizationMode) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let packed = MLX.quantized(weight, groupSize: groupSize, bits: bits, mode: mode)
        scales = packed.1
        biases = packed.2
        super.init(weight: packed.0)
        freeze()
    }
    public override func callAsFunction(_ x: MLXArray, indices: MLXArray, sorted: Bool = false)
        -> MLXArray
    {
        gatherQuantizedMM(
            x, weight, scales: scales, biases: biases, rhsIndices: indices,
            groupSize: groupSize, bits: bits, mode: mode, sortedIndices: sorted)
    }
}
