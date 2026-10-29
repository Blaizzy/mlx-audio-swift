import Foundation
import MLX
import MLXNN

// Copyright © 2023-2024 Apple Inc.
// Expert projections and batching adapted from mlx-lm (MIT).
class MingSwitchLinear: Module, Quantizable {
    let weight: MLXArray
    init(input: Int, output: Int, experts: Int) {
        weight = MLXRandom.normal([experts, output, input]) * sqrt(1 / Float(input))
    }
    init(weight: MLXArray) { self.weight = weight }
    func callAsFunction(_ x: MLXArray, indices: MLXArray, sorted: Bool = false) -> MLXArray {
        gatherMM(x, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
    }
    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        MingQuantizedSwitchLinear(weight: weight, groupSize: groupSize, bits: bits, mode: mode)
    }
}
final class MingQuantizedSwitchLinear: MingSwitchLinear, Quantized {
    let scales: MLXArray
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    init(weight: MLXArray, groupSize: Int, bits: Int, mode: QuantizationMode) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let packed = MLX.quantized(weight, groupSize: groupSize, bits: bits, mode: mode)
        scales = packed.1
        biases = packed.2
        super.init(weight: packed.0)
        freeze()
    }
    override func callAsFunction(_ x: MLXArray, indices: MLXArray, sorted: Bool = false) -> MLXArray
    {
        gatherQuantizedMM(
            x, weight, scales: scales, biases: biases, rhsIndices: indices,
            groupSize: groupSize, bits: bits, mode: mode, sortedIndices: sorted)
    }
}
final class MingSwitchMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: MingSwitchLinear
    @ModuleInfo(key: "up_proj") var up: MingSwitchLinear
    @ModuleInfo(key: "down_proj") var down: MingSwitchLinear
    let limit: Float?
    convenience init(_ c: MingTinyTextConfiguration, limit: Float?) {
        self.init(hidden: c.hiddenSize, intermediate: c.moeIntermediateSize, experts: c.numExperts, limit: limit)
    }
    init(hidden: Int, intermediate: Int, experts: Int, limit: Float? = nil) {
        self.limit = limit
        _gate.wrappedValue = MingSwitchLinear(
            input: hidden, output: intermediate, experts: experts)
        _up.wrappedValue = MingSwitchLinear(
            input: hidden, output: intermediate, experts: experts)
        _down.wrappedValue = MingSwitchLinear(
            input: intermediate, output: hidden, experts: experts)
    }
    private func activate(up: MLXArray, gate: MLXArray) -> MLXArray {
        var g = silu(gate), u = up
        if let limit, limit > 0 { g = minimum(g, limit); u = clip(u, min: -limit, max: limit) }
        return g * u
    }
    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var input = x.expandedDimensions(axes: [-2, -3])
        var selected = indices
        let sorted = indices.size >= 64
        var inverse: MLXArray?
        if sorted {
            let flat = indices.reshaped(-1)
            let order = argSort(flat)
            inverse = argSort(order)
            input = input.reshaped(-1, 1, x.dim(-1))[order.floorDivide(indices.dim(-1))]
            selected = flat[order]
        }
        let activated = activate(up: up(input, indices: selected, sorted: sorted),
                     gate: gate(input, indices: selected, sorted: sorted))
        var output = down(activated, indices: selected, sorted: sorted)
        if let inverse {
            output = output[inverse].reshaped(indices.shape + [1, x.dim(-1)])
        }
        return output.squeezed(axis: -2)
    }
}

// Retain the established internal names for other model families sharing these projections.
typealias MingTinySwitchLinear = MingSwitchLinear
typealias MingTinyQuantizedSwitchLinear = MingQuantizedSwitchLinear
typealias MingTinySwitchMLP = MingSwitchMLP
