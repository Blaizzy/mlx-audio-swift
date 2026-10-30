import Foundation
import MLX
import MLXAudioCore
import MLXNN

// Copyright © 2023-2024 Apple Inc. Batching adapted from mlx-lm (MIT).

typealias MingSwitchLinear = ExpertLinear
typealias MingQuantizedSwitchLinear = QuantizedExpertLinear

final class MingSwitchMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: MingSwitchLinear
    @ModuleInfo(key: "up_proj") var up: MingSwitchLinear
    @ModuleInfo(key: "down_proj") var down: MingSwitchLinear
    let limit: Float?
    convenience init(_ c: MingTinyTextConfiguration, limit: Float?) {
        self.init(
            hidden: c.hiddenSize, intermediate: c.moeIntermediateSize, experts: c.numExperts,
            limit: limit)
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
        var g = silu(gate)
        var u = up
        if let limit, limit > 0 {
            g = minimum(g, limit)
            u = clip(u, min: -limit, max: limit)
        }
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
        let activated = activate(
            up: up(input, indices: selected, sorted: sorted),
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
