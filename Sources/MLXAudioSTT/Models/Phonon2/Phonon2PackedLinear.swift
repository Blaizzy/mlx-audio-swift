// Packed layer integration adapted from fermion-research 0.2.11,
// Copyright 2026 Fermion Research, Apache-2.0.
// SPDX-License-Identifier: Apache-2.0
import MLX
import MLXNN

/// A linear or kernel-1 convolution preserving Phonon-2's learned five-value palette.
final class Phonon2PackedLinear: Module, UnaryLayer {
    @ParameterInfo(key: "base_q") var baseQ: MLXArray
    @ParameterInfo(key: "residual_q") var residualQ: MLXArray
    @ParameterInfo(key: "row_lo") var rowLo: MLXArray
    @ParameterInfo(key: "row_hi") var rowHi: MLXArray
    let bias: MLXArray?
    let inputDimensions: Int
    let outputDimensions: Int

    private var _baseScales: MLXArray?
    private var _residualScales: MLXArray?

    init(inputDimensions: Int, outputDimensions: Int, bias: Bool) {
        precondition(inputDimensions > 0 && inputDimensions % 128 == 0 && outputDimensions > 0)
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
        _baseQ.wrappedValue = MLXArray.zeros([outputDimensions, inputDimensions / 16], dtype: .uint32)
        _residualQ.wrappedValue = MLXArray.zeros([outputDimensions, inputDimensions / 16], dtype: .uint32)
        _rowLo.wrappedValue = MLXArray.zeros([outputDimensions], dtype: .float16)
        _rowHi.wrappedValue = MLXArray.zeros([outputDimensions], dtype: .float16)
        self.bias = bias ? MLXArray.zeros([outputDimensions], dtype: .bfloat16) : nil
    }

    func materialize() {
        let shape = [outputDimensions, inputDimensions / 128]
        // Subtract after converting to fp32: rounding hi-lo in fp16 changes learned weights.
        let lo = rowLo.asType(.float32)
        let delta = rowHi.asType(.float32) - lo
        _baseScales = contiguous(broadcast(lo.expandedDimensions(axis: 1), to: shape))
        _residualScales = contiguous(broadcast(delta.expandedDimensions(axis: 1), to: shape))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(_baseScales != nil && _residualScales != nil, "Materialize packed scales after loading weights")
        let baseScales = _baseScales!
        let residualScales = _residualScales!
        let input = x.asType(.float32)
        let a = quantizedMM(input, baseQ, scales: baseScales, biases: -baseScales, groupSize: 128, bits: 2)
        let b = quantizedMM(input, residualQ, scales: residualScales, biases: -residualScales, groupSize: 128, bits: 2)
        var y = a + b
        if let bias { y = y + bias.asType(.float32) }
        return y.asType(x.dtype)
    }
}
