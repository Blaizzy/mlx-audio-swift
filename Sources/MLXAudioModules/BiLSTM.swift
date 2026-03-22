import Foundation
@preconcurrency import MLX
import MLXNN

public class BiLSTM: Module {
    public let inputSize: Int
    public let hiddenSize: Int

    public var Wx_forward: MLXArray
    public var Wh_forward: MLXArray
    public var bias_ih_forward: MLXArray
    public var bias_hh_forward: MLXArray

    public var Wx_backward: MLXArray
    public var Wh_backward: MLXArray
    public var bias_ih_backward: MLXArray
    public var bias_hh_backward: MLXArray

    public init(inputSize: Int, hiddenSize: Int) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        let scale = 1.0 / Float(hiddenSize).squareRoot()

        Wx_forward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, inputSize])
        Wh_forward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
        bias_ih_forward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
        bias_hh_forward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])

        Wx_backward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, inputSize])
        Wh_backward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize, hiddenSize])
        bias_ih_backward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
        bias_hh_backward = MLXRandom.uniform(low: -scale, high: scale, [4 * hiddenSize])
    }

    private func forwardDirection(_ x: MLXArray) -> MLXArray {
        let xProj = MLX.addmm(bias_ih_forward + bias_hh_forward, x, Wx_forward.transposed())

        let seqLen = x.shape[x.ndim - 2]
        let batchSize = x.ndim == 3 ? x.shape[0] : 1
        var hidden = MLXArray.zeros([batchSize, hiddenSize])
        var cell = MLXArray.zeros([batchSize, hiddenSize])
        var allHidden = [MLXArray]()
        allHidden.reserveCapacity(seqLen)

        for idx in 0..<seqLen {
            var ifgo = xProj[.ellipsis, idx, 0...]
            ifgo = ifgo + MLX.matmul(hidden, Wh_forward.transposed())

            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])

            cell = f * cell + i * g
            hidden = o * MLX.tanh(cell)
            allHidden.append(hidden)
        }

        return MLX.stacked(allHidden, axis: -2)
    }

    private func backwardDirection(_ x: MLXArray) -> MLXArray {
        let xProj = MLX.addmm(bias_ih_backward + bias_hh_backward, x, Wx_backward.transposed())

        let seqLen = x.shape[x.ndim - 2]
        let batchSize = x.ndim == 3 ? x.shape[0] : 1
        var hidden = MLXArray.zeros([batchSize, hiddenSize])
        var cell = MLXArray.zeros([batchSize, hiddenSize])
        var allHidden = [MLXArray]()
        allHidden.reserveCapacity(seqLen)

        for idx in stride(from: seqLen - 1, through: 0, by: -1) {
            var ifgo = xProj[.ellipsis, idx, 0...]
            ifgo = ifgo + MLX.matmul(hidden, Wh_backward.transposed())

            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])

            cell = f * cell + i * g
            hidden = o * MLX.tanh(cell)
            allHidden.append(hidden)
        }

        return MLX.stacked(allHidden.reversed(), axis: -2)
    }

    public func callAsFunction(_ x: MLXArray) -> (MLXArray, ()) {
        var input = x
        if input.ndim == 2 {
            input = input.expandedDimensions(axis: 0)
        }
        let fwd = forwardDirection(input)
        let bwd = backwardDirection(input)
        let output = MLX.concatenated([fwd, bwd], axis: -1)
        return (output, ())
    }
}
