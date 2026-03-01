import Foundation
import MLX
import MLXNN

typealias ParakeetLSTMState = (hidden: MLXArray?, cell: MLXArray?)

final class ParakeetStackedLSTM: Module {
    let numLayers: Int
    let batchFirst: Bool

    @ModuleInfo(key: "lstm") var layers: [LSTM]

    init(
        inputSize: Int,
        hiddenSize: Int,
        numLayers: Int = 1,
        bias: Bool = true,
        batchFirst: Bool = true
    ) {
        self.numLayers = numLayers
        self.batchFirst = batchFirst

        var lstmLayers: [LSTM] = []
        lstmLayers.reserveCapacity(numLayers)
        for i in 0..<numLayers {
            let inSize = i == 0 ? inputSize : hiddenSize
            lstmLayers.append(LSTM(inputSize: inSize, hiddenSize: hiddenSize, bias: bias))
        }
        self._layers.wrappedValue = lstmLayers
    }

    func callAsFunction(_ x: MLXArray, state: ParakeetLSTMState? = nil) -> (MLXArray, ParakeetLSTMState) {
        var output = batchFirst ? x.transposed(1, 0, 2) : x

        var hiddenByLayer: [MLXArray?] = Array(repeating: nil, count: numLayers)
        var cellByLayer: [MLXArray?] = Array(repeating: nil, count: numLayers)

        if let hidden = state?.hidden, hidden.shape[0] == numLayers {
            let split = hidden.split(parts: numLayers, axis: 0)
            for i in 0..<numLayers {
                hiddenByLayer[i] = split[i].squeezed(axis: 0)
            }
        }
        if let cell = state?.cell, cell.shape[0] == numLayers {
            let split = cell.split(parts: numLayers, axis: 0)
            for i in 0..<numLayers {
                cellByLayer[i] = split[i].squeezed(axis: 0)
            }
        }

        var nextHidden: [MLXArray] = []
        var nextCell: [MLXArray] = []

        for i in 0..<numLayers {
            let layer = layers[i]
            let (allH, allC) = layer(output, hidden: hiddenByLayer[i], cell: cellByLayer[i])
            output = allH
            nextHidden.append(allH[allH.shape[0] - 1])
            nextCell.append(allC[allC.shape[0] - 1])
        }

        if batchFirst {
            output = output.transposed(1, 0, 2)
        }

        return (
            output,
            (
                hidden: nextHidden.isEmpty ? nil : MLX.stacked(nextHidden, axis: 0),
                cell: nextCell.isEmpty ? nil : MLX.stacked(nextCell, axis: 0)
            )
        )
    }
}

final class ParakeetRNNTPrediction: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "dec_rnn") var decRnn: ParakeetStackedLSTM

    init(args: ParakeetPredictConfig) {
        let embeddingCount = args.blankAsPad ? args.vocabSize + 1 : args.vocabSize
        self._embed.wrappedValue = Embedding(embeddingCount: embeddingCount, dimensions: args.prednet.predHidden)
        self._decRnn.wrappedValue = ParakeetStackedLSTM(
            inputSize: args.prednet.predHidden,
            hiddenSize: args.prednet.rnnHiddenSize ?? args.prednet.predHidden,
            numLayers: args.prednet.predRnnLayers
        )
    }
}

final class ParakeetPredictNetwork: Module {
    let predHidden: Int
    @ModuleInfo(key: "prediction") var prediction: ParakeetRNNTPrediction

    init(args: ParakeetPredictConfig) {
        self.predHidden = args.prednet.predHidden
        self._prediction.wrappedValue = ParakeetRNNTPrediction(args: args)
    }

    func callAsFunction(_ token: MLXArray?, state: ParakeetLSTMState? = nil) -> (MLXArray, ParakeetLSTMState) {
        let embedded: MLXArray
        if let token {
            embedded = prediction.embed(token)
        } else {
            let batch = state?.hidden?.shape[1] ?? 1
            embedded = MLXArray.zeros([batch, 1, predHidden], type: Float.self)
        }
        return prediction.decRnn(embedded, state: state)
    }
}

final class ParakeetJointNetwork: Module {
    let numClasses: Int

    @ModuleInfo(key: "pred") var pred: Linear
    @ModuleInfo(key: "enc") var enc: Linear
    @ModuleInfo(key: "joint_net") var outputProj: Linear

    let activationName: String

    init(args: ParakeetJointConfig) {
        self.numClasses = args.numClasses + 1 + args.numExtraOutputs
        self.activationName = args.jointnet.activation.lowercased()

        self._pred.wrappedValue = Linear(args.jointnet.predHidden, args.jointnet.jointHidden)
        self._enc.wrappedValue = Linear(args.jointnet.encoderHidden, args.jointnet.jointHidden)
        self._outputProj.wrappedValue = Linear(args.jointnet.jointHidden, numClasses)
    }

    func callAsFunction(_ encOut: MLXArray, _ predOut: MLXArray) -> MLXArray {
        let encProjected = enc(encOut)
        let predProjected = pred(predOut)
        var x = encProjected.expandedDimensions(axis: 2) + predProjected.expandedDimensions(axis: 1)

        switch activationName {
        case "relu":
            x = relu(x)
        case "sigmoid":
            x = sigmoid(x)
        default:
            x = tanh(x)
        }

        return outputProj(x)
    }
}
