import Foundation
import MLX
import MLXNN

// MARK: - Duration Encoder

class KokoroDurationEncoder {
    var lstms: [(isLSTM: Bool, lstm: KokoroBiLSTM?, adaLN: KokoroAdaLayerNorm?)] = []

    init(weights: [String: MLXArray], dModel: Int, styDim: Int, nlayers: Int) {
        for i in 0..<(nlayers * 2) {
            if i % 2 == 0 {
                let lstm = KokoroBiLSTM(
                    inputSize: dModel + styDim, hiddenSize: dModel / 2,
                    wxForward: weights["predictor.text_encoder.lstms.\(i).weight_ih_l0"]!,
                    whForward: weights["predictor.text_encoder.lstms.\(i).weight_hh_l0"]!,
                    biasIhForward: weights["predictor.text_encoder.lstms.\(i).bias_ih_l0"]!,
                    biasHhForward: weights["predictor.text_encoder.lstms.\(i).bias_hh_l0"]!,
                    wxBackward: weights["predictor.text_encoder.lstms.\(i).weight_ih_l0_reverse"]!,
                    whBackward: weights["predictor.text_encoder.lstms.\(i).weight_hh_l0_reverse"]!,
                    biasIhBackward: weights["predictor.text_encoder.lstms.\(i).bias_ih_l0_reverse"]!,
                    biasHhBackward: weights["predictor.text_encoder.lstms.\(i).bias_hh_l0_reverse"]!
                )
                lstms.append((isLSTM: true, lstm: lstm, adaLN: nil))
            } else {
                let adaLN = KokoroAdaLayerNorm(
                    weight: weights["predictor.text_encoder.lstms.\(i).fc.weight"]!,
                    bias: weights["predictor.text_encoder.lstms.\(i).fc.bias"]!
                )
                lstms.append((isLSTM: false, lstm: nil, adaLN: adaLN))
            }
        }
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray, textLengths: MLXArray, m: MLXArray) -> MLXArray {
        var x = x.transposed(2, 0, 1)
        let s = MLX.broadcast(style, to: [x.shape[0], x.shape[1], style.shape[style.shape.count - 1]])
        x = MLX.concatenated([x, s], axis: -1)
        x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(1, 0, 2), MLXArray.zeros(like: x), x)
        x = x.transposed(1, 2, 0)

        for block in lstms {
            if block.isLSTM, let lstm = block.lstm {
                x = x.transposed(0, 2, 1)[0]
                let (lstmOutput, _) = lstm(x)
                x = lstmOutput.transposed(0, 2, 1)
                let xPad = MLXArray.zeros([x.shape[0], x.shape[1], m.shape[m.shape.count - 1]])
                xPad[0..<x.shape[0], 0..<x.shape[1], 0..<x.shape[2]] = x
                x = xPad
            } else if let adaLN = block.adaLN {
                x = adaLN(x.transposed(0, 2, 1), style).transposed(0, 2, 1)
                x = MLX.concatenated([x, s.transposed(1, 2, 0)], axis: 1)
                x = MLX.where(m.expandedDimensions(axes: [-1]).transposed(0, 2, 1), MLXArray.zeros(like: x), x)
            }
        }
        return x.transposed(0, 2, 1)
    }
}

// MARK: - Text Encoder

class KokoroTextEncoder {
    let embedding: Embedding
    let cnn: [[(type: String, conv: KokoroConvWeighted?, norm: KokoroLayerNorm?, actv: LeakyReLU?)]]
    let lstm: KokoroBiLSTM

    init(weights: [String: MLXArray], channels: Int, kernelSize: Int, depth: Int) {
        embedding = Embedding(weight: weights["text_encoder.embedding.weight"]!)
        let padding = (kernelSize - 1) / 2

        var cnnLayers: [[(type: String, conv: KokoroConvWeighted?, norm: KokoroLayerNorm?, actv: LeakyReLU?)]] = []
        for i in 0..<depth {
            cnnLayers.append([
                (type: "conv", conv: KokoroConvWeighted(
                    weightG: weights["text_encoder.cnn.\(i).0.weight_g"]!,
                    weightV: weights["text_encoder.cnn.\(i).0.weight_v"]!,
                    bias: weights["text_encoder.cnn.\(i).0.bias"]!,
                    padding: padding
                ), norm: nil, actv: nil),
                (type: "norm", conv: nil, norm: KokoroLayerNorm(
                    weight: weights["text_encoder.cnn.\(i).1.gamma"]!,
                    bias: weights["text_encoder.cnn.\(i).1.beta"]!
                ), actv: nil),
                (type: "actv", conv: nil, norm: nil, actv: LeakyReLU(negativeSlope: 0.2)),
            ])
        }
        cnn = cnnLayers

        lstm = KokoroBiLSTM(
            inputSize: channels, hiddenSize: channels / 2,
            wxForward: weights["text_encoder.lstm.weight_ih_l0"]!,
            whForward: weights["text_encoder.lstm.weight_hh_l0"]!,
            biasIhForward: weights["text_encoder.lstm.bias_ih_l0"]!,
            biasHhForward: weights["text_encoder.lstm.bias_hh_l0"]!,
            wxBackward: weights["text_encoder.lstm.weight_ih_l0_reverse"]!,
            whBackward: weights["text_encoder.lstm.weight_hh_l0_reverse"]!,
            biasIhBackward: weights["text_encoder.lstm.bias_ih_l0_reverse"]!,
            biasHhBackward: weights["text_encoder.lstm.bias_hh_l0_reverse"]!
        )
    }

    func callAsFunction(_ x: MLXArray, inputLengths: MLXArray, m: MLXArray) -> MLXArray {
        var x = embedding(x).transposed(0, 2, 1)
        let mask = m.expandedDimensions(axis: 1)
        x = MLX.where(mask, 0.0, x)

        for convBlock in cnn {
            for layer in convBlock {
                switch layer.type {
                case "conv":
                    x = MLX.swappedAxes(x, 2, 1)
                    x = layer.conv!(x, conv: MLX.conv1d)
                    x = MLX.swappedAxes(x, 2, 1)
                case "norm":
                    x = MLX.swappedAxes(x, 2, 1)
                    x = layer.norm!(x)
                    x = MLX.swappedAxes(x, 2, 1)
                case "actv":
                    x = layer.actv!(x)
                default: break
                }
                x = MLX.where(mask, 0.0, x)
            }
        }

        x = MLX.swappedAxes(x, 2, 1)
        let (lstmOutput, _) = lstm(x)
        x = MLX.swappedAxes(lstmOutput, 2, 1)

        let xPad = MLX.zeros([x.shape[0], x.shape[1], mask.shape[mask.shape.count - 1]])
        xPad._updateInternal(x)
        return MLX.where(mask, 0.0, xPad)
    }
}

// MARK: - Prosody Predictor

class KokoroProsodyPredictor {
    let shared: KokoroBiLSTM
    let f0Blocks: [KokoroAdainResBlk1d]
    let nBlocks: [KokoroAdainResBlk1d]
    let f0Proj: KokoroConv1d
    let nProj: KokoroConv1d

    init(weights: [String: MLXArray], styleDim: Int, dHid: Int) {
        shared = KokoroBiLSTM(
            inputSize: dHid + styleDim, hiddenSize: dHid / 2,
            wxForward: weights["predictor.shared.weight_ih_l0"]!,
            whForward: weights["predictor.shared.weight_hh_l0"]!,
            biasIhForward: weights["predictor.shared.bias_ih_l0"]!,
            biasHhForward: weights["predictor.shared.bias_hh_l0"]!,
            wxBackward: weights["predictor.shared.weight_ih_l0_reverse"]!,
            whBackward: weights["predictor.shared.weight_hh_l0_reverse"]!,
            biasIhBackward: weights["predictor.shared.bias_ih_l0_reverse"]!,
            biasHhBackward: weights["predictor.shared.bias_hh_l0_reverse"]!
        )

        f0Blocks = [
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "predictor.F0.0", dimIn: dHid, dimOut: dHid, styleDim: styleDim),
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "predictor.F0.1", dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: "true"),
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "predictor.F0.2", dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim),
        ]

        nBlocks = [
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "predictor.N.0", dimIn: dHid, dimOut: dHid, styleDim: styleDim),
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "predictor.N.1", dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: "true"),
            KokoroAdainResBlk1d(weights: weights, keyPrefix: "predictor.N.2", dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim),
        ]

        f0Proj = KokoroConv1d(
            weight: weights["predictor.F0_proj.weight"]!,
            bias: weights["predictor.F0_proj.bias"]!
        )
        nProj = KokoroConv1d(
            weight: weights["predictor.N_proj.weight"]!,
            bias: weights["predictor.N_proj.bias"]!
        )
    }

    func predict(x: MLXArray, s: MLXArray) -> (f0: MLXArray, n: MLXArray) {
        let (x1, _) = shared(x.transposed(0, 2, 1))

        var f0Val = x1.transposed(0, 2, 1)
        for block in f0Blocks { f0Val = block(x: f0Val, s: s) }
        f0Val = MLX.swappedAxes(f0Val, 2, 1)
        f0Val = f0Proj(f0Val)
        f0Val = MLX.swappedAxes(f0Val, 2, 1)

        var nVal = x1.transposed(0, 2, 1)
        for block in nBlocks { nVal = block(x: nVal, s: s) }
        nVal = MLX.swappedAxes(nVal, 2, 1)
        nVal = nProj(nVal)
        nVal = MLX.swappedAxes(nVal, 2, 1)

        return (f0Val.squeezed(axis: 1), nVal.squeezed(axis: 1))
    }
}
