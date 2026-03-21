import Foundation
@preconcurrency import MLX
import MLXNN

class KittenLinearNorm: Module {
    @ModuleInfo(key: "linear_layer") var linearLayer: Linear

    init(inDim: Int, outDim: Int) {
        _linearLayer = ModuleInfo(wrappedValue: Linear(inDim, outDim), key: "linear_layer")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearLayer(x)
    }
}

class KittenAdaLayerNorm: Module {
    let channels: Int
    let eps: Float
    @ModuleInfo var fc: Linear

    init(styleDim: Int, channels: Int, eps: Float = 1e-5) {
        self.channels = channels
        self.eps = eps
        _fc = ModuleInfo(wrappedValue: Linear(styleDim, channels * 2))
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
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

class KittenTextEncoder: Module {
    @ModuleInfo var embedding: Embedding
    // cnn is list[list[Module]] in Python: cnn.0.0 = ConvWeighted, cnn.0.1 = LayerNorm
    @ModuleInfo var cnn: [[Module]]
    @ModuleInfo var lstm: KittenBiLSTM

    init(channels: Int, kernelSize: Int, depth: Int, nSymbols: Int) {
        _embedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: nSymbols, dimensions: channels))
        let padding = (kernelSize - 1) / 2
        _cnn = ModuleInfo(wrappedValue: (0..<depth).map { _ -> [Module] in
            [
                KittenConvWeighted(inChannels: channels, outChannels: channels, kernelSize: kernelSize, padding: padding),
                LayerNorm(dimensions: channels),
            ]
        })
        _lstm = ModuleInfo(wrappedValue: KittenBiLSTM(inputSize: channels, hiddenSize: channels / 2))
    }

    func callAsFunction(_ x: MLXArray, inputLengths: MLXArray, mask: MLXArray) -> MLXArray {
        var h = embedding(x)
        h = h.transposed(0, 2, 1)
        let m = mask.expandedDimensions(axis: 1)
        h = MLX.where(m, MLXArray(Float(0)), h)

        for block in cnn {
            let conv = block[0] as! KittenConvWeighted
            let ln = block[1] as! LayerNorm

            h = conv(h.swappedAxes(2, 1), op: .conv1d).swappedAxes(2, 1)
            h = ln(h.swappedAxes(2, 1)).swappedAxes(2, 1)
            h = MLX.where(m, MLXArray(Float(0)), h)
            h = kittenLeakyRelu(h, negativeSlope: 0.2)
            h = MLX.where(m, MLXArray(Float(0)), h)
        }

        h = h.swappedAxes(2, 1)
        let (lstmOut, _) = lstm(h)
        h = lstmOut.swappedAxes(2, 1)

        let padded = MLXArray.zeros([h.shape[0], h.shape[1], mask.shape[mask.ndim - 1]])
        let validLen = min(h.shape[2], padded.shape[2])
        var result = padded
        result[0..., 0..., ..<validLen] = h[0..., 0..., ..<validLen]
        result = MLX.where(m, MLXArray(Float(0)), result)
        return result
    }
}

class KittenDurationEncoder: Module {
    let dModel: Int
    let styDim: Int
    let nLayers: Int
    // Flat array: [LSTM, AdaLayerNorm, LSTM, AdaLayerNorm, ...]
    @ModuleInfo var lstms: [Module]

    init(styDim: Int, dModel: Int, nLayers: Int, dropout: Float = 0.1) {
        self.dModel = dModel
        self.styDim = styDim
        self.nLayers = nLayers
        var allModules = [Module]()
        for _ in 0..<nLayers {
            allModules.append(KittenBiLSTM(inputSize: dModel + styDim, hiddenSize: dModel / 2))
            allModules.append(KittenAdaLayerNorm(styleDim: styDim, channels: dModel))
        }
        _lstms = ModuleInfo(wrappedValue: allModules)
    }

    func callAsFunction(_ x: MLXArray, style: MLXArray, textLengths: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x.transposed(2, 0, 1)
        let s = MLX.broadcast(style, to: [h.shape[0], h.shape[1], style.shape[style.ndim - 1]])
        h = MLX.concatenated([h, s], axis: -1)
        let mExpanded = mask[.ellipsis, (.newAxis)].transposed(1, 0, 2)
        h = MLX.where(mExpanded, MLXArray(Float(0)), h)
        h = h.transposed(1, 2, 0)

        for i in 0..<nLayers {
            let lstmLayer = lstms[i * 2] as! KittenBiLSTM
            let normLayer = lstms[i * 2 + 1] as! KittenAdaLayerNorm

            // Batch dim dropped here — matches Python: x.transpose(0, 2, 1)[0]
            let input = h.transposed(0, 2, 1)[0]
            let (out, _) = lstmLayer(input)
            h = out.transposed(0, 2, 1)
            let padded = MLXArray.zeros([h.shape[0], h.shape[1], mask.shape[mask.ndim - 1]])
            let validLen = min(h.shape[2], padded.shape[2])
            var result = padded
            result[0..., 0..., ..<validLen] = h[0..., 0..., ..<validLen]
            h = result

            // AdaLayerNorm step
            h = normLayer(h.transposed(0, 2, 1), style).transposed(0, 2, 1)
            h = MLX.concatenated([h, s.transposed(1, 2, 0)], axis: 1)
            let maskT = mask[.ellipsis, (.newAxis)].transposed(0, 2, 1)
            h = MLX.where(maskT, MLXArray(Float(0)), h)
        }
        return h.transposed(0, 2, 1)
    }
}

class KittenProsodyPredictor: Module {
    @ModuleInfo(key: "text_encoder") var textEncoder: KittenDurationEncoder
    @ModuleInfo var lstm: KittenBiLSTM
    @ModuleInfo(key: "duration_proj") var durationProj: KittenLinearNorm
    @ModuleInfo var shared: KittenBiLSTM
    @ModuleInfo(key: "F0") var f0Blocks: [KittenAdainResBlk1d]
    @ModuleInfo(key: "N") var nBlocks: [KittenAdainResBlk1d]
    @ModuleInfo(key: "F0_proj") var f0Proj: Conv1d
    @ModuleInfo(key: "N_proj") var nProj: Conv1d

    init(styleDim: Int, dHid: Int, nLayers: Int, maxDur: Int, dropout: Float = 0.0) {
        _textEncoder = ModuleInfo(wrappedValue: KittenDurationEncoder(styDim: styleDim, dModel: dHid, nLayers: nLayers, dropout: dropout), key: "text_encoder")
        _lstm = ModuleInfo(wrappedValue: KittenBiLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2))
        _durationProj = ModuleInfo(wrappedValue: KittenLinearNorm(inDim: dHid, outDim: maxDur), key: "duration_proj")
        _shared = ModuleInfo(wrappedValue: KittenBiLSTM(inputSize: dHid + styleDim, hiddenSize: dHid / 2))

        _f0Blocks = ModuleInfo(wrappedValue: [
            KittenAdainResBlk1d(dimIn: dHid, dimOut: dHid, styleDim: styleDim, dropoutP: dropout),
            KittenAdainResBlk1d(dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: true, dropoutP: dropout),
            KittenAdainResBlk1d(dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim, dropoutP: dropout),
        ], key: "F0")
        _nBlocks = ModuleInfo(wrappedValue: [
            KittenAdainResBlk1d(dimIn: dHid, dimOut: dHid, styleDim: styleDim, dropoutP: dropout),
            KittenAdainResBlk1d(dimIn: dHid, dimOut: dHid / 2, styleDim: styleDim, upsample: true, dropoutP: dropout),
            KittenAdainResBlk1d(dimIn: dHid / 2, dimOut: dHid / 2, styleDim: styleDim, dropoutP: dropout),
        ], key: "N")
        _f0Proj = ModuleInfo(wrappedValue: Conv1d(inputChannels: dHid / 2, outputChannels: 1, kernelSize: 1, padding: 0), key: "F0_proj")
        _nProj = ModuleInfo(wrappedValue: Conv1d(inputChannels: dHid / 2, outputChannels: 1, kernelSize: 1, padding: 0), key: "N_proj")
    }

    func f0Ntrain(_ x: MLXArray, _ s: MLXArray) -> (MLXArray, MLXArray) {
        let (sharedOut, _) = shared(x.transposed(0, 2, 1))
        var f0 = sharedOut.transposed(0, 2, 1)
        for block in f0Blocks {
            f0 = block(f0, s)
        }
        f0 = f0Proj(f0.swappedAxes(2, 1)).swappedAxes(2, 1)

        var n = sharedOut.transposed(0, 2, 1)
        for block in nBlocks {
            n = block(n, s)
        }
        n = nProj(n.swappedAxes(2, 1)).swappedAxes(2, 1)

        return (f0.squeezed(axis: 1), n.squeezed(axis: 1))
    }
}

