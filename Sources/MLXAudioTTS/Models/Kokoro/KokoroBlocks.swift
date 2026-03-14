import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

// MARK: - Instance Normalization

class KokoroInstanceNorm1d {
    let numFeatures: Int
    let eps: Float

    init(numFeatures: Int, eps: Float = 1e-5) {
        self.numFeatures = numFeatures
        self.eps = eps
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let x: MLXArray
        let wasTwoD = input.ndim == 2
        if wasTwoD {
            x = input.expandedDimensions(axis: 0)
        } else {
            x = input
        }
        // x is [batch, features, length] — normalize over length (axis 2)
        let mean = MLX.mean(x, axes: [2], keepDims: true)
        let variance = MLX.variance(x, axes: [2], keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)
        return wasTwoD ? normalized.squeezed(axes: [0]) : normalized
    }
}

// MARK: - Layer Norm (Inference-only wrapper)

class KokoroLayerNorm: Module {
    let weight: MLXArray?
    let bias: MLXArray?
    let eps: Float

    init(weight: MLXArray, bias: MLXArray?, eps: Float = 1e-5) {
        self.weight = weight
        self.bias = bias
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

// MARK: - Conv1d with Weight Normalization

class KokoroConvWeighted: Module {
    var weightG: MLXArray
    var weightV: MLXArray
    var bias: MLXArray?
    let stride: Int
    let padding: Int
    let dilation: Int
    let outputPadding: Int
    let groups: Int

    init(
        weightG: MLXArray, weightV: MLXArray, bias: MLXArray?,
        stride: Int = 1, padding: Int = 1, dilation: Int = 1,
        outputPadding: Int = 0, groups: Int = 1
    ) {
        self.weightG = weightG
        self.weightV = weightV
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.outputPadding = outputPadding
        self.groups = groups
        super.init()
    }

    private func computeWeight() -> MLXArray {
        let axes = Array(1..<weightV.ndim)
        let normV = MLX.sqrt(MLX.sum(weightV * weightV, axes: axes, keepDims: true))
        return (weightV / (normV + 1e-7)) * weightG
    }

    /// Forward pass for regular conv1d.
    func callAsFunction(
        _ x: MLXArray,
        conv: (MLXArray, MLXArray, Int, Int, Int, Int, StreamOrDevice) -> MLXArray
    ) -> MLXArray {
        let w = computeWeight()
        let reshapedBias = bias?.reshaped([1, 1, -1])
        let useWeight = (x.shape.last == w.shape.last || groups > 1) ? w : w.transposed()
        let result = conv(x, useWeight, stride, padding, dilation, groups, .default)
        return reshapedBias.map { result + $0 } ?? result
    }

    /// Forward pass for transposed conv1d (with output padding parameter).
    func callAsFunction(
        _ x: MLXArray,
        conv: (MLXArray, MLXArray, Int, Int, Int, Int, Int, StreamOrDevice) -> MLXArray
    ) -> MLXArray {
        let w = computeWeight()
        let reshapedBias = bias?.reshaped([1, 1, -1])
        let useWeight = (x.shape.last == w.shape.last || groups > 1) ? w : w.transposed()
        let result = conv(x, useWeight, stride, padding, dilation, outputPadding, groups, .default)
        return reshapedBias.map { result + $0 } ?? result
    }
}

// MARK: - Simple Conv1d (inference-only)

class KokoroConv1d {
    let weight: MLXArray
    let bias: MLXArray?
    let padding: Int
    let dilation: Int
    let stride: Int
    let groups: Int

    init(
        weight: MLXArray, bias: MLXArray? = nil,
        stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1
    ) {
        self.weight = weight
        self.bias = bias
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias { y = y + bias }
        return y
    }
}

// MARK: - Adaptive Instance Normalization (1D)

class KokoroAdaIN1d {
    private let norm: KokoroInstanceNorm1d
    private let fc: Linear

    init(numFeatures: Int, fcWeight: MLXArray, fcBias: MLXArray) {
        norm = KokoroInstanceNorm1d(numFeatures: numFeatures)
        fc = Linear(weight: fcWeight, bias: fcBias)
    }

    func callAsFunction(_ x: MLXArray, s: MLXArray) -> MLXArray {
        let h = fc(s).expandedDimensions(axes: [2])
        let split = h.split(parts: 2, axis: 1)
        return (1 + split[0]) * norm(x) + split[1]
    }
}

// MARK: - Adaptive Layer Normalization

class KokoroAdaLayerNorm: Module {
    let eps: Float
    let fc: Linear

    init(weight: MLXArray, bias: MLXArray?, eps: Float = 1e-5) {
        self.eps = eps
        fc = Linear(weight: weight, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let h = fc(s).reshaped([fc(s).shape[0], fc(s).shape[1], 1])
        let split = h.split(parts: 2, axis: 1)
        let gamma = split[0].transposed(2, 0, 1)
        let beta = split[1].transposed(2, 0, 1)
        let mean = MLX.mean(x, axes: [-1], keepDims: true)
        let variance = MLX.variance(x, axes: [-1], keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)
        return (1 + gamma) * normalized + beta
    }
}

// MARK: - UpSample1d

class KokoroUpSample1d {
    private let layerType: String
    private let interpolate: Upsample

    init(layerType: String) {
        self.layerType = layerType
        interpolate = Upsample(scaleFactor: 2.0, mode: .nearest)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layerType == "none" ? x : interpolate(x)
    }
}

// MARK: - Reflection Pad 1D

class KokoroReflectionPad1d: Module {
    let padding: IntOrPair

    init(padding: (Int, Int)) {
        self.padding = IntOrPair([padding.0, padding.1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.padded(x, widths: [IntOrPair([0, 0]), IntOrPair([0, 0]), padding])
    }
}

// MARK: - Bidirectional LSTM

class KokoroBiLSTM: Module {
    let inputSize: Int
    let hiddenSize: Int

    var wxForward: MLXArray
    var whForward: MLXArray
    var biasIhForward: MLXArray?
    var biasHhForward: MLXArray?
    var wxBackward: MLXArray
    var whBackward: MLXArray
    var biasIhBackward: MLXArray?
    var biasHhBackward: MLXArray?

    init(
        inputSize: Int, hiddenSize: Int,
        wxForward: MLXArray, whForward: MLXArray,
        biasIhForward: MLXArray? = nil, biasHhForward: MLXArray? = nil,
        wxBackward: MLXArray, whBackward: MLXArray,
        biasIhBackward: MLXArray? = nil, biasHhBackward: MLXArray? = nil
    ) {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.wxForward = wxForward
        self.whForward = whForward
        self.biasIhForward = biasIhForward
        self.biasHhForward = biasHhForward
        self.wxBackward = wxBackward
        self.whBackward = whBackward
        self.biasIhBackward = biasIhBackward
        self.biasHhBackward = biasHhBackward
        super.init()
    }

    private func processDirection(
        _ x: MLXArray, wx: MLXArray, wh: MLXArray,
        biasIh: MLXArray?, biasHh: MLXArray?, reverse: Bool
    ) -> (MLXArray, MLXArray) {
        let xProj: MLXArray
        if let biasIh, let biasHh {
            xProj = MLX.addMM(biasIh + biasHh, x, wx.transposed())
        } else {
            xProj = MLX.matmul(x, wx.transposed())
        }

        let seqLen = x.shape[x.shape.count - 2]
        var allHidden: [MLXArray] = []
        var allCell: [MLXArray] = []
        var h = MLXArray.zeros([x.shape[0], hiddenSize])
        var c = MLXArray.zeros([x.shape[0], hiddenSize])

        let indices = reverse ? Array(stride(from: seqLen - 1, through: 0, by: -1)) : Array(0..<seqLen)
        for idx in indices {
            var ifgo = xProj[0..., idx, 0...]
            ifgo = ifgo + MLX.matmul(h, wh.transposed())
            let gates = MLX.split(ifgo, parts: 4, axis: -1)
            let i = MLX.sigmoid(gates[0])
            let f = MLX.sigmoid(gates[1])
            let g = MLX.tanh(gates[2])
            let o = MLX.sigmoid(gates[3])
            c = f * c + i * g
            h = o * MLX.tanh(c)
            if reverse {
                allHidden.insert(h, at: 0)
                allCell.insert(c, at: 0)
            } else {
                allHidden.append(h)
                allCell.append(c)
            }
        }
        return (MLX.stacked(allHidden, axis: -2), MLX.stacked(allCell, axis: -2))
    }

    func callAsFunction(
        _ x: MLXArray
    ) -> (MLXArray, ((MLXArray, MLXArray), (MLXArray, MLXArray))) {
        let input = x.ndim == 2 ? x.expandedDimensions(axis: 0) : x
        let (fwdH, fwdC) = processDirection(
            input, wx: wxForward, wh: whForward,
            biasIh: biasIhForward, biasHh: biasHhForward, reverse: false
        )
        let (bwdH, bwdC) = processDirection(
            input, wx: wxBackward, wh: whBackward,
            biasIh: biasIhBackward, biasHh: biasHhBackward, reverse: true
        )
        let output = MLX.concatenated([fwdH, bwdH], axis: -1)
        return (
            output,
            (
                (fwdH[0..., -1, 0...], fwdC[0..., -1, 0...]),
                (bwdH[0..., 0, 0...], bwdC[0..., 0, 0...])
            )
        )
    }
}

// MARK: - AdaIN Residual Block 1D

class KokoroAdainResBlk1d {
    let actv: LeakyReLU
    let dimIn: Int
    let upsampleType: String
    let upsample: KokoroUpSample1d
    let learnedSc: Bool
    let pool: Module?
    let poolConv: KokoroConvWeighted?

    var conv1: KokoroConvWeighted
    var conv2: KokoroConvWeighted
    var norm1: KokoroAdaIN1d
    var norm2: KokoroAdaIN1d
    var conv1x1: KokoroConvWeighted?

    init(
        weights: [String: MLXArray], keyPrefix: String,
        dimIn: Int, dimOut: Int, styleDim: Int = 64,
        actv: LeakyReLU = LeakyReLU(negativeSlope: 0.2),
        upsample: String = "none"
    ) {
        self.actv = actv
        self.dimIn = dimIn
        upsampleType = upsample
        self.upsample = KokoroUpSample1d(layerType: upsample)
        learnedSc = dimIn != dimOut

        if upsample == "none" {
            pool = nil
            poolConv = nil
        } else {
            pool = nil
            poolConv = KokoroConvWeighted(
                weightG: weights["\(keyPrefix).pool.weight_g"]!,
                weightV: weights["\(keyPrefix).pool.weight_v"]!,
                bias: weights["\(keyPrefix).pool.bias"]!,
                stride: 2, padding: 1, groups: dimIn
            )
        }

        conv1 = KokoroConvWeighted(
            weightG: weights["\(keyPrefix).conv1.weight_g"]!,
            weightV: weights["\(keyPrefix).conv1.weight_v"]!,
            bias: weights["\(keyPrefix).conv1.bias"]!, padding: 1
        )
        conv2 = KokoroConvWeighted(
            weightG: weights["\(keyPrefix).conv2.weight_g"]!,
            weightV: weights["\(keyPrefix).conv2.weight_v"]!,
            bias: weights["\(keyPrefix).conv2.bias"]!, padding: 1
        )
        norm1 = KokoroAdaIN1d(
            numFeatures: dimIn,
            fcWeight: weights["\(keyPrefix).norm1.fc.weight"]!,
            fcBias: weights["\(keyPrefix).norm1.fc.bias"]!
        )
        norm2 = KokoroAdaIN1d(
            numFeatures: dimIn,
            fcWeight: weights["\(keyPrefix).norm2.fc.weight"]!,
            fcBias: weights["\(keyPrefix).norm2.fc.bias"]!
        )
        if learnedSc {
            conv1x1 = KokoroConvWeighted(
                weightG: weights["\(keyPrefix).conv1x1.weight_g"]!,
                weightV: weights["\(keyPrefix).conv1x1.weight_v"]!,
                bias: nil, padding: 0
            )
        }
    }

    func shortcut(_ x: MLXArray) -> MLXArray {
        var out = MLX.swappedAxes(x, 2, 1)
        out = upsample(out)
        out = MLX.swappedAxes(out, 2, 1)
        if let conv1x1 {
            out = MLX.swappedAxes(out, 2, 1)
            out = conv1x1(out, conv: MLX.conv1d)
            out = MLX.swappedAxes(out, 2, 1)
        }
        return out
    }

    func residual(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var out = norm1(x, s: s)
        out = actv(out)
        out = MLX.swappedAxes(out, 2, 1)
        if upsampleType != "none", let poolConv {
            out = poolConv(out, conv: MLX.convTransposed1d)
            out = MLX.padded(out, widths: [IntOrPair([0, 0]), IntOrPair([1, 0]), IntOrPair([0, 0])])
        }
        out = MLX.swappedAxes(out, 2, 1)
        out = MLX.swappedAxes(out, 2, 1)
        out = conv1(out, conv: MLX.conv1d)
        out = MLX.swappedAxes(out, 2, 1)
        out = norm2(out, s: s)
        out = actv(out)
        out = MLX.swappedAxes(out, 2, 1)
        out = conv2(out, conv: MLX.conv1d)
        out = MLX.swappedAxes(out, 2, 1)
        return out
    }

    func callAsFunction(x: MLXArray, s: MLXArray) -> MLXArray {
        (residual(x, s) + shortcut(x)) / sqrt(2.0)
    }
}

// MARK: - AdaIN Residual Block 1 (for Generator)

class KokoroAdaINResBlock1 {
    var convs1: [KokoroConvWeighted] = []
    var convs2: [KokoroConvWeighted] = []
    var adain1: [KokoroAdaIN1d] = []
    var adain2: [KokoroAdaIN1d] = []
    var alpha1: [MLXArray] = []
    var alpha2: [MLXArray] = []

    init(
        weights: [String: MLXArray], keyPrefix: String,
        channels: Int, kernelSize: Int = 3,
        dilation: [Int] = [1, 3, 5], styleDim: Int = 64
    ) {
        for i in 0..<3 {
            let pad = (kernelSize * dilation[i] - dilation[i]) / 2
            convs1.append(KokoroConvWeighted(
                weightG: weights["\(keyPrefix).convs1.\(i).weight_g"]!,
                weightV: weights["\(keyPrefix).convs1.\(i).weight_v"]!,
                bias: weights["\(keyPrefix).convs1.\(i).bias"]!,
                padding: pad, dilation: dilation[i]
            ))
            convs2.append(KokoroConvWeighted(
                weightG: weights["\(keyPrefix).convs2.\(i).weight_g"]!,
                weightV: weights["\(keyPrefix).convs2.\(i).weight_v"]!,
                bias: weights["\(keyPrefix).convs2.\(i).bias"]!,
                padding: (kernelSize - 1) / 2
            ))
            adain1.append(KokoroAdaIN1d(
                numFeatures: channels,
                fcWeight: weights["\(keyPrefix).adain1.\(i).fc.weight"]!,
                fcBias: weights["\(keyPrefix).adain1.\(i).fc.bias"]!
            ))
            adain2.append(KokoroAdaIN1d(
                numFeatures: channels,
                fcWeight: weights["\(keyPrefix).adain2.\(i).fc.weight"]!,
                fcBias: weights["\(keyPrefix).adain2.\(i).fc.bias"]!
            ))
            alpha1.append(weights["\(keyPrefix).alpha1.\(i)"]!)
            alpha2.append(weights["\(keyPrefix).alpha2.\(i)"]!)
        }
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var result = x
        for i in 0..<convs1.count {
            var xt = adain1[i](result, s: s)
            xt = xt + (1 / alpha1[i]) * (MLX.sin(alpha1[i] * xt).pow(2))
            xt = MLX.swappedAxes(xt, 2, 1)
            xt = convs1[i](xt, conv: MLX.conv1d)
            xt = MLX.swappedAxes(xt, 2, 1)
            xt = adain2[i](xt, s: s)
            xt = xt + (1 / alpha2[i]) * (MLX.sin(alpha2[i] * xt).pow(2))
            xt = MLX.swappedAxes(xt, 2, 1)
            xt = convs2[i](xt, conv: MLX.conv1d)
            xt = MLX.swappedAxes(xt, 2, 1)
            result = xt + result
        }
        return result
    }
}

// MARK: - Interpolation

func kokoroInterpolate(
    input: MLXArray, size: [Int]? = nil, scaleFactor: [Float]? = nil,
    mode: String = "nearest", alignCorners: Bool? = nil
) -> MLXArray {
    let spatialDims = input.ndim - 2
    guard spatialDims >= 1 else { fatalError("Expected at least 3D input") }
    guard (size == nil) != (scaleFactor == nil) else {
        fatalError("Exactly one of size or scaleFactor must be defined")
    }

    var outputSize: [Int]
    if let scaleFactor {
        let factors = scaleFactor.count == 1
            ? Array(repeating: scaleFactor[0], count: spatialDims) : scaleFactor
        outputSize = (0..<spatialDims).map { max(1, Int(ceil(Float(input.shape[$0 + 2]) * factors[$0]))) }
    } else {
        let s = size!
        outputSize = s.count == 1 ? Array(repeating: s[0], count: spatialDims) : s
    }

    guard spatialDims == 1 else { fatalError("Only 1D interpolation supported") }
    return kokoroInterpolate1d(input: input, size: outputSize[0], mode: mode, alignCorners: alignCorners)
}

private func kokoroInterpolate1d(
    input: MLXArray, size: Int, mode: String = "linear", alignCorners: Bool? = nil
) -> MLXArray {
    let shape = input.shape
    let batchSize = shape[0], channels = shape[1], inWidth = shape[2]
    let outSize = max(1, size)
    let inputWidth = max(1, inWidth)

    if mode == "nearest" {
        if outSize == 1 {
            return input[0..., 0..., MLXArray(converting: [0]).asType(.int32)]
        }
        let scale = Float(inputWidth) / Float(outSize)
        let indices = MLX.clip(
            MLX.floor(MLXArray(0..<outSize).asType(.float32) * scale).asType(.int32),
            min: 0, max: inputWidth - 1
        )
        return input[0..., 0..., indices]
    }

    // Linear interpolation
    var x: MLXArray
    if alignCorners == true && outSize > 1 {
        x = MLXArray(0..<outSize).asType(.float32) * (Float(inputWidth - 1) / Float(outSize - 1))
    } else if outSize == 1 {
        x = MLXArray(converting: [0.0]).asType(.float32)
    } else {
        x = MLXArray(0..<outSize).asType(.float32) * (Float(inputWidth) / Float(outSize))
        if alignCorners != true {
            x = x + 0.5 * (Float(inputWidth) / Float(outSize)) - 0.5
        }
    }

    if inputWidth == 1 {
        return MLX.broadcast(input, to: [batchSize, channels, outSize])
    }

    let xLow = MLX.floor(x).asType(.int32)
    let xHigh = MLX.minimum(xLow + 1, MLXArray(inputWidth - 1, dtype: .int32))
    let xFrac = x - xLow.asType(.float32)
    let yLow = input[0..., 0..., xLow]
    let yHigh = input[0..., 0..., xHigh]
    return yLow * (1 - xFrac).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
         + yHigh * xFrac.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
}
