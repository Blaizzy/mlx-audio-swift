//
//  DACVAELayers.swift
//  MLXAudioCodecs
//
// Created by Prince Canuma on 04/01/2026.
//

import Foundation
import MLX
import MLXNN

// MARK: - Weight Normalization Helper

/// Compute weight normalization factor for DACVAE.
func dacvaeNormalizeWeight(_ x: MLXArray, exceptDim: Int = 0) -> MLXArray {
    guard x.ndim == 3 else {
        fatalError("Input tensor must have 3 dimensions")
    }
    let axes = (0..<x.ndim).filter { $0 != exceptDim }
    return MLX.sqrt(MLX.sum(x * x, axes: axes, keepDims: true))
}

// MARK: - Snake Activation

/// Snake activation function for DACVAE.
func dacvaeSnake(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    let recip = 1.0 / (alpha + 1e-9)
    let sinVal = MLX.sin(alpha * x)
    return x + recip * (sinVal * sinVal)
}

/// Snake activation for 1D signals (DACVAE).
public class DACVAESnake1d: Module {
    @ModuleInfo(key: "alpha") var alpha: MLXArray

    public init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.ones([1, 1, channels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return dacvaeSnake(x, alpha: alpha)
    }
}

// MARK: - Weight-Normalized Conv1d

/// Weight-normalized 1D convolution with optional causal padding (DACVAE).
public class DACVAEWNConv1d: Module {
    let kernelSize: Int
    let dilation: Int
    let strideVal: Int
    let causal: Bool
    let padMode: String
    let useWeightNorm: Bool
    var paddingVal: Int

    @ModuleInfo(key: "weight_g") var weightG: MLXArray?
    @ModuleInfo(key: "weight_v") var weightV: MLXArray?
    @ModuleInfo(key: "weight") var weight: MLXArray?
    @ModuleInfo(key: "bias") var biasParam: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        bias: Bool = true,
        causal: Bool = false,
        padMode: String = "none",
        norm: String = "weight_norm"
    ) {
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.strideVal = stride
        self.causal = causal
        self.padMode = padMode
        self.useWeightNorm = norm == "weight_norm"

        // Calculate padding for pad_mode="none"
        if padMode == "none" {
            self.paddingVal = (kernelSize - stride) * dilation / 2
        } else {
            self.paddingVal = 0
        }

        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, kernelSize, inChannels]
        )

        if useWeightNorm {
            self._weightG.wrappedValue = dacvaeNormalizeWeight(weightInit)
            self._weightV.wrappedValue = weightInit / (self._weightG.wrappedValue! + 1e-12)
            self._weight.wrappedValue = nil
        } else {
            self._weight.wrappedValue = weightInit
            self._weightG.wrappedValue = nil
            self._weightV.wrappedValue = nil
        }

        self._biasParam.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    private func getWeight() -> MLXArray {
        if useWeightNorm, let g = weightG, let v = weightV {
            return g * v / dacvaeNormalizeWeight(v)
        }
        return weight!
    }

    private func autoPad(_ x: MLXArray) -> MLXArray {
        if padMode == "none" {
            return x
        }

        let length = x.shape[1]
        let effectiveKernelSize = (kernelSize - 1) * dilation + 1
        let paddingTotal = effectiveKernelSize - strideVal
        let nFrames = Float(length - effectiveKernelSize + paddingTotal) / Float(strideVal) + 1
        let idealLength = (Int(ceil(nFrames)) - 1) * strideVal + (kernelSize - paddingTotal)
        let extraPadding = max(0, idealLength - length)

        var padLeft: Int
        var padRight: Int

        if causal {
            // Causal: all padding on left
            padLeft = paddingTotal
            padRight = extraPadding
        } else {
            // Non-causal: symmetric padding
            padRight = extraPadding / 2
            padLeft = paddingTotal - padRight + extraPadding - padRight
        }

        if padLeft > 0 || padRight > 0 {
            return MLX.padded(x, widths: [.init(0), .init((padLeft, padRight)), .init(0)])
        }

        return x
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = autoPad(x)
        let w = getWeight()
        h = MLX.conv1d(h, w, stride: strideVal, padding: paddingVal, dilation: dilation)
        if let b = biasParam {
            h = h + b
        }
        return h
    }
}

// MARK: - Weight-Normalized ConvTranspose1d

/// Weight-normalized transposed 1D convolution with optional causal unpadding (DACVAE).
public class DACVAEWNConvTranspose1d: Module {
    let kernelSize: Int
    let dilation: Int
    let strideVal: Int
    let causal: Bool
    let padMode: String
    let useWeightNorm: Bool
    var paddingVal: Int

    @ModuleInfo(key: "weight_g") var weightG: MLXArray?
    @ModuleInfo(key: "weight_v") var weightV: MLXArray?
    @ModuleInfo(key: "weight") var weight: MLXArray?
    @ModuleInfo(key: "bias") var biasParam: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        bias: Bool = true,
        causal: Bool = false,
        padMode: String = "none",
        norm: String = "weight_norm"
    ) {
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.strideVal = stride
        self.causal = causal
        self.padMode = padMode
        self.useWeightNorm = norm == "weight_norm"

        // Calculate padding for pad_mode="none"
        if padMode == "none" {
            self.paddingVal = (stride + 1) / 2
        } else {
            self.paddingVal = 0
        }

        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, kernelSize, inChannels]
        )

        if useWeightNorm {
            self._weightG.wrappedValue = dacvaeNormalizeWeight(weightInit, exceptDim: 2)
            self._weightV.wrappedValue = weightInit / (self._weightG.wrappedValue! + 1e-12)
            self._weight.wrappedValue = nil
        } else {
            self._weight.wrappedValue = weightInit
            self._weightG.wrappedValue = nil
            self._weightV.wrappedValue = nil
        }

        self._biasParam.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    private func getWeight() -> MLXArray {
        if useWeightNorm, let g = weightG, let v = weightV {
            return g * v / dacvaeNormalizeWeight(v, exceptDim: 2)
        }
        return weight!
    }

    private func unpad(_ x: MLXArray) -> MLXArray {
        if padMode == "none" {
            return x
        }

        let length = x.shape[1]
        let paddingTotal = kernelSize - strideVal

        var paddingLeft: Int
        var paddingRight: Int

        if causal {
            // Causal: remove padding from end
            paddingRight = paddingTotal
            paddingLeft = 0
        } else {
            // Non-causal: remove from both sides
            paddingRight = paddingTotal / 2
            paddingLeft = paddingTotal - paddingRight
        }

        let endIdx = length - paddingRight
        if endIdx > paddingLeft {
            return x[0..., paddingLeft..<endIdx, 0...]
        }
        return x
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = getWeight()
        // Perform transposed convolution manually since MLX doesn't have conv_transpose1d
        var h = convTranspose1d(x, weight: w, stride: strideVal, padding: paddingVal)
        if let b = biasParam {
            h = h + b.reshaped([1, 1, -1])
        }
        return unpad(h)
    }

    /// Manual transposed 1D convolution implementation.
    private func convTranspose1d(_ x: MLXArray, weight: MLXArray, stride: Int, padding: Int) -> MLXArray {
        // x shape: (batch, length, inChannels) - NLC format
        // weight shape: (outChannels, kernelSize, inChannels)
        let batch = x.shape[0]
        let length = x.shape[1]
        let inChannels = x.shape[2]
        let outChannels = weight.shape[0]
        let kSize = weight.shape[1]

        // Calculate output length
        let outputLength = (length - 1) * stride - 2 * padding + kSize

        // Transpose input to NCL for easier computation
        let xT = x.transposed(0, 2, 1)  // (batch, inChannels, length)

        // Use scatter-based transpose convolution
        let xData = xT.asArray(Float.self)
        let wData = weight.asArray(Float.self)
        var outData = [Float](repeating: 0, count: batch * outChannels * outputLength)

        for b in 0..<batch {
            for t in 0..<length {
                for oc in 0..<outChannels {
                    for k in 0..<kSize {
                        let outT = t * stride + k - padding
                        if outT >= 0 && outT < outputLength {
                            for ic in 0..<inChannels {
                                let xIdx = b * inChannels * length + ic * length + t
                                let wIdx = oc * kSize * inChannels + k * inChannels + ic
                                let oIdx = b * outChannels * outputLength + oc * outputLength + outT
                                outData[oIdx] += xData[xIdx] * wData[wIdx]
                            }
                        }
                    }
                }
            }
        }

        let output = MLXArray(outData).reshaped([batch, outChannels, outputLength])

        // Transpose back to NLC
        return output.transposed(0, 2, 1)
    }
}

// MARK: - DACVAE ELU Activation

/// Exponential Linear Unit activation for DACVAE.
public class DACVAEElu: Module {
    let alpha: Float

    public init(alpha: Float = 1.0) {
        self.alpha = alpha
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLX.where(x .> 0, x, alpha * (MLX.exp(x) - 1))
    }
}

// MARK: - Residual Unit

/// Residual unit with dilated convolutions supporting Snake or ELU activation.
public class DACVAEResidualUnit: Module {
    let trueSkip: Bool
    let actType: String

    @ModuleInfo(key: "act1") var act1: Module
    @ModuleInfo(key: "conv1") var conv1: DACVAEWNConv1d
    @ModuleInfo(key: "act2") var act2: Module
    @ModuleInfo(key: "conv2") var conv2: DACVAEWNConv1d

    public init(
        dim: Int = 16,
        kernel: Int = 7,
        dilation: Int = 1,
        act: String = "Snake",
        compress: Int = 1,
        causal: Bool = false,
        padMode: String = "none",
        norm: String = "weight_norm",
        trueSkip: Bool = false
    ) {
        self.trueSkip = trueSkip
        self.actType = act

        let hidden = dim / compress

        // First activation + conv
        if act == "Snake" {
            self._act1.wrappedValue = DACVAESnake1d(channels: dim)
        } else {
            self._act1.wrappedValue = DACVAEElu()
        }

        self._conv1.wrappedValue = DACVAEWNConv1d(
            inChannels: dim,
            outChannels: hidden,
            kernelSize: kernel,
            dilation: dilation,
            causal: causal,
            padMode: padMode,
            norm: norm
        )

        // Second activation + conv
        if act == "Snake" {
            self._act2.wrappedValue = DACVAESnake1d(channels: hidden)
        } else {
            self._act2.wrappedValue = DACVAEElu()
        }

        self._conv2.wrappedValue = DACVAEWNConv1d(
            inChannels: hidden,
            outChannels: dim,
            kernelSize: 1,
            causal: causal,
            padMode: padMode,
            norm: norm
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        if let snakeAct = act1 as? DACVAESnake1d {
            y = snakeAct(x)
        } else if let eluAct = act1 as? DACVAEElu {
            y = eluAct(x)
        } else {
            y = x
        }

        y = conv1(y)

        if let snakeAct = act2 as? DACVAESnake1d {
            y = snakeAct(y)
        } else if let eluAct = act2 as? DACVAEElu {
            y = eluAct(y)
        }

        y = conv2(y)

        if trueSkip {
            return x
        }

        // Handle padding differences
        let pad = (x.shape[1] - y.shape[1]) / 2
        var xTrimmed = x
        if pad > 0 {
            xTrimmed = x[0..., pad..<(x.shape[1] - pad), 0...]
        }
        return xTrimmed + y
    }
}
