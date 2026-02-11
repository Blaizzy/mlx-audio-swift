//
//  EncodecLayers.swift
//  MLXAudioCodecs
//
//  Ported from mlx-audio Python implementation
//

import Foundation
import MLX
import MLXNN

// MARK: - Encodec LSTM

/// LSTM layer for Encodec.
public class EncodecLSTM: Module {
    let hiddenSize: Int
    var Wx: MLXArray
    var Wh: MLXArray
    var bias: MLXArray?

    public init(inputSize: Int, hiddenSize: Int, bias: Bool = true) {
        self.hiddenSize = hiddenSize
        self.Wx = MLXArray.zeros([4 * hiddenSize, inputSize])
        self.Wh = MLXArray.zeros([4 * hiddenSize, hiddenSize])
        self.bias = bias ? MLXArray.zeros([4 * hiddenSize]) : nil
    }

    public func callAsFunction(_ x: MLXArray, hidden: MLXArray? = nil, cell: MLXArray? = nil) -> MLXArray {
        var xProj: MLXArray
        if let b = bias {
            xProj = MLX.addMM(b, x, Wx.T)
        } else {
            xProj = MLX.matmul(x, Wx.T)
        }

        var allHidden: [MLXArray] = []
        let B = x.shape[0]
        var currentCell = cell ?? MLXArray.zeros([B, hiddenSize])
        var currentHidden = hidden

        for t in 0..<x.shape[1] {
            let xT = xProj[0..., t, 0...]

            var hProj: MLXArray
            if let h = currentHidden {
                hProj = MLX.matmul(h, Wh.T)
            } else {
                hProj = MLXArray.zeros([B, 4 * hiddenSize])
            }

            let gates = xT + hProj

            // Split gates: i, f, g, o
            let i = sigmoid(gates[0..., 0..<hiddenSize])
            let f = sigmoid(gates[0..., hiddenSize..<(2 * hiddenSize)])
            let g = tanh(gates[0..., (2 * hiddenSize)..<(3 * hiddenSize)])
            let o = sigmoid(gates[0..., (3 * hiddenSize)...])

            currentCell = f * currentCell + i * g
            currentHidden = o * tanh(currentCell)
            allHidden.append(currentHidden!)
        }

        return MLX.stacked(allHidden, axis: 1)
    }
}

// MARK: - Encodec LSTM Block

/// LSTM block with residual connection for Encodec.
public class EncodecLSTMBlock: Module {
    @ModuleInfo(key: "lstm") var lstm: [EncodecLSTM]

    public init(config: EncodecConfig, dimension: Int) {
        self._lstm.wrappedValue = (0..<config.numLstmLayers).map { _ in
            EncodecLSTM(inputSize: dimension, hiddenSize: dimension)
        }
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates
        for lstmLayer in lstm {
            h = lstmLayer(h)
        }
        return h + hiddenStates
    }
}

// MARK: - Encodec Conv1d

/// Conv1d with asymmetric or causal padding and normalization.
public class EncodecConv1d: Module {
    let causal: Bool
    let padMode: String
    let normType: String
    let stride: Int
    let kernelSizeEffective: Int
    let paddingTotal: Int

    @ModuleInfo(key: "conv") var conv: MLXNN.Conv1d
    @ModuleInfo(key: "norm") var norm: GroupNorm?

    public init(
        config: EncodecConfig,
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1
    ) {
        self.causal = config.useCausalConv
        self.padMode = config.padMode
        self.normType = config.normType
        self.stride = stride

        // Effective kernel size with dilations
        self.kernelSizeEffective = (kernelSize - 1) * dilation + 1
        self.paddingTotal = kernelSize - stride

        self._conv.wrappedValue = MLXNN.Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            dilation: dilation
        )

        if normType == "time_group_norm" {
            self._norm.wrappedValue = GroupNorm(groupCount: 1, dimensions: outChannels, pytorchCompatible: true)
        } else {
            self._norm.wrappedValue = nil
        }
    }

    private func getExtraPaddingForConv1d(_ hiddenStates: MLXArray) -> Int {
        let length = hiddenStates.shape[1]
        let nFrames = Float(length - kernelSizeEffective + paddingTotal) / Float(stride) + 1
        let nFramesInt = Int(ceil(nFrames)) - 1
        let idealLength = nFramesInt * stride + kernelSizeEffective - paddingTotal
        return max(0, idealLength - length)
    }

    private func pad1d(_ hiddenStates: MLXArray, paddings: (Int, Int), mode: String) -> MLXArray {
        if mode != "reflect" {
            // Zero padding
            return MLX.padded(hiddenStates, widths: [.init(0), .init((paddings.0, paddings.1)), .init(0)])
        }

        // Reflect padding - manually build reflected values
        let length = hiddenStates.shape[1]

        // For reflect padding, we need indices: [n, n-1, ..., 2, 1] for left pad
        // and [length-2, length-3, ...] for right pad
        var parts: [MLXArray] = []

        // Left padding (reflect from position 1)
        if paddings.0 > 0 {
            var leftIndices: [Int] = []
            for i in 0..<paddings.0 {
                let idx = min(paddings.0 - i, length - 1)
                leftIndices.append(idx)
            }
            let leftSlices = leftIndices.map { hiddenStates[0..., $0..<($0+1), 0...] }
            if !leftSlices.isEmpty {
                let prefix = MLX.concatenated(leftSlices, axis: 1)
                parts.append(prefix)
            }
        }

        parts.append(hiddenStates)

        // Right padding (reflect from position length-2)
        if paddings.1 > 0 {
            var rightIndices: [Int] = []
            for i in 0..<paddings.1 {
                let idx = max(length - 2 - i, 0)
                rightIndices.append(idx)
            }
            let rightSlices = rightIndices.map { hiddenStates[0..., $0..<($0+1), 0...] }
            if !rightSlices.isEmpty {
                let suffix = MLX.concatenated(rightSlices, axis: 1)
                parts.append(suffix)
            }
        }

        return MLX.concatenated(parts, axis: 1)
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let extraPadding = getExtraPaddingForConv1d(hiddenStates)

        var h: MLXArray
        if causal {
            // Left padding for causal
            h = pad1d(hiddenStates, paddings: (paddingTotal, extraPadding), mode: padMode)
        } else {
            // Asymmetric padding required for odd strides
            let paddingRight = paddingTotal / 2
            let paddingLeft = paddingTotal - paddingRight
            h = pad1d(hiddenStates, paddings: (paddingLeft, paddingRight + extraPadding), mode: padMode)
        }

        h = conv(h)

        if let normLayer = norm {
            h = normLayer(h)
        }

        return h
    }
}

// MARK: - Encodec ConvTranspose1d Wrapper

/// ConvTranspose1d with asymmetric or causal padding and normalization.
public class EncodecConvTranspose1dLayer: Module {
    let causal: Bool
    let trimRightRatio: Float
    let normType: String
    let paddingTotal: Int

    @ModuleInfo(key: "conv") var conv: EncodecBaseConvTranspose1d
    @ModuleInfo(key: "norm") var norm: GroupNorm?

    public init(
        config: EncodecConfig,
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1
    ) {
        self.causal = config.useCausalConv
        self.trimRightRatio = config.trimRightRatio
        self.normType = config.normType
        self.paddingTotal = kernelSize - stride

        self._conv.wrappedValue = EncodecBaseConvTranspose1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride
        )

        if normType == "time_group_norm" {
            self._norm.wrappedValue = GroupNorm(groupCount: 1, dimensions: outChannels, pytorchCompatible: true)
        } else {
            self._norm.wrappedValue = nil
        }
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = conv(hiddenStates)

        if let normLayer = norm {
            h = normLayer(h)
        }

        let paddingRight: Int
        if causal {
            paddingRight = Int(ceil(Float(paddingTotal) * trimRightRatio))
        } else {
            paddingRight = paddingTotal / 2
        }

        let paddingLeft = paddingTotal - paddingRight

        let end = h.shape[1] - paddingRight
        if end > paddingLeft {
            h = h[0..., paddingLeft..<end, 0...]
        }
        return h
    }
}

// MARK: - Encodec Resnet Block

/// Residual block from SEANet model as used by EnCodec.
public class EncodecResnetBlock: Module {
    @ModuleInfo(key: "block") var block: [Module]
    @ModuleInfo(key: "shortcut") var shortcut: Module

    public init(config: EncodecConfig, dim: Int, dilations: [Int]) {
        let kernelSizes = [config.residualKernelSize, 1]

        guard kernelSizes.count == dilations.count else {
            fatalError("Number of kernel sizes should match number of dilations")
        }

        let hidden = dim / config.compress
        var blockLayers: [Module] = []

        for (i, (kernelSize, dilation)) in zip(kernelSizes, dilations).enumerated() {
            let inChs = i == 0 ? dim : hidden
            let outChs = i == kernelSizes.count - 1 ? dim : hidden
            blockLayers.append(ELU())
            blockLayers.append(EncodecConv1d(
                config: config,
                inChannels: inChs,
                outChannels: outChs,
                kernelSize: kernelSize,
                dilation: dilation
            ))
        }

        self._block.wrappedValue = blockLayers

        if config.useConvShortcut {
            self._shortcut.wrappedValue = EncodecConv1d(
                config: config,
                inChannels: dim,
                outChannels: dim,
                kernelSize: 1
            )
        } else {
            self._shortcut.wrappedValue = EncodecIdentity()
        }
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let residual = hiddenStates
        var h = hiddenStates

        for layer in block {
            if let elu = layer as? ELU {
                h = elu(h)
            } else if let conv = layer as? EncodecConv1d {
                h = conv(h)
            }
        }

        if let shortcutConv = shortcut as? EncodecConv1d {
            return shortcutConv(residual) + h
        } else {
            return residual + h
        }
    }
}

// MARK: - Helper Identity Module

/// Identity module that returns input unchanged.
public class EncodecIdentity: Module {
    public override init() {
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x
    }
}

// MARK: - ELU Activation

/// Exponential Linear Unit activation.
public class ELU: Module {
    let alpha: Float

    public init(alpha: Float = 1.0) {
        self.alpha = alpha
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLX.where(x .> 0, x, alpha * (MLX.exp(x) - 1))
    }
}

// MARK: - EncodecBaseConvTranspose1d

/// 1D Transposed Convolution layer for Encodec.
public class EncodecBaseConvTranspose1d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var biasParam: MLXArray?

    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let strideVal: Int
    let paddingVal: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        bias: Bool = true
    ) {
        self.inChannels = inputChannels
        self.outChannels = outputChannels
        self.kernelSize = kernelSize
        self.strideVal = stride
        self.paddingVal = padding

        // Initialize weights
        let scale = sqrt(2.0 / Float(inputChannels * kernelSize))
        self._weight.wrappedValue = MLXRandom.normal([outputChannels, kernelSize, inputChannels]) * scale

        if bias {
            self._biasParam.wrappedValue = MLXArray.zeros([outputChannels])
        } else {
            self._biasParam.wrappedValue = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x shape: (batch, length, channels) - NLC format
        let batch = x.shape[0]
        let length = x.shape[1]

        // Calculate output length
        let outputLength = (length - 1) * strideVal - 2 * paddingVal + kernelSize

        // Transpose to NCL for convolution
        let h = x.transposed(0, 2, 1)  // (batch, channels, length)

        // Use scatter-based transpose convolution
        let xData = h.asArray(Float.self)
        let wData = weight.asArray(Float.self)
        var outData = [Float](repeating: 0, count: batch * outChannels * outputLength)

        for b in 0..<batch {
            for t in 0..<length {
                for oc in 0..<outChannels {
                    for k in 0..<kernelSize {
                        let outT = t * strideVal + k - paddingVal
                        if outT >= 0 && outT < outputLength {
                            for ic in 0..<inChannels {
                                let xIdx = b * inChannels * length + ic * length + t
                                let wIdx = oc * kernelSize * inChannels + k * inChannels + ic
                                let oIdx = b * outChannels * outputLength + oc * outputLength + outT
                                outData[oIdx] += xData[xIdx] * wData[wIdx]
                            }
                        }
                    }
                }
            }
        }

        var output = MLXArray(outData).reshaped([batch, outChannels, outputLength])

        // Add bias
        if let b = biasParam {
            output = output + b.reshaped([1, outChannels, 1])
        }

        // Transpose back to NLC
        return output.transposed(0, 2, 1)
    }
}
