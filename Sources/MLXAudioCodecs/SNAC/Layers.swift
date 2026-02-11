//
//  Layers.swift
//  MLXAudio
//
//  Created by Prince Canuma on 29/12/25.
//

import Foundation
import MLX
import MLXNN

// MARK: - Sequential (matches Python nn.Sequential serialization)

/// A Sequential container that wraps modules in a `layers` array
/// to match Python's nn.Sequential weight serialization format.
public class Sequential: Module, UnaryLayer {
    let layers: [Module]

    public init(_ layers: [Module]) {
        self.layers = layers
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        for layer in layers {
            result = (layer as! UnaryLayer).callAsFunction(result)
        }
        eval(result)
        return result
    }
}

// MARK: - Helper Functions

func normalizeWeight(_ x: MLXArray, exceptDim: Int = 0) -> MLXArray {
    guard x.ndim == 3 else {
        fatalError("Input tensor must have 3 dimensions")
    }

    let axes = (0..<x.ndim).filter { $0 != exceptDim }
    return sqrt(sum(pow(x, MLXArray(2)), axes: axes, keepDims: true))
}

func snake(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    let shape = x.shape
    var x = x.reshaped([shape[0], shape[1], -1])
    let recip = 1.0 / (alpha + 1e-9)
    x = x + recip * pow(sin(alpha * x), MLXArray(2))
    return x.reshaped(shape)
}

// MARK: - Weight Normalized Conv1d

class WNConv1d: Module, UnaryLayer {
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    var bias: MLXArray?

    let kernelSize: Int
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        let scale = sqrt(1.0 / Double(inChannels * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, kernelSize, inChannels / groups]
        )
        self._weightG.wrappedValue = normalizeWeight(weightInit)
        self._weightV.wrappedValue = weightInit / (self._weightG.wrappedValue + 1e-12)
        self.bias = bias ? MLX.zeros([outChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Ensure input is 3D: [batch, in_channels, time]
        let x3d: MLXArray
        if x.ndim == 2 {
            // [in_channels, time] -> [1, in_channels, time]
            x3d = x.reshaped([1, x.shape[0], x.shape[1]])
        } else {
            x3d = x
        }
        let xT = x3d.transposed(axes: [0, 2, 1]) // [batch, time, in_channels]
        let normV = normalizeWeight(weightV)
        let weight = weightG * weightV / (normV + 1e-12)
        var y = MLX.conv1d(
            xT,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let bias = bias {
            y = y + bias
        }
        // Output shape is [batch, time, outChannels], transpose to [batch, outChannels, time]
        return y.transposed(axes: [0, 2, 1])
    }
}

// MARK: - Weight Normalized ConvTranspose1d

public class WNConvTranspose1d: Module, UnaryLayer {
    var bias: MLXArray?
    let kernelSize: Int
    let padding: Int
    let dilation: Int
    let stride: Int
    let outputPadding: Int
    let groups: Int
    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        outputPadding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.bias = bias ? zeros([outChannels]) : nil
        self.kernelSize = kernelSize
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.outputPadding = outputPadding
        self.groups = groups

        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [inChannels, kernelSize, outChannels / groups]
        )
        self._weightG.wrappedValue = normalizeWeight(weightInit, exceptDim: 0)
        self._weightV.wrappedValue = weightInit / (self._weightG.wrappedValue + 1e-12)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input is NCT [batch, channels, time], transpose to NTC for convTransposed1d
        let xT = x.transposed(axes: [0, 2, 1])  // NCT -> NTC

        let weight = weightG * weightV / normalizeWeight(weightV, exceptDim: 0)
        // MLX uses (out_channels, kernel_size, in_channels) format
        let weightSwapped = weight.swappedAxes(0, 2)
        var y = MLX.convTransposed1d(
            xT,
            weightSwapped,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let bias = bias {
            y = y + bias
        }
        // Output is NTC, transpose back to NCT
        return y.transposed(axes: [0, 2, 1])
    }
}


// MARK: - Snake Activation

public class Snake1d: Module, UnaryLayer {
    var alpha: MLXArray

    public init(channels: Int) {
        self.alpha = ones([1, channels, 1])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return snake(x, alpha: alpha)
    }
}

// MARK: - Residual Unit

public class ResidualUnit: Module, UnaryLayer {
    let block: Sequential

    public init(dim: Int = 16, dilation: Int = 1, kernel: Int = 7, groups: Int = 1) {
        let pad = ((kernel - 1) * dilation) / 2
        self.block = Sequential([
            Snake1d(channels: dim),
            WNConv1d(
                inChannels: dim,
                outChannels: dim,
                kernelSize: kernel,
                padding: pad,
                dilation: dilation,
                groups: groups
            ),
            Snake1d(channels: dim),
            WNConv1d(inChannels: dim, outChannels: dim, kernelSize: 1)
        ])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = block(x)

        let pad = (x.shape[x.ndim - 1] - y.shape[y.ndim - 1]) / 2
        var xPadded = x
        if pad > 0 {
            xPadded = x[.ellipsis, pad..<(-pad)]
        }
        return xPadded + y
    }
}

// MARK: - Encoder Block

public class EncoderBlock: Module, UnaryLayer {
    let block: Sequential

    public init(outputDim: Int = 16, inputDim: Int? = nil, stride: Int = 1, groups: Int = 1) {
        let inputDim = inputDim ?? outputDim / 2
        self.block = Sequential([
            ResidualUnit(dim: inputDim, dilation: 1, groups: groups),
            ResidualUnit(dim: inputDim, dilation: 3, groups: groups),
            ResidualUnit(dim: inputDim, dilation: 9, groups: groups),
            Snake1d(channels: inputDim),
            WNConv1d(
                inChannels: inputDim,
                outChannels: outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0))
            )
        ])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return block(x)
    }
}

// MARK: - Noise Block

public class NoiseBlock: Module, UnaryLayer {
    let linear: WNConv1d

    public init(dim: Int) {
        self.linear = WNConv1d(inChannels: dim, outChannels: dim, kernelSize: 1, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.shape[0]
        let T = x.shape[2]

        let noise = MLXRandom.normal([B, 1, T])
        let h = linear(x)
        let n = noise * h
        return x + n
    }
}

// MARK: - Decoder Block

public class DecoderBlock: Module, UnaryLayer {
    let block: Sequential

    public init(inputDim: Int = 16, outputDim: Int = 8, stride: Int = 1, noise: Bool = false, groups: Int = 1) {
        var layers: [Module] = [
            Snake1d(channels: inputDim),
            WNConvTranspose1d(
                inChannels: inputDim,
                outChannels: outputDim,
                kernelSize: 2 * stride,
                stride: stride,
                padding: Int(ceil(Double(stride) / 2.0)),
                outputPadding: stride % 2
            )
        ]

        if noise {
            layers.append(NoiseBlock(dim: outputDim))
        }

        layers.append(contentsOf: [
            ResidualUnit(dim: outputDim, dilation: 1, groups: groups),
            ResidualUnit(dim: outputDim, dilation: 3, groups: groups),
            ResidualUnit(dim: outputDim, dilation: 9, groups: groups)
        ])

        self.block = Sequential(layers)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return block(x)
    }
}

// MARK: - Encoder

public class Encoder: Module, UnaryLayer {
    let block: Sequential

    public init(
        dModel: Int = 64,
        strides: [Int] = [3, 3, 7, 7],
        depthwise: Bool = false,
        attnWindowSize: Int? = 32
    ) {
        var layers: [Module] = [
            WNConv1d(inChannels: 1, outChannels: dModel, kernelSize: 7, padding: 3)
        ]

        var currentDModel = dModel
        for stride in strides {
            currentDModel *= 2
            let groups = depthwise ? currentDModel / 2 : 1
            layers.append(EncoderBlock(outputDim: currentDModel, stride: stride, groups: groups))
        }

        if let attnWindowSize = attnWindowSize {
            layers.append(LocalMHA(dim: currentDModel, windowSize: attnWindowSize))
        }

        let groups = depthwise ? currentDModel : 1
        layers.append(
            WNConv1d(
                inChannels: currentDModel,
                outChannels: currentDModel,
                kernelSize: 7,
                padding: 3,
                groups: groups
            )
        )

        self.block = Sequential(layers)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return block(x)
    }
}

// MARK: - Decoder

public class Decoder: Module, UnaryLayer {
    let model: Sequential

    public init(
        inputChannel: Int,
        channels: Int,
        rates: [Int],
        noise: Bool = false,
        depthwise: Bool = false,
        attnWindowSize: Int? = 32,
        dOut: Int = 1
    ) {
        var layers: [Module]

        if depthwise {
            layers = [
                WNConv1d(
                    inChannels: inputChannel,
                    outChannels: inputChannel,
                    kernelSize: 7,
                    padding: 3,
                    groups: inputChannel
                ),
                WNConv1d(inChannels: inputChannel, outChannels: channels, kernelSize: 1)
            ]
        } else {
            layers = [
                WNConv1d(inChannels: inputChannel, outChannels: channels, kernelSize: 7, padding: 3)
            ]
        }

        if let attnWindowSize = attnWindowSize {
            layers.append(LocalMHA(dim: channels, windowSize: attnWindowSize))
        }

        for (i, stride) in rates.enumerated() {
            let inputDim = channels / Int(pow(2.0, Double(i)))
            let outputDim = channels / Int(pow(2.0, Double(i + 1)))
            let groups = depthwise ? outputDim : 1
            layers.append(
                DecoderBlock(inputDim: inputDim, outputDim: outputDim, stride: stride, noise: noise, groups: groups)
            )
        }

        let finalOutputDim = channels / Int(pow(2.0, Double(rates.count)))
        layers.append(contentsOf: [
            Snake1d(channels: finalOutputDim),
            WNConv1d(inChannels: finalOutputDim, outChannels: dOut, kernelSize: 7, padding: 3),
            Tanh()
        ])

        self.model = Sequential(layers)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return model(x)
    }
}
