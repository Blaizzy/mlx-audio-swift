import Foundation
import MLX
import MLXNN

@inline(__always)
func fishS1NormalizeWeight(_ x: MLXArray, exceptDim: Int = 0) -> MLXArray {
    let axes = (0..<x.ndim).filter { $0 != exceptDim }
    return MLX.sqrt(MLX.sum(x * x, axes: axes, keepDims: true))
}

@inline(__always)
func fishS1Conv1dTorchToMLX(_ weight: MLXArray) -> MLXArray {
    weight.transposed(0, 2, 1)
}

@inline(__always)
func fishS1ConvTranspose1dTorchToMLX(_ weight: MLXArray) -> MLXArray {
    weight.transposed(1, 2, 0)
}

@inline(__always)
func fishS1FindMultiple(_ n: Int, _ k: Int) -> Int {
    n.isMultiple(of: k) ? n : n + k - (n % k)
}

func fishS1Unpad1d(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    let end = x.shape[2] - right
    return x[0..., 0..., left..<end]
}

func fishS1ExtraPaddingForConv1d(
    _ x: MLXArray,
    kernelSize: Int,
    stride: Int,
    paddingTotal: Int = 0
) -> Int {
    let length = x.shape[2]
    let nFrames = Double(length - kernelSize + paddingTotal) / Double(stride) + 1.0
    let idealLength = (ceil(nFrames) - 1.0) * Double(stride) + Double(kernelSize - paddingTotal)
    return max(0, Int(idealLength - Double(length)))
}

@inline(__always)
func fishS1CallUnary(_ module: Module, _ x: MLXArray) -> MLXArray {
    (module as! UnaryLayer).callAsFunction(x)
}

public final class FishS1Identity: Module, UnaryLayer {
    public override init() {}

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x
    }
}

public final class FishS1Conv1dTorch: Module, UnaryLayer {
    public let stride: Int
    public let padding: Int
    public let dilation: Int
    public let groups: Int
    public let kernelSize: Int

    public var weight: MLXArray
    public var bias: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernelSize = kernelSize

        let inPerGroup = inChannels / max(groups, 1)
        let scale = sqrt(1.0 / Double(inPerGroup * kernelSize))
        self.weight = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, inPerGroup, kernelSize]
        )
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xNLC = x.transposed(0, 2, 1)
        var y = MLX.conv1d(
            xNLC,
            fishS1Conv1dTorchToMLX(weight),
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let bias {
            y = y + bias
        }
        return y.transposed(0, 2, 1)
    }
}

public final class FishS1ConvTranspose1dTorch: Module, UnaryLayer {
    public let stride: Int
    public let padding: Int
    public let dilation: Int
    public let groups: Int
    public let kernelSize: Int
    public let outputPadding: Int

    public var weight: MLXArray
    public var bias: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernelSize = kernelSize
        self.outputPadding = stride > 1 ? 1 : 0

        let outPerGroup = outChannels / max(groups, 1)
        let scale = sqrt(1.0 / Double(outPerGroup * kernelSize))
        self.weight = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [inChannels, outPerGroup, kernelSize]
        )
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xNLC = x.transposed(0, 2, 1)
        var y = MLX.convTransposed1d(
            xNLC,
            fishS1ConvTranspose1dTorchToMLX(weight),
            stride: stride,
            padding: padding,
            dilation: dilation,
            outputPadding: outputPadding,
            groups: groups
        )
        if let bias {
            y = y + bias
        }
        return y.transposed(0, 2, 1)
    }
}

public final class FishS1WNConv1d: Module, UnaryLayer {
    public let stride: Int
    public let padding: Int
    public let dilation: Int
    public let groups: Int
    public let kernelSize: Int

    public var weight_g: MLXArray
    public var weight_v: MLXArray
    public var bias: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernelSize = kernelSize

        let inPerGroup = inChannels / max(groups, 1)
        let scale = sqrt(1.0 / Double(inPerGroup * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, inPerGroup, kernelSize]
        )
        self.weight_g = fishS1NormalizeWeight(weightInit, exceptDim: 0)
        self.weight_v = weightInit / (weight_g + 1e-12)
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xNLC = x.transposed(0, 2, 1)
        let weight = weight_g * weight_v / (fishS1NormalizeWeight(weight_v, exceptDim: 0) + 1e-12)
        var y = MLX.conv1d(
            xNLC,
            fishS1Conv1dTorchToMLX(weight),
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let bias {
            y = y + bias
        }
        return y.transposed(0, 2, 1)
    }
}

public final class FishS1WNConvTranspose1d: Module, UnaryLayer {
    public let stride: Int
    public let padding: Int
    public let dilation: Int
    public let groups: Int
    public let kernelSize: Int
    public let outputPadding: Int

    public var weight_g: MLXArray
    public var weight_v: MLXArray
    public var bias: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernelSize = kernelSize
        self.outputPadding = stride > 1 ? 1 : 0

        let outPerGroup = outChannels / max(groups, 1)
        let scale = sqrt(1.0 / Double(outPerGroup * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [inChannels, outPerGroup, kernelSize]
        )
        self.weight_g = fishS1NormalizeWeight(weightInit, exceptDim: 0)
        self.weight_v = weightInit / (weight_g + 1e-12)
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xNLC = x.transposed(0, 2, 1)
        let weight = weight_g * weight_v / (fishS1NormalizeWeight(weight_v, exceptDim: 0) + 1e-12)
        var y = MLX.convTransposed1d(
            xNLC,
            fishS1ConvTranspose1dTorchToMLX(weight),
            stride: stride,
            padding: padding,
            dilation: dilation,
            outputPadding: outputPadding,
            groups: groups
        )
        if let bias {
            y = y + bias
        }
        return y.transposed(0, 2, 1)
    }
}

public final class FishS1Snake1d: Module, UnaryLayer {
    public var alpha: MLXArray

    public init(channels: Int) {
        self.alpha = MLXArray.ones([1, channels, 1])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + (1.0 / (alpha + 1e-9)) * MLX.square(MLX.sin(alpha * x))
    }
}

public final class FishS1CausalConvNet: Module, UnaryLayer {
    @ModuleInfo(key: "conv") public var conv: FishS1Conv1dTorch

    public let stride: Int
    public let kernelSize: Int
    public let padding: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        dilation: Int = 1,
        stride: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.kernelSize = (kernelSize - 1) * dilation + 1
        self.padding = self.kernelSize - stride
        self._conv = ModuleInfo(wrappedValue: FishS1Conv1dTorch(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        ))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let extra = fishS1ExtraPaddingForConv1d(x, kernelSize: kernelSize, stride: stride, paddingTotal: padding)
        let padded = MLX.padded(
            x,
            widths: [
                IntOrPair(0),
                IntOrPair(0),
                IntOrPair((padding, extra))
            ]
        )
        return conv(padded)
    }
}

public final class FishS1CausalTransConvNet: Module, UnaryLayer {
    @ModuleInfo(key: "conv") public var conv: FishS1ConvTranspose1dTorch

    public let stride: Int
    public let kernelSize: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        dilation: Int = 1,
        stride: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.stride = stride
        self.kernelSize = kernelSize
        self._conv = ModuleInfo(wrappedValue: FishS1ConvTranspose1dTorch(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias
        ))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv(x)
        let totalPad = kernelSize - stride
        let right = Int(ceil(Double(totalPad)))
        let left = totalPad - right
        return fishS1Unpad1d(y, left: left, right: right)
    }
}

public final class FishS1CausalWNConv1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") public var conv: FishS1CausalConvNet

    public var weight_g: MLXArray
    public var weight_v: MLXArray
    public var bias: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let convModule = FishS1CausalConvNet(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            dilation: dilation,
            stride: stride,
            groups: groups,
            bias: bias
        )
        let weight = convModule.conv.weight
        self.weight_g = fishS1NormalizeWeight(weight, exceptDim: 0)
        self.weight_v = weight / (weight_g + 1e-12)
        self.bias = convModule.conv.bias
        convModule.conv.weight = weight_v
        convModule.conv.bias = self.bias
        self._conv = ModuleInfo(wrappedValue: convModule)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv.conv.weight = weight_g * weight_v / (fishS1NormalizeWeight(weight_v, exceptDim: 0) + 1e-12)
        conv.conv.bias = bias
        return conv(x)
    }
}

public final class FishS1CausalWNConvTranspose1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") public var conv: FishS1CausalTransConvNet

    public var weight_g: MLXArray
    public var weight_v: MLXArray
    public var bias: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let convModule = FishS1CausalTransConvNet(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            dilation: dilation,
            stride: stride,
            groups: groups,
            bias: bias
        )
        let weight = convModule.conv.weight
        self.weight_g = fishS1NormalizeWeight(weight, exceptDim: 0)
        self.weight_v = weight / (weight_g + 1e-12)
        self.bias = convModule.conv.bias
        convModule.conv.weight = weight_v
        convModule.conv.bias = self.bias
        self._conv = ModuleInfo(wrappedValue: convModule)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv.conv.weight = weight_g * weight_v / (fishS1NormalizeWeight(weight_v, exceptDim: 0) + 1e-12)
        conv.conv.bias = bias
        return conv(x)
    }
}
