import Foundation
import MLX
import MLXNN

@inline(__always)
func bigVGANNormalizeWeight(_ x: MLXArray, exceptDim: Int = 0) -> MLXArray {
    let axes = (0..<x.ndim).filter { $0 != exceptDim }
    return MLX.sqrt(MLX.sum(x * x, axes: axes, keepDims: true))
}

private func bigVGANSinc(_ x: Double) -> Double {
    if abs(x) < 1e-12 {
        return 1.0
    }
    return sin(Double.pi * x) / (Double.pi * x)
}

private func bigVGANBesselI0(_ x: Double) -> Double {
    let y = (x * x) / 4.0
    var term = 1.0
    var sum = 1.0
    for k in 1...40 {
        let kk = Double(k)
        term *= y / (kk * kk)
        sum += term
        if term < 1e-12 * sum {
            break
        }
    }
    return sum
}

private func bigVGANKaiserWindow(size: Int, beta: Double) -> [Float] {
    if size <= 1 {
        return [1.0]
    }

    let denom = bigVGANBesselI0(beta)
    let half = Double(size - 1) / 2.0
    return (0..<size).map { idx in
        let ratio = (Double(idx) - half) / half
        let value = bigVGANBesselI0(beta * sqrt(Swift.max(0.0, 1.0 - ratio * ratio))) / denom
        return Float(value)
    }
}

func bigVGANKaiserSincFilter1d(cutoff: Double, halfWidth: Double, kernelSize: Int) -> MLXArray {
    let even = kernelSize.isMultiple(of: 2)
    let halfSize = kernelSize / 2
    let deltaF = 4.0 * halfWidth
    let a = 2.285 * Double(max(halfSize - 1, 0)) * Double.pi * deltaF + 7.95

    let beta: Double
    if a > 50.0 {
        beta = 0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        beta = 0.5842 * pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0)
    } else {
        beta = 0.0
    }

    let window = bigVGANKaiserWindow(size: kernelSize, beta: beta)
    let time: [Double] = (0..<kernelSize).map { idx in
        if even {
            return Double(idx - halfSize) + 0.5
        } else {
            return Double(idx - halfSize)
        }
    }

    guard cutoff > 0 else {
        return MLXArray.zeros([1, kernelSize, 1], dtype: .float32)
    }

    var filter = zip(window, time).map { windowValue, timeValue in
        Float(2.0 * cutoff * Double(windowValue) * bigVGANSinc(2.0 * cutoff * timeValue))
    }
    let sumValue = Swift.max(filter.reduce(0, +), 1e-12)
    filter = filter.map { $0 / sumValue }
    return MLXArray(filter).reshaped([1, kernelSize, 1])
}

public final class BigVGANPeriodicActivation: Module, UnaryLayer {
    public let alphaLogscale: Bool
    public let useBeta: Bool

    public var alpha: MLXArray
    public var beta: MLXArray

    public init(channels: Int, alphaLogscale: Bool, useBeta: Bool) {
        self.alphaLogscale = alphaLogscale
        self.useBeta = useBeta
        let initial = alphaLogscale ? MLXArray.zeros([channels]) : MLXArray.ones([channels])
        self.alpha = initial
        self.beta = initial
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var alphaValue = alpha.reshaped([1, 1, alpha.shape[0]])
        var betaValue = useBeta ? beta.reshaped([1, 1, beta.shape[0]]) : alphaValue
        if alphaLogscale {
            alphaValue = MLX.exp(alphaValue)
            betaValue = MLX.exp(betaValue)
        }
        let recip = 1.0 / (betaValue + 1e-9)
        let sine = MLX.sin(x * alphaValue)
        return x + recip * (sine * sine)
    }
}

public final class BigVGANWNConv1d: Module, UnaryLayer {
    public let kernelSize: Int
    public let stride: Int
    public let padding: Int
    public let dilation: Int
    public let groups: Int

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
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        let scale = sqrt(1.0 / Double((inChannels / Swift.max(groups, 1)) * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, kernelSize, inChannels / groups]
        )
        self.weight_g = bigVGANNormalizeWeight(weightInit)
        self.weight_v = weightInit / (self.weight_g + 1e-12)
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = weight_g * weight_v / (bigVGANNormalizeWeight(weight_v) + 1e-12)
        var y = MLX.conv1d(
            x,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let bias {
            y = y + bias
        }
        return y
    }
}

public final class BigVGANWNConvTranspose1d: Module, UnaryLayer {
    public let kernelSize: Int
    public let stride: Int
    public let padding: Int
    public let dilation: Int
    public let outputPadding: Int
    public let groups: Int

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
        outputPadding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.outputPadding = outputPadding
        self.groups = groups

        let scale = sqrt(1.0 / Double((inChannels / Swift.max(groups, 1)) * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outChannels, kernelSize, inChannels / groups]
        )
        self.weight_g = bigVGANNormalizeWeight(weightInit, exceptDim: 2)
        self.weight_v = weightInit / (self.weight_g + 1e-12)
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = weight_g * weight_v / (bigVGANNormalizeWeight(weight_v, exceptDim: 2) + 1e-12)
        var y = MLX.convTransposed1d(
            x,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            outputPadding: outputPadding,
            groups: groups
        )
        if let bias {
            y = y + bias
        }
        return y
    }
}

public final class BigVGANLowPassFilter1d: Module, UnaryLayer {
    public let stride: Int
    public let padding: Bool
    public let paddingMode: PadMode
    public let padLeft: Int
    public let padRight: Int
    public let filter: MLXArray

    public init(
        cutoff: Double = 0.5,
        halfWidth: Double = 0.6,
        stride: Int = 1,
        padding: Bool = true,
        paddingMode: PadMode = .edge,
        kernelSize: Int = 12
    ) {
        self.stride = stride
        self.padding = padding
        self.paddingMode = paddingMode
        let even = kernelSize.isMultiple(of: 2)
        self.padLeft = kernelSize / 2 - (even ? 1 : 0)
        self.padRight = kernelSize / 2
        self.filter = bigVGANKaiserSincFilter1d(cutoff: cutoff, halfWidth: halfWidth, kernelSize: kernelSize)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.shape[2]
        var work = x
        if padding {
            work = MLX.padded(
                work,
                widths: [IntOrPair(0), IntOrPair((padLeft, padRight)), IntOrPair(0)],
                mode: paddingMode
            )
        }
        let expanded = MLX.broadcast(filter, to: [channels, filter.shape[1], filter.shape[2]])
        return MLX.conv1d(work, expanded, stride: stride, groups: channels)
    }
}

public final class BigVGANUpSample1d: Module, UnaryLayer {
    public let ratio: Int
    public let kernelSize: Int
    public let stride: Int
    public let pad: Int
    public let padLeft: Int
    public let padRight: Int
    public let filter: MLXArray

    public init(ratio: Int = 2, kernelSize: Int? = nil) {
        self.ratio = ratio
        self.kernelSize = kernelSize ?? ((6 * ratio) / 2) * 2
        self.stride = ratio
        self.pad = self.kernelSize / ratio - 1
        self.padLeft = pad * stride + (self.kernelSize - stride) / 2
        self.padRight = pad * stride + (self.kernelSize - stride + 1) / 2
        self.filter = bigVGANKaiserSincFilter1d(
            cutoff: 0.5 / Double(ratio),
            halfWidth: 0.6 / Double(ratio),
            kernelSize: self.kernelSize
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.shape[2]
        var work = MLX.padded(
            x,
            widths: [IntOrPair(0), IntOrPair((pad, pad)), IntOrPair(0)],
            mode: .edge
        )
        let expanded = MLX.broadcast(filter, to: [channels, filter.shape[1], filter.shape[2]])
        work = Float(ratio) * MLX.convTransposed1d(work, expanded, stride: stride, groups: channels)

        let end = work.shape[1] - padRight
        if end <= padLeft {
            return work
        }
        return work[0..., padLeft..<end, 0...]
    }
}

public final class BigVGANDownSample1d: Module, UnaryLayer {
    @ModuleInfo(key: "lowpass") public var lowpass: BigVGANLowPassFilter1d

    public init(ratio: Int = 2, kernelSize: Int? = nil) {
        let resolvedKernel = kernelSize ?? ((6 * ratio) / 2) * 2
        self._lowpass = ModuleInfo(wrappedValue: BigVGANLowPassFilter1d(
            cutoff: 0.5 / Double(ratio),
            halfWidth: 0.6 / Double(ratio),
            stride: ratio,
            kernelSize: resolvedKernel
        ))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        lowpass(x)
    }
}

public final class BigVGANActivation1d: Module, UnaryLayer {
    @ModuleInfo(key: "act") public var act: BigVGANPeriodicActivation
    @ModuleInfo(key: "upsample") public var upsample: BigVGANUpSample1d
    @ModuleInfo(key: "downsample") public var downsample: BigVGANDownSample1d

    public init(
        channels: Int,
        activation: BigVGANActivationType,
        snakeLogscale: Bool,
        upRatio: Int = 2,
        downRatio: Int = 2,
        upKernelSize: Int = 12,
        downKernelSize: Int = 12
    ) {
        self._act = ModuleInfo(wrappedValue: BigVGANPeriodicActivation(
            channels: channels,
            alphaLogscale: snakeLogscale,
            useBeta: activation == .snakebeta
        ))
        self._upsample = ModuleInfo(wrappedValue: BigVGANUpSample1d(ratio: upRatio, kernelSize: upKernelSize))
        self._downsample = ModuleInfo(wrappedValue: BigVGANDownSample1d(ratio: downRatio, kernelSize: downKernelSize))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downsample(act(upsample(x)))
    }
}

public final class BigVGANUpsampleStage: Module, UnaryLayer {
    @ModuleInfo(key: "0") public var conv: BigVGANWNConvTranspose1d

    public init(conv: BigVGANWNConvTranspose1d) {
        self._conv = ModuleInfo(wrappedValue: conv)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x)
    }
}
