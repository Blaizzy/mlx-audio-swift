import Foundation
import MLX
import MLXNN

private func reflectPad1d(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    let length = x.shape[2]
    if left == 0, right == 0 {
        return x
    }

    let leftIndices = stride(from: left, to: 0, by: -1).map { $0 }
    let middleIndices = Array(0..<length)
    let rightIndices = stride(from: length - 2, to: length - 2 - right, by: -1).map { $0 }
    let indices = leftIndices + middleIndices + rightIndices
    let indexArray = MLXArray(indices.map(Int32.init))

    return take(x, indexArray, axis: 2)
}

private func pad1d(
    _ x: MLXArray,
    paddings: (Int, Int),
    mode: String = "constant",
    value _: Float = 0
) -> MLXArray {
    let length = x.shape[2]
    let left = paddings.0
    let right = paddings.1

    precondition(left >= 0 && right >= 0, "Padding must be non-negative")

    if mode == "reflect" {
        let maxPad = max(left, right)
        var extraPad = 0
        var src = x

        if length <= maxPad {
            extraPad = maxPad - length + 1
            src = MLX.padded(src, widths: [.init(0), .init(0), .init((0, extraPad))])
        }

        var padded = reflectPad1d(src, left: left, right: right)
        if extraPad > 0 {
            let newLength = padded.shape[2] - extraPad
            padded = padded[0..., 0..., 0..<newLength]
        }
        return padded
    }

    return MLX.padded(x, widths: [.init(0), .init(0), .init((left, right))])
}

private func getExtraPaddingForConv1d(
    _ x: MLXArray,
    kernelSize: Int,
    stride: Int,
    paddingTotal: Int = 0
) -> Int {
    let length = x.shape[2]
    let nFrames = Float(length - kernelSize + paddingTotal) / Float(stride) + 1
    let idealLength = (Int(ceil(nFrames)) - 1) * stride + (kernelSize - paddingTotal)
    return idealLength - length
}

/// Conv1d with asymmetric padding that mirrors SAM-Audio reference behavior.
public final class SAMConv1d: Module {
    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let stride: Int
    let dilation: Int

    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.dilation = dilation
        self._weight.wrappedValue = MLXArray.zeros([outChannels, kernelSize, inChannels])
        self._bias.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.ndim == 3, "Expected input shape (batch, channels, length)")

        let effectiveKernel = (kernelSize - 1) * dilation + 1
        let paddingTotal = effectiveKernel - stride
        let extraPadding = getExtraPaddingForConv1d(
            x,
            kernelSize: effectiveKernel,
            stride: stride,
            paddingTotal: paddingTotal
        )

        let paddingRight = paddingTotal / 2
        let paddingLeft = paddingTotal - paddingRight
        var padded = pad1d(x, paddings: (paddingLeft, paddingRight + extraPadding))

        padded = padded.transposed(0, 2, 1)
        var out = MLX.conv1d(
            padded,
            weight,
            stride: stride,
            dilation: dilation
        )
        out = out.transposed(0, 2, 1)

        if let bias {
            out = out + bias.reshaped([1, outChannels, 1])
        }
        return out
    }
}

/// Convolution block: GroupNorm -> SiLU -> Conv1d.
public final class ConvBlock1d: Module {
    @ModuleInfo(key: "groupnorm") var groupNorm: GroupNorm
    @ModuleInfo(key: "project") var project: SAMConv1d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        dilation: Int = 1,
        numGroups: Int = 8
    ) {
        self._groupNorm.wrappedValue = GroupNorm(
            groupCount: numGroups,
            dimensions: inChannels,
            pytorchCompatible: true
        )
        self._project.wrappedValue = SAMConv1d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            dilation: dilation
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var xT = x.transposed(0, 2, 1)
        xT = groupNorm(xT)
        var out = xT.transposed(0, 2, 1)
        out = silu(out)
        return project(out)
    }
}

/// Residual block with two ConvBlock1d layers.
public final class ResnetBlock1d: Module {
    @ModuleInfo(key: "block1") var block1: ConvBlock1d
    @ModuleInfo(key: "block2") var block2: ConvBlock1d
    @ModuleInfo(key: "to_out") var toOut: SAMConv1d?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int = 3,
        stride: Int = 1,
        dilation: Int = 1,
        numGroups: Int = 8
    ) {
        self._block1.wrappedValue = ConvBlock1d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            dilation: dilation,
            numGroups: numGroups
        )
        self._block2.wrappedValue = ConvBlock1d(
            inChannels: outChannels,
            outChannels: outChannels,
            numGroups: numGroups
        )
        self._toOut.wrappedValue = inChannels != outChannels
            ? SAMConv1d(inChannels: inChannels, outChannels: outChannels, kernelSize: 1)
            : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = block1(x)
        h = block2(h)
        let residual = toOut?(x) ?? x
        return h + residual
    }
}

/// Patcher used by the DiT to preprocess input features.
public final class Patcher: Module {
    let patchSize: Int
    @ModuleInfo(key: "block") var block: ResnetBlock1d

    public init(
        inChannels: Int,
        outChannels: Int,
        patchSize: Int
    ) {
        precondition(outChannels % patchSize == 0, "outChannels must be divisible by patchSize")
        self.patchSize = patchSize
        self._block.wrappedValue = ResnetBlock1d(
            inChannels: inChannels,
            outChannels: outChannels / patchSize,
            numGroups: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = block(x)
        let batch = out.shape[0]
        let channels = out.shape[1]
        let length = out.shape[2]
        precondition(length % patchSize == 0, "Length must be divisible by patchSize")

        let newLength = length / patchSize
        out = out.reshaped([batch, channels, newLength, patchSize])
        out = out.transposed(0, 1, 3, 2)
        out = out.reshaped([batch, channels * patchSize, newLength])
        return out
    }
}
