import Foundation
import MLX
import MLXNN

// MARK: - Helper Functions

/// Compute L2 normalization factor across all dims except exceptDim.
func fishNormalizeWeight(_ x: MLXArray, exceptDim: Int = 0) -> MLXArray {
    let axes = (0..<x.ndim).filter { $0 != exceptDim }
    return MLX.sqrt(MLX.sum(x * x, axes: axes, keepDims: true))
}

/// Snake activation: x + (1/alpha) * sin(alpha * x)^2
func fishSnake(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    let recip = 1.0 / (alpha + 1e-9)
    let sinVal = MLX.sin(alpha * x)
    return x + recip * (sinVal * sinVal)
}

/// Remove padding from last dimension (NLC layout).
func fishUnpad1d(_ x: MLXArray, paddings: (Int, Int)) -> MLXArray {
    let (padLeft, padRight) = paddings
    let end = x.dim(-1) - padRight
    if padLeft == 0 && padRight == 0 { return x }
    if padRight == 0 {
        return x[0..., 0..., padLeft...]
    }
    return x[0..., 0..., padLeft..<end]
}

/// Calculate extra padding for conv1d to ensure proper output alignment.
func fishGetExtraPaddingForConv1d(
    _ x: MLXArray, kernelSize: Int, stride: Int, paddingTotal: Int
) -> Int {
    let length = x.dim(-1)
    let nFrames = Float(length + paddingTotal - kernelSize) / Float(stride) + 1.0
    let idealLength = (Int(ceil(nFrames)) - 1) * stride + kernelSize - paddingTotal
    return max(idealLength - length, 0)
}

// MARK: - Snake1d Activation

public class FishSnake1d: Module {
    @ModuleInfo(key: "alpha") var alpha: MLXArray

    public init(channels: Int) {
        // Shape [1, channels, 1] for NCL broadcasting
        self._alpha.wrappedValue = MLXArray.ones([1, channels, 1])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        fishSnake(x, alpha: alpha)
    }
}

// MARK: - Conv1d with PyTorch Weight Layout

/// Conv1d that stores weights in PyTorch [out, in/groups, k] layout
/// and transposes for MLX conv1d (which expects [out, k, in/groups]).
public class FishConv1dTorch: Module {
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var biasParam: MLXArray?

    public init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, padding: Int = 0, dilation: Int = 1,
        groups: Int = 1, bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        // PyTorch layout: [out, in/groups, k]
        let scale = sqrt(1.0 / Float(inChannels / groups * kernelSize))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale,
            [outChannels, inChannels / groups, kernelSize]
        )
        self._biasParam.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, C, T] (NCL) -> transpose to [B, T, C] (NLC) for MLX
        var h = x.transposed(0, 2, 1)
        // Weight: [out, in/groups, k] -> [out, k, in/groups] for MLX
        let w = weight.transposed(0, 2, 1)
        h = MLX.conv1d(h, w, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias = biasParam {
            h = h + bias
        }
        // Back to NCL
        return h.transposed(0, 2, 1)
    }
}

/// ConvTranspose1d with PyTorch weight layout.
public class FishConvTranspose1dTorch: Module {
    let stride: Int
    let padding: Int
    let groups: Int

    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var biasParam: MLXArray?

    public init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, padding: Int = 0,
        groups: Int = 1, bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.groups = groups

        // PyTorch layout: [in, out/groups, k]
        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale,
            [inChannels, outChannels / groups, kernelSize]
        )
        self._biasParam.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: NCL -> NLC
        var h = x.transposed(0, 2, 1)
        // Weight: [in, out/groups, k] -> [out/groups, k, in] for MLX convTranspose1d
        let w = weight.transposed(1, 2, 0)
        h = MLX.convTransposed1d(h, w, stride: stride, padding: padding, groups: groups)
        if let bias = biasParam {
            h = h + bias
        }
        return h.transposed(0, 2, 1)
    }
}

// MARK: - Weight-Normalized Convolutions

/// Weight-normalized Conv1d for Fish S1 DAC.
/// Stores weight_g (magnitude) and weight_v (direction) in PyTorch [out, in/groups, k] layout.
public class FishWNConv1d: Module {
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    @ModuleInfo(key: "bias") var biasParam: MLXArray?

    public init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, padding: Int = 0, dilation: Int = 1,
        groups: Int = 1, bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        // PyTorch layout: [out, in/groups, k]
        let scale = sqrt(1.0 / Float(inChannels / groups * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale, high: scale,
            [outChannels, inChannels / groups, kernelSize]
        )
        self._weightG.wrappedValue = fishNormalizeWeight(weightInit)
        self._weightV.wrappedValue = weightInit / (fishNormalizeWeight(weightInit) + 1e-12)
        self._biasParam.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Compute normalized weight
        let w = weightG * weightV / (fishNormalizeWeight(weightV) + 1e-12)
        // x: NCL -> NLC, weight: [out, in/groups, k] -> [out, k, in/groups]
        var h = x.transposed(0, 2, 1)
        let wt = w.transposed(0, 2, 1)
        h = MLX.conv1d(h, wt, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias = biasParam {
            h = h + bias
        }
        return h.transposed(0, 2, 1)
    }
}

/// Weight-normalized ConvTranspose1d for Fish S1 DAC.
public class FishWNConvTranspose1d: Module {
    let stride: Int
    let padding: Int
    let groups: Int

    @ModuleInfo(key: "weight_g") var weightG: MLXArray
    @ModuleInfo(key: "weight_v") var weightV: MLXArray
    @ModuleInfo(key: "bias") var biasParam: MLXArray?

    public init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, padding: Int = 0,
        groups: Int = 1, bias: Bool = true
    ) {
        self.stride = stride
        self.padding = padding
        self.groups = groups

        // PyTorch layout: [in, out/groups, k]
        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        let weightInit = MLXRandom.uniform(
            low: -scale, high: scale,
            [inChannels, outChannels / groups, kernelSize]
        )
        self._weightG.wrappedValue = fishNormalizeWeight(weightInit)
        self._weightV.wrappedValue = weightInit / (fishNormalizeWeight(weightInit) + 1e-12)
        self._biasParam.wrappedValue = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = weightG * weightV / (fishNormalizeWeight(weightV) + 1e-12)
        var h = x.transposed(0, 2, 1)
        // Weight: [in, out/groups, k] -> [out/groups, k, in]
        let wt = w.transposed(1, 2, 0)
        h = MLX.convTransposed1d(h, wt, stride: stride, padding: padding, groups: groups)
        if let bias = biasParam {
            h = h + bias
        }
        return h.transposed(0, 2, 1)
    }
}

// MARK: - Causal Convolution Wrappers

/// Causal Conv1d: pads (kernel_size - stride) on the left.
public class FishCausalConvNet: Module {
    let kernelSize: Int
    let stride: Int
    let dilation: Int

    @ModuleInfo(key: "conv") var conv: FishConv1dTorch

    public init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, dilation: Int = 1, groups: Int = 1, bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.dilation = dilation
        self._conv.wrappedValue = FishConv1dTorch(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: 0,
            dilation: dilation, groups: groups, bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let paddingTotal = (kernelSize - stride) * dilation
        let extraPad = fishGetExtraPaddingForConv1d(x, kernelSize: kernelSize, stride: stride, paddingTotal: paddingTotal)

        // Pad on left (causal) and extra on right
        var h = x
        if paddingTotal + extraPad > 0 {
            h = MLX.padded(h, widths: [.init(0), .init(0), .init((paddingTotal, extraPad))])
        }
        return conv(h)
    }
}

/// Causal ConvTranspose1d: removes (kernel_size - stride) from right.
public class FishCausalTransConvNet: Module {
    let kernelSize: Int
    let stride: Int

    @ModuleInfo(key: "conv") var conv: FishConvTranspose1dTorch

    public init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, groups: Int = 1, bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self._conv.wrappedValue = FishConvTranspose1dTorch(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: 0,
            groups: groups, bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        let padRight = kernelSize - stride
        if padRight > 0 {
            h = fishUnpad1d(h, paddings: (0, padRight))
        }
        return h
    }
}

/// Causal weight-normalized Conv1d.
public class FishCausalWNConv1d: Module {
    let kernelSize: Int
    let stride: Int
    let dilation: Int

    @ModuleInfo(key: "conv") var conv: FishWNConv1d

    public init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, padding: Int = 0, dilation: Int = 1,
        groups: Int = 1, bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self.dilation = dilation
        self._conv.wrappedValue = FishWNConv1d(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: 0,
            dilation: dilation, groups: groups, bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let paddingTotal = (kernelSize - stride) * dilation
        let extraPad = fishGetExtraPaddingForConv1d(x, kernelSize: kernelSize, stride: stride, paddingTotal: paddingTotal)

        var h = x
        if paddingTotal + extraPad > 0 {
            h = MLX.padded(h, widths: [.init(0), .init(0), .init((paddingTotal, extraPad))])
        }
        return conv(h)
    }
}

/// Causal weight-normalized ConvTranspose1d.
public class FishCausalWNConvTranspose1d: Module {
    let kernelSize: Int
    let stride: Int

    @ModuleInfo(key: "conv") var conv: FishWNConvTranspose1d

    public init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, groups: Int = 1, bias: Bool = true
    ) {
        self.kernelSize = kernelSize
        self.stride = stride
        self._conv.wrappedValue = FishWNConvTranspose1d(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: 0,
            groups: groups, bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        let padRight = kernelSize - stride
        if padRight > 0 {
            h = fishUnpad1d(h, paddings: (0, padRight))
        }
        return h
    }
}

// MARK: - RMS Normalization

public class FishTFRMSNorm: Module {
    let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray

    public init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rms = MLX.sqrt(MLX.mean(x * x, axis: -1, keepDims: true) + eps)
        return (x / rms) * weight
    }
}

// MARK: - LayerScale

public class FishLayerScale: Module {
    @ModuleInfo(key: "gamma") var gamma: MLXArray

    public init(dim: Int, initValue: Float = 1e-2) {
        self._gamma.wrappedValue = initValue * MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

// MARK: - ConvNeXt Block

/// ConvNeXt block with causal depthwise conv, LayerNorm, and pointwise convolutions.
public class FishConvNeXtBlock: Module {
    let dim: Int

    @ModuleInfo(key: "dwconv") var dwconv: FishCausalConvNet
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    @ModuleInfo(key: "gamma") var gamma: MLXArray?

    public init(dim: Int, kernelSize: Int = 7, layerScaleInitValue: Float = 1e-6) {
        self.dim = dim
        self._dwconv.wrappedValue = FishCausalConvNet(
            inChannels: dim, outChannels: dim, kernelSize: kernelSize,
            groups: dim
        )
        self._norm.wrappedValue = LayerNorm(dimensions: dim)
        self._pwconv1.wrappedValue = Linear(dim, dim * 4)
        self._pwconv2.wrappedValue = Linear(dim * 4, dim)
        if layerScaleInitValue > 0 {
            self._gamma.wrappedValue = layerScaleInitValue * MLXArray.ones([dim])
        } else {
            self._gamma.wrappedValue = nil
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        // x: NCL
        var h = dwconv(x)
        // NCL -> NLC for LayerNorm and pointwise
        h = h.transposed(0, 2, 1)
        h = norm(h)
        h = pwconv1(h)
        h = MLXNN.gelu(h)
        h = pwconv2(h)
        if let g = gamma {
            h = h * g
        }
        // NLC -> NCL
        h = h.transposed(0, 2, 1)
        return residual + h
    }
}
