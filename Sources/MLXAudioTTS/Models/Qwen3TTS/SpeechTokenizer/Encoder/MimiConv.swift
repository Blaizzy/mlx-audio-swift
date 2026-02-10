//
//  MimiConv.swift
//  MLXAudio
//
//  Mimi codec convolution modules for the encoder pipeline.
//  Ported from mlx_audio/codec/models/mimi/modules/conv.py
//
//  Architecture: All convolutions operate in NCL format [batch, channels, time]
//  and internally swap to NLC for MLX's conv1d.
//

import Foundation
import MLX
import MLXNN

// MARK: - MimiConv1d

/// Custom Conv1d that handles NCL<->NLC format conversion.
///
/// Input/output: NCL [batch, channels, time]
/// MLX conv1d operates on NLC internally.
public class MimiConv1d: Module, UnaryLayer {
    var weight: MLXArray
    var bias: MLXArray?
    let padding: Int
    let groups: Int
    let stride: Int
    let dilation: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.dilation = dilation

        // Weight shape: [outChannels, kernelSize, inChannels/groups] (MLX format)
        self.weight = MLXArray.zeros([outChannels, kernelSize, inChannels / groups])
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // NCL -> NLC
        var y = conv1d(
            x.transposed(0, 2, 1),
            self.weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let b = bias {
            y = y + b
        }
        // NLC -> NCL
        return y.transposed(0, 2, 1)
    }
}

// MARK: - Helper Functions

/// Calculate extra padding needed for proper conv1d output length.
func mimiGetExtraPaddingForConv1d(
    length: Int,
    kernelSize: Int,
    stride: Int,
    paddingTotal: Int
) -> Int {
    let nFrames = max(Float(length + paddingTotal - kernelSize), 0) / Float(stride) + 1.0
    let idealLength = (Int(ceil(nFrames)) - 1) * stride + kernelSize - paddingTotal
    return max(0, idealLength - length)
}

/// Unpad a 1D tensor along the last axis.
func mimiUnpad1d(_ x: MLXArray, unpadLeft: Int, unpadRight: Int) -> MLXArray {
    let left = unpadLeft
    let right = x.shape[x.ndim - 1] - unpadRight
    return x[.ellipsis, left..<right]
}

// MARK: - MimiStreamableConv1d

/// Causal streaming conv1d with proper padding calculation.
///
/// Only the non-streaming `__call__` path is needed for encoding (all-at-once).
public class MimiStreamableConv1d: Module, UnaryLayer {
    let causal: Bool
    let kernelSize: Int
    let padMode: String

    @ModuleInfo(key: "conv") var conv: MimiNormConv1d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        causal: Bool = true,
        padMode: String = "constant"
    ) {
        self.causal = causal
        self.kernelSize = kernelSize
        self.padMode = padMode

        self._conv.wrappedValue = MimiNormConv1d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            dilation: dilation,
            groups: groups,
            bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let effectiveKernel = (kernelSize - 1) * conv.conv.dilation + 1
        let paddingTotal = effectiveKernel - conv.conv.stride
        let extraPadding = mimiGetExtraPaddingForConv1d(
            length: x.shape[x.ndim - 1],
            kernelSize: effectiveKernel,
            stride: conv.conv.stride,
            paddingTotal: paddingTotal
        )

        let paddingLeft: Int
        let paddingRight: Int
        if causal {
            paddingLeft = paddingTotal
            paddingRight = 0
        } else {
            paddingRight = paddingTotal / 2
            paddingLeft = paddingTotal - paddingRight
        }

        // Pad along last axis (time dimension in NCL format)
        let padded: MLXArray
        if padMode == "edge" {
            padded = edgePad1dNCL(x, padLeft: paddingLeft, padRight: paddingRight + extraPadding)
        } else {
            padded = constantPad1dNCL(x, padLeft: paddingLeft, padRight: paddingRight + extraPadding)
        }

        return conv(padded)
    }
}

// MARK: - MimiStreamableConvTranspose1d

/// Causal streaming transposed conv1d with unpadding.
public class MimiStreamableConvTranspose1d: Module, UnaryLayer {
    let causal: Bool
    let kernelSize: Int

    @ModuleInfo(key: "convtr") var convtr: MimiNormConvTranspose1d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        causal: Bool = true
    ) {
        self.causal = causal
        self.kernelSize = kernelSize

        self._convtr.wrappedValue = MimiNormConvTranspose1d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            groups: groups,
            bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let stride = convtr.convtr.stride
        let paddingTotal = max(kernelSize - stride, 0)
        let y = convtr(x)
        if causal {
            return mimiUnpad1d(y, unpadLeft: 0, unpadRight: paddingTotal)
        } else {
            let unpadRight = paddingTotal / 2
            let unpadLeft = paddingTotal - unpadRight
            return mimiUnpad1d(y, unpadLeft: unpadLeft, unpadRight: unpadRight)
        }
    }
}

// MARK: - MimiNormConv1d

/// Thin wrapper around MimiConv1d (norm is identity in current config).
public class MimiNormConv1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: MimiConv1d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = MimiConv1d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            groups: groups,
            dilation: dilation,
            bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}

// MARK: - MimiNormConvTranspose1d

/// Thin wrapper around MimiConvTranspose1d.
public class MimiNormConvTranspose1d: Module, UnaryLayer {
    @ModuleInfo(key: "convtr") var convtr: MimiConvTranspose1d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._convtr.wrappedValue = MimiConvTranspose1d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            groups: groups,
            bias: bias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return convtr(x)
    }
}

// MARK: - MimiConvTranspose1d

/// Transposed Conv1d with NCL<->NLC format handling and depthwise support.
public class MimiConvTranspose1d: Module, UnaryLayer {
    var weight: MLXArray
    var bias: MLXArray?
    let padding: Int
    let groups: Int
    let stride: Int
    let kernelSize: Int
    let inChannels: Int
    let outChannels: Int

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.kernelSize = kernelSize
        self.inChannels = inChannels
        self.outChannels = outChannels

        // Weight shape: [outChannels, kernelSize, inChannels/groups]
        self.weight = MLXArray.zeros([outChannels, kernelSize, inChannels / groups])
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // NCL -> NLC
        var y = convTransposed1d(
            x.transposed(0, 2, 1),
            weight,
            stride: stride,
            padding: padding,
            groups: groups
        )
        if let b = bias {
            y = y + b
        }
        // NLC -> NCL
        return y.transposed(0, 2, 1)
    }
}

// MARK: - MimiConvDownsample1d

/// Downsampling via StreamableConv1d with kernel = 2 * stride.
public class MimiConvDownsample1d: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: MimiStreamableConv1d

    public init(stride: Int, dim: Int, causal: Bool = true) {
        self._conv.wrappedValue = MimiStreamableConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: 2 * stride,
            stride: stride,
            dilation: 1,
            groups: 1,
            bias: false,
            causal: causal,
            padMode: "edge"
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return conv(x)
    }
}

// MARK: - Edge Padding Helper

/// Edge (replicate) padding along the time dimension for NCL format.
func edgePad1dNCL(_ x: MLXArray, padLeft: Int, padRight: Int) -> MLXArray {
    guard padLeft > 0 || padRight > 0 else { return x }

    var result = x
    if padLeft > 0 {
        // Replicate the first time step
        let firstStep = x[0..., 0..., 0..<1]  // [batch, channels, 1]
        let leftPad = broadcast(firstStep, to: [x.shape[0], x.shape[1], padLeft])
        result = concatenated([leftPad, result], axis: 2)
    }
    if padRight > 0 {
        let lastStep = x[0..., 0..., (x.shape[2] - 1)..<x.shape[2]]
        let rightPad = broadcast(lastStep, to: [x.shape[0], x.shape[1], padRight])
        result = concatenated([result, rightPad], axis: 2)
    }
    return result
}
