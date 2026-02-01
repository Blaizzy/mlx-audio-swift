//
//  Convolutions.swift
//  MLXAudio
//
//  Causal convolution primitives for Speech Tokenizer Decoder.
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Depthwise Convolution Weight Container

/// Container for depthwise conv weights to match PyTorch key structure.
///
/// Weight shape: [outChannels, kernel, inPerGroup] = [channels, kernel, 1] for depthwise
public class DepthwiseConvWeight: Module {
    var weight: MLXArray
    var bias: MLXArray

    public init(outChannels: Int, kernelSize: Int, inPerGroup: Int) {
        self.weight = MLXArray.zeros([outChannels, kernelSize, inPerGroup])
        self.bias = MLXArray.zeros([outChannels])
    }
}

// MARK: - CausalConv1d

/// Causal 1D convolution with proper padding.
///
/// Supports grouped convolutions for depthwise convs.
/// Input format: NCL (batch, channels, time)
/// Output format: NCL (batch, channels, time)
///
/// Causal padding ensures output at time t only depends on inputs at times <= t.
public class CausalConv1d: Module, UnaryLayer {
    public let groups: Int
    public let inChannels: Int
    public let outChannels: Int
    public let stride: Int
    public let kernelSizeVal: Int
    public let kernelSize: Int  // Effective kernel size with dilation
    public let dilation: Int
    public let padding: Int

    // Either regular Conv1d or DepthwiseConvWeight for grouped conv
    // Note: In Python both use key "conv", but Swift needs unique keys.
    // Weight sanitization maps "conv" -> "conv" for groups==1 and "conv" -> "depthwiseConv" for groups>1
    @ModuleInfo(key: "conv") var conv: Conv1d?
    @ModuleInfo(key: "depthwiseConv") var depthwiseConv: DepthwiseConvWeight?

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1
    ) {
        self.groups = groups
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.stride = stride
        self.kernelSizeVal = kernelSize
        self.kernelSize = (kernelSize - 1) * dilation + 1  // Effective kernel size
        self.dilation = dilation
        self.padding = self.kernelSize - stride

        if groups == 1 {
            // Regular convolution
            self._conv.wrappedValue = Conv1d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: kernelSize,
                stride: stride,
                padding: 0,
                dilation: dilation
            )
            self._depthwiseConv.wrappedValue = nil
        } else {
            // Grouped/depthwise convolution - implement manually
            let inPerGroup = inChannels / groups
            self._conv.wrappedValue = nil
            self._depthwiseConv.wrappedValue = DepthwiseConvWeight(
                outChannels: outChannels,
                kernelSize: kernelSize,
                inPerGroup: inPerGroup
            )
        }
    }

    /// Calculate extra padding needed for proper output length.
    private func getExtraPadding(length: Int) -> Int {
        let nFrames = Float(length - kernelSize + padding) / Float(stride) + 1
        let idealLength = (Int(ceil(nFrames)) - 1) * stride + (kernelSize - padding)
        return idealLength - length
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)
        let extraPadding = getExtraPadding(length: x.shape[2])

        // Pad on the left (causal) using constantPad1dNCL from utils
        let padded = constantPad1dNCL(x, padLeft: padding, padRight: extraPadding)

        if groups == 1 {
            // MLX Conv1d expects [batch, time, channels] (NLC format)
            var out = padded.transposed(0, 2, 1)  // NCL -> NLC
            out = conv!(out)
            out = out.transposed(0, 2, 1)  // NLC -> NCL
            return out
        } else {
            // Depthwise convolution: apply each filter to its corresponding channel
            // x: [batch, channels, time], weight: [channels, kernel, 1]
            let channels = padded.shape[1]
            let time = padded.shape[2]
            let kSize = depthwiseConv!.weight.shape[1]

            // Calculate output time dimension
            let outputTime = time - kSize + 1

            // Create sliding windows: [batch, channels, outputTime, kernel]
            var windowSlices: [MLXArray] = []
            for i in 0..<kSize {
                windowSlices.append(padded[0..., 0..., i..<(i + outputTime)])
            }
            let windows = stacked(windowSlices, axis: -1)

            // Apply weights: weight is [channels, kernel, 1] -> squeeze to [channels, kernel]
            let w = depthwiseConv!.weight.squeezed(axis: -1)  // [channels, kernel]

            // Multiply and sum: [batch, channels, outputTime, kernel] * [1, channels, 1, kernel] -> [batch, channels, outputTime]
            let wExpanded = w.reshaped([1, channels, 1, kSize])
            var out = sum(windows * wExpanded, axis: -1)

            // Add bias: [channels] -> [1, channels, 1]
            let biasExpanded = depthwiseConv!.bias.reshaped([1, channels, 1])
            out = out + biasExpanded

            return out
        }
    }
}

// MARK: - CausalTransposeConv1d

/// Causal transposed 1D convolution for upsampling.
///
/// Input format: NCL (batch, channels, time)
/// Output format: NCL (batch, channels, time * stride)
///
/// Trims from the right to maintain causal behavior.
public class CausalTransposeConv1d: Module, UnaryLayer {
    public let trimRight: Int

    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1
    ) {
        // Trim from the right for causal behavior (matches Encodec/DAC/Mimi)
        self.trimRight = kernelSize - stride

        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)
        // MLX ConvTransposed1d expects [batch, time, channels] (NLC format)
        var out = x.transposed(0, 2, 1)  // NCL -> NLC
        out = conv(out)
        out = out.transposed(0, 2, 1)  // NLC -> NCL

        // Trim from right for causal behavior
        if trimRight > 0 {
            let newLen = out.shape[2] - trimRight
            out = out[0..., 0..., 0..<newLen]
        }

        return out
    }
}
