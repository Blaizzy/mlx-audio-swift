//
//  SpeakerEncoder.swift
//  MLXAudio
//
//  ECAPA-TDNN speaker encoder for Qwen3-TTS.
//  Ported from mlx_audio/tts/models/qwen3_tts/speaker_encoder.py
//

import Foundation
import MLX
import MLXNN

// MARK: - TimeDelayNetBlock

/// TDNN block with 1D convolution, reflect padding, and ReLU activation.
///
/// Shape flow (NCL format):
/// - Input: [batch, in_channels, time]
/// - After transpose: [batch, time, in_channels] (NLC for MLX Conv1d)
/// - After reflect pad: [batch, time + 2*pad, in_channels]
/// - After conv: [batch, time, out_channels]
/// - After transpose back: [batch, out_channels, time]
/// - After ReLU: [batch, out_channels, time]
public class TimeDelayNetBlock: Module, UnaryLayer {
    public let pad: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        dilation: Int
    ) {
        // Compute "same" padding amount
        self.pad = (kernelSize - 1) * dilation / 2
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: 1,
            padding: 0,
            dilation: dilation
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)
        var out = x.transposed(0, 2, 1)  // NCL -> NLC
        out = reflectPad1d(out, pad: pad)
        out = conv(out)
        out = out.transposed(0, 2, 1)  // NLC -> NCL
        return relu(out)
    }
}

// MARK: - Res2NetBlock

/// Res2Net block for multi-scale feature extraction.
///
/// The input is split into `scale` chunks along the channel dimension.
/// Each chunk (except the first) is processed through a TDNN block,
/// with the output of the previous chunk added to the input.
///
/// Shape flow:
/// - Input: [batch, channels, time]
/// - Split: [scale × [batch, channels/scale, time]]
/// - After processing: [batch, channels, time]
public class Res2NetBlock: Module, UnaryLayer {
    public let scale: Int
    @ModuleInfo(key: "blocks") var blocks: [TimeDelayNetBlock]

    public init(
        inChannels: Int,
        outChannels: Int,
        scale: Int = 8,
        kernelSize: Int = 3,
        dilation: Int = 1
    ) {
        self.scale = scale
        let inChannel = inChannels / scale
        let hiddenChannel = outChannels / scale

        var blockList: [TimeDelayNetBlock] = []
        for _ in 0..<(scale - 1) {
            blockList.append(TimeDelayNetBlock(
                inChannels: inChannel,
                outChannels: hiddenChannel,
                kernelSize: kernelSize,
                dilation: dilation
            ))
        }
        self._blocks.wrappedValue = blockList
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let chunks = split(x, parts: scale, axis: 1)
        var outputs: [MLXArray] = []

        var outputPart: MLXArray? = nil
        for i in 0..<chunks.count {
            let chunk = chunks[i]
            if i == 0 {
                outputPart = chunk
            } else if i == 1 {
                outputPart = blocks[i - 1](chunk)
            } else {
                outputPart = blocks[i - 1](chunk + outputPart!)
            }
            outputs.append(outputPart!)
        }

        return concatenated(outputs, axis: 1)
    }
}

// MARK: - SqueezeExcitationBlock

/// Squeeze-and-excitation block for channel attention.
///
/// Shape flow:
/// - Input: [batch, channels, time]
/// - Global avg pool: [batch, channels, 1]
/// - Transpose for conv: [batch, 1, channels]
/// - After conv1 + ReLU: [batch, 1, se_channels]
/// - After conv2 + Sigmoid: [batch, 1, out_channels]
/// - Transpose back: [batch, out_channels, 1]
/// - Output (x * se): [batch, channels, time]
public class SqueezeExcitationBlock: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    public init(inChannels: Int, seChannels: Int, outChannels: Int) {
        self._conv1.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: seChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: seChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)
        // Global average pooling
        let xMean = mean(x, axis: 2, keepDims: true)  // [batch, channels, 1]
        // SE path - transpose for MLX Conv1d (NLC format)
        var se = xMean.transposed(0, 2, 1)  // [batch, 1, channels]
        se = relu(conv1(se))
        se = sigmoid(conv2(se))
        se = se.transposed(0, 2, 1)  // [batch, channels, 1]
        return x * se
    }
}

// MARK: - SqueezeExcitationRes2NetBlock

/// TDNN-Res2Net-TDNN-SE block used in ECAPA-TDNN.
///
/// Architecture: TDNN -> Res2Net -> TDNN -> SE -> Residual
///
/// Shape flow:
/// - Input: [batch, in_channels, time]
/// - After processing: [batch, out_channels, time]
/// - Output (with residual): [batch, out_channels, time]
public class SqueezeExcitationRes2NetBlock: Module, UnaryLayer {
    public let outChannels: Int
    @ModuleInfo(key: "tdnn1") var tdnn1: TimeDelayNetBlock
    @ModuleInfo(key: "res2net_block") var res2netBlock: Res2NetBlock
    @ModuleInfo(key: "tdnn2") var tdnn2: TimeDelayNetBlock
    @ModuleInfo(key: "se_block") var seBlock: SqueezeExcitationBlock

    public init(
        inChannels: Int,
        outChannels: Int,
        res2netScale: Int = 8,
        seChannels: Int = 128,
        kernelSize: Int = 3,
        dilation: Int = 1
    ) {
        self.outChannels = outChannels

        self._tdnn1.wrappedValue = TimeDelayNetBlock(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: 1,
            dilation: 1
        )
        self._res2netBlock.wrappedValue = Res2NetBlock(
            inChannels: outChannels,
            outChannels: outChannels,
            scale: res2netScale,
            kernelSize: kernelSize,
            dilation: dilation
        )
        self._tdnn2.wrappedValue = TimeDelayNetBlock(
            inChannels: outChannels,
            outChannels: outChannels,
            kernelSize: 1,
            dilation: 1
        )
        self._seBlock.wrappedValue = SqueezeExcitationBlock(
            inChannels: outChannels,
            seChannels: seChannels,
            outChannels: outChannels
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = tdnn1(x)
        out = res2netBlock(out)
        out = tdnn2(out)
        out = seBlock(out)
        return out + residual
    }
}

// MARK: - AttentiveStatisticsPooling

/// Attentive statistics pooling layer.
///
/// Computes attention-weighted mean and standard deviation
/// to produce a fixed-size output from variable-length input.
///
/// Shape flow:
/// - Input: [batch, channels, time]
/// - Attention features: [batch, channels*3, time]
/// - After TDNN: [batch, attention_channels, time]
/// - After conv: [batch, channels, time]
/// - After softmax: [batch, channels, time]
/// - Weighted mean: [batch, channels, 1]
/// - Weighted std: [batch, channels, 1]
/// - Output: [batch, channels*2, 1]
public class AttentiveStatisticsPooling: Module, UnaryLayer {
    let eps: Float = 1e-12
    @ModuleInfo(key: "tdnn") var tdnn: TimeDelayNetBlock
    @ModuleInfo(key: "conv") var conv: Conv1d

    public init(channels: Int, attentionChannels: Int = 128) {
        self._tdnn.wrappedValue = TimeDelayNetBlock(
            inChannels: channels * 3,
            outChannels: attentionChannels,
            kernelSize: 1,
            dilation: 1
        )
        self._conv.wrappedValue = Conv1d(
            inputChannels: attentionChannels,
            outputChannels: channels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let batch = x.shape[0]
        let channels = x.shape[1]
        let seqLength = x.shape[2]

        // Compute mean and std
        let xMean = mean(x, axis: 2, keepDims: true)  // [batch, channels, 1]
        let xVar = MLX.variance(x, axis: 2, keepDims: true)
        let xStd = sqrt(xVar + eps)  // [batch, channels, 1]

        // Expand to match sequence length
        let meanExpanded = broadcast(xMean, to: [batch, channels, seqLength])
        let stdExpanded = broadcast(xStd, to: [batch, channels, seqLength])

        // Concatenate features
        var attention = concatenated([x, meanExpanded, stdExpanded], axis: 1)

        // Apply attention
        attention = tdnn(attention)
        attention = tanh(attention)
        // Conv expects NLC format
        attention = attention.transposed(0, 2, 1)  // NCL -> NLC
        attention = conv(attention)
        attention = attention.transposed(0, 2, 1)  // NLC -> NCL
        attention = softmax(attention, axis: 2)

        // Compute weighted mean and std
        let weightedMean = sum(attention * x, axis: 2, keepDims: true)
        let diff = x - weightedMean
        let weightedVar = sum(attention * (diff * diff), axis: 2, keepDims: true)
        let weightedStd = sqrt(maximum(weightedVar, MLXArray(eps)))

        // Concatenate mean and std
        let pooled = concatenated([weightedMean, weightedStd], axis: 1)
        return pooled
    }
}

// MARK: - Qwen3TTSSpeakerEncoder

/// ECAPA-TDNN speaker encoder for Qwen3-TTS.
///
/// Architecture:
/// - Initial TDNN layer (mel_dim -> 512)
/// - 3x SE-Res2Net blocks (512 -> 512 each)
/// - Multi-layer feature aggregation (concat + TDNN)
/// - Attentive statistics pooling
/// - Final FC layer (1536*2 -> 1024)
///
/// Shape flow:
/// - Input: [batch, time, mel_dim]
/// - After transpose: [batch, mel_dim, time]
/// - After blocks: [batch, 512, time] (3 times)
/// - After MFA: [batch, 1536, time]
/// - After ASP: [batch, 3072, 1]
/// - After FC: [batch, 1024, 1]
/// - Output: [batch, 1024]
public class Qwen3TTSSpeakerEncoder: Module, UnaryLayer {
    public let config: Qwen3TTSSpeakerEncoderConfig
    public let channels: [Int]

    // Store blocks as a module list
    @ModuleInfo(key: "blocks") var blocks: [Module]

    // Multi-layer feature aggregation
    @ModuleInfo(key: "mfa") var mfa: TimeDelayNetBlock

    // Attentive Statistical Pooling
    @ModuleInfo(key: "asp") var asp: AttentiveStatisticsPooling

    // Final linear transformation
    @ModuleInfo(key: "fc") var fc: Conv1d

    public init(config: Qwen3TTSSpeakerEncoderConfig) {
        self.config = config
        self.channels = config.encChannels

        // Build blocks
        var blockList: [Module] = []

        // Initial TDNN layer
        blockList.append(TimeDelayNetBlock(
            inChannels: config.melDim,
            outChannels: config.encChannels[0],
            kernelSize: config.encKernelSizes[0],
            dilation: config.encDilations[0]
        ))

        // SE-Res2Net layers
        for i in 1..<(config.encChannels.count - 1) {
            blockList.append(SqueezeExcitationRes2NetBlock(
                inChannels: config.encChannels[i - 1],
                outChannels: config.encChannels[i],
                res2netScale: config.encRes2netScale,
                seChannels: config.encSeChannels,
                kernelSize: config.encKernelSizes[i],
                dilation: config.encDilations[i]
            ))
        }

        self._blocks.wrappedValue = blockList

        // Multi-layer feature aggregation
        let lastChannels = config.encChannels[config.encChannels.count - 1]
        self._mfa.wrappedValue = TimeDelayNetBlock(
            inChannels: lastChannels,
            outChannels: lastChannels,
            kernelSize: config.encKernelSizes[config.encKernelSizes.count - 1],
            dilation: config.encDilations[config.encDilations.count - 1]
        )

        // Attentive Statistical Pooling
        self._asp.wrappedValue = AttentiveStatisticsPooling(
            channels: lastChannels,
            attentionChannels: config.encAttentionChannels
        )

        // Final linear transformation
        self._fc.wrappedValue = Conv1d(
            inputChannels: lastChannels * 2,
            outputChannels: config.encDim,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: Mel spectrogram [batch, time, mel_dim]

        // Transpose to [batch, channels, time]
        var out = x.transposed(0, 2, 1)

        var hiddenStatesList: [MLXArray] = []
        for block in blocks {
            if let tdnn = block as? TimeDelayNetBlock {
                out = tdnn(out)
            } else if let seRes2Net = block as? SqueezeExcitationRes2NetBlock {
                out = seRes2Net(out)
            }
            hiddenStatesList.append(out)
        }

        // Multi-layer feature aggregation (concatenate SE-Res2Net outputs)
        // Skip the first block (initial TDNN) output
        out = concatenated(Array(hiddenStatesList[1...]), axis: 1)
        out = mfa(out)

        // Attentive Statistical Pooling
        out = asp(out)

        // Final linear transformation - Conv expects NLC format
        out = out.transposed(0, 2, 1)  // NCL -> NLC
        out = fc(out)
        out = out.transposed(0, 2, 1)  // NLC -> NCL

        // Squeeze time dimension
        out = squeezed(out, axis: -1)
        return out
    }

    /// Check if Conv1d weights are already in MLX format.
    ///
    /// MLX format: (out_channels, kernel_size, in_channels)
    /// PyTorch format: (out_channels, in_channels, kernel_size)
    ///
    /// Uses heuristics: kernel_size is typically smaller than in_channels.
    private static func isMLXFormat(_ arr: MLXArray) -> Bool {
        guard arr.ndim == 3 else { return false }

        let shape = arr.shape
        let dim2 = shape[1]
        let dim3 = shape[2]

        if dim2 == 1 {
            // Pattern: (out, 1, dim3)
            // If dim3 is large, likely in_channels -> MLX format (out, kernel=1, in)
            return dim3 > 64
        } else if dim3 == 1 {
            // Pattern: (out, dim2, 1)
            // If dim2 is large, likely in_channels -> PyTorch format (out, in, kernel=1)
            return dim2 <= 64
        }

        // General heuristic: kernel_size < in_channels is more common
        // So if middle dimension is smaller, it's likely already MLX format
        return dim2 < dim3
    }

    /// Sanitize weights from PyTorch/safetensors format to MLX format.
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            // Only process speaker_encoder weights
            guard key.hasPrefix("speaker_encoder.") else { continue }

            // Remove prefix
            let newKey = String(key.dropFirst("speaker_encoder.".count))

            // Handle Conv1d weight transposition
            // PyTorch Conv1d: [out_channels, in_channels, kernel_size]
            // MLX Conv1d: [out_channels, kernel_size, in_channels]
            if newKey.hasSuffix(".weight") && value.ndim == 3 {
                // Only transpose if not already in MLX format
                if isMLXFormat(value) {
                    sanitized[newKey] = value
                } else {
                    // Transpose from [out, in, k] to [out, k, in]
                    sanitized[newKey] = value.transposed(0, 2, 1)
                }
            } else {
                sanitized[newKey] = value
            }
        }

        return sanitized
    }
}
