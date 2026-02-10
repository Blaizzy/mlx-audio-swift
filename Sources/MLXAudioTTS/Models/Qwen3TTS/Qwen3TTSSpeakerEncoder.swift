// Port of mlx_audio/tts/models/qwen3_tts/speaker_encoder.py
// ECAPA-TDNN speaker encoder that extracts x-vector speaker embeddings from mel spectrograms.

@preconcurrency import MLX
import MLXNN
import Foundation

// MARK: - Reflect Padding Helper

/// Apply reflect padding to the time dimension (axis=1) in NLC format.
///
/// - Parameters:
///   - x: Input tensor `[batch, time, channels]`
///   - pad: Number of samples to pad on each side
/// - Returns: Padded tensor `[batch, time + 2*pad, channels]`
func reflectPad1d(_ x: MLXArray, pad: Int) -> MLXArray {
    if pad <= 0 { return x }
    // Mirror without repeating the boundary element.
    // Python: left = x[:, 1:pad+1, :][:, ::-1, :]
    //         right = x[:, -(pad+1):-1, :][:, ::-1, :]
    let left = x[0..., 1 ..< (pad + 1), 0...][0..., .stride(by: -1), 0...]
    let right = x[0..., (-(pad + 1)) ..< (-1), 0...][0..., .stride(by: -1), 0...]
    return concatenated([left, x, right], axis: 1)
}

// MARK: - TimeDelayNetBlock

/// TDNN block with 1D convolution, reflect padding, and ReLU activation.
final class TimeDelayNetBlock: Module {
    let pad: Int

    @ModuleInfo var conv: Conv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int) {
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
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
final class Res2NetBlock: Module {
    let scale: Int
    @ModuleInfo var blocks: [TimeDelayNetBlock]

    init(inChannels: Int, outChannels: Int, scale: Int = 8, kernelSize: Int = 3, dilation: Int = 1) {
        self.scale = scale
        let inChannel = inChannels / scale
        let hiddenChannel = outChannels / scale

        self._blocks.wrappedValue = (0 ..< scale - 1).map { _ in
            TimeDelayNetBlock(
                inChannels: inChannel,
                outChannels: hiddenChannel,
                kernelSize: kernelSize,
                dilation: dilation
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let chunks = split(x, parts: scale, axis: 1)
        var outputs = [MLXArray]()
        var outputPart: MLXArray? = nil

        for i in 0 ..< chunks.count {
            if i == 0 {
                outputPart = chunks[i]
            } else if i == 1 {
                outputPart = blocks[i - 1](chunks[i])
            } else {
                outputPart = blocks[i - 1](chunks[i] + outputPart!)
            }
            outputs.append(outputPart!)
        }

        return concatenated(outputs, axis: 1)
    }
}

// MARK: - SqueezeExcitationBlock

/// Squeeze-and-excitation block for channel attention.
final class SqueezeExcitationBlock: Module {
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d

    init(inChannels: Int, seChannels: Int, outChannels: Int) {
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
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
final class SqueezeExcitationRes2NetBlock: Module {
    let outChannels: Int

    @ModuleInfo var tdnn1: TimeDelayNetBlock
    @ModuleInfo(key: "res2net_block") var res2netBlock: Res2NetBlock
    @ModuleInfo var tdnn2: TimeDelayNetBlock
    @ModuleInfo(key: "se_block") var seBlock: SqueezeExcitationBlock

    init(
        inChannels: Int,
        outChannels: Int,
        res2netScale: Int = 8,
        seChannels: Int = 128,
        kernelSize: Int = 3,
        dilation: Int = 1
    ) {
        self.outChannels = outChannels

        self._tdnn1.wrappedValue = TimeDelayNetBlock(
            inChannels: inChannels, outChannels: outChannels, kernelSize: 1, dilation: 1
        )
        self._res2netBlock.wrappedValue = Res2NetBlock(
            inChannels: outChannels, outChannels: outChannels,
            scale: res2netScale, kernelSize: kernelSize, dilation: dilation
        )
        self._tdnn2.wrappedValue = TimeDelayNetBlock(
            inChannels: outChannels, outChannels: outChannels, kernelSize: 1, dilation: 1
        )
        self._seBlock.wrappedValue = SqueezeExcitationBlock(
            inChannels: outChannels, seChannels: seChannels, outChannels: outChannels
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
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
final class AttentiveStatisticsPooling: Module {
    let eps: Float = 1e-12

    @ModuleInfo var tdnn: TimeDelayNetBlock
    @ModuleInfo var conv: Conv1d

    init(channels: Int, attentionChannels: Int = 128) {
        self._tdnn.wrappedValue = TimeDelayNetBlock(
            inChannels: channels * 3, outChannels: attentionChannels, kernelSize: 1, dilation: 1
        )
        self._conv.wrappedValue = Conv1d(
            inputChannels: attentionChannels,
            outputChannels: channels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time]
        let batch = x.dim(0)
        let channels = x.dim(1)
        let seqLength = x.dim(2)

        // Compute mean and std
        let xMean = mean(x, axis: 2, keepDims: true)
        let xStd = sqrt(MLX.variance(x, axis: 2, keepDims: true) + eps)

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
        let weightedVar = sum(attention * (x - weightedMean) ** 2, axis: 2, keepDims: true)
        let weightedStd = sqrt(clip(weightedVar, min: eps))

        // Concatenate mean and std
        let pooled = concatenated([weightedMean, weightedStd], axis: 1)
        return pooled
    }
}

// MARK: - Qwen3TTSSpeakerEncoder

/// ECAPA-TDNN speaker encoder for Qwen3-TTS.
///
/// Architecture:
/// ```
/// mel spectrogram [batch, time, mel_dim(128)]
///   -> transpose to [batch, mel_dim(128), time]
///   -> TimeDelayNetBlock (mel_dim=128 -> 512, kernel=5, dilation=1)
///   -> 3x SqueezeExcitationRes2NetBlock (512->512, kernels=[3,3,3], dilations=[2,3,4])
///   -> concatenate hidden states from SE-Res2Net blocks
///   -> TimeDelayNetBlock (MFA: 1536->1536, kernel=1, dilation=1)
///   -> AttentiveStatisticsPooling (1536 -> 3072 via mean+std)
///   -> Conv1d (3072 -> enc_dim, kernel=1)
///   -> squeeze -> [batch, enc_dim]
/// ```
final class Qwen3TTSSpeakerEncoder: Module {
    let config: Qwen3TTSSpeakerEncoderConfig
    let channels: [Int]

    @ModuleInfo var blocks: [Module]
    @ModuleInfo var mfa: TimeDelayNetBlock
    @ModuleInfo var asp: AttentiveStatisticsPooling
    @ModuleInfo var fc: Conv1d

    init(config: Qwen3TTSSpeakerEncoderConfig) {
        self.config = config
        self.channels = config.encChannels

        // Build blocks
        var blocksList = [Module]()

        // Initial TDNN layer
        blocksList.append(
            TimeDelayNetBlock(
                inChannels: config.melDim,
                outChannels: config.encChannels[0],
                kernelSize: config.encKernelSizes[0],
                dilation: config.encDilations[0]
            )
        )

        // SE-Res2Net layers
        for i in 1 ..< config.encChannels.count - 1 {
            blocksList.append(
                SqueezeExcitationRes2NetBlock(
                    inChannels: config.encChannels[i - 1],
                    outChannels: config.encChannels[i],
                    res2netScale: config.encRes2netScale,
                    seChannels: config.encSeChannels,
                    kernelSize: config.encKernelSizes[i],
                    dilation: config.encDilations[i]
                )
            )
        }

        self._blocks.wrappedValue = blocksList

        // Multi-layer feature aggregation
        self._mfa.wrappedValue = TimeDelayNetBlock(
            inChannels: config.encChannels.last!,
            outChannels: config.encChannels.last!,
            kernelSize: config.encKernelSizes.last!,
            dilation: config.encDilations.last!
        )

        // Attentive Statistical Pooling
        self._asp.wrappedValue = AttentiveStatisticsPooling(
            channels: config.encChannels.last!,
            attentionChannels: config.encAttentionChannels
        )

        // Final linear transformation
        self._fc.wrappedValue = Conv1d(
            inputChannels: config.encChannels.last! * 2,
            outputChannels: config.encDim,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    /// Forward pass.
    ///
    /// - Parameter x: Mel spectrogram `[batch, time, mel_dim]`
    /// - Returns: Speaker embedding `[batch, enc_dim]`
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Transpose to [batch, channels, time]
        var out = x.transposed(0, 2, 1)

        var hiddenStatesList = [MLXArray]()
        for layer in blocks {
            if let tdnn = layer as? TimeDelayNetBlock {
                out = tdnn(out)
            } else if let seRes2net = layer as? SqueezeExcitationRes2NetBlock {
                out = seRes2net(out)
            }
            hiddenStatesList.append(out)
        }

        // Multi-layer feature aggregation (concatenate SE-Res2Net outputs, skip first TDNN)
        out = concatenated(Array(hiddenStatesList.dropFirst()), axis: 1)
        out = mfa(out)

        // Attentive Statistical Pooling
        out = asp(out)

        // Final linear transformation - Conv expects NLC format
        out = out.transposed(0, 2, 1)  // NCL -> NLC
        out = fc(out)
        out = out.transposed(0, 2, 1)  // NLC -> NCL

        // Squeeze time dimension
        out = out.squeezed(axis: -1)
        return out
    }

    /// Sanitize weights from PyTorch format to MLX format.
    ///
    /// - Removes the `speaker_encoder.` prefix from keys.
    /// - Transposes Conv1d weights from PyTorch `[out, in, kernel]` to MLX `[out, kernel, in]`.
    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (k, v) in weights {
            guard k.hasPrefix("speaker_encoder.") else { continue }

            let newKey = String(k.dropFirst("speaker_encoder.".count))

            var value = v
            // Handle all Conv1d weights: PyTorch [out, in, kernel] -> MLX [out, kernel, in]
            if newKey.hasSuffix(".weight") && v.ndim == 3 {
                // Check if already in MLX format by comparing dimensions.
                // PyTorch: [out_channels, in_channels, kernel_size]
                // MLX:     [out_channels, kernel_size, in_channels]
                // For kernel_size=1, the last two dims are ambiguous. We transpose
                // only if the shape looks like PyTorch format (dim(1) > dim(2) for
                // typical conv layers where in_channels > kernel_size).
                // Following the Python reference: always transpose unless already correct.
                value = v.transposed(0, 2, 1)
            }

            sanitized[newKey] = value
        }

        return sanitized
    }
}
