//
//  DecoderBlocks.swift
//  MLXAudio
//
//  HiFi-GAN style decoder blocks for Speech Tokenizer Decoder.
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
//

import Foundation
import MLX
import MLXNN

// MARK: - DecoderResidualUnit

/// Residual unit for decoder.
///
/// Architecture:
/// - SnakeBeta activation
/// - CausalConv1d (kernel=7, dilation=dilation)
/// - SnakeBeta activation
/// - CausalConv1d (kernel=1)
/// - Residual connection
///
/// PyTorch weight keys:
/// - act1.alpha, act1.beta (SnakeBeta)
/// - conv1.conv.weight, conv1.conv.bias (CausalConv1d)
/// - act2.alpha, act2.beta (SnakeBeta)
/// - conv2.conv.weight, conv2.conv.bias (CausalConv1d)
public class DecoderResidualUnit: Module, UnaryLayer {
    public let dim: Int
    public let dilation: Int

    @ModuleInfo(key: "act1") var act1: SnakeBeta
    @ModuleInfo(key: "conv1") var conv1: CausalConv1d
    @ModuleInfo(key: "act2") var act2: SnakeBeta
    @ModuleInfo(key: "conv2") var conv2: CausalConv1d

    public init(dim: Int, dilation: Int = 1) {
        self.dim = dim
        self.dilation = dilation

        self._act1.wrappedValue = SnakeBeta(channels: dim)
        self._conv1.wrappedValue = CausalConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: 7,
            dilation: dilation
        )
        self._act2.wrappedValue = SnakeBeta(channels: dim)
        self._conv2.wrappedValue = CausalConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: 1
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)
        return x + conv2(act2(conv1(act1(x))))
    }
}

// MARK: - DecoderBlockUpsample

/// Upsample layer wrapper for decoder block.
///
/// Implements causal transpose conv with right trim.
/// kernel_size = 2 * upsample_rate for smooth upsampling.
///
/// PyTorch weight keys: conv.weight, conv.bias
public class DecoderBlockUpsample: Module, UnaryLayer {
    public let inDim: Int
    public let outDim: Int
    public let upsampleRate: Int
    public let trimRight: Int

    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    public init(inDim: Int, outDim: Int, upsampleRate: Int) {
        self.inDim = inDim
        self.outDim = outDim
        self.upsampleRate = upsampleRate

        let kernelSize = 2 * upsampleRate
        // Trim = kernel_size - stride for causal behavior
        self.trimRight = kernelSize - upsampleRate

        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inDim,
            outputChannels: outDim,
            kernelSize: kernelSize,
            stride: upsampleRate,
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

// MARK: - DecoderBlock

/// Decoder block with upsampling (HiFi-GAN style).
///
/// Architecture:
/// - block[0]: SnakeBeta activation
/// - block[1]: DecoderBlockUpsample (transpose conv)
/// - block[2-4]: 3x DecoderResidualUnit with dilations [1, 3, 9]
///
/// Channel progression:
/// - Block 0: 1536 -> 768, stride=8
/// - Block 1: 768 -> 384, stride=5
/// - Block 2: 384 -> 192, stride=4
/// - Block 3: 192 -> 96, stride=3
public class DecoderBlock: Module, UnaryLayer {
    public let inDim: Int
    public let outDim: Int
    public let upsampleRate: Int

    // PyTorch uses self.block as a ModuleList
    // block[0]: SnakeBeta
    // block[1]: DecoderBlockUpsample
    // block[2-4]: DecoderResidualUnit
    @ModuleInfo(key: "block") var block: [Module]

    public init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        // Channel dimensions: decoder_dim / 2^layer_idx -> decoder_dim / 2^(layer_idx+1)
        self.inDim = config.decoderDim / (1 << layerIdx)
        self.outDim = config.decoderDim / (1 << (layerIdx + 1))
        self.upsampleRate = config.upsampleRates[layerIdx]

        var blockLayers: [Module] = []

        // block[0]: SnakeBeta activation
        blockLayers.append(SnakeBeta(channels: inDim))

        // block[1]: Upsample
        blockLayers.append(DecoderBlockUpsample(
            inDim: inDim,
            outDim: outDim,
            upsampleRate: upsampleRate
        ))

        // block[2-4]: Residual units with dilations [1, 3, 9]
        for dilation in [1, 3, 9] {
            blockLayers.append(DecoderResidualUnit(dim: outDim, dilation: dilation))
        }

        self._block.wrappedValue = blockLayers
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x

        for layer in block {
            if let snakeBeta = layer as? SnakeBeta {
                out = snakeBeta(out)
            } else if let upsample = layer as? DecoderBlockUpsample {
                out = upsample(out)
            } else if let residual = layer as? DecoderResidualUnit {
                out = residual(out)
            }
        }
        return out
    }
}

// MARK: - DecoderInitialConv

/// Initial decoder convolution.
///
/// Transforms latent_dim (1024) to decoder_dim (1536).
/// Uses causal padding with kernel_size=7.
///
/// PyTorch key: decoder.decoder.0.conv.{weight,bias}
public class DecoderInitialConv: Module, UnaryLayer {
    public let latentDim: Int
    public let decoderDim: Int
    public let kernelSize: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    public init(latentDim: Int, decoderDim: Int, kernelSize: Int = 7) {
        self.latentDim = latentDim
        self.decoderDim = decoderDim
        self.kernelSize = kernelSize

        // Note: Causal padding is handled inline during forward pass
        self._conv.wrappedValue = Conv1d(
            inputChannels: latentDim,
            outputChannels: decoderDim,
            kernelSize: kernelSize,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)

        // Causal padding (left pad on time dimension)
        let padded = constantPad1dNCL(x, padLeft: kernelSize - 1, padRight: 0)

        // MLX Conv1d expects [batch, time, channels] (NLC format)
        var out = padded.transposed(0, 2, 1)  // NCL -> NLC
        out = conv(out)
        out = out.transposed(0, 2, 1)  // NLC -> NCL

        return out
    }
}

// MARK: - DecoderOutputSnake

/// Output SnakeBeta layer for decoder.
///
/// Applied before the final output convolution.
///
/// PyTorch key: decoder.decoder.5.{alpha, beta}
public class DecoderOutputSnake: Module, UnaryLayer {
    public let channels: Int
    private let eps: Float = 1e-9

    var alpha: MLXArray
    var beta: MLXArray

    public init(channels: Int) {
        self.channels = channels
        // Initialize with zeros: exp(0) = 1
        self.alpha = MLXArray.zeros([channels])
        self.beta = MLXArray.zeros([channels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)

        let alphaExp = exp(alpha).reshaped([1, -1, 1])
        let betaExp = exp(beta).reshaped([1, -1, 1])

        // SnakeBeta formula
        return x + (1.0 / (betaExp + eps)) * pow(sin(x * alphaExp), 2)
    }
}

// MARK: - DecoderOutputConv

/// Output convolution layer for decoder.
///
/// Reduces channels to 1 (mono audio output).
/// Uses causal padding with kernel_size=7.
///
/// PyTorch key: decoder.decoder.6.conv.{weight, bias}
public class DecoderOutputConv: Module, UnaryLayer {
    public let channels: Int
    public let kernelSize: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    public init(channels: Int, kernelSize: Int = 7) {
        self.channels = channels
        self.kernelSize = kernelSize

        // Output is mono (1 channel)
        self._conv.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: 1,
            kernelSize: kernelSize,
            padding: 0
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)

        // Causal padding (left pad on time dimension)
        let padded = constantPad1dNCL(x, padLeft: kernelSize - 1, padRight: 0)

        // MLX Conv1d expects [batch, time, channels] (NLC format)
        var out = padded.transposed(0, 2, 1)  // NCL -> NLC
        out = conv(out)
        out = out.transposed(0, 2, 1)  // NLC -> NCL

        return out
    }
}
