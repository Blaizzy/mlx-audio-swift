//
//  MimiSeanetEncoder.swift
//  MLXAudio
//
//  SEANet encoder for the Mimi codec encoder pipeline.
//  Ported from mlx_audio/codec/models/mimi/modules/seanet.py (encoder parts only)
//
//  Architecture:
//    init_conv1d -> [EncoderLayer × 4] -> ELU -> final_conv1d
//  Each EncoderLayer = [ResnetBlock × nresidual] -> ELU -> downsample_conv
//

import Foundation
import MLX
import MLXNN

// MARK: - MimiSeanetResnetBlock

/// Residual block with ELU activation for SEANet.
///
/// block[0]: ELU -> Conv(ksize=residual_ksize, dilation=d) -> hidden_dim
/// block[1]: ELU -> Conv(ksize=1, dilation=1) -> dim
/// + skip connection (true_skip = identity, else 1x1 conv)
public class MimiSeanetResnetBlock: Module {
    @ModuleInfo(key: "block") var block: [MimiStreamableConv1d]
    @ModuleInfo(key: "shortcut") var shortcut: MimiStreamableConv1d?

    public init(
        dim: Int,
        kernelSizesAndDilations: [(Int, Int)],
        compress: Int = 2,
        trueSkip: Bool = true,
        causal: Bool = true,
        padMode: String = "constant"
    ) {
        let hidden = dim / compress
        var blockLayers: [MimiStreamableConv1d] = []

        for (i, (ksize, dilation)) in kernelSizesAndDilations.enumerated() {
            let inCh = (i == 0) ? dim : hidden
            let outCh = (i == kernelSizesAndDilations.count - 1) ? dim : hidden
            blockLayers.append(MimiStreamableConv1d(
                inChannels: inCh,
                outChannels: outCh,
                kernelSize: ksize,
                stride: 1,
                dilation: dilation,
                groups: 1,
                bias: true,
                causal: causal,
                padMode: padMode
            ))
        }
        self._block.wrappedValue = blockLayers

        if trueSkip {
            self._shortcut.wrappedValue = nil
        } else {
            self._shortcut.wrappedValue = MimiStreamableConv1d(
                inChannels: dim,
                outChannels: dim,
                kernelSize: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                bias: true,
                causal: causal,
                padMode: padMode
            )
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var xs = x
        for b in block {
            xs = b(elu(xs))
        }
        if let sc = shortcut {
            residual = sc(residual)
        }
        return xs + residual
    }
}

// MARK: - MimiEncoderLayer

/// Single encoder layer: residual blocks + downsample convolution.
///
/// The ratios are reversed for encoding (downsample instead of upsample).
public class MimiEncoderLayer: Module {
    @ModuleInfo(key: "residuals") var residuals: [MimiSeanetResnetBlock]
    @ModuleInfo(key: "downsample") var downsample: MimiStreamableConv1d

    public init(
        ratio: Int,
        mult: Int,
        numFilters: Int,
        numResidualLayers: Int = 1,
        residualKernelSize: Int = 3,
        dilationBase: Int = 2,
        compress: Int = 2,
        trueSkip: Bool = true,
        causal: Bool = true,
        padMode: String = "constant"
    ) {
        let dim = mult * numFilters
        var blocks: [MimiSeanetResnetBlock] = []
        var dilation = 1
        for _ in 0..<numResidualLayers {
            blocks.append(MimiSeanetResnetBlock(
                dim: dim,
                kernelSizesAndDilations: [(residualKernelSize, dilation), (1, 1)],
                compress: compress,
                trueSkip: trueSkip,
                causal: causal,
                padMode: padMode
            ))
            dilation *= dilationBase
        }
        self._residuals.wrappedValue = blocks

        self._downsample.wrappedValue = MimiStreamableConv1d(
            inChannels: dim,
            outChannels: dim * 2,
            kernelSize: ratio * 2,
            stride: ratio,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: true,
            padMode: padMode
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var xs = x
        for r in residuals {
            xs = r(xs)
        }
        return downsample(elu(xs))
    }
}

// MARK: - MimiSeanetEncoder

/// Full SEANet encoder: init_conv -> encoder layers -> ELU -> final_conv.
///
/// Configured from Qwen3TTSTokenizerEncoderConfig:
/// - channels=1, numFilters=64, ratios=[8,6,5,4] (reversed for downsampling → [4,5,6,8])
/// - kernelSize=7, residualKernelSize=3, lastKernelSize=3
/// - compress=2, dilationBase=2, numResidualLayers=1
/// - hiddenSize=512 (output dimension)
public class MimiSeanetEncoder: Module {
    @ModuleInfo(key: "init_conv1d") var initConv1d: MimiStreamableConv1d
    @ModuleInfo(key: "layers") var layers: [MimiEncoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: MimiStreamableConv1d

    public init(config: Qwen3TTSTokenizerEncoderConfig) {
        var mult = 1
        let causal = config.useCausalConv
        let trueSkip = !config.useConvShortcut
        let padMode = "constant"

        self._initConv1d.wrappedValue = MimiStreamableConv1d(
            inChannels: config.audioChannels,
            outChannels: mult * config.numFilters,
            kernelSize: config.kernelSize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: causal,
            padMode: padMode
        )

        // Ratios are reversed for encoding (downsample)
        var encoderLayers: [MimiEncoderLayer] = []
        for ratio in config.upsamplingRatios.reversed() {
            encoderLayers.append(MimiEncoderLayer(
                ratio: ratio,
                mult: mult,
                numFilters: config.numFilters,
                numResidualLayers: config.numResidualLayers,
                residualKernelSize: config.residualKernelSize,
                dilationBase: config.dilationGrowthRate,
                compress: config.compress,
                trueSkip: trueSkip,
                causal: causal,
                padMode: padMode
            ))
            mult *= 2
        }
        self._layers.wrappedValue = encoderLayers

        self._finalConv1d.wrappedValue = MimiStreamableConv1d(
            inChannels: mult * config.numFilters,
            outChannels: config.hiddenSize,
            kernelSize: config.lastKernelSize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: causal,
            padMode: padMode
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var xs = initConv1d(x)
        for layer in layers {
            xs = layer(xs)
        }
        xs = elu(xs)
        return finalConv1d(xs)
    }
}

// MARK: - ELU Activation

/// ELU activation function: x if x > 0, else alpha * (exp(x) - 1)
func elu(_ x: MLXArray, alpha: Float = 1.0) -> MLXArray {
    return MLX.where(x .> 0, x, MLXArray(alpha) * (exp(x) - 1))
}
