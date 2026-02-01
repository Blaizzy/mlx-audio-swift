//
//  ConvNeXtBlock.swift
//  MLXAudio
//
//  ConvNeXt-style block for Speech Tokenizer Decoder.
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
//

import Foundation
import MLX
import MLXNN

// MARK: - ConvNeXtBlock

/// ConvNeXt-style block with depthwise convolution and inverted bottleneck.
///
/// Architecture:
/// 1. Depthwise conv (groups=channels) for spatial mixing
/// 2. LayerNorm (channel-last)
/// 3. Linear (channels -> 4*channels)
/// 4. GELU activation
/// 5. Linear (4*channels -> channels)
/// 6. LayerScale (learnable gamma)
/// 7. Residual connection
///
/// Input format: NCL (batch, channels, time)
/// Output format: NCL (batch, channels, time)
public class ConvNeXtBlock: Module, UnaryLayer {
    public let dim: Int

    @ModuleInfo(key: "dwconv") var dwconv: CausalConv1d
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear

    /// LayerScale parameter - initialized to small value for stable training
    var gamma: MLXArray

    /// Initialize ConvNeXtBlock.
    ///
    /// - Parameters:
    ///   - dim: Number of input/output channels
    ///   - layerScaleInit: Initial value for LayerScale gamma (default 1e-6)
    public init(dim: Int, layerScaleInit: Float = 1e-6) {
        self.dim = dim

        // Depthwise convolution (groups = channels for depthwise)
        self._dwconv.wrappedValue = CausalConv1d(
            inChannels: dim,
            outChannels: dim,
            kernelSize: 7,
            stride: 1,
            dilation: 1,
            groups: dim
        )

        // LayerNorm applied to channel dimension (last axis after transpose)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)

        // Pointwise convolutions (implemented as Linear for channel-last format)
        // Inverted bottleneck: expand to 4x, then project back
        self._pwconv1.wrappedValue = Linear(dim, dim * 4)
        self._pwconv2.wrappedValue = Linear(dim * 4, dim)

        // LayerScale: learnable per-channel scaling initialized to small value
        self.gamma = MLXArray.ones([dim]) * layerScaleInit
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)
        let residual = x

        // Depthwise convolution for spatial mixing
        var out = dwconv(x)

        // Transpose to channel-last for LayerNorm and Linear layers
        out = out.transposed(0, 2, 1)  // NCL -> NLC [batch, time, channels]

        // LayerNorm
        out = norm(out)

        // Inverted bottleneck MLP
        out = pwconv1(out)
        out = gelu(out)
        out = pwconv2(out)

        // LayerScale
        out = gamma * out

        // Transpose back to NCL
        out = out.transposed(0, 2, 1)  // NLC -> NCL [batch, channels, time]

        // Residual connection
        return residual + out
    }
}
