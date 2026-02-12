//
//  VoxtralRealtimeEncoder.swift
//  MLXAudioSTT
//

import Foundation
import MLX
import MLXNN

// MARK: - Interleaved (GPT-J) RoPE

/// Interleaved (GPT-J style) RoPE: rotates consecutive pairs (x[0],x[1]), (x[2],x[3]), ...
func interleavedRope(
    _ x: MLXArray, cos: MLXArray, sin: MLXArray, nHeads: Int, headDim: Int
) -> MLXArray {
    let seqLen = x.dim(0)
    let reshaped = x.reshaped(seqLen, nHeads, headDim)
    let x1 = reshaped[0..., 0..., .stride(from: 0, to: headDim, by: 2)]
    let x2 = reshaped[0..., 0..., .stride(from: 1, to: headDim, by: 2)]
    let cosExp = cos[0..., .newAxis, 0...]
    let sinExp = sin[0..., .newAxis, 0...]
    let o1 = x1 * cosExp - x2 * sinExp
    let o2 = x2 * cosExp + x1 * sinExp
    let stacked = MLX.stacked([o1, o2], axis: -1)
    return stacked.reshaped(seqLen, nHeads, headDim).reshaped(seqLen, nHeads * headDim)
}

func computeRopeFreqs(positions: MLXArray, headDim: Int, theta: Float) -> (MLXArray, MLXArray) {
    let freqs =
        1.0
        / MLX.pow(
            MLXArray(theta),
            MLX.arange(0, headDim, step: 2).asType(.float32) / MLXArray(Float(headDim)))
    let angles = positions[0..., .newAxis].asType(.float32) * freqs[.newAxis, 0...]
    return (MLX.cos(angles), MLX.sin(angles))
}

// MARK: - CausalConv1d

class CausalConv1d: Module {
    let padding: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1) {
        self.padding = kernelSize - stride
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var padded = x
        if padding > 0 {
            padded = MLX.padded(x, widths: [.init((0, 0)), .init((padding, 0)), .init((0, 0))])
        }
        return conv(padded)
    }
}

// MARK: - Encoder Attention

class VoxtralEncoderAttention: Module {
    let nHeads: Int
    let headDim: Int
    let slidingWindow: Int
    let ropeTheta: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ config: VoxtralEncoderConfig) {
        self.nHeads = config.nHeads
        self.headDim = config.headDim
        self.slidingWindow = config.slidingWindow
        self.ropeTheta = config.ropeTheta
        let attnDim = config.nHeads * config.headDim

        // Selective biases: wq, wv, wo have bias; wk does NOT
        self._wq.wrappedValue = Linear(config.dim, attnDim, bias: true)
        self._wk.wrappedValue = Linear(config.dim, attnDim, bias: false)
        self._wv.wrappedValue = Linear(config.dim, attnDim, bias: true)
        self._wo.wrappedValue = Linear(attnDim, config.dim, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray, ropeCos: MLXArray, ropeSin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let seqLen = x.dim(0)
        var q = wq(x)
        var k = wk(x)
        let v = wv(x)

        q = interleavedRope(q, cos: ropeCos, sin: ropeSin, nHeads: nHeads, headDim: headDim)
        k = interleavedRope(k, cos: ropeCos, sin: ropeSin, nHeads: nHeads, headDim: headDim)

        let qr = q.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        let kr = k.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        let vr = v.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)

        let scale = 1.0 / sqrt(Float(headDim))
        let attnOut = MLXFast.scaledDotProductAttention(
            queries: qr, keys: kr, values: vr, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(seqLen, nHeads * headDim)

        return wo(attnOut)
    }
}

// MARK: - Encoder Layer

class VoxtralEncoderLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralEncoderAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    // SwiGLU FFN: w1=gate (no bias), w3=up (no bias), w2=down (bias)
    @ModuleInfo(key: "feed_forward_w1") var feedForwardW1: Linear
    @ModuleInfo(key: "feed_forward_w3") var feedForwardW3: Linear
    @ModuleInfo(key: "feed_forward_w2") var feedForwardW2: Linear

    init(_ config: VoxtralEncoderConfig) {
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._attention.wrappedValue = VoxtralEncoderAttention(config)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._feedForwardW1.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW3.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW2.wrappedValue = Linear(config.hiddenDim, config.dim, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray, ropeCos: MLXArray, ropeSin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var h = attentionNorm(x)
        h = attention(h, ropeCos: ropeCos, ropeSin: ropeSin, mask: mask)
        var out = x + h

        h = ffnNorm(out)
        let gate = silu(feedForwardW1(h))
        let up = feedForwardW3(h)
        out = out + feedForwardW2(gate * up)

        return out
    }
}

// MARK: - Audio Encoder

public class VoxtralAudioEncoder: Module {
    let config: VoxtralEncoderConfig

    @ModuleInfo(key: "conv_layers_0_conv") var convLayers0: CausalConv1d
    @ModuleInfo(key: "conv_layers_1_conv") var convLayers1: CausalConv1d

    @ModuleInfo(key: "transformer_layers") var transformerLayers: [VoxtralEncoderLayer]
    @ModuleInfo(key: "transformer_norm") var transformerNorm: RMSNorm

    @ModuleInfo(key: "audio_language_projection_0") var audioLangProj0: Linear
    @ModuleInfo(key: "audio_language_projection_2") var audioLangProj2: Linear

    public init(_ config: VoxtralEncoderConfig) {
        self.config = config
        self._convLayers0.wrappedValue = CausalConv1d(
            inChannels: 128, outChannels: config.dim, kernelSize: 3, stride: 1)
        self._convLayers1.wrappedValue = CausalConv1d(
            inChannels: config.dim, outChannels: config.dim, kernelSize: 3, stride: 2)
        self._transformerLayers.wrappedValue = (0..<config.nLayers).map { _ in
            VoxtralEncoderLayer(config)
        }
        self._transformerNorm.wrappedValue = RMSNorm(
            dimensions: config.dim, eps: config.normEps)
        let adapterInputDim = config.dim * config.downsampleFactor
        let decoderDim = 3072
        self._audioLangProj0.wrappedValue = Linear(adapterInputDim, decoderDim, bias: false)
        self._audioLangProj2.wrappedValue = Linear(decoderDim, decoderDim, bias: false)
    }

    func convStem(_ mel: MLXArray) -> MLXArray {
        var x = mel.T.expandedDimensions(axis: 0)
        x = gelu(convLayers0(x))
        x = gelu(convLayers1(x))
        x = x.squeezed(axis: 0)  // [seq, dim]
        let trunc = x.dim(0) % config.downsampleFactor
        if trunc > 0 {
            x = x[trunc...]
        }
        return x
    }

    func downsampleAndProject(_ encoded: MLXArray) -> MLXArray {
        let seqLen = encoded.dim(0)
        let ds = config.downsampleFactor
        let dsLen = seqLen / ds
        if dsLen == 0 { return encoded[0..<0] }
        let x = encoded[0..<(dsLen * ds)].reshaped(dsLen, config.dim * ds)
        return audioLangProj2(gelu(audioLangProj0(x)))
    }

    func encodeFull(_ convOut: MLXArray) -> MLXArray {
        let seqLen = convOut.dim(0)
        let positions = MLX.arange(seqLen)
        let (ropeCos, ropeSin) = computeRopeFreqs(
            positions: positions, headDim: config.headDim, theta: config.ropeTheta)
        var x = convOut
        for layer in transformerLayers {
            x = layer(x, ropeCos: ropeCos, ropeSin: ropeSin, mask: .causal)
        }
        x = transformerNorm(x)
        return downsampleAndProject(x)
    }

    func encodeChunks(_ convOut: MLXArray) -> MLXArray {
        let seqLen = convOut.dim(0)
        let sw = config.slidingWindow

        var chunks: [MLXArray] = []
        for chunkStart in stride(from: 0, to: seqLen, by: sw) {
            let chunkEnd = min(chunkStart + sw, seqLen)
            var x = convOut[chunkStart..<chunkEnd]
            let positions = MLX.arange(chunkStart, chunkEnd)
            let (ropeCos, ropeSin) = computeRopeFreqs(
                positions: positions, headDim: config.headDim, theta: config.ropeTheta)
            for layer in transformerLayers {
                x = layer(x, ropeCos: ropeCos, ropeSin: ropeSin, mask: .causal)
            }
            let normed = transformerNorm(x)
            chunks.append(normed)
            // Force evaluation to bound graph size for long audio
            MLX.eval(normed)
        }
        return MLX.concatenated(chunks, axis: 0)
    }

    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        let convOut = convStem(mel)
        let seqLen = convOut.dim(0)

        if seqLen <= config.slidingWindow {
            return encodeFull(convOut)
        } else {
            let encoded = encodeChunks(convOut)
            return downsampleAndProject(encoded)
        }
    }
}
