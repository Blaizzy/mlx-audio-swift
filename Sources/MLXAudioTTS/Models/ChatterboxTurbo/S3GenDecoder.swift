//
//  S3GenDecoder.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN

private func s3Mish(_ x: MLXArray) -> MLXArray {
    x * MLX.tanh(MLX.log(MLX.exp(x) + MLXArray(Float(1.0))))
}

private func downsampleMask(_ mask: MLXArray, stride: Int) -> MLXArray {
    let length = mask.shape[2]
    let indices = Swift.stride(from: 0, to: length, by: stride).map { Int32($0) }
    let indexArray = MLXArray(indices)
    return mask[0..., 0..., indexArray]
}

private func applyMasked(_ module: Module, x: MLXArray, mask: MLXArray) -> MLXArray {
    if let block = module as? Block1D {
        return block(x, mask: mask)
    }
    if let block = module as? CausalBlock1D {
        return block(x, mask: mask)
    }
    fatalError("Unsupported masked module: \(type(of: module))")
}

private func applyUnary(_ module: Module, x: MLXArray) -> MLXArray {
    if let layer = module as? Conv1dPT {
        return layer(x)
    }
    if let layer = module as? ConvTranspose1dPT {
        return layer(x)
    }
    if let layer = module as? CausalConv1d {
        return layer(x)
    }
    if let layer = module as? Downsample1D {
        return layer(x)
    }
    if let layer = module as? Upsample1D {
        return layer(x)
    }
    fatalError("Unsupported unary module: \(type(of: module))")
}

final class Conv1dPT: Module {
    @ModuleInfo(key: "conv") private var conv: MLXNN.Conv1d

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1
    ) {
        self._conv.wrappedValue = MLXNN.Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        h = conv(h)
        return h.transposed(0, 2, 1)
    }
}

final class ConvTranspose1dPT: Module {
    @ModuleInfo(key: "conv") private var conv: ConvTransposed1d

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0
    ) {
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        h = conv(h)
        return h.transposed(0, 2, 1)
    }
}

func s3genSinusoidalPosEmb(_ timesteps: MLXArray, dim: Int, scale: Float = 1000) -> MLXArray {
    var t = timesteps
    if t.ndim == 0 {
        t = t.expandedDimensions(axis: 0)
    }

    let halfDim = dim / 2
    let emb = logf(10000) / Float(halfDim - 1)
    let expRange = MLXArray.arange(0.0, Double(halfDim), step: 1.0, dtype: .float32)
    let freqs = MLX.exp(expRange * MLXArray(-emb))
    let scaled = MLXArray(scale) * t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    return MLX.concatenated([MLX.sin(scaled), MLX.cos(scaled)], axis: -1)
}

final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") private var linear1: Linear
    @ModuleInfo(key: "linear_2") private var linear2: Linear

    init(inChannels: Int, timeEmbedDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linear1(x)
        h = silu(h)
        return linear2(h)
    }
}

final class CausalConv1d: Module {
    private let causalPadding: Int
    @ModuleInfo(key: "conv") private var conv: Conv1dPT

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1
    ) {
        self.causalPadding = (kernelSize - 1) * dilation
        self._conv.wrappedValue = Conv1dPT(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [.init(0), .init(0), .init((causalPadding, 0))])
        return conv(padded)
    }
}

final class Block1D: Module {
    @ModuleInfo(key: "conv") private var conv: Conv1dPT
    @ModuleInfo(key: "norm") private var norm: GroupNorm

    init(dim: Int, dimOut: Int, groups: Int = 8) {
        self._conv.wrappedValue = Conv1dPT(inChannels: dim, outChannels: dimOut, kernelSize: 3, padding: 1)
        self._norm.wrappedValue = GroupNorm(groupCount: groups, dimensions: dimOut, pytorchCompatible: true)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x * mask
        h = conv(h)
        h = h.transposed(0, 2, 1)
        h = norm(h)
        h = h.transposed(0, 2, 1)
        h = s3Mish(h)
        return h * mask
    }
}

final class CausalBlock1D: Module {
    private let conv: CausalConv1d
    private let norm: LayerNorm

    init(dim: Int, dimOut: Int, groups: Int = 8) {
        self.conv = CausalConv1d(inChannels: dim, outChannels: dimOut, kernelSize: 3)
        self.norm = LayerNorm(dimensions: dimOut)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x * mask
        h = conv(h)
        h = h.transposed(0, 2, 1)
        h = norm(h)
        h = h.transposed(0, 2, 1)
        h = s3Mish(h)
        return h * mask
    }
}

final class ResnetBlock1D: Module {
    @ModuleInfo(key: "mlp") private var mlp: [Linear]
    @ModuleInfo(key: "block1") private var block1: Module
    @ModuleInfo(key: "block2") private var block2: Module
    @ModuleInfo(key: "res_conv") private var resConv: Conv1dPT

    init(
        dim: Int,
        dimOut: Int,
        timeEmbedDim: Int,
        causal: Bool,
        groups: Int = 8
    ) {
        self._mlp.wrappedValue = [Linear(timeEmbedDim, dimOut)]
        if causal {
            self._block1.wrappedValue = CausalBlock1D(dim: dim, dimOut: dimOut, groups: groups)
            self._block2.wrappedValue = CausalBlock1D(dim: dimOut, dimOut: dimOut, groups: groups)
        } else {
            self._block1.wrappedValue = Block1D(dim: dim, dimOut: dimOut, groups: groups)
            self._block2.wrappedValue = Block1D(dim: dimOut, dimOut: dimOut, groups: groups)
        }
        self._resConv.wrappedValue = Conv1dPT(inChannels: dim, outChannels: dimOut, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
        var h = applyMasked(block1, x: x, mask: mask)
        h = h + mlp[0](s3Mish(timeEmb)).expandedDimensions(axis: 2)
        h = applyMasked(block2, x: h, mask: mask)
        return h + resConv(x * mask)
    }
}

final class Downsample1D: Module {
    @ModuleInfo(key: "conv") private var conv: Conv1dPT

    init(dim: Int) {
        self._conv.wrappedValue = Conv1dPT(inChannels: dim, outChannels: dim, kernelSize: 3, stride: 2, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

final class Upsample1D: Module {
    @ModuleInfo(key: "conv") private var conv: ConvTranspose1dPT

    init(dim: Int) {
        self._conv.wrappedValue = ConvTranspose1dPT(inChannels: dim, outChannels: dim, kernelSize: 4, stride: 2, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

final class SelfAttention1D: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "to_q") private var toQ: Linear
    @ModuleInfo(key: "to_k") private var toK: Linear
    @ModuleInfo(key: "to_v") private var toV: Linear
    @ModuleInfo(key: "to_out") private var toOut: [Linear]

    init(dim: Int, numHeads: Int = 8, headDim: Int = 64, dropout: Float = 0.0) {
        self.numHeads = numHeads
        self.headDim = headDim
        let innerDim = numHeads * headDim
        self.scale = pow(Float(headDim), -0.5)

        self._toQ.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toK.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toV.wrappedValue = Linear(dim, innerDim, bias: false)
        self._toOut.wrappedValue = [Linear(innerDim, dim)]
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        var q = toQ(x)
        var k = toK(x)
        var v = toV(x)

        q = q.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)

        var attn = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * MLXArray(scale)
        if let mask {
            let maskExpanded = mask.expandedDimensions(axis: 1).expandedDimensions(axis: 1)
            attn = MLX.where(maskExpanded .> MLXArray(Float(0)), attn, MLXArray(Float(-1e9)))
        }
        attn = MLX.softmax(attn, axis: -1)
        var out = MLX.matmul(attn, v)
        out = out.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return toOut[0](out)
    }
}

final class S3DecoderGELU: Module {
    @ModuleInfo(key: "proj") private var proj: Linear

    init(dimIn: Int, dimOut: Int) {
        self._proj.wrappedValue = Linear(dimIn, dimOut)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        gelu(proj(x))
    }
}

final class FeedForward: Module {
    @ModuleInfo(key: "net") private var net: [Module]

    init(dim: Int, mult: Int = 4) {
        let inner = dim * mult
        self._net.wrappedValue = [
            S3DecoderGELU(dimIn: dim, dimOut: inner),
            Linear(inner, dim)
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = (net[0] as! S3DecoderGELU)(x)
        h = (net[1] as! Linear)(h)
        return h
    }
}

final class S3GenTransformerBlock: Module {
    private let attn1: SelfAttention1D
    private let ff: FeedForward
    @ModuleInfo(key: "norm1") private var norm1: LayerNorm
    @ModuleInfo(key: "norm3") private var norm3: LayerNorm

    init(dim: Int, numHeads: Int = 8, headDim: Int = 64, ffMult: Int = 4) {
        self.attn1 = SelfAttention1D(dim: dim, numHeads: numHeads, headDim: headDim)
        self.ff = FeedForward(dim: dim, mult: ffMult)
        self._norm1.wrappedValue = LayerNorm(dimensions: dim)
        self._norm3.wrappedValue = LayerNorm(dimensions: dim)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x + attn1(norm1(x), mask: mask)
        h = h + ff(norm3(h))
        return h
    }
}

final class DownBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: ResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [S3GenTransformerBlock]
    @ModuleInfo(key: "downsample") var downsample: Module

    init(
        inputChannel: Int,
        outputChannel: Int,
        timeEmbedDim: Int,
        causal: Bool,
        nBlocks: Int,
        numHeads: Int,
        attentionHeadDim: Int,
        isLast: Bool
    ) {
        self._resnet.wrappedValue = ResnetBlock1D(
            dim: inputChannel,
            dimOut: outputChannel,
            timeEmbedDim: timeEmbedDim,
            causal: causal
        )
        self._transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in
            S3GenTransformerBlock(dim: outputChannel, numHeads: numHeads, headDim: attentionHeadDim)
        }
        if isLast {
            self._downsample.wrappedValue = causal
                ? CausalConv1d(inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3)
                : Conv1dPT(inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3, padding: 1)
        } else {
            self._downsample.wrappedValue = Downsample1D(dim: outputChannel)
        }
    }
}

final class MidBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: ResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [S3GenTransformerBlock]

    init(
        channels: Int,
        timeEmbedDim: Int,
        causal: Bool,
        nBlocks: Int,
        numHeads: Int,
        attentionHeadDim: Int
    ) {
        self._resnet.wrappedValue = ResnetBlock1D(
            dim: channels,
            dimOut: channels,
            timeEmbedDim: timeEmbedDim,
            causal: causal
        )
        self._transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in
            S3GenTransformerBlock(dim: channels, numHeads: numHeads, headDim: attentionHeadDim)
        }
    }
}

final class UpBlock: Module {
    @ModuleInfo(key: "resnet") var resnet: ResnetBlock1D
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [S3GenTransformerBlock]
    @ModuleInfo(key: "upsample") var upsample: Module

    init(
        inputChannel: Int,
        outputChannel: Int,
        timeEmbedDim: Int,
        causal: Bool,
        nBlocks: Int,
        numHeads: Int,
        attentionHeadDim: Int,
        isLast: Bool
    ) {
        self._resnet.wrappedValue = ResnetBlock1D(
            dim: inputChannel,
            dimOut: outputChannel,
            timeEmbedDim: timeEmbedDim,
            causal: causal
        )
        self._transformerBlocks.wrappedValue = (0..<nBlocks).map { _ in
            S3GenTransformerBlock(dim: outputChannel, numHeads: numHeads, headDim: attentionHeadDim)
        }
        if isLast {
            self._upsample.wrappedValue = causal
                ? CausalConv1d(inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3)
                : Conv1dPT(inChannels: outputChannel, outChannels: outputChannel, kernelSize: 3, padding: 1)
        } else {
            self._upsample.wrappedValue = Upsample1D(dim: outputChannel)
        }
    }
}

final class ConditionalDecoder: Module {
    let inChannels: Int
    let outChannels: Int
    let causal: Bool
    let meanflow: Bool

    @ModuleInfo(key: "time_mlp") private var timeMlp: TimestepEmbedding
    @ModuleInfo(key: "down_blocks") private var downBlocks: [DownBlock]
    @ModuleInfo(key: "mid_blocks") private var midBlocks: [MidBlock]
    @ModuleInfo(key: "up_blocks") private var upBlocks: [UpBlock]
    @ModuleInfo(key: "final_block") private var finalBlock: Module
    @ModuleInfo(key: "final_proj") private var finalProj: Conv1dPT
    @ModuleInfo(key: "time_embed_mixer") private var timeEmbedMixer: Linear?

    init(
        inChannels: Int = 320,
        outChannels: Int = 80,
        causal: Bool = true,
        channels: [Int] = [256],
        attentionHeadDim: Int = 64,
        nBlocks: Int = 4,
        numMidBlocks: Int = 12,
        numHeads: Int = 8,
        meanflow: Bool = false
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.causal = causal
        self.meanflow = meanflow

        let timeEmbedDim = channels[0] * 4
        self._timeMlp.wrappedValue = TimestepEmbedding(inChannels: inChannels, timeEmbedDim: timeEmbedDim)

        var outputChannel = inChannels
        var downBlocks: [DownBlock] = []
        for (index, ch) in channels.enumerated() {
            let inputChannel = outputChannel
            outputChannel = ch
            let isLast = index == channels.count - 1
            downBlocks.append(DownBlock(
                inputChannel: inputChannel,
                outputChannel: outputChannel,
                timeEmbedDim: timeEmbedDim,
                causal: causal,
                nBlocks: nBlocks,
                numHeads: numHeads,
                attentionHeadDim: attentionHeadDim,
                isLast: isLast
            ))
        }
        self._downBlocks.wrappedValue = downBlocks

        var midBlocks: [MidBlock] = []
        for _ in 0..<numMidBlocks {
            midBlocks.append(MidBlock(
                channels: channels.last ?? outputChannel,
                timeEmbedDim: timeEmbedDim,
                causal: causal,
                nBlocks: nBlocks,
                numHeads: numHeads,
                attentionHeadDim: attentionHeadDim
            ))
        }
        self._midBlocks.wrappedValue = midBlocks

        let channelsUp = channels.reversed() + [channels.first ?? outputChannel]
        var upBlocks: [UpBlock] = []
        for index in 0..<(channelsUp.count - 1) {
            let inputChannel = channelsUp[index] * 2
            let outputChannel = channelsUp[index + 1]
            let isLast = index == channelsUp.count - 2
            upBlocks.append(UpBlock(
                inputChannel: inputChannel,
                outputChannel: outputChannel,
                timeEmbedDim: timeEmbedDim,
                causal: causal,
                nBlocks: nBlocks,
                numHeads: numHeads,
                attentionHeadDim: attentionHeadDim,
                isLast: isLast
            ))
        }
        self._upBlocks.wrappedValue = upBlocks

        self._finalBlock.wrappedValue = causal
            ? CausalBlock1D(dim: channelsUp.last ?? outputChannel, dimOut: channelsUp.last ?? outputChannel)
            : Block1D(dim: channelsUp.last ?? outputChannel, dimOut: channelsUp.last ?? outputChannel)
        self._finalProj.wrappedValue = Conv1dPT(inChannels: channelsUp.last ?? outputChannel, outChannels: outChannels, kernelSize: 1)

        if meanflow {
            self._timeEmbedMixer.wrappedValue = Linear(timeEmbedDim * 2, timeEmbedDim, bias: false)
        } else {
            self._timeEmbedMixer.wrappedValue = nil
        }
    }

    func callAsFunction(
        x: MLXArray,
        mask: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil,
        r: MLXArray? = nil
    ) -> MLXArray {
        var timeEmb = s3genSinusoidalPosEmb(t, dim: inChannels)
        timeEmb = timeMlp(timeEmb)

        if meanflow, let r, let timeEmbedMixer {
            let rEmb = timeMlp(s3genSinusoidalPosEmb(r, dim: inChannels))
            let concat = MLX.concatenated([timeEmb, rEmb], axis: -1)
            timeEmb = timeEmbedMixer(concat)
        }

        var inputs: [MLXArray] = [x, mu]
        if let spks {
            let expanded = spks.expandedDimensions(axis: 2).broadcasted(to: [spks.dim(0), spks.dim(1), x.dim(2)])
            inputs.append(expanded)
        }
        if let cond {
            inputs.append(cond)
        }

        var h = MLX.concatenated(inputs, axis: 1)

        var hiddens: [MLXArray] = []
        var masks: [MLXArray] = [mask]

        for down in downBlocks {
            let maskDown = masks.last!
            h = down.resnet(h, mask: maskDown, timeEmb: timeEmb)

            var tState = h.transposed(0, 2, 1)
            let maskT = maskDown[0..., 0, 0...]
            for block in down.transformerBlocks {
                tState = block(tState, mask: maskT)
            }
            h = tState.transposed(0, 2, 1)

            hiddens.append(h)
            h = applyUnary(down.downsample, x: h * maskDown)
            masks.append(downsampleMask(maskDown, stride: 2))
        }

        masks.removeLast()
        let maskMid = masks.last ?? mask

        for mid in midBlocks {
            h = mid.resnet(h, mask: maskMid, timeEmb: timeEmb)
            var tState = h.transposed(0, 2, 1)
            let maskT = maskMid[0..., 0, 0...]
            for block in mid.transformerBlocks {
                tState = block(tState, mask: maskT)
            }
            h = tState.transposed(0, 2, 1)
        }

        var lastMaskUp: MLXArray = mask
        for up in upBlocks {
            let maskUp = masks.popLast() ?? mask
            lastMaskUp = maskUp
            let skip = hiddens.popLast() ?? h
            h = h[0..., 0..., 0..<skip.shape[2]]
            h = MLX.concatenated([h, skip], axis: 1)

            h = up.resnet(h, mask: maskUp, timeEmb: timeEmb)
            var tState = h.transposed(0, 2, 1)
            let maskT = maskUp[0..., 0, 0...]
            for block in up.transformerBlocks {
                tState = block(tState, mask: maskT)
            }
            h = tState.transposed(0, 2, 1)

            h = applyUnary(up.upsample, x: h * maskUp)
        }

        h = applyMasked(finalBlock, x: h, mask: lastMaskUp)
        return finalProj(h * lastMaskUp) * mask
    }
}
