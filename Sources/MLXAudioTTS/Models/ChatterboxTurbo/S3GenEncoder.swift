//
//  S3GenEncoder.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN

private func s3ReverseAxis0(_ x: MLXArray) -> MLXArray {
    let length = x.shape[0]
    let indices = (0..<length).map { Int32(length - 1 - $0) }
    return x[MLXArray(indices)]
}

final class EspnetRelPositionalEncoding: Module {
    let dModel: Int
    let xscale: Float
    private var pe: MLXArray?

    init(dModel: Int, dropout: Float = 0.1, maxLen: Int = 5000) {
        self.dModel = dModel
        self.xscale = pow(Float(dModel), 0.5)
        self.pe = nil
        super.init()
        extendPe(size: maxLen)
    }

    private func extendPe(size: Int) {
        if let pe, pe.shape[1] >= size * 2 - 1 {
            return
        }

        let position = MLXArray.arange(0.0, Double(size), step: 1.0, dtype: .float32)
            .expandedDimensions(axis: 1)
        let divTerm = MLX.exp(
            MLXArray.arange(0.0, Double(dModel), step: 2.0, dtype: .float32)
                * (MLXArray(Float(-log(10000.0))) / MLXArray(Float(dModel)))
        )

        let pePositiveSin = MLX.sin(position * divTerm)
        let pePositiveCos = MLX.cos(position * divTerm)
        let pePositive = MLX.concatenated(
            [pePositiveSin.expandedDimensions(axis: 2), pePositiveCos.expandedDimensions(axis: 2)],
            axis: -1
        ).reshaped([size, dModel])

        let peNegativeSin = MLX.sin(-position * divTerm)
        let peNegativeCos = MLX.cos(-position * divTerm)
        let peNegative = MLX.concatenated(
            [peNegativeSin.expandedDimensions(axis: 2), peNegativeCos.expandedDimensions(axis: 2)],
            axis: -1
        ).reshaped([size, dModel])

        let pePositiveFlipped = s3ReverseAxis0(pePositive)
        let peNegativeTail = peNegative[1..., 0...]
        let peConcat = MLX.concatenated([pePositiveFlipped, peNegativeTail], axis: 0)
        self.pe = peConcat.expandedDimensions(axis: 0)
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let t = x.shape[1]
        extendPe(size: t)

        let scaled = x * MLXArray(xscale)

        guard let pe else {
            return (scaled, MLXArray.zeros([1, 2 * t - 1, dModel], type: Float.self))
        }

        let center = pe.shape[1] / 2
        let start = center - t + 1
        let end = center + t
        let posEmb = pe[0..., start..<end, 0...]
        return (scaled, posEmb)
    }
}

final class LinearInput: Module {
    @ModuleInfo(key: "linear") private var linear: Linear
    @ModuleInfo(key: "norm") private var norm: LayerNorm
    @ModuleInfo(key: "pos_enc") private var posEnc: EspnetRelPositionalEncoding

    init(inputSize: Int, outputSize: Int, dropout: Float = 0.1) {
        self._linear.wrappedValue = Linear(inputSize, outputSize)
        self._norm.wrappedValue = LayerNorm(dimensions: outputSize, eps: 1e-5)
        self._posEnc.wrappedValue = EspnetRelPositionalEncoding(dModel: outputSize, dropout: dropout)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        var h = linear(x)
        h = norm(h)
        let (scaled, posEmb) = posEnc(h)
        return (scaled, posEmb, mask)
    }
}

final class RelPositionMultiHeadedAttention: Module {
    let nHead: Int
    let dK: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") private var linearQ: Linear
    @ModuleInfo(key: "linear_k") private var linearK: Linear
    @ModuleInfo(key: "linear_v") private var linearV: Linear
    @ModuleInfo(key: "linear_out") private var linearOut: Linear
    @ModuleInfo(key: "linear_pos") private var linearPos: Linear
    @ModuleInfo(key: "pos_bias_u") private var posBiasU: MLXArray
    @ModuleInfo(key: "pos_bias_v") private var posBiasV: MLXArray

    init(nHead: Int, nFeat: Int, dropoutRate: Float = 0.0, keyBias: Bool = true) {
        self.nHead = nHead
        self.dK = nFeat / nHead
        self.scale = pow(Float(dK), -0.5)

        self._linearQ.wrappedValue = Linear(nFeat, nFeat)
        self._linearK.wrappedValue = Linear(nFeat, nFeat, bias: keyBias)
        self._linearV.wrappedValue = Linear(nFeat, nFeat)
        self._linearOut.wrappedValue = Linear(nFeat, nFeat)
        self._linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._posBiasU.wrappedValue = MLXArray.zeros([nHead, dK], type: Float.self)
        self._posBiasV.wrappedValue = MLXArray.zeros([nHead, dK], type: Float.self)
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let heads = x.shape[1]
        let t1 = x.shape[2]
        let t2 = x.shape[3]

        let zeroPad = MLXArray.zeros([batch, heads, t1, 1], type: Float.self)
        var xPadded = MLX.concatenated([zeroPad, x], axis: -1)
        xPadded = xPadded.reshaped([batch, heads, t2 + 1, t1])
        xPadded = xPadded[0..., 0..., 1..., 0...]
        var shifted = xPadded.reshaped([batch, heads, t1, t2])
        let keep = t2 / 2 + 1
        shifted = shifted[0..., 0..., 0..., 0..<keep]
        return shifted
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray? = nil,
        posEmb: MLXArray? = nil
    ) -> MLXArray {
        let batch = x.shape[0]
        let time = x.shape[1]
        let dim = x.shape[2]

        let q = linearQ(x).reshaped([batch, time, nHead, dK])
        let k = linearK(x).reshaped([batch, time, nHead, dK]).transposed(0, 2, 1, 3)
        let v = linearV(x).reshaped([batch, time, nHead, dK]).transposed(0, 2, 1, 3)

        let qWithBiasU = (q + posBiasU).transposed(0, 2, 1, 3)
        let matrixAC = MLX.matmul(qWithBiasU, k.transposed(0, 1, 3, 2))

        var scores = matrixAC * MLXArray(scale)

        if let posEmb {
            let tPos = posEmb.shape[1]
            let p = linearPos(posEmb)
                .reshaped([1, tPos, nHead, dK])
                .transposed(0, 2, 1, 3)

            let qWithBiasV = (q + posBiasV).transposed(0, 2, 1, 3)
            var matrixBD = MLX.matmul(qWithBiasV, p.transposed(0, 1, 3, 2))
            if matrixAC.shape != matrixBD.shape {
                matrixBD = relShift(matrixBD)
            }
            scores = (matrixAC + matrixBD) * MLXArray(scale)
        }

        if let mask {
            let maskExpanded: MLXArray
            if mask.ndim == 2 {
                maskExpanded = mask.expandedDimensions(axis: 1).expandedDimensions(axis: 1)
            } else {
                maskExpanded = mask.expandedDimensions(axis: 1)
            }
            scores = MLX.where(maskExpanded .> MLXArray(Float(0.0)), scores, MLXArray(Float(-1e9)))
        }

        let attn = MLX.softmax(scores, axis: -1)
        let out = MLX.matmul(attn, v).transposed(0, 2, 1, 3).reshaped([batch, time, dim])
        return linearOut(out)
    }
}

final class PositionwiseFeedForward: Module {
    @ModuleInfo(key: "w_1") private var w1: Linear
    @ModuleInfo(key: "w_2") private var w2: Linear

    init(dModel: Int, dInner: Int, dropout: Float = 0.1) {
        self._w1.wrappedValue = Linear(dModel, dInner)
        self._w2.wrappedValue = Linear(dInner, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)))
    }
}

final class ConformerEncoderLayer: Module {
    @ModuleInfo(key: "norm_mha") private var normMha: LayerNorm
    @ModuleInfo(key: "self_attn") private var selfAttn: RelPositionMultiHeadedAttention
    @ModuleInfo(key: "norm_ff") private var normFf: LayerNorm
    @ModuleInfo(key: "feed_forward") private var feedForward: PositionwiseFeedForward
    let size: Int

    init(size: Int, nHead: Int, dInner: Int, dropoutRate: Float = 0.1, keyBias: Bool = true) {
        self.size = size
        self._normMha.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
        self._selfAttn.wrappedValue = RelPositionMultiHeadedAttention(
            nHead: nHead,
            nFeat: size,
            dropoutRate: dropoutRate,
            keyBias: keyBias
        )
        self._normFf.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
        self._feedForward.wrappedValue = PositionwiseFeedForward(dModel: size, dInner: dInner, dropout: dropoutRate)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, posEmb: MLXArray? = nil) -> MLXArray {
        let residualAttn = x
        let attnOut = selfAttn(normMha(x), mask: mask, posEmb: posEmb)
        let afterAttn = residualAttn + attnOut

        let residualFf = afterAttn
        let ffOut = feedForward(normFf(afterAttn))
        return residualFf + ffOut
    }
}

final class PreLookaheadLayer: Module {
    let preLookaheadLen: Int
    @ModuleInfo(key: "conv1") private var conv1: Conv1d
    @ModuleInfo(key: "conv2") private var conv2: Conv1d

    init(channels: Int, preLookaheadLen: Int = 3) {
        self.preLookaheadLen = preLookaheadLen
        self._conv1.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: preLookaheadLen + 1,
            stride: 1,
            padding: 0
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            stride: 1,
            padding: 0
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = MLX.padded(x, widths: [.init(0), .init((0, preLookaheadLen)), .init(0)])
        out = leakyRelu(conv1(out), negativeSlope: 0.1)
        out = MLX.padded(out, widths: [.init(0), .init((2, 0)), .init(0)])
        out = conv2(out)
        return out + x
    }
}

final class Upsample1DEncoder: Module {
    let stride: Int
    @ModuleInfo(key: "conv") private var conv: Conv1d

    init(channels: Int, stride: Int = 2) {
        self.stride = stride
        self._conv.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: stride * 2 + 1,
            stride: 1,
            padding: 0
        )
    }

    private func repeatTime(_ x: MLXArray, count: Int) -> MLXArray {
        let batch = x.shape[0]
        let time = x.shape[1]
        let channels = x.shape[2]
        let expanded = x.expandedDimensions(axis: 2)
        let broadcasted = expanded.broadcasted(to: [batch, time, count, channels])
        return broadcasted.reshaped([batch, time * count, channels])
    }

    func callAsFunction(_ x: MLXArray, xLens: MLXArray) -> (MLXArray, MLXArray) {
        var h = repeatTime(x, count: stride)
        h = MLX.padded(h, widths: [.init(0), .init((stride * 2, 0)), .init(0)])
        h = conv(h)
        return (h, xLens * MLXArray(Int32(stride)))
    }
}

final class UpsampleConformerEncoder: Module {
    let outputSize: Int

    @ModuleInfo(key: "embed") private var embed: LinearInput
    @ModuleInfo(key: "pre_lookahead_layer") private var preLookaheadLayer: PreLookaheadLayer
    @ModuleInfo(key: "encoders") private var encoders: [ConformerEncoderLayer]
    @ModuleInfo(key: "up_layer") private var upLayer: Upsample1DEncoder
    @ModuleInfo(key: "up_embed") private var upEmbed: LinearInput
    @ModuleInfo(key: "up_encoders") private var upEncoders: [ConformerEncoderLayer]
    @ModuleInfo(key: "after_norm") private var afterNorm: LayerNorm

    init(
        inputSize: Int = 512,
        outputSize: Int = 512,
        attentionHeads: Int = 8,
        linearUnits: Int = 2048,
        numBlocks: Int = 6,
        dropoutRate: Float = 0.1
    ) {
        self.outputSize = outputSize

        self._embed.wrappedValue = LinearInput(inputSize: inputSize, outputSize: outputSize, dropout: dropoutRate)
        self._preLookaheadLayer.wrappedValue = PreLookaheadLayer(channels: outputSize, preLookaheadLen: 3)
        self._encoders.wrappedValue = (0..<numBlocks).map { _ in
            ConformerEncoderLayer(
                size: outputSize,
                nHead: attentionHeads,
                dInner: linearUnits,
                dropoutRate: dropoutRate
            )
        }
        self._upLayer.wrappedValue = Upsample1DEncoder(channels: outputSize, stride: 2)
        self._upEmbed.wrappedValue = LinearInput(inputSize: inputSize, outputSize: outputSize, dropout: dropoutRate)
        self._upEncoders.wrappedValue = (0..<4).map { _ in
            ConformerEncoderLayer(
                size: outputSize,
                nHead: attentionHeads,
                dInner: linearUnits,
                dropoutRate: dropoutRate
            )
        }
        self._afterNorm.wrappedValue = LayerNorm(dimensions: outputSize)
    }

    func callAsFunction(_ xs: MLXArray, xsLens: MLXArray) -> (MLXArray, MLXArray) {
        let batch = xs.shape[0]
        let time = xs.shape[1]

        let seqRange = MLXArray.arange(0, time, dtype: .int32)
        let seqRangeExpand = seqRange.expandedDimensions(axis: 0).broadcasted(to: [batch, time])
        let lensExpand = xsLens.expandedDimensions(axis: -1)
        var mask = seqRangeExpand .< lensExpand
        mask = mask.expandedDimensions(axis: 1)

        var (h, posEmb, _) = embed(xs, mask: mask)
        h = preLookaheadLayer(h)

        let mask1d = mask[0..., 0, 0...]
        for layer in encoders {
            h = layer(h, mask: mask1d, posEmb: posEmb)
        }

        let (up, upLens) = upLayer(h, xLens: xsLens)
        h = up

        let time2 = h.shape[1]
        let seqRange2 = MLXArray.arange(0, time2, dtype: .int32)
        let seqRangeExpand2 = seqRange2.expandedDimensions(axis: 0).broadcasted(to: [batch, time2])
        var mask2 = seqRangeExpand2 .< upLens.expandedDimensions(axis: -1)
        mask2 = mask2.expandedDimensions(axis: 1)

        (h, posEmb, _) = upEmbed(h, mask: mask2)
        let mask2d = mask2[0..., 0, 0...]
        for layer in upEncoders {
            h = layer(h, mask: mask2d, posEmb: posEmb)
        }

        h = afterNorm(h)
        return (h, mask2)
    }
}
