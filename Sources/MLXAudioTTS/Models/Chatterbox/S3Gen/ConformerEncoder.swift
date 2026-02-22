// Copyright © 2025 Anthropic. All rights reserved.
// Ported from Python mlx-audio chatterbox s3gen/transformer/

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Positional Encodings

/// Sinusoidal positional encoding base class.
class S3GenPositionalEncoding: Module {
    let dModel: Int
    let xscale: Float
    let dropoutRate: Float
    let maxLen: Int
    var pe: MLXArray

    init(dModel: Int, dropoutRate: Float, maxLen: Int = 5000) {
        self.dModel = dModel
        self.xscale = sqrt(Float(dModel))
        self.dropoutRate = dropoutRate
        self.maxLen = maxLen
        self.pe = Self.createPE(maxLen: maxLen, dModel: dModel)
    }

    static func createPE(maxLen: Int, dModel: Int) -> MLXArray {
        let position = MLXArray(0 ..< maxLen).asType(.float32).expandedDimensions(axis: 1)
        let divTerm = MLX.exp(
            MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) })
                * Float(-log(10000.0) / Float(dModel))
        )
        let peSin = MLX.sin(position * divTerm)
        let peCos = MLX.cos(position * divTerm)

        // Interleave sin and cos
        // Stack along last dim then reshape
        let stacked = MLX.stacked([peSin, peCos], axis: -1) // (maxLen, dModel/2, 2)
        let pe = stacked.reshaped(maxLen, dModel)
        return pe.expandedDimensions(axis: 0) // (1, maxLen, dModel)
    }

    func positionEncoding(offset: Int, size: Int) -> MLXArray {
        return pe[0..., offset ..< (offset + size), 0...]
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let posEmb = positionEncoding(offset: offset, size: x.dim(1))
        let out = x * xscale + posEmb
        return (out, posEmb)
    }
}

/// Relative positional encoding - does NOT add pos_emb to input.
class S3GenRelPositionalEncoding: S3GenPositionalEncoding {
    override func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let out = x * xscale
        let posEmb = positionEncoding(offset: offset, size: x.dim(1))
        return (out, posEmb)
    }
}

/// ESPnet-style relative positional encoding with bidirectional positions.
class S3GenEspnetRelPositionalEncoding: Module {
    let dModel: Int
    let xscale: Float
    let dropoutRate: Float
    let maxLen: Int
    var pe: MLXArray

    init(dModel: Int, dropoutRate: Float, maxLen: Int = 5000) {
        self.dModel = dModel
        self.xscale = sqrt(Float(dModel))
        self.dropoutRate = dropoutRate
        self.maxLen = maxLen
        self.pe = Self.createRelPE(size: maxLen, dModel: dModel)
    }

    static func createRelPE(size: Int, dModel: Int) -> MLXArray {
        let position = MLXArray(0 ..< size).asType(.float32).expandedDimensions(axis: 1)
        let divTerm = MLX.exp(
            MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) })
                * Float(-log(10000.0) / Float(dModel))
        )

        // Positive positions
        let posSin = MLX.sin(position * divTerm)
        let posCos = MLX.cos(position * divTerm)
        let posStacked = MLX.stacked([posSin, posCos], axis: -1).reshaped(size, dModel)

        // Negative positions
        let negSin = MLX.sin(-1.0 * position * divTerm)
        let negCos = MLX.cos(-1.0 * position * divTerm)
        let negStacked = MLX.stacked([negSin, negCos], axis: -1).reshaped(size, dModel)

        // Reverse positive, skip first of negative, concatenate
        let reversedIndices = MLXArray(Array((0..<size).reversed().map { Int32($0) }))
        let posReversed = posStacked.take(reversedIndices, axis: 0)
            .expandedDimensions(axis: 0)
        let negSkipped = negStacked[1...].expandedDimensions(axis: 0)

        return MLX.concatenated([posReversed, negSkipped], axis: 1)
    }

    func positionEncoding(size: Int, offset: Int = 0) -> MLXArray {
        let center = pe.dim(1) / 2
        let start = center - size + 1
        let end = center + size
        return pe[0..., start ..< end, 0...]
    }

    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
        let out = x * xscale
        let posEmb = positionEncoding(size: x.dim(1), offset: offset)
        return (out, posEmb)
    }
}

// MARK: - Subsampling

/// Linear transform without subsampling - used as input embedding.
class S3GenLinearNoSubsampling: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    let dropoutRate: Float
    let posEnc: Module  // One of the positional encoding types

    init(idim: Int, odim: Int, dropoutRate: Float, posEnc: Module) {
        self._linear.wrappedValue = Linear(idim, odim)
        self._norm.wrappedValue = LayerNorm(dimensions: odim, eps: 1e-5)
        self.dropoutRate = dropoutRate
        self.posEnc = posEnc
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray, offset: Int = 0
    ) -> (MLXArray, MLXArray, MLXArray) {
        var out = linear(x)
        out = norm(out)

        // Apply positional encoding based on type
        let posEmb: MLXArray
        if let espnet = posEnc as? S3GenEspnetRelPositionalEncoding {
            (out, posEmb) = espnet(out, offset: offset)
        } else if let rel = posEnc as? S3GenRelPositionalEncoding {
            (out, posEmb) = rel(out, offset: offset)
        } else if let base = posEnc as? S3GenPositionalEncoding {
            (out, posEmb) = base(out, offset: offset)
        } else {
            posEmb = MLX.zeros([1, x.dim(1), x.dim(2)])
        }

        return (out, posEmb, mask)
    }
}

// MARK: - Feed Forward

/// Positionwise feed forward layer.
class S3GenPositionwiseFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear
    let activation: (MLXArray) -> MLXArray
    let dropoutRate: Float

    init(idim: Int, hiddenUnits: Int, dropoutRate: Float, activation: @escaping (MLXArray) -> MLXArray) {
        self._w1.wrappedValue = Linear(idim, hiddenUnits)
        self._w2.wrappedValue = Linear(hiddenUnits, idim)
        self.activation = activation
        self.dropoutRate = dropoutRate
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return w2(activation(w1(x)))
    }
}

// MARK: - Multi-Head Attention

/// Multi-head attention layer.
class S3GenMultiHeadedAttention: Module {
    let dK: Int
    let h: Int
    let dropoutRate: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(nHead: Int, nFeat: Int, dropoutRate: Float, keyBias: Bool = true) {
        self.dK = nFeat / nHead
        self.h = nHead
        self.dropoutRate = dropoutRate

        self._linearQ.wrappedValue = Linear(nFeat, nFeat)
        self._linearK.wrappedValue = Linear(nFeat, nFeat, bias: keyBias)
        self._linearV.wrappedValue = Linear(nFeat, nFeat)
        self._linearOut.wrappedValue = Linear(nFeat, nFeat)
    }

    func forwardQKV(query: MLXArray, key: MLXArray, value: MLXArray)
        -> (MLXArray, MLXArray, MLXArray)
    {
        let B = query.dim(0)
        var q = linearQ(query).reshaped(B, -1, h, dK)
        var k = linearK(key).reshaped(B, -1, h, dK)
        var v = linearV(value).reshaped(B, -1, h, dK)

        q = q.transposed(0, 2, 1, 3) // (B, h, T1, dK)
        k = k.transposed(0, 2, 1, 3) // (B, h, T2, dK)
        v = v.transposed(0, 2, 1, 3) // (B, h, T2, dK)

        return (q, k, v)
    }

    func forwardAttention(value: MLXArray, scores: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = value.dim(0)
        var attn: MLXArray

        if let mask = mask, mask.dim(2) > 0 {
            let expandedMask = mask.expandedDimensions(axis: 1)
            let truncatedMask = expandedMask[0..., 0..., 0..., ..<scores.dim(-1)]
            let maskedScores = MLX.where(truncatedMask .== 0, MLXArray(-1e9), scores)
            attn = softmax(maskedScores, axis: -1)
            attn = MLX.where(truncatedMask .== 0, MLXArray(Float(0)), attn)
        } else {
            attn = softmax(scores, axis: -1)
        }

        var x = matmul(attn, value)  // (B, h, T1, dK)
        x = x.transposed(0, 2, 1, 3) // (B, T1, h, dK)
        x = x.reshaped(B, -1, h * dK)

        return linearOut(x)
    }

    func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray,
        mask: MLXArray? = nil, posEmb: MLXArray? = nil, cache: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        var (q, k, v) = forwardQKV(query: query, key: key, value: value)

        if let cache = cache, cache.dim(0) > 0 {
            let parts = cache.split(parts: 2, axis: -1)
            k = MLX.concatenated([parts[0], k], axis: 2)
            v = MLX.concatenated([parts[1], v], axis: 2)
        }

        let newCache = MLX.concatenated([k, v], axis: -1)
        let scale = Float(1.0 / sqrt(Float(dK)))
        let scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale

        return (forwardAttention(value: v, scores: scores, mask: mask), newCache)
    }
}

/// Multi-head attention with relative positional encoding.
class S3GenRelPositionMultiHeadedAttention: S3GenMultiHeadedAttention {
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    override init(nHead: Int, nFeat: Int, dropoutRate: Float, keyBias: Bool = true) {
        let dK = nFeat / nHead
        let scale = Float(sqrt(6.0 / Float(nHead + dK)))
        self._linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._posBiasU.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [nHead, dK])
        self._posBiasV.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [nHead, dK])
        super.init(nHead: nHead, nFeat: nFeat, dropoutRate: dropoutRate, keyBias: keyBias)
    }

    func relShift(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), heads = x.dim(1), T1 = x.dim(2), T2 = x.dim(3)
        let zeroPad = MLXArray.zeros([B, heads, T1, 1])
        let xPadded = MLX.concatenated([zeroPad, x], axis: -1)
        let reshaped = xPadded.reshaped(B, heads, T2 + 1, T1)
        let shifted = reshaped[0..., 0..., 1..., 0...].reshaped(B, heads, T1, T2)
        return shifted[0..., 0..., 0..., ..<(T2 / 2 + 1)]
    }

    override func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray,
        mask: MLXArray? = nil, posEmb: MLXArray? = nil, cache: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        var (q, k, v) = forwardQKV(query: query, key: key, value: value)
        q = q.transposed(0, 2, 1, 3) // (B, T1, h, dK)

        if let cache = cache, cache.dim(0) > 0 {
            let parts = cache.split(parts: 2, axis: -1)
            k = MLX.concatenated([parts[0], k], axis: 2)
            v = MLX.concatenated([parts[1], v], axis: 2)
        }

        let newCache = MLX.concatenated([k, v], axis: -1)

        guard let posEmb = posEmb else {
            fatalError("pos_emb required for RelPositionMultiHeadedAttention")
        }

        let nBatchPos = posEmb.dim(0)
        var p = linearPos(posEmb).reshaped(nBatchPos, -1, h, dK)
        p = p.transposed(0, 2, 1, 3) // (B, h, T, dK)

        let qWithBiasU = (q + posBiasU).transposed(0, 2, 1, 3) // (B, h, T1, dK)
        let qWithBiasV = (q + posBiasV).transposed(0, 2, 1, 3)

        let matrixAC = matmul(qWithBiasU, k.transposed(0, 1, 3, 2))
        var matrixBD = matmul(qWithBiasV, p.transposed(0, 1, 3, 2))

        if matrixAC.shape != matrixBD.shape {
            matrixBD = relShift(matrixBD)
        }

        let scale = Float(1.0 / sqrt(Float(dK)))
        let scores = (matrixAC + matrixBD) * scale

        return (forwardAttention(value: v, scores: scores, mask: mask), newCache)
    }
}

// MARK: - Convolution Module

/// Convolution module for Conformer.
class S3GenConvolutionModule: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d
    let channels: Int
    let kernelSize: Int
    let lorder: Int
    let useLayerNorm: Bool
    let activation: (MLXArray) -> MLXArray

    // Norm stored as either type
    @ModuleInfo(key: "norm") var normModule: Module

    init(
        channels: Int, kernelSize: Int = 15,
        activation: @escaping (MLXArray) -> MLXArray = { silu($0) },
        norm: String = "batch_norm", causal: Bool = false, bias: Bool = true
    ) {
        self.channels = channels
        self.kernelSize = kernelSize
        self.activation = activation

        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: 2 * channels,
            kernelSize: 1, stride: 1, padding: 0, bias: bias)

        if causal {
            self.lorder = kernelSize - 1
            self._depthwiseConv.wrappedValue = Conv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, stride: 1, padding: 0,
                groups: channels, bias: bias)
        } else {
            self.lorder = 0
            let padding = (kernelSize - 1) / 2
            self._depthwiseConv.wrappedValue = Conv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, stride: 1, padding: padding,
                groups: channels, bias: bias)
        }

        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 1, stride: 1, padding: 0, bias: bias)

        if norm == "batch_norm" {
            self.useLayerNorm = false
            self._normModule.wrappedValue = BatchNorm(featureCount: channels)
        } else {
            self.useLayerNorm = true
            self._normModule.wrappedValue = LayerNorm(dimensions: channels)
        }
    }

    func callAsFunction(_ x: MLXArray, maskPad: MLXArray? = nil, cache: MLXArray? = nil)
        -> (MLXArray, MLXArray)
    {
        // x: (B, T, C), need (B, C, T) for processing
        var h = x.transposed(0, 2, 1)

        if let maskPad = maskPad, maskPad.dim(2) > 0 {
            h = MLX.where(maskPad, h, MLXArray(Float(0)))
        }

        // Causal padding
        let newCache: MLXArray
        if lorder > 0 {
            if cache == nil || cache!.dim(2) == 0 {
                h = MLX.padded(h, widths: [.init(0), .init(0), .init((lorder, 0))])
            } else {
                h = MLX.concatenated([cache!, h], axis: 2)
            }
            newCache = h[0..., 0..., (-lorder)...]
        } else {
            newCache = MLXArray.zeros([0, 0, 0])
        }

        // GLU: pointwise expansion
        // (B, C, T) -> swap to (B, T, C) for Conv1d
        h = h.transposed(0, 2, 1)
        h = pointwiseConv1(h) // (B, T, 2C)
        h = h.transposed(0, 2, 1) // (B, 2C, T)

        // GLU split
        let halfC = channels
        let h1 = h[0..., ..<halfC, 0...]
        let h2 = h[0..., halfC..., 0...]
        h = h1 * sigmoid(h2) // (B, C, T)

        // Depthwise conv: swap to (B, T, C)
        h = h.transposed(0, 2, 1)
        h = depthwiseConv(h)
        h = h.transposed(0, 2, 1) // (B, C, T)

        // Normalization - cast to concrete type since normModule is typed as Module
        if useLayerNorm {
            h = h.transposed(0, 2, 1) // (B, T, C)
            if let ln = normModule as? LayerNorm {
                h = ln(h)
            }
            h = activation(h)
            h = h.transposed(0, 2, 1) // (B, C, T)
        } else {
            if let bn = normModule as? BatchNorm {
                h = bn(h)
            }
            h = activation(h)
        }

        // Pointwise compression
        h = h.transposed(0, 2, 1)
        h = pointwiseConv2(h)
        h = h.transposed(0, 2, 1) // (B, C, T)

        if let maskPad = maskPad, maskPad.dim(2) > 0 {
            h = MLX.where(maskPad, h, MLXArray(Float(0)))
        }

        return (h.transposed(0, 2, 1), newCache) // (B, T, C)
    }
}

// MARK: - Conformer Encoder Layer

/// Conformer encoder layer combining self-attention, convolution, and feed-forward.
class S3GenConformerEncoderLayer: Module {
    let size: Int
    let normalizeBefore: Bool
    let dropoutRate: Float
    let ffScale: Float

    @ModuleInfo(key: "self_attn") var selfAttn: Module // MultiHeadedAttention or RelPosition variant
    @ModuleInfo(key: "feed_forward") var feedForward: S3GenPositionwiseFeedForward
    @ModuleInfo(key: "norm_ff") var normFF: LayerNorm
    @ModuleInfo(key: "norm_mha") var normMHA: LayerNorm

    // Optional modules
    var feedForwardMacaron: S3GenPositionwiseFeedForward?
    var normFFMacaron: LayerNorm?
    var convModule: S3GenConvolutionModule?
    var normConv: LayerNorm?
    var normFinal: LayerNorm?

    init(
        size: Int,
        selfAttn: Module,
        feedForward: S3GenPositionwiseFeedForward,
        feedForwardMacaron: S3GenPositionwiseFeedForward? = nil,
        convModule: S3GenConvolutionModule? = nil,
        dropoutRate: Float = 0.1,
        normalizeBefore: Bool = true
    ) {
        self.size = size
        self.normalizeBefore = normalizeBefore
        self.dropoutRate = dropoutRate

        self._selfAttn.wrappedValue = selfAttn
        self._feedForward.wrappedValue = feedForward
        self._normFF.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
        self._normMHA.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)

        if let ffm = feedForwardMacaron {
            self.feedForwardMacaron = ffm
            self.normFFMacaron = LayerNorm(dimensions: size, eps: 1e-12)
            self.ffScale = 0.5
        } else {
            self.ffScale = 1.0
        }

        if let conv = convModule {
            self.convModule = conv
            self.normConv = LayerNorm(dimensions: size, eps: 1e-12)
            self.normFinal = LayerNorm(dimensions: size, eps: 1e-12)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray, posEmb: MLXArray, maskPad: MLXArray? = nil,
        attCache: MLXArray? = nil, cnnCache: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        var x = x

        // Macaron-style feed-forward (optional, half-step)
        if let ffm = feedForwardMacaron, let normM = normFFMacaron {
            let residual = x
            if normalizeBefore { x = normM(x) }
            let ffOut = ffm(x)
            x = residual + ffScale * ffOut
            if !normalizeBefore { x = normM(x) }
        }

        // Multi-headed self-attention
        let residualMHA = x
        if normalizeBefore { x = normMHA(x) }

        let xAtt: MLXArray
        let newAttCache: MLXArray
        if let relAttn = selfAttn as? S3GenRelPositionMultiHeadedAttention {
            (xAtt, newAttCache) = relAttn(
                query: x, key: x, value: x,
                mask: mask, posEmb: posEmb, cache: attCache)
        } else if let mha = selfAttn as? S3GenMultiHeadedAttention {
            (xAtt, newAttCache) = mha(
                query: x, key: x, value: x,
                mask: mask, posEmb: posEmb, cache: attCache)
        } else {
            fatalError("Unknown attention type")
        }

        x = residualMHA + xAtt
        if !normalizeBefore { x = normMHA(x) }

        // Convolution module (optional)
        var newCnnCache = MLXArray.zeros([0, 0, 0])
        if let conv = convModule, let normC = normConv, let normF = normFinal {
            let residualConv = x
            if normalizeBefore { x = normC(x) }
            let (convOut, cnnCacheOut) = conv(x, maskPad: maskPad, cache: cnnCache)
            newCnnCache = cnnCacheOut
            x = residualConv + convOut
            if !normalizeBefore { x = normC(x) }
            // Feed-forward
            let residualFF = x
            if normalizeBefore { x = normFF(x) }
            let ffOut = feedForward(x)
            x = residualFF + ffScale * ffOut
            if !normalizeBefore { x = normFF(x) }
            // Final norm
            x = normF(x)
        } else {
            // Feed-forward without conv module
            let residualFF = x
            if normalizeBefore { x = normFF(x) }
            let ffOut = feedForward(x)
            x = residualFF + ffScale * ffOut
            if !normalizeBefore { x = normFF(x) }
        }

        return (x, mask, newAttCache, newCnnCache)
    }
}

// MARK: - Upsample 1D

/// 1D upsampling using repeat + conv.
class S3GenUpsample1D: Module {
    let channels: Int
    let outChannels: Int
    let strideVal: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, outChannels: Int, stride: Int = 2) {
        self.channels = channels
        self.outChannels = outChannels
        self.strideVal = stride

        self._conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: outChannels,
            kernelSize: stride * 2 + 1, stride: 1, padding: 0)
    }

    func callAsFunction(_ inputs: MLXArray, inputLengths: MLXArray)
        -> (MLXArray, MLXArray)
    {
        // inputs: (B, C, T) PyTorch format
        let B = inputs.dim(0), C = inputs.dim(1)

        // Repeat each timestep stride times
        var outputs = MLX.repeated(inputs, count: strideVal, axis: 2)

        // Pad on the left
        outputs = MLX.padded(outputs, widths: [.init(0), .init(0), .init((strideVal * 2, 0))])

        // Transpose to (B, T, C) for Conv1d
        outputs = outputs.transposed(0, 2, 1)
        outputs = conv(outputs)
        outputs = outputs.transposed(0, 2, 1) // back to (B, C, T)

        return (outputs, inputLengths * strideVal)
    }
}

// MARK: - Pre-Lookahead Layer

/// Pre-lookahead layer for causal processing.
class S3GenPreLookaheadLayer: Module {
    let channels: Int
    let preLookaheadLen: Int

    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(channels: Int, preLookaheadLen: Int = 1) {
        self.channels = channels
        self.preLookaheadLen = preLookaheadLen

        self._conv1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: preLookaheadLen + 1, stride: 1, padding: 0)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 3, stride: 1, padding: 0)
    }

    func callAsFunction(_ inputs: MLXArray, context: MLXArray? = nil) -> MLXArray {
        var outputs = inputs // (B, T, C)

        if context == nil || context!.dim(1) == 0 {
            outputs = MLX.padded(outputs, widths: [.init(0), .init((0, preLookaheadLen)), .init(0)])
        } else {
            outputs = MLX.concatenated([outputs, context!], axis: 1)
        }

        outputs = leakyRelu(conv1(outputs))

        // Causal pad on left
        outputs = MLX.padded(outputs, widths: [.init(0), .init((2, 0)), .init(0)])
        outputs = conv2(outputs)

        return outputs + inputs
    }
}

// MARK: - Helper Functions

/// Create padding mask from lengths.
func s3genMakePadMask(lengths: MLXArray, maxLen: Int) -> MLXArray {
    let batchSize = lengths.dim(0)
    let ml = maxLen > 0 ? maxLen : Int(lengths.max().item(Int.self))
    let seqRange = MLXArray(0 ..< ml).expandedDimensions(axis: 0)
    let seqRangeExpanded = MLX.broadcast(seqRange, to: [batchSize, ml])
    let seqLengthExpanded = lengths.expandedDimensions(axis: -1)
    return seqRangeExpanded .>= seqLengthExpanded
}

/// Create subsequent chunk mask for streaming.
func s3genSubsequentChunkMask(size: Int, chunkSize: Int, numLeftChunks: Int = -1) -> MLXArray {
    let posIdx = MLXArray(0 ..< size)
    let blockValue = ((posIdx / chunkSize) + 1) * chunkSize
    return posIdx.expandedDimensions(axis: 0) .< blockValue.expandedDimensions(axis: 1)
}

/// Apply optional chunk mask for streaming/training.
func s3genAddOptionalChunkMask(
    xs: MLXArray, masks: MLXArray,
    useDynamicChunk: Bool, useDynamicLeftChunk: Bool,
    decodingChunkSize: Int, staticChunkSize: Int,
    numDecodingLeftChunks: Int
) -> MLXArray {
    if useDynamicChunk {
        let maxLen = xs.dim(1)
        let chunkSize: Int
        let numLeftChunks: Int
        if decodingChunkSize < 0 {
            chunkSize = maxLen
            numLeftChunks = -1
        } else if decodingChunkSize > 0 {
            chunkSize = decodingChunkSize
            numLeftChunks = numDecodingLeftChunks
        } else {
            chunkSize = maxLen
            numLeftChunks = -1
        }
        var chunkMasks = s3genSubsequentChunkMask(size: xs.dim(1), chunkSize: chunkSize, numLeftChunks: numLeftChunks)
        chunkMasks = chunkMasks.expandedDimensions(axis: 0)
        return masks & chunkMasks
    } else if staticChunkSize > 0 {
        var chunkMasks = s3genSubsequentChunkMask(
            size: xs.dim(1), chunkSize: staticChunkSize, numLeftChunks: numDecodingLeftChunks)
        chunkMasks = chunkMasks.expandedDimensions(axis: 0)
        return masks & chunkMasks
    } else {
        return masks
    }
}

// MARK: - UpsampleConformerEncoder

/// Full Conformer encoder with upsampling for speech synthesis.
class UpsampleConformerEncoder: Module {
    let outputSizeVal: Int
    let normalizeBefore: Bool
    let staticChunkSize: Int
    let useDynamicChunk: Bool
    let useDynamicLeftChunk: Bool
    let numEncoders: Int
    let numUpEncoders: Int
    let upsampleStride: Int

    @ModuleInfo(key: "embed") var embed: S3GenLinearNoSubsampling
    @ModuleInfo(key: "up_embed") var upEmbed: S3GenLinearNoSubsampling
    @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm
    @ModuleInfo(key: "pre_lookahead_layer") var preLookaheadLayer: S3GenPreLookaheadLayer
    @ModuleInfo(key: "up_layer") var upLayer: S3GenUpsample1D
    @ModuleInfo(key: "encoders") var encoders: [S3GenConformerEncoderLayer]
    @ModuleInfo(key: "up_encoders") var upEncoders: [S3GenConformerEncoderLayer]

    init(
        inputSize: Int = 512, outputSize: Int = 512,
        attentionHeads: Int = 8, linearUnits: Int = 2048,
        numBlocks: Int = 6, numUpBlocks: Int = 4,
        dropoutRate: Float = 0.1, positionalDropoutRate: Float = 0.1,
        attentionDropoutRate: Float = 0.1,
        posEncLayerType: String = "rel_pos_espnet",
        normalizeBefore: Bool = true,
        staticChunkSize: Int = 0,
        useDynamicChunk: Bool = false, useDynamicLeftChunk: Bool = false,
        macaronStyle: Bool = false,
        selfattentionLayerType: String = "rel_selfattn",
        useCnnModule: Bool = false, cnnModuleKernel: Int = 15,
        causal: Bool = false, cnnModuleNorm: String = "batch_norm",
        keyBias: Bool = true, preLookaheadLen: Int = 3,
        upsampleStride: Int = 2
    ) {
        self.outputSizeVal = outputSize
        self.normalizeBefore = normalizeBefore
        self.staticChunkSize = staticChunkSize
        self.useDynamicChunk = useDynamicChunk
        self.useDynamicLeftChunk = useDynamicLeftChunk
        self.numEncoders = numBlocks
        self.numUpEncoders = numUpBlocks
        self.upsampleStride = upsampleStride

        // Create positional encoding
        let posEnc: Module
        if posEncLayerType == "rel_pos_espnet" {
            posEnc = S3GenEspnetRelPositionalEncoding(dModel: outputSize, dropoutRate: positionalDropoutRate)
        } else {
            posEnc = S3GenRelPositionalEncoding(dModel: outputSize, dropoutRate: positionalDropoutRate)
        }

        self._embed.wrappedValue = S3GenLinearNoSubsampling(
            idim: inputSize, odim: outputSize, dropoutRate: dropoutRate, posEnc: posEnc)

        self._afterNorm.wrappedValue = LayerNorm(dimensions: outputSize, eps: 1e-5)

        let activation: (MLXArray) -> MLXArray = { silu($0) }

        self._preLookaheadLayer.wrappedValue = S3GenPreLookaheadLayer(
            channels: outputSize, preLookaheadLen: preLookaheadLen)

        self._upLayer.wrappedValue = S3GenUpsample1D(
            channels: outputSize, outChannels: outputSize, stride: upsampleStride)

        // Up positional encoding
        let upPosEnc: Module
        if posEncLayerType == "rel_pos_espnet" {
            upPosEnc = S3GenEspnetRelPositionalEncoding(dModel: outputSize, dropoutRate: positionalDropoutRate)
        } else {
            upPosEnc = S3GenRelPositionalEncoding(dModel: outputSize, dropoutRate: positionalDropoutRate)
        }

        self._upEmbed.wrappedValue = S3GenLinearNoSubsampling(
            idim: inputSize, odim: outputSize, dropoutRate: dropoutRate, posEnc: upPosEnc)

        // Create encoder layers
        var encoderLayers = [S3GenConformerEncoderLayer]()
        for i in 0 ..< numBlocks {
            let attnModule: Module
            if selfattentionLayerType == "rel_selfattn" {
                attnModule = S3GenRelPositionMultiHeadedAttention(
                    nHead: attentionHeads, nFeat: outputSize,
                    dropoutRate: attentionDropoutRate, keyBias: keyBias)
            } else {
                attnModule = S3GenMultiHeadedAttention(
                    nHead: attentionHeads, nFeat: outputSize,
                    dropoutRate: attentionDropoutRate, keyBias: keyBias)
            }

            let ff = S3GenPositionwiseFeedForward(
                idim: outputSize, hiddenUnits: linearUnits,
                dropoutRate: dropoutRate, activation: activation)

            let ffMacaron: S3GenPositionwiseFeedForward? = macaronStyle
                ? S3GenPositionwiseFeedForward(
                    idim: outputSize, hiddenUnits: linearUnits,
                    dropoutRate: dropoutRate, activation: activation)
                : nil

            let convMod: S3GenConvolutionModule? = useCnnModule
                ? S3GenConvolutionModule(
                    channels: outputSize, kernelSize: cnnModuleKernel,
                    activation: activation, norm: cnnModuleNorm, causal: causal)
                : nil

            let layer = S3GenConformerEncoderLayer(
                size: outputSize, selfAttn: attnModule,
                feedForward: ff, feedForwardMacaron: ffMacaron,
                convModule: convMod, dropoutRate: dropoutRate,
                normalizeBefore: normalizeBefore)

            encoderLayers.append(layer)
        }
        self._encoders.wrappedValue = encoderLayers

        // Create up-encoder layers
        var upEncoderLayers = [S3GenConformerEncoderLayer]()
        for i in 0 ..< numUpBlocks {
            let attnModule: Module
            if selfattentionLayerType == "rel_selfattn" {
                attnModule = S3GenRelPositionMultiHeadedAttention(
                    nHead: attentionHeads, nFeat: outputSize,
                    dropoutRate: attentionDropoutRate, keyBias: keyBias)
            } else {
                attnModule = S3GenMultiHeadedAttention(
                    nHead: attentionHeads, nFeat: outputSize,
                    dropoutRate: attentionDropoutRate, keyBias: keyBias)
            }

            let ff = S3GenPositionwiseFeedForward(
                idim: outputSize, hiddenUnits: linearUnits,
                dropoutRate: dropoutRate, activation: activation)

            let ffMacaron: S3GenPositionwiseFeedForward? = macaronStyle
                ? S3GenPositionwiseFeedForward(
                    idim: outputSize, hiddenUnits: linearUnits,
                    dropoutRate: dropoutRate, activation: activation)
                : nil

            let convMod: S3GenConvolutionModule? = useCnnModule
                ? S3GenConvolutionModule(
                    channels: outputSize, kernelSize: cnnModuleKernel,
                    activation: activation, norm: cnnModuleNorm, causal: causal)
                : nil

            let layer = S3GenConformerEncoderLayer(
                size: outputSize, selfAttn: attnModule,
                feedForward: ff, feedForwardMacaron: ffMacaron,
                convModule: convMod, dropoutRate: dropoutRate,
                normalizeBefore: normalizeBefore)

            upEncoderLayers.append(layer)
        }
        self._upEncoders.wrappedValue = upEncoderLayers
    }

    func callAsFunction(
        xs: MLXArray, xsLens: MLXArray,
        context: MLXArray? = nil,
        decodingChunkSize: Int = 0,
        numDecodingLeftChunks: Int = -1,
        streaming: Bool = false
    ) -> (MLXArray, MLXArray) {
        let T = xs.dim(1)
        var masks = MLX.logicalNot(s3genMakePadMask(lengths: xsLens, maxLen: T))
        masks = masks.expandedDimensions(axis: 1) // (B, 1, T)

        var (out, posEmb, masksOut) = embed(xs, mask: masks)
        masks = masksOut

        let effectiveChunkSize = streaming ? staticChunkSize : 0

        var chunkMasks = s3genAddOptionalChunkMask(
            xs: out, masks: masks,
            useDynamicChunk: useDynamicChunk,
            useDynamicLeftChunk: useDynamicLeftChunk,
            decodingChunkSize: decodingChunkSize,
            staticChunkSize: effectiveChunkSize,
            numDecodingLeftChunks: numDecodingLeftChunks)

        // Pre-lookahead
        var embeddedContext: MLXArray? = nil
        if let ctx = context, ctx.dim(1) > 0 {
            let ctxMasks = MLXArray.ones([1, 1, ctx.dim(1)]).asType(.bool)
            (embeddedContext, _, _) = embed(ctx, mask: ctxMasks, offset: out.dim(1))
        }

        out = preLookaheadLayer(out, context: embeddedContext)

        // Forward through main encoder layers
        for i in 0 ..< numEncoders {
            (out, chunkMasks, _, _) = encoders[i](out, mask: chunkMasks, posEmb: posEmb)
        }

        // Upsample
        var upOut = out.transposed(0, 2, 1) // (B, D, T)
        var xsLensUp: MLXArray
        (upOut, xsLensUp) = upLayer(upOut, inputLengths: xsLens)
        upOut = upOut.transposed(0, 2, 1) // (B, T', D)

        let newT = upOut.dim(1)
        masks = MLX.logicalNot(s3genMakePadMask(lengths: xsLensUp, maxLen: newT))
        masks = masks.expandedDimensions(axis: 1)

        (out, posEmb, masks) = upEmbed(upOut, mask: masks)

        let effectiveUpChunkSize = effectiveChunkSize * upsampleStride

        chunkMasks = s3genAddOptionalChunkMask(
            xs: out, masks: masks,
            useDynamicChunk: useDynamicChunk,
            useDynamicLeftChunk: useDynamicLeftChunk,
            decodingChunkSize: decodingChunkSize,
            staticChunkSize: effectiveUpChunkSize,
            numDecodingLeftChunks: numDecodingLeftChunks)

        // Forward through up-encoder layers
        for i in 0 ..< numUpEncoders {
            (out, chunkMasks, _, _) = upEncoders[i](out, mask: chunkMasks, posEmb: posEmb)
        }

        if normalizeBefore {
            out = afterNorm(out)
        }

        return (out, masks)
    }
}
