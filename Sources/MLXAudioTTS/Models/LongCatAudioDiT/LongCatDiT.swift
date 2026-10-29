import Foundation
import MLX
import MLXNN

func longCatLayerNorm(_ x: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
    let f = x.asType(.float32)
    return ((f - mean(f, axis: -1, keepDims: true)) * rsqrt(variance(f, axis: -1, keepDims: true) + epsilon)).asType(x.dtype)
}

final class LongCatRMSNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    let epsilon: Float
    init(_ dim: Int, epsilon: Float) {
        self.epsilon = epsilon; _weight.wrappedValue = .ones([dim])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * rsqrt(mean(f.square(), axis: -1, keepDims: true) + epsilon)).asType(x.dtype) * weight
    }
}

final class LongCatEmbedder: Module {
    @ModuleInfo var proj: [Linear]
    init(_ input: Int, _ output: Int) {
        _proj.wrappedValue = [Linear(input, output), Linear(output, output)]
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        if let mask { h = MLX.where(mask.expandedDimensions(axis: -1), h, 0) }
        h = proj[1](silu(proj[0](h)))
        if let mask { h = MLX.where(mask.expandedDimensions(axis: -1), h, 0) }
        return h
    }
}

final class LongCatTimeEmbedding: Module, UnaryLayer {
    @ModuleInfo(key: "time_mlp") var mlp: [Linear]
    let frequencyDim: Int
    init(_ dim: Int, frequencyDim: Int = 256) {
        self.frequencyDim = frequencyDim
        _mlp.wrappedValue = [Linear(frequencyDim, dim), Linear(dim, dim)]
    }
    func callAsFunction(_ time: MLXArray) -> MLXArray {
        let half = frequencyDim / 2
        let frequencies = exp(MLXArray(0..<half).asType(.float32) * (-log(Float(10000)) / Float(half - 1)))
        let angles = 1000 * time.expandedDimensions(axis: -1) * frequencies
        return mlp[1](silu(mlp[0](concatenated([sin(angles), cos(angles)], axis: -1))))
    }
}

final class LongCatAdaMLP: Module, UnaryLayer {
    @ModuleInfo var mlp: [Linear]
    init(_ dim: Int, bias: Bool = true) {
        _mlp.wrappedValue = [Linear(dim, dim * 6, bias: bias)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { mlp[0](silu(x)) }
}

final class LongCatGRN: Module, UnaryLayer {
    @ParameterInfo var gamma: MLXArray
    @ParameterInfo var beta: MLXArray
    init(_ dim: Int) {
        _gamma.wrappedValue = .zeros([1, 1, dim])
        _beta.wrappedValue = .zeros([1, 1, dim])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gx = sqrt(sum(x.square(), axis: 1, keepDims: true))
        let nx = gx / (mean(gx, axis: -1, keepDims: true) + 1e-6)
        return gamma * x * nx + beta + x
    }
}

final class LongCatConvNeXt: Module, UnaryLayer {
    @ParameterInfo(key: "dwconv_weight") var depthwiseWeight: MLXArray
    @ParameterInfo(key: "dwconv_bias") var depthwiseBias: MLXArray
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var pwconv1: Linear
    @ModuleInfo var pwconv2: Linear
    @ModuleInfo var grn: LongCatGRN
    let dim: Int
    init(_ dim: Int, bias: Bool, epsilon: Float) {
        self.dim = dim
        _depthwiseWeight.wrappedValue = .zeros([dim, 7, 1])
        _depthwiseBias.wrappedValue = .zeros([dim])
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: epsilon)
        _pwconv1.wrappedValue = Linear(dim, dim * 2, bias: bias)
        _pwconv2.wrappedValue = Linear(dim * 2, dim, bias: bias)
        _grn.wrappedValue = LongCatGRN(dim * 2)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = conv1d(x, depthwiseWeight, padding: 3, groups: dim) + depthwiseBias
        return x + pwconv2(grn(silu(pwconv1(norm(h)))))
    }
}

func longCatRoPE(_ x: MLXArray) -> MLXArray {
    // Qwen-style half rotation, computed in Float32 before restoring the input dtype.
    let dim = x.dim(-1), length = x.dim(-2)
    let inverse = pow(MLXArray(Float(100000)), -MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32) / Float(dim))
    let frequencies = MLXArray(0..<length).asType(.float32).expandedDimensions(axis: 1) * inverse
    let angles = concatenated([frequencies, frequencies], axis: -1).expandedDimensions(axes: [0, 1])
    let halves = split(x, parts: 2, axis: -1)
    let rotated = concatenated([-halves[1], halves[0]], axis: -1).asType(.float32)
    return (x.asType(.float32) * cos(angles) + rotated * sin(angles)).asType(x.dtype)
}

final class LongCatAttention: Module {
    @ModuleInfo(key: "to_q") var query: Linear
    @ModuleInfo(key: "to_k") var key: Linear
    @ModuleInfo(key: "to_v") var value: Linear
    @ModuleInfo(key: "to_out") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: LongCatRMSNorm?
    @ModuleInfo(key: "k_norm") var keyNorm: LongCatRMSNorm?
    let heads: Int
    let headDim: Int
    init(_ c: LongCatConfiguration) {
        let d = c.ditDim
        heads = c.ditHeads; headDim = d / heads
        _query.wrappedValue = Linear(d, d, bias: c.ditBias)
        _key.wrappedValue = Linear(d, d, bias: c.ditBias)
        _value.wrappedValue = Linear(d, d, bias: c.ditBias)
        _output.wrappedValue = Linear(d, d, bias: c.ditBias)
        _queryNorm.wrappedValue = c.ditQkNorm ? LongCatRMSNorm(d, epsilon: c.ditEps) : nil
        _keyNorm.wrappedValue = c.ditQkNorm ? LongCatRMSNorm(d, epsilon: c.ditEps) : nil
    }
    func callAsFunction(_ x: MLXArray, condition: MLXArray? = nil, mask: MLXArray? = nil) -> MLXArray {
        let c = condition ?? x, b = x.dim(0), length = x.dim(1)
        var q = query(x), k = key(c)
        if let queryNorm, let keyNorm { q = queryNorm(q); k = keyNorm(k) }
        q = longCatRoPE(q.reshaped(b, length, heads, headDim).transposed(0, 2, 1, 3))
        k = longCatRoPE(k.reshaped(b, c.dim(1), heads, headDim).transposed(0, 2, 1, 3))
        let v = value(c).reshaped(b, c.dim(1), heads, headDim).transposed(0, 2, 1, 3)
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) / sqrt(Float(headDim))
        if let mask { scores = MLX.where(mask.expandedDimensions(axes: [1, 2]), scores, -Float.infinity) }
        let weights = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        return output(matmul(weights, v).transposed(0, 2, 1, 3).reshaped(b, length, heads * headDim))
    }
}

final class LongCatFFN: Module, UnaryLayer {
    @ModuleInfo var ff: [Linear]
    init(_ c: LongCatConfiguration) {
        let inner = Int(Float(c.ditDim) * c.ditFfMult)
        _ff.wrappedValue = [Linear(c.ditDim, inner, bias: c.ditBias), Linear(inner, c.ditDim, bias: c.ditBias)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { ff[1](geluApproximate(ff[0](x))) }
}

final class LongCatDiTBlock: Module {
    @ParameterInfo(key: "adaln_scale_shift") var scaleShift: MLXArray?
    @ModuleInfo(key: "adaln_mlp") var adaln: LongCatAdaMLP?
    @ModuleInfo(key: "self_attn") var selfAttention: LongCatAttention
    @ModuleInfo(key: "cross_attn") var crossAttention: LongCatAttention?
    @ModuleInfo(key: "cross_attn_norm") var crossNorm: LayerNorm?
    @ModuleInfo(key: "cross_attn_norm_c") var crossConditionNorm: LayerNorm?
    @ModuleInfo var ffn: LongCatFFN
    let epsilon: Float
    init(_ c: LongCatConfiguration) {
        epsilon = c.ditEps
        _scaleShift.wrappedValue = c.ditAdalnType == "global" ? .zeros([c.ditDim * 6]) : nil
        _adaln.wrappedValue = c.ditAdalnType == "local" ? LongCatAdaMLP(c.ditDim) : nil
        _selfAttention.wrappedValue = LongCatAttention(c)
        _crossAttention.wrappedValue = c.ditCrossAttn ? LongCatAttention(c) : nil
        _crossNorm.wrappedValue = c.ditCrossAttn && c.ditCrossAttnNorm ? LayerNorm(dimensions: c.ditDim, eps: c.ditEps) : nil
        _crossConditionNorm.wrappedValue = c.ditCrossAttn && c.ditCrossAttnNorm ? LayerNorm(dimensions: c.ditDim, eps: c.ditEps) : nil
        _ffn.wrappedValue = LongCatFFN(c)
    }
    func callAsFunction(_ x: MLXArray, time: MLXArray, text: MLXArray, mask: MLXArray?, textMask: MLXArray?, global: MLXArray?) -> MLXArray {
        let modulation = global.map { $0 + scaleShift! } ?? adaln!(time)
        let parts = split(modulation, parts: 6, axis: -1).map { $0.expandedDimensions(axis: 1) }
        var h = x + parts[0] * selfAttention(longCatLayerNorm(x, epsilon: epsilon) * (1 + parts[1]) + parts[2], mask: mask)
        if let crossAttention {
            h = h + crossAttention(crossNorm?(h) ?? h, condition: crossConditionNorm?(text) ?? text, mask: textMask)
        }
        return h + parts[3] * ffn(longCatLayerNorm(h, epsilon: epsilon) * (1 + parts[4]) + parts[5])
    }
}

final class LongCatFinalNorm: Module {
    @ModuleInfo var linear: Linear
    let epsilon: Float
    init(_ c: LongCatConfiguration) {
        epsilon = c.ditEps; _linear.wrappedValue = Linear(c.ditDim, c.ditDim * 2)
    }
    func callAsFunction(_ x: MLXArray, condition: MLXArray) -> MLXArray {
        let parts = split(linear(silu(condition)), parts: 2, axis: -1)
        return longCatLayerNorm(x, epsilon: epsilon) * (1 + parts[0].expandedDimensions(axis: 1)) + parts[1].expandedDimensions(axis: 1)
    }
}

final class LongCatDiT: Module {
    @ModuleInfo(key: "time_embed") var timeEmbedding: LongCatTimeEmbedding
    @ModuleInfo(key: "input_embed") var inputEmbedding: LongCatEmbedder
    @ModuleInfo(key: "text_embed") var textEmbedding: LongCatEmbedder
    @ModuleInfo var blocks: [LongCatDiTBlock]
    @ModuleInfo(key: "norm_out") var norm: LongCatFinalNorm
    @ModuleInfo(key: "proj_out") var projection: Linear
    @ModuleInfo(key: "adaln_global_mlp") var globalAdaLN: LongCatAdaMLP?
    @ModuleInfo(key: "text_conv_layer") var textConvolution: [LongCatConvNeXt]
    @ModuleInfo(key: "latent_embed") var latentEmbedding: LongCatEmbedder?
    @ModuleInfo(key: "latent_cond_embedder") var latentConditionEmbedding: LongCatEmbedder?
    let config: LongCatConfiguration
    init(_ c: LongCatConfiguration) {
        config = c
        _timeEmbedding.wrappedValue = LongCatTimeEmbedding(c.ditDim)
        _inputEmbedding.wrappedValue = LongCatEmbedder(c.latentDim, c.ditDim)
        _textEmbedding.wrappedValue = LongCatEmbedder(c.ditTextDim, c.ditDim)
        _blocks.wrappedValue = (0..<c.ditDepth).map { _ in LongCatDiTBlock(c) }
        _norm.wrappedValue = LongCatFinalNorm(c)
        _projection.wrappedValue = Linear(c.ditDim, c.latentDim)
        _globalAdaLN.wrappedValue = c.ditAdalnType == "global" ? LongCatAdaMLP(c.ditDim) : nil
        _textConvolution.wrappedValue = c.ditTextConv ? (0..<4).map { _ in LongCatConvNeXt(c.ditDim, bias: c.ditBias, epsilon: c.ditEps) } : []
        _latentEmbedding.wrappedValue = c.ditUseLatentCondition ? LongCatEmbedder(c.latentDim, c.ditDim) : nil
        _latentConditionEmbedding.wrappedValue = c.ditUseLatentCondition ? LongCatEmbedder(c.ditDim * 2, c.ditDim) : nil
    }
    func callAsFunction(_ input: MLXArray, text: MLXArray, textLength: MLXArray, time: MLXArray,
                        mask: MLXArray?, textMask: MLXArray?, latentCondition: MLXArray?, returnLayer: Int? = nil) -> (MLXArray, MLXArray?) {
        let t = timeEmbedding(time.ndim == 0 ? broadcast(time, to: [input.dim(0)]) : time)
        var condition = textEmbedding(text, mask: textMask)
        for conv in textConvolution { condition = conv(condition) }
        if !textConvolution.isEmpty, let textMask { condition = MLX.where(textMask.expandedDimensions(axis: -1), condition, 0) }
        var x = inputEmbedding(input, mask: mask)
        if let latentCondition, let latentEmbedding, let latentConditionEmbedding {
            x = latentConditionEmbedding(concatenated([x, latentEmbedding(latentCondition, mask: mask)], axis: -1))
        }
        let skip = x
        let normCondition = config.ditAdalnUseTextCond && globalAdaLN != nil
            ? t + sum(condition, axis: 1) / textLength.asType(condition.dtype).expandedDimensions(axis: -1) : t
        let global = globalAdaLN?(normCondition)
        var hidden: MLXArray?
        for (i, block) in blocks.enumerated() {
            x = block(x, time: t, text: condition, mask: mask, textMask: textMask, global: global)
            if returnLayer == i + 1 {
                hidden = x
                if config.ditLongSkip { x = x + skip }
            }
        }
        if config.ditLongSkip { x = x + skip }
        return (projection(norm(x, condition: normCondition)), hidden)
    }
}
