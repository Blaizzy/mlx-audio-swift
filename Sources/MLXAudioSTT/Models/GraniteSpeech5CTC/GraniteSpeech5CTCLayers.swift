import Foundation
import MLX
import MLXNN

final class Granite5EvalBatchNorm: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    init(_ dimensions: Int) {
        _weight.wrappedValue = .ones([dimensions])
        _bias.wrappedValue = .zeros([dimensions])
        _runningMean.wrappedValue = .zeros([dimensions])
        _runningVar.wrappedValue = .ones([dimensions])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = ((x.asType(.float32) - runningMean.asType(.float32))
            * rsqrt(runningVar.asType(.float32) + 1e-5)) * weight.asType(.float32) + bias.asType(.float32)
        return y.asType(x.dtype)
    }
}

final class Granite5FeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    init(_ c: GraniteSpeech5CTCEncoderConfig) {
        _linear1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: c.attentionBias)
        _linear2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: c.attentionBias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

final class Granite5Attention: Module {
    let context: Int
    let heads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var out: Linear
    @ModuleInfo(key: "rel_pos_emb") var relative: Embedding
    init(_ c: GraniteSpeech5CTCEncoderConfig) {
        context = c.contextSize; heads = c.numAttentionHeads; headDim = c.headDim
        scale = pow(Float(c.headDim), -0.5)
        let inner = heads * headDim
        _q.wrappedValue = Linear(c.hiddenSize, inner, bias: false)
        _k.wrappedValue = Linear(c.hiddenSize, inner, bias: false)
        _v.wrappedValue = Linear(c.hiddenSize, inner, bias: false)
        _out.wrappedValue = Linear(inner, c.hiddenSize)
        _relative.wrappedValue = Embedding(embeddingCount: 2 * c.maxPositionEmbeddings + 1, dimensions: headDim)
    }
    func callAsFunction(_ input: MLXArray, distances: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = input.dim(0), t = input.dim(1), d = input.dim(2)
        let pad = (context - t % context) % context
        var x = input, attentionMask = mask
        if pad > 0 {
            x = padded(x, widths: [.init((0,0)),.init((0,pad)),.init((0,0))])
            let valid = attentionMask ?? MLXArray.ones([b,t], dtype: .bool)
            attentionMask = padded(valid.asType(.bool), widths: [.init((0,0)),.init((0,pad))])
        }
        let blocks = x.dim(1) / context
        func split(_ x: MLXArray) -> MLXArray {
            x.reshaped(b,blocks,context,heads,headDim).transposed(0,1,3,2,4)
        }
        let queries = split(q(x)), keys = split(k(x)), values = split(v(x))
        let relativeEmbeddings = relative(distances) * scale
        var positionBias = MLX.einsum("bmhcd,crd->bmhcr", queries, relativeEmbeddings)
        if let attentionMask {
            let valid = attentionMask.reshaped(b,blocks,context).expandedDimensions(axes: [2,3])
            let minimum: Float = positionBias.dtype == .float16 ? -65504
                : (positionBias.dtype == .bfloat16 ? -3.38953139e38 : -Float.greatestFiniteMagnitude)
            positionBias = MLX.where(valid, positionBias, MLXArray(minimum).asType(positionBias.dtype))
        }
        let scores = matmul(queries, keys.transposed(0,1,2,4,3)) * scale + positionBias
        let probs = softmax(scores.asType(.float32), axis: -1).asType(queries.dtype)
        let result = matmul(probs, values).transposed(0,1,3,2,4).reshaped(b,blocks * context,d)
        return out(result[0...,0..<t,0...])
    }
}

final class Granite5Convolution: Module {
    let padding: Int
    @ModuleInfo(key: "pointwise_lin1") var first: Linear
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo var norm: Granite5EvalBatchNorm
    @ModuleInfo(key: "pointwise_lin2") var last: Linear
    init(_ c: GraniteSpeech5CTCEncoderConfig, stride: Int) {
        let inner = c.hiddenSize * c.convExpansionFactor
        padding = (c.convKernelSize - 1) / 2
        _first.wrappedValue = Linear(c.hiddenSize, inner * 2)
        _depthwise.wrappedValue = Conv1d(inputChannels: inner, outputChannels: inner,
            kernelSize: c.convKernelSize, stride: stride, groups: inner, bias: false)
        _norm.wrappedValue = Granite5EvalBatchNorm(inner)
        _last.wrappedValue = Linear(inner, c.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let parts = split(first(x), parts: 2, axis: -1)
        var h = parts[0] * sigmoid(parts[1])
        if let mask { h = MLX.where(mask.expandedDimensions(axis: -1), h, zeros(like: h)) }
        h = padded(h, widths: [.init((0,0)),.init((padding,padding)),.init((0,0))])
        return last(silu(norm(depthwise(h))))
    }
}

final class Granite5Block: Module {
    let subsample: Bool
    @ModuleInfo(key: "feed_forward1") var ff1: Granite5FeedForward
    @ModuleInfo(key: "self_attn") var attention: Granite5Attention
    @ModuleInfo var conv: Granite5Convolution
    @ModuleInfo(key: "feed_forward2") var ff2: Granite5FeedForward
    @ModuleInfo(key: "norm_feed_forward1") var normFF1: LayerNorm
    @ModuleInfo(key: "norm_self_att") var normAttention: LayerNorm
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "norm_feed_forward2") var normFF2: LayerNorm
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    init(_ c: GraniteSpeech5CTCEncoderConfig, subsample: Bool) {
        self.subsample = subsample
        _ff1.wrappedValue = Granite5FeedForward(c)
        _attention.wrappedValue = Granite5Attention(c)
        _conv.wrappedValue = Granite5Convolution(c, stride: subsample ? 2 : 1)
        _ff2.wrappedValue = Granite5FeedForward(c)
        _normFF1.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
        _normAttention.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
        _normConv.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
        _normFF2.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
        _normOut.wrappedValue = LayerNorm(dimensions: c.hiddenSize)
    }
    func callAsFunction(_ x: MLXArray, distances: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + 0.5 * ff1(normFF1(x))
        h = h + attention(normAttention(h), distances: distances, mask: mask)
        let convolution = conv(normConv(h), mask: mask)
        if subsample {
            let half = h.dim(1) / 2
            h = h[0...,0..<(2 * half),0...].reshaped(h.dim(0),half,2,h.dim(2)).mean(axis: 2)
                + convolution[0...,0..<half,0...]
        } else { h = h + convolution }
        return normOut(h + 0.5 * ff2(normFF2(h)))
    }
}

public final class GraniteSpeech5CTCEncoder: Module {
    let config: GraniteSpeech5CTCEncoderConfig
    private let _distances: MLXArray
    @ModuleInfo(key: "input_linear") var input: Linear
    @ModuleInfo var layers: [Granite5Block]
    @ModuleInfo var out: Linear
    @ModuleInfo(key: "out_mid") var outMid: Linear
    public init(_ c: GraniteSpeech5CTCEncoderConfig) {
        config = c
        _input.wrappedValue = Linear(c.numMelBins * 4, c.hiddenSize)
        _layers.wrappedValue = (0..<c.numHiddenLayers).map { Granite5Block(c, subsample: c.subsampleLayers.contains($0)) }
        _out.wrappedValue = Linear(c.hiddenSize, c.vocabSize)
        _outMid.wrappedValue = Linear(c.vocabSize, c.hiddenSize)
        let positions = arange(c.contextSize, dtype: .int32)
        _distances = positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0) + c.maxPositionEmbeddings
    }
    public func callAsFunction(_ features: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let dtype = (input as? QuantizedLinear)?.scales.dtype ?? input.weight.dtype
        var h = input(features.asType(dtype)), mask = attentionMask?.asType(.bool)
        if let mask { h = MLX.where(mask.expandedDimensions(axis: -1), h, zeros(like: h)) }
        for (index,layer) in layers.enumerated() {
            h = layer(h, distances: _distances, mask: mask)
            if layer.subsample, let current = mask {
                let half = current.dim(1) / 2
                mask = current[0...,0..<(2 * half)].reshaped(current.dim(0),half,2).all(axis: 2)
            }
            if index + 1 == config.numHiddenLayers / 2 {
                let probabilities = softmax(out(h).asType(.float32), axis: -1).asType(h.dtype)
                h = h + outMid(probabilities)
            }
        }
        return h
    }
}
