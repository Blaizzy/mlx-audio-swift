import Foundation
import MLX
import MLXNN

/// UMT5 keeps a separate relative-position bias in every encoder block.
final class LongCatT5Norm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    let epsilon: Float
    init(_ dim: Int, epsilon: Float) {
        self.epsilon = epsilon
        self._weight.wrappedValue = .ones([dim])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * rsqrt(mean(x.asType(.float32).square(), axis: -1, keepDims: true) + epsilon) * weight
    }
}

final class LongCatT5Dense: Module, UnaryLayer {
    @ModuleInfo(key: "wi_0") var wi0: Linear?
    @ModuleInfo(key: "wi_1") var wi1: Linear?
    @ModuleInfo var wi: Linear?
    @ModuleInfo var wo: Linear
    init(_ c: LongCatTextEncoderConfiguration) {
        _wi0.wrappedValue = c.isGatedAct ? Linear(c.dModel, c.dFf, bias: false) : nil
        _wi1.wrappedValue = c.isGatedAct ? Linear(c.dModel, c.dFf, bias: false) : nil
        _wi.wrappedValue = c.isGatedAct ? nil : Linear(c.dModel, c.dFf, bias: false)
        _wo.wrappedValue = Linear(c.dFf, c.dModel, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let wi0, let wi1 { return wo(gelu(wi0(x)) * wi1(x)) }
        return wo(gelu(wi!(x)))
    }
}

final class LongCatT5Attention: Module {
    @ModuleInfo var q: Linear
    @ModuleInfo var k: Linear
    @ModuleInfo var v: Linear
    @ModuleInfo var o: Linear
    @ModuleInfo(key: "relative_attention_bias") var relativeBias: Embedding
    let heads: Int
    let headDim: Int
    let buckets: Int
    let maxDistance: Int
    init(_ c: LongCatTextEncoderConfiguration) {
        heads = c.numHeads; headDim = c.dKv
        buckets = c.relativeAttentionNumBuckets; maxDistance = c.relativeAttentionMaxDistance
        _q.wrappedValue = Linear(c.dModel, heads * headDim, bias: false)
        _k.wrappedValue = Linear(c.dModel, heads * headDim, bias: false)
        _v.wrappedValue = Linear(c.dModel, heads * headDim, bias: false)
        _o.wrappedValue = Linear(heads * headDim, c.dModel, bias: false)
        _relativeBias.wrappedValue = Embedding(embeddingCount: buckets, dimensions: heads)
    }
    static func relativeBucket(_ position: Int, buckets: Int, maxDistance: Int) -> Int {
        let half = buckets / 2, exact = half / 2, distance = abs(position)
        let bucket = distance < exact ? distance : min(half - 1,
            exact + Int(log(Double(distance) / Double(exact)) / log(Double(maxDistance) / Double(exact)) * Double(half - exact)))
        return bucket + (position > 0 ? half : 0)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.dim(0), l = x.dim(1)
        let query = q(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        let key = k(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        let value = v(x).reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3)
        let indices = (0..<l).flatMap { i in (0..<l).map { j in Int32(Self.relativeBucket(j - i, buckets: buckets, maxDistance: maxDistance)) } }
        let bias = relativeBias(MLXArray(indices).reshaped(l, l)).transposed(2, 0, 1).expandedDimensions(axis: 0)
        var scores = matmul(query, key.transposed(0, 1, 3, 2)) + bias
        if let mask { scores = scores + (1 - mask.asType(.float32)).expandedDimensions(axes: [1, 2]) * -1e9 }
        let weights = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        return o(matmul(weights, value).transposed(0, 2, 1, 3).reshaped(b, l, heads * headDim))
    }
}

final class LongCatT5Block: Module {
    @ModuleInfo(key: "SelfAttention") var attention: LongCatT5Attention
    @ModuleInfo(key: "layer_norm_sa") var attentionNorm: LongCatT5Norm
    @ModuleInfo(key: "DenseReluDense") var dense: LongCatT5Dense
    @ModuleInfo(key: "layer_norm_ff") var feedForwardNorm: LongCatT5Norm
    init(_ c: LongCatTextEncoderConfiguration) {
        _attention.wrappedValue = LongCatT5Attention(c)
        _attentionNorm.wrappedValue = LongCatT5Norm(c.dModel, epsilon: c.layerNormEpsilon)
        _dense.wrappedValue = LongCatT5Dense(c)
        _feedForwardNorm.wrappedValue = LongCatT5Norm(c.dModel, epsilon: c.layerNormEpsilon)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let h = x + attention(attentionNorm(x), mask: mask)
        return h + dense(feedForwardNorm(h))
    }
}

final class LongCatUMT5Encoder: Module {
    @ModuleInfo var shared: Embedding
    @ModuleInfo var block: [LongCatT5Block]
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LongCatT5Norm
    init(_ c: LongCatTextEncoderConfiguration) {
        _shared.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.dModel)
        _block.wrappedValue = (0..<c.numLayers).map { _ in LongCatT5Block(c) }
        _finalNorm.wrappedValue = LongCatT5Norm(c.dModel, epsilon: c.layerNormEpsilon)
    }
    func callAsFunction(_ ids: MLXArray, mask: MLXArray?) -> (MLXArray, MLXArray) {
        let embedding = shared(ids)
        var x = embedding
        for layer in block { x = layer(x, mask: mask) }
        return (finalNorm(x), embedding)
    }
}
