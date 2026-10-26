import Foundation
import MLX
import MLXAudioCore
import MLXNN

final class Audio8Norm: Module, UnaryLayer {
    var weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float) { self.weight = .ones([dim]); self.eps = eps }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * rsqrt(mean(f * f, axis: -1, keepDims: true) + eps)).asType(x.dtype) * weight
    }
}

/// Audio8 rounds the rotary table to BF16 before applying it in Float32.
final class Audio8RoPE {
    let cosine: MLXArray
    let sine: MLXArray
    init(length: Int, dim: Int, base: Float) {
        let frequency = 1 / pow(MLXArray(base), MLX.arange(0, dim, step: 2, dtype: .float32) / Float(dim))
        let phase = MLX.outer(MLX.arange(length, dtype: .float32), frequency)
        cosine = cos(phase).asType(.bfloat16)
        sine = sin(phase).asType(.bfloat16)
        eval(cosine, sine)
    }
    func apply(_ x: MLXArray, offset: Int) -> MLXArray {
        let length = x.shape[1], half = x.shape[3] / 2
        let c = cosine[offset..<(offset + length)].asType(.float32).reshaped([1, length, 1, half])
        let s = sine[offset..<(offset + length)].asType(.float32).reshaped([1, length, 1, half])
        let y = x.asType(.float32).reshaped(x.shape.dropLast() + [half, 2])
        let a = y[0..., 0..., 0..., 0..., 0], b = y[0..., 0..., 0..., 0..., 1]
        return stacked([a * c - b * s, b * c + a * s], axis: -1).reshaped(x.shape).asType(x.dtype)
    }
}

final class Audio8Attention: Module {
    @ModuleInfo var wqkv: Linear
    @ModuleInfo var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Audio8Norm?
    @ModuleInfo(key: "k_norm") var kNorm: Audio8Norm?
    let heads: Int, kvHeads: Int, headDim: Int
    init(dim: Int, heads: Int, kvHeads: Int, headDim: Int, qkvBias: Bool = false,
         outputBias: Bool = false, qkNorm: Bool = false, eps: Float = 1e-5) {
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
        _wqkv.wrappedValue = Linear(dim, (heads + 2 * kvHeads) * headDim, bias: qkvBias)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: outputBias)
        _qNorm.wrappedValue = qkNorm ? Audio8Norm(headDim, eps: eps) : nil
        _kNorm.wrappedValue = qkNorm ? Audio8Norm(headDim, eps: eps) : nil
    }
    func callAsFunction(_ x: MLXArray, rope: Audio8RoPE, offset: Int,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache? = nil) -> MLXArray {
        let b = x.shape[0], t = x.shape[1], qSize = heads * headDim, kvSize = kvHeads * headDim
        let qkv = wqkv(x)
        var q = qkv[0..., 0..., 0..<qSize].reshaped([b, t, heads, headDim])
        var k = qkv[0..., 0..., qSize..<(qSize + kvSize)].reshaped([b, t, kvHeads, headDim])
        var v = qkv[0..., 0..., (qSize + kvSize)...].reshaped([b, t, kvHeads, headDim]).transposed(0, 2, 1, 3)
        if let qNorm, let kNorm { q = qNorm(q); k = kNorm(k) }
        q = rope.apply(q, offset: offset).transposed(0, 2, 1, 3)
        k = rope.apply(k, offset: offset).transposed(0, 2, 1, 3)
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                    scale: 1 / sqrt(Float(headDim)), mask: mask)
        return wo(y.transposed(0, 2, 1, 3).reshaped([b, t, qSize]))
    }
}

final class Audio8FeedForward: Module, UnaryLayer {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear
    init(dim: Int, intermediate: Int) {
        _w1.wrappedValue = Linear(dim, intermediate, bias: false)
        _w2.wrappedValue = Linear(intermediate, dim, bias: false)
        _w3.wrappedValue = Linear(dim, intermediate, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class Audio8LayerScale: Module, UnaryLayer {
    var gamma: MLXArray
    init(_ dim: Int) { gamma = MLXArray.ones([dim]) * 0.01 }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

final class Audio8Block: Module {
    let attention: Audio8Attention
    @ModuleInfo(key: "feed_forward") var feedForward: Audio8FeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: Audio8Norm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: Audio8Norm
    @ModuleInfo(key: "attention_layer_scale") var attentionScale: Audio8LayerScale?
    @ModuleInfo(key: "ffn_layer_scale") var ffnScale: Audio8LayerScale?
    init(dim: Int, intermediate: Int, heads: Int, kvHeads: Int, headDim: Int,
         qkvBias: Bool = false, outputBias: Bool = false, qkNorm: Bool = false,
         eps: Float = 1e-5, layerScale: Bool = false) {
        attention = Audio8Attention(dim: dim, heads: heads, kvHeads: kvHeads, headDim: headDim,
                qkvBias: qkvBias, outputBias: outputBias, qkNorm: qkNorm, eps: eps)
        _feedForward.wrappedValue = Audio8FeedForward(dim: dim, intermediate: intermediate)
        _attentionNorm.wrappedValue = Audio8Norm(dim, eps: eps)
        _ffnNorm.wrappedValue = Audio8Norm(dim, eps: eps)
        _attentionScale.wrappedValue = layerScale ? Audio8LayerScale(dim) : nil
        _ffnScale.wrappedValue = layerScale ? Audio8LayerScale(dim) : nil
    }
    func callAsFunction(_ x: MLXArray, rope: Audio8RoPE, offset: Int = 0,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache? = nil) -> MLXArray {
        var a = attention(attentionNorm(x), rope: rope, offset: offset, mask: mask, cache: cache)
        if let attentionScale { a = attentionScale(a) }
        let h = x + a
        var f = feedForward(ffnNorm(h))
        if let ffnScale { f = ffnScale(f) }
        return h + f
    }
}

final class Audio8Backbone: Module {
    let config: Audio8Config
    @ModuleInfo var embeddings: Embedding
    @ModuleInfo(key: "codebook_embeddings") var codebookEmbeddings: Embedding
    let layers: [Audio8Block]
    let norm: Audio8Norm
    @ModuleInfo(key: "fast_project_in") var fastProjectIn: Linear?
    @ModuleInfo(key: "fast_embeddings") var fastEmbeddings: Embedding
    @ModuleInfo(key: "fast_layers") var fastLayers: [Audio8Block]
    @ModuleInfo(key: "fast_norm") var fastNorm: Audio8Norm
    @ModuleInfo(key: "fast_output") var fastOutput: Linear
    let slowRoPE: Audio8RoPE, fastRoPE: Audio8RoPE
    init(_ c: Audio8Config) {
        config = c
        _embeddings.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.dim)
        _codebookEmbeddings.wrappedValue = Embedding(embeddingCount: c.codebookSize * c.numCodebooks, dimensions: c.dim)
        layers = (0..<c.nLayer).map { _ in Audio8Block(dim: c.dim, intermediate: c.intermediateSize,
            heads: c.nHead, kvHeads: c.nLocalHeads, headDim: c.headDim, qkvBias: c.attentionQKVBias,
            outputBias: c.attentionOBias, qkNorm: c.attentionQKNorm, eps: c.normEps) }
        norm = Audio8Norm(c.dim, eps: c.normEps)
        _fastProjectIn.wrappedValue = c.fastDim == c.dim ? nil : Linear(c.dim, c.fastDim)
        _fastEmbeddings.wrappedValue = Embedding(embeddingCount: c.codebookSize, dimensions: c.fastDim)
        _fastLayers.wrappedValue = (0..<c.nFastLayer).map { _ in Audio8Block(dim: c.fastDim,
            intermediate: c.fastIntermediateSize, heads: c.fastNHead, kvHeads: c.fastNLocalHeads,
            headDim: c.fastHeadDim, qkvBias: c.fastAttentionQKVBias, outputBias: c.fastAttentionOBias,
            qkNorm: c.fastAttentionQKNorm, eps: c.normEps) }
        _fastNorm.wrappedValue = Audio8Norm(c.fastDim, eps: c.normEps)
        _fastOutput.wrappedValue = Linear(c.fastDim, c.codebookSize, bias: false)
        slowRoPE = Audio8RoPE(length: c.maxSeqLen, dim: c.headDim, base: c.ropeBase)
        fastRoPE = Audio8RoPE(length: c.numCodebooks, dim: c.fastHeadDim, base: c.ropeBase)
    }
    func embed(_ ids: MLXArray) -> MLXArray {
        let c = config, tokens = ids[0..., 0, 0...]
        let book = (0..<c.numCodebooks).map { codebookEmbeddings(ids[0..., $0 + 1, 0...] + Int32($0 * c.codebookSize)) }
        let total = stacked(book, axis: 1).sum(axis: 1)
        let semantic = (tokens .>= c.semanticBeginID) .&& (tokens .<= c.semanticEndID)
        return embeddings(tokens) + which(semantic.expandedDimensions(axis: -1), total, 0)
    }
    func slow(_ ids: MLXArray, offset: Int = 0, cache: [KVCache]? = nil) -> (MLXArray, MLXArray) {
        var h = embed(ids)
        let t = ids.shape[2]
        let q = MLX.arange(offset, offset + t).expandedDimensions(axis: 1)
        let k = MLX.arange(offset + t).expandedDimensions(axis: 0)
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.array((k .<= q).reshaped([1, 1, t, offset + t]))
        for (i, layer) in layers.enumerated() { h = layer(h, rope: slowRoPE, offset: offset, mask: mask, cache: cache?[i]) }
        let normalized = norm(h)
        return (embeddings.asLinear(normalized), cFastHidden(h, normalized))
    }
    private func cFastHidden(_ h: MLXArray, _ n: MLXArray) -> MLXArray { config.normFastlayerInput ? n : h }
    func fast(_ h: MLXArray, position: Int, cache: [KVCache]) -> MLXArray {
        var x = h
        for (i, layer) in fastLayers.enumerated() { x = layer(x, rope: fastRoPE, offset: position, mask: .none, cache: cache[i]) }
        return fastOutput(fastNorm(x))[0..., -1, 0...]
    }
}
