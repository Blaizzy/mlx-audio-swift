import Foundation
import MLX
import MLXNN
import MLXRandom
import MLXAudioCore

// Match the fused SwiGLU used by the reference Mistral backbone in BF16.
private let voxtralTTSSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = MLX.compile { (gate: MLXArray, up: MLXArray) in
    silu(gate) * up
}

final class VoxtralTTSFeedForward: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear
    let fusedActivation: Bool
    init(dim: Int, hidden: Int, bias: Bool = false, fusedActivation: Bool = false) {
        self.fusedActivation = fusedActivation
        _w1.wrappedValue = Linear(dim, hidden, bias: bias)
        _w2.wrappedValue = Linear(hidden, dim, bias: bias)
        _w3.wrappedValue = Linear(dim, hidden, bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = w1(x), up = w3(x)
        return w2(fusedActivation ? voxtralTTSSwiGLU(gate,up) : silu(gate) * up)
    }
}

final class VoxtralTTSAttention: Module {
    @ModuleInfo var wq: Linear
    @ModuleInfo var wk: Linear
    @ModuleInfo var wv: Linear
    @ModuleInfo var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    let heads: Int, kvHeads: Int, headDim: Int
    let rope: RoPE?
    init(dim: Int, heads: Int, kvHeads: Int, headDim: Int, bias: Bool = false,
         ropeTheta: Float? = nil, qkEps: Float? = nil) {
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
        _wq.wrappedValue = Linear(dim, heads * headDim, bias: bias)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: bias)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: bias)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: bias)
        rope = ropeTheta.map { RoPE(dimensions: headDim, traditional: true, base: $0) }
        if let eps = qkEps {
            _qNorm.wrappedValue = RMSNorm(dimensions: heads * headDim, eps: eps)
            _kNorm.wrappedValue = RMSNorm(dimensions: kvHeads * headDim, eps: eps)
        }
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache? = nil,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
                        slopes: MLXArray? = nil, window: Int = 0) -> MLXArray {
        let b = x.dim(0), t = x.dim(1)
        var q = wq(x), k = wk(x)
        if let qNorm, let kNorm { q = qNorm(q); k = kNorm(k) }
        q = q.reshaped(b,t,heads,headDim).transposed(0,2,1,3)
        k = k.reshaped(b,t,kvHeads,headDim).transposed(0,2,1,3)
        var v = wv(x).reshaped(b,t,kvHeads,headDim).transposed(0,2,1,3)
        if let rope {
            q = applyRotaryPosition(rope, to: q, offset: cache?.ropeOffset)
            k = applyRotaryPosition(rope, to: k, offset: cache?.ropeOffset)
        }
        let output: MLXArray
        if let slopes {
            // The codec accumulates ALiBi scores and softmax in float32.
            if kvHeads < heads { k = repeated(k, count: heads / kvHeads, axis: 1); v = repeated(v, count: heads / kvHeads, axis: 1) }
            let positions = MLXArray(0..<t)
            let distance = positions[.newAxis, 0...] - positions[0..., .newAxis]
            let bias = slopes[0..., .newAxis, .newAxis] * distance.asType(.float32)[.newAxis]
            let invalid = logicalOr(distance .> 0, distance .< -window)
            let additive = MLX.where(invalid, MLXArray(Float(-1e9)), MLXArray(Float(0)))
            let scores = matmul(q, k.transposed(0,1,3,2)) * pow(Float(headDim), -0.5) + bias + additive
            let probabilities = softmax(scores.asType(.float32), axis: -1).asType(x.dtype)
            output = matmul(probabilities, v)
        } else if rope != nil {
            output = attentionWithCacheUpdate(queries: q, keys: k, values: v,
                                              cache: cache, scale: pow(Float(headDim), -0.5), mask: mask)
        } else {
            // Match the reference acoustic head's explicit score/softmax path.
            if kvHeads < heads { k = repeated(k, count: heads / kvHeads, axis: 1); v = repeated(v, count: heads / kvHeads, axis: 1) }
            output = matmul(softmax(matmul(q, k.transposed(0,1,3,2)) * pow(Float(headDim), -0.5), axis: -1), v)
        }
        return wo(output.transposed(0,2,1,3).reshaped(b,t,-1))
    }
}

final class VoxtralTTSBlock: Module {
    let attention: VoxtralTTSAttention
    @ModuleInfo(key: "feed_forward") var feedForward: VoxtralTTSFeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ParameterInfo(key: "attention_scale") var attentionScale: MLXArray?
    @ParameterInfo(key: "ffn_scale") var ffnScale: MLXArray?
    init(dim: Int, hidden: Int, heads: Int, kvHeads: Int, headDim: Int, eps: Float,
         bias: Bool = false, ropeTheta: Float? = nil, qkEps: Float? = nil, layerScale: Float? = nil, fusedFeedForward: Bool = false) {
        attention = VoxtralTTSAttention(dim: dim, heads: heads, kvHeads: kvHeads, headDim: headDim,
                                       bias: bias, ropeTheta: ropeTheta, qkEps: qkEps)
        _feedForward.wrappedValue = VoxtralTTSFeedForward(dim: dim, hidden: hidden, bias: bias, fusedActivation: fusedFeedForward)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        if let layerScale { _attentionScale.wrappedValue = .ones([dim]) * layerScale; _ffnScale.wrappedValue = .ones([dim]) * layerScale }
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache? = nil,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
                        slopes: MLXArray? = nil, window: Int = 0) -> MLXArray {
        var a = attention(attentionNorm(x), cache: cache, mask: mask, slopes: slopes, window: window)
        if let attentionScale { a = a * attentionScale }
        let h = x + a
        var f = feedForward(ffnNorm(h))
        if let ffnScale { f = f * ffnScale }
        return h + f
    }
}

final class VoxtralTTSBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [VoxtralTTSBlock]
    let norm: RMSNorm
    init(_ c: VoxtralTTSConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.dim)
        layers = (0..<c.layers).map { _ in VoxtralTTSBlock(dim: c.dim, hidden: c.hiddenDim,
            heads: c.heads, kvHeads: c.kvHeads, headDim: c.headDim, eps: c.normEps,
            bias: c.useBiases, ropeTheta: c.ropeTheta, fusedFeedForward: true) }
        norm = RMSNorm(dimensions: c.dim, eps: c.normEps)
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache.first)
        for (i, layer) in layers.enumerated() { h = layer(h, cache: cache[i], mask: mask) }
        return norm(h)
    }
}

final class VoxtralTTSAcousticHead: Module {
    @ModuleInfo(key: "input_projection") var inputProjection: Linear
    @ModuleInfo(key: "llm_projection") var llmProjection: Linear
    @ModuleInfo(key: "time_projection") var timeProjection: Linear
    @ModuleInfo(key: "semantic_codebook_output") var semanticOutput: Linear
    @ModuleInfo(key: "acoustic_codebook_output") var acousticOutput: Linear
    let layers: [VoxtralTTSBlock]
    let norm: RMSNorm
    let configuration: VoxtralTTSConfiguration
    private let _frequencies: MLXArray
    init(_ c: VoxtralTTSConfiguration) {
        configuration = c
        _inputProjection.wrappedValue = Linear(c.acousticCodebooks, c.acousticDim, bias: false)
        _llmProjection.wrappedValue = Linear(c.dim, c.acousticDim, bias: false)
        _timeProjection.wrappedValue = Linear(c.acousticDim, c.acousticDim, bias: false)
        _semanticOutput.wrappedValue = Linear(c.dim, (c.semanticSize / 128 + 1) * 128, bias: false)
        _acousticOutput.wrappedValue = Linear(c.acousticDim, c.acousticCodebooks, bias: false)
        layers = (0..<c.acousticLayers).map { _ in VoxtralTTSBlock(dim: c.acousticDim, hidden: c.acousticHiddenDim,
            heads: c.acousticHeads, kvHeads: c.acousticKVHeads, headDim: c.acousticHeadDim,
            eps: c.normEps, bias: c.useBiases) }
        norm = RMSNorm(dimensions: c.acousticDim, eps: c.normEps)
        let half = c.acousticDim / 2
        _frequencies = exp(MLXArray(0..<half).asType(.float32) * (-log(Float(10000)) / Float(half)))
    }
    func velocity(_ noise: MLXArray, time: Float, hidden: MLXArray) -> MLXArray {
        let phase = _frequencies * time
        let timeEmbedding = concatenated([cos(phase), sin(phase)]).reshaped(1,-1)
        let timeFeatures = broadcast(timeProjection(timeEmbedding), to: [noise.dim(0), configuration.acousticDim])
        var h = stacked([inputProjection(noise), timeFeatures, llmProjection(hidden)], axis: 1)
        for layer in layers { h = layer(h) }
        return acousticOutput(norm(h)[0...,0,0...])
    }
    func semantic(_ hidden: MLXArray) -> MLXArray {
        let logits = semanticOutput(hidden).asType(.float32)
        let ids = MLXArray(0..<logits.dim(-1))
        let allowed = logicalAnd(ids .> 0, ids .< configuration.semanticSize + 2)
        return argMax(MLX.where(allowed, logits, MLXArray(Float(-1e9))), axis: -1)
    }
    func decodeFrame(_ hidden: MLXArray, options: VoxtralTTSGenerationOptions, noise: MLXArray? = nil) -> MLXArray {
        let c = configuration, b = hidden.dim(0)
        let semanticCode = semantic(hidden)
        var x = noise ?? MLXRandom.normal([b,c.acousticCodebooks]) * c.sigmaMax
        let conditioned = concatenated([hidden, .zeros(like: hidden)], axis: 0)
        let dt = 1 / Float(options.denoisingSteps - 1)
        for i in 0..<(options.denoisingSteps - 1) {
            let v = velocity(concatenated([x,x], axis: 0), time: Float(i) * dt, hidden: conditioned)
            x = x + (options.guidanceScale * v[0..<b] + (1 - options.guidanceScale) * v[b...]) * dt
        }
        let levels = Float(c.acousticSize - 1)
        let codes = clip(round((clip(x, min: -1, max: 1) + 1) * (levels / 2)), min: 0, max: levels).asType(.int32) + 2
        return concatenated([semanticCode.asType(.int32).reshaped(b,1), codes], axis: -1)
    }
}
