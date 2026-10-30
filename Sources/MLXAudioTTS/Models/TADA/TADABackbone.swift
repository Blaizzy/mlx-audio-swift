import Foundation
import MLX
import MLXNN

final class TADAMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ dim: Int, _ inner: Int) {
        _gate = ModuleInfo(wrappedValue: Linear(dim, inner, bias: false), key: "gate_proj")
        _up = ModuleInfo(wrappedValue: Linear(dim, inner, bias: false), key: "up_proj")
        _down = ModuleInfo(wrappedValue: Linear(inner, dim, bias: false), key: "down_proj")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}
final class TADAAttention: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var out: Linear
    let c: TADAConfiguration
    private let _frequencies: MLXArray?
    init(_ c: TADAConfiguration) {
        self.c = c
        _frequencies = c.ropeScaling.map { scaling in
            MLXArray((0 ..< c.headDim / 2).map { i -> Float in
                let frequency = pow(c.ropeTheta, Float(2 * i) / Float(c.headDim))
                let wavelength = 2 * Float.pi * frequency
                if wavelength > scaling.originalContext / scaling.lowFreqFactor { return frequency * scaling.factor }
                if wavelength < scaling.originalContext / scaling.highFreqFactor { return frequency }
                let smooth = (scaling.originalContext / wavelength - scaling.lowFreqFactor) / (scaling.highFreqFactor - scaling.lowFreqFactor)
                return frequency / ((1 - smooth) / scaling.factor + smooth)
            })
        }
        _q = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.numAttentionHeads * c.headDim, bias: false), key: "q_proj")
        _k = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.numKeyValueHeads * c.headDim, bias: false), key: "k_proj")
        _v = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.numKeyValueHeads * c.headDim, bias: false), key: "v_proj")
        _out = ModuleInfo(wrappedValue: Linear(c.numAttentionHeads * c.headDim, c.hiddenSize, bias: false), key: "o_proj")
    }
    func callAsFunction(_ x: MLXArray, cache: inout (MLXArray, MLXArray)?) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), offset = cache?.0.dim(2) ?? 0
        var queries = q(x).reshaped(b, n, c.numAttentionHeads, c.headDim).transposed(0, 2, 1, 3)
        var keys = k(x).reshaped(b, n, c.numKeyValueHeads, c.headDim).transposed(0, 2, 1, 3)
        var values = v(x).reshaped(b, n, c.numKeyValueHeads, c.headDim).transposed(0, 2, 1, 3)
        queries = MLXFast.RoPE(queries, dimensions: c.headDim, traditional: false, base: _frequencies == nil ? c.ropeTheta : nil, scale: 1, offset: offset, freqs: _frequencies)
        keys = MLXFast.RoPE(keys, dimensions: c.headDim, traditional: false, base: _frequencies == nil ? c.ropeTheta : nil, scale: 1, offset: offset, freqs: _frequencies)
        if let previous = cache {
            keys = concatenated([previous.0, keys], axis: 2)
            values = concatenated([previous.1, values], axis: 2)
        }
        cache = (keys, values)
        let mask: MLXArray? = n > 1 ? MLXArray(0) + MLX.where(
            MLXArray(0 ..< n).expandedDimensions(axis: 1) + offset .>= MLXArray(0 ..< (offset + n)),
            MLXArray(Float(0)), MLXArray(-Float.infinity)) : nil
        let y = MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values,
            scale: 1 / sqrt(Float(c.headDim)), mask: mask)
        return out(y.transposed(0, 2, 1, 3).reshaped(b, n, -1))
    }
}
final class TADALayer: Module {
    @ModuleInfo(key: "self_attn") var attention: TADAAttention
    let mlp: TADAMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(_ c: TADAConfiguration) {
        _attention = ModuleInfo(wrappedValue: TADAAttention(c), key: "self_attn")
        mlp = TADAMLP(c.hiddenSize, c.intermediateSize)
        _inputNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "input_layernorm")
        _postNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "post_attention_layernorm")
    }
    func callAsFunction(_ x: MLXArray, cache: inout (MLXArray, MLXArray)?) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: &cache)
        return h + mlp(postNorm(h))
    }
}
final class TADALlama: Module {
    @ModuleInfo(key: "embed_tokens") var embeddings: Embedding
    let layers: [TADALayer]
    let norm: RMSNorm
    init(_ c: TADAConfiguration) {
        _embeddings = ModuleInfo(wrappedValue: Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize), key: "embed_tokens")
        layers = (0 ..< c.numHiddenLayers).map { _ in TADALayer(c) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray, cache: inout [(MLXArray, MLXArray)?]) -> MLXArray {
        var h = x
        for i in layers.indices { h = layers[i](h, cache: &cache[i]) }
        return norm(h)
    }
}
final class TADAHeadLayer: Module {
    let ffn: TADAMLP
    let norm: RMSNorm
    @ModuleInfo(key: "adaLN_modulation_linear") var modulation: Linear
    init(_ c: TADAConfiguration, dim: Int) {
        ffn = TADAMLP(dim, Int(Float(dim) * c.headFFNRatio))
        norm = RMSNorm(dimensions: dim, eps: c.rmsNormEps)
        _modulation = ModuleInfo(wrappedValue: Linear(dim, 3 * dim, bias: false), key: "adaLN_modulation_linear")
    }
    func callAsFunction(_ x: MLXArray, _ condition: MLXArray) -> MLXArray {
        let m = split(modulation(silu(condition)), parts: 3, axis: -1)
        return x + m[2] * ffn(norm(x) * (1 + m[1]) + m[0])
    }
}
final class TADATimeEmbedder: Module {
    let mlp_0: Linear
    let mlp_2: Linear
    init(_ dim: Int) { mlp_0 = Linear(256, dim, bias: false); mlp_2 = Linear(dim, dim, bias: false) }
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let frequencies = exp(-log(Float(10000)) * MLXArray(0 ..< 128).asType(.float32) / 128)
        let phase = t.asType(.float32).expandedDimensions(axis: -1) * frequencies
        return mlp_2(silu(mlp_0(concatenated([cos(phase), sin(phase)], axis: -1))))
    }
}
final class TADAFinalHead: Module {
    let linear: Linear
    @ModuleInfo(key: "adaLN_modulation_linear") var modulation: Linear
    let eps: Float
    init(_ c: TADAConfiguration, dim: Int) {
        eps = c.rmsNormEps
        linear = Linear(dim, c.latentSize, bias: false)
        _modulation = ModuleInfo(wrappedValue: Linear(dim, 2 * dim, bias: false), key: "adaLN_modulation_linear")
    }
    func callAsFunction(_ x: MLXArray, _ condition: MLXArray) -> MLXArray {
        let m = split(modulation(silu(condition)), parts: 2, axis: -1)
        let normalized = x * rsqrt(mean(square(x.asType(.float32)), axis: -1, keepDims: true) + eps).asType(x.dtype)
        return linear(normalized * (1 + m[1]) + m[0])
    }
}
final class TADADiffusionHead: Module {
    @ModuleInfo(key: "noisy_images_proj") var noisy: Linear
    @ModuleInfo(key: "cond_proj") var cond: Linear
    @ModuleInfo(key: "t_embedder") var time: TADATimeEmbedder
    let layers: [TADAHeadLayer]
    @ModuleInfo(key: "final_layer") var final: TADAFinalHead
    init(_ c: TADAConfiguration) {
        let dim = c.bottleneckDim ?? c.hiddenSize
        _noisy = ModuleInfo(wrappedValue: Linear(c.latentSize, dim, bias: false), key: "noisy_images_proj")
        _cond = ModuleInfo(wrappedValue: Linear(dim, dim, bias: false), key: "cond_proj")
        _time = ModuleInfo(wrappedValue: TADATimeEmbedder(dim), key: "t_embedder")
        layers = (0 ..< c.headLayers).map { _ in TADAHeadLayer(c, dim: dim) }
        _final = ModuleInfo(wrappedValue: TADAFinalHead(c, dim: dim), key: "final_layer")
    }
    func callAsFunction(_ x: MLXArray, time t: MLXArray, condition: MLXArray) -> MLXArray {
        let c = cond(condition) + time(t)
        var h = noisy(x)
        for layer in layers { h = layer(h, c) }
        return final(h, c)
    }
}
