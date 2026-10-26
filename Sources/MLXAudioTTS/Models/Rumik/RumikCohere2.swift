// Cohere2 architecture adapted from mlx-lm (Apple Inc.).
import Foundation
import MLX
import MLXNN
import MLXAudioCore

private let rumikSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray =
    MLX.compile(shapeless: true) { gate, up in silu(gate) * up }

final class RumikCohereAttention: Module {
    let config: RumikConfiguration
    let isLocal: Bool
    let rope: RoPE
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear

    init(_ c: RumikConfiguration, index: Int) {
        config = c
        isLocal = (index + 1) % c.slidingWindowPattern != 0
        rope = RoPE(dimensions: c.headDim, traditional: true, base: c.ropeTheta)
        _query.wrappedValue = Linear(c.hiddenSize, c.attentionHeads * c.headDim, bias: c.attentionBias)
        _key.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: c.attentionBias)
        _value.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: c.attentionBias)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.attentionBias)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        // Construct masks before updating the ring; single-token masks must honor ring order.
        let mask = createAttentionMask(h: x, cache: cache,
                                       windowSize: isLocal ? config.slidingWindow : nil)
        var q = query(x).reshaped(b, n, config.attentionHeads, config.headDim).transposed(0, 2, 1, 3)
        var k = key(x).reshaped(b, n, config.kvHeads, config.headDim).transposed(0, 2, 1, 3)
        let v = value(x).reshaped(b, n, config.kvHeads, config.headDim).transposed(0, 2, 1, 3)
        if isLocal {
            q = rope(q, offset: cache?.offset ?? 0)
            k = rope(k, offset: cache?.offset ?? 0)
        }
        // Match Cohere2's fp16 SDPA promotion while retaining the projected dtype.
        let dtype = q.dtype
        let y = attentionWithCacheUpdate(
            queries: dtype == .float16 ? q.asType(.float32) : q, keys: k, values: v,
            cache: cache, scale: pow(Float(config.headDim), -0.5), mask: mask)
            .asType(dtype).transposed(0, 2, 1, 3).reshaped(b, n, config.hiddenSize)
        return output(y)
    }
}

final class RumikCohereMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: RumikConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(rumikSwiGLU(gate(x), up(x))) }
}

final class RumikCohereBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: RumikCohereAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: LayerNorm
    let mlp: RumikCohereMLP
    init(_ c: RumikConfiguration, index: Int) {
        _attention.wrappedValue = RumikCohereAttention(c, index: index)
        _inputNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps, bias: c.layerNormBias)
        mlp = RumikCohereMLP(c)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let h = inputNorm(x)
        return attention(h, cache: cache) + mlp(h) + x
    }
}

final class RumikCohereModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedding: Embedding
    let layers: [RumikCohereBlock]
    let norm: LayerNorm
    init(_ c: RumikConfiguration) {
        _embedding.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        layers = (0..<c.hiddenLayers).map { RumikCohereBlock(c, index: $0) }
        norm = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps, bias: c.layerNormBias)
    }
    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedding(tokens)
        for (i, layer) in layers.enumerated() { h = layer(h, cache: cache?[i]) }
        return norm(h)
    }
}

/// An audio-only tied projection retaining packed weights for quantized checkpoints.
struct RumikAudioHead {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    init(embedding: Embedding, config: RumikConfiguration) {
        let ids = MLXArray((config.firstUnitID...config.lastUnitID).map(Int32.init) + [Int32(config.audioEndTokenID)])
        weight = embedding.weight[ids]
        if let q = embedding as? QuantizedEmbedding {
            scales = q.scales[ids]
            biases = q.biases.map { $0[ids] }
            groupSize = q.groupSize; bits = q.bits; mode = q.mode
        } else {
            scales = nil; biases = nil; groupSize = 64; bits = 4; mode = .affine
        }
        eval(weight)
        if let scales { eval(scales) }
        if let biases { eval(biases) }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let scales {
            return quantizedMM(x, weight, scales: scales, biases: biases,
                               groupSize: groupSize, bits: bits, mode: mode)
        }
        return matmul(x, weight.T)
    }
}
