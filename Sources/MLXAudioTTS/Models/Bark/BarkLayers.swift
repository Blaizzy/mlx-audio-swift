import MLX
import MLXNN
import MLXAudioCore

final class BarkAttention: Module {
    @ModuleInfo(key: "att_proj") var input: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int
    let causal: Bool

    init(_ config: BarkStageConfig, causal: Bool) {
        heads = config.heads; self.causal = causal
        _input.wrappedValue = Linear(config.width, 3 * config.width, bias: config.bias)
        _output.wrappedValue = Linear(config.width, config.width, bias: config.bias)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache? = nil) -> MLXArray {
        let (b, t, d) = (x.dim(0), x.dim(1), x.dim(2))
        let qkv = split(input(x), parts: 3, axis: -1).map {
            $0.reshaped([b, t, heads, d / heads]).transposed(0, 2, 1, 3)
        }
        let (k, v) = cache?.update(keys: qkv[1], values: qkv[2]) ?? (qkv[1], qkv[2])
        let mask: MLXArray?
        if causal && t > 1 {
            let positions = MLXArray(0..<k.dim(2))
            let queries = MLXArray((k.dim(2) - t)..<k.dim(2))
            mask = positions.expandedDimensions(axis: 0) .<= queries.expandedDimensions(axis: 1)
        } else { mask = nil }
        let y = MLXFast.scaledDotProductAttention(queries: qkv[0], keys: k, values: v,
            scale: 1 / Float(d / heads).squareRoot(), mask: mask)
        return output(y.transposed(0, 2, 1, 3).reshaped([b, t, d]))
    }
}

final class BarkMLP: Module {
    @ModuleInfo(key: "in_proj") var input: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    init(_ config: BarkStageConfig) {
        _input.wrappedValue = Linear(config.width, 4 * config.width, bias: false)
        _output.wrappedValue = Linear(4 * config.width, config.width, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(gelu(input(x))) }
}

final class BarkBlock: Module {
    @ModuleInfo(key: "layernorm_1") var norm1: LayerNorm
    @ModuleInfo(key: "layernorm_2") var norm2: LayerNorm
    @ModuleInfo var attn: BarkAttention
    @ModuleInfo var mlp: BarkMLP
    init(_ config: BarkStageConfig, causal: Bool) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-5, bias: !causal)
        _norm2.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-5, bias: !causal)
        _attn.wrappedValue = BarkAttention(config, causal: causal)
        _mlp.wrappedValue = BarkMLP(config)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache? = nil) -> MLXArray {
        let h = x + attn(norm1(x), cache: cache)
        return h + mlp(norm2(h))
    }
}

final class BarkGPT: Module {
    @ModuleInfo(key: "input_embeds_layer") var embedding: Embedding
    @ModuleInfo(key: "position_embeds_layer") var positions: Embedding
    @ModuleInfo var layers: [BarkBlock]
    @ModuleInfo(key: "layernorm_final") var norm: LayerNorm
    @ModuleInfo(key: "lm_head") var head: Linear
    let config: BarkStageConfig
    init(_ config: BarkStageConfig) {
        self.config = config
        _embedding.wrappedValue = Embedding(embeddingCount: config.inputVocabSize, dimensions: config.width)
        _positions.wrappedValue = Embedding(embeddingCount: config.blockSize, dimensions: config.width)
        _layers.wrappedValue = (0..<config.layers).map { _ in BarkBlock(config, causal: true) }
        _norm.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-5, bias: false)
        _head.wrappedValue = Linear(config.width, config.outputVocabSize, bias: false)
    }
    func makeCache() -> [KVCache] { layers.map { _ in KVCacheSimple() } }
    func callAsFunction(_ tokens: MLXArray, mergeContext: Bool = false, cache: [KVCache]? = nil) -> MLXArray {
        let offset = cache?.first?.offset ?? 0
        var x: MLXArray
        if mergeContext && offset == 0 {
            x = concatenated([
                embedding(tokens[0..., 0..<256]) + embedding(tokens[0..., 256..<512]),
                embedding(tokens[0..., 512...])], axis: 1)
        } else { x = embedding(tokens) }
        x = x + positions(MLXArray(offset..<(offset + x.dim(1))))
        for (i, layer) in layers.enumerated() { x = layer(x, cache: cache?[i]) }
        return head(norm(x)[0..., (-1)..., 0...])
    }
}

final class BarkFineGPT: Module {
    @ModuleInfo(key: "input_embeds_layers") var embeddings: [Embedding]
    @ModuleInfo(key: "position_embeds_layer") var positions: Embedding
    @ModuleInfo var layers: [BarkBlock]
    @ModuleInfo(key: "layernorm_final") var norm: LayerNorm
    @ModuleInfo(key: "lm_heads") var heads: [Linear]
    let config: BarkStageConfig
    init(_ config: BarkStageConfig) {
        self.config = config
        _embeddings.wrappedValue = (0..<config.codebooks).map { _ in
            Embedding(embeddingCount: config.inputVocabSize, dimensions: config.width)
        }
        _positions.wrappedValue = Embedding(embeddingCount: config.blockSize, dimensions: config.width)
        _layers.wrappedValue = (0..<config.layers).map { _ in BarkBlock(config, causal: false) }
        _norm.wrappedValue = LayerNorm(dimensions: config.width, eps: 1e-5)
        _heads.wrappedValue = (config.givenCodebooks..<config.codebooks).map { _ in
            Linear(config.width, config.outputVocabSize, bias: false)
        }
    }
    func callAsFunction(_ tokens: MLXArray, codebook: Int) -> MLXArray {
        var x = embeddings[0](tokens[0..., 0..., 0])
        for i in 1...codebook { x = x + embeddings[i](tokens[0..., 0..., i]) }
        x = x + positions(MLXArray(0..<tokens.dim(1)))
        for layer in layers { x = layer(x) }
        return heads[codebook - config.givenCodebooks](norm(x))
    }
}
