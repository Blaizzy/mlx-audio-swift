import MLX
import MLXNN

public class T5DecoderLayer: Module {
    @ModuleInfo(key: "self_attention") var selfAttention: T5Attention
    @ModuleInfo(key: "cross_attention") var crossAttention: T5Attention
    @ModuleInfo var dense: T5DenseActivation
    let ln1: RMSNorm
    let ln2: RMSNorm
    let ln3: RMSNorm

    public init(config: T5Config) {
        self._selfAttention.wrappedValue = T5Attention(config: config)
        self._crossAttention.wrappedValue = T5Attention(config: config)
        self._dense.wrappedValue = T5DenseActivation(config: config)
        self.ln1 = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self.ln2 = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self.ln3 = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
    }

    public func callAsFunction(
        _ x: MLXArray,
        memory: MLXArray,
        selfAttnMask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> (MLXArray, KVCache) {
        var x = x
        let y = ln1(x)
        let (selfAttnOut, newCache) = selfAttention(
            queries: y, keys: y, values: y,
            mask: selfAttnMask, cache: cache
        )
        x = x + selfAttnOut

        let z = ln2(x)
        let (crossAttnOut, _) = crossAttention(
            queries: z, keys: memory, values: memory
        )
        x = x + crossAttnOut

        let w = ln3(x)
        x = x + dense(w)

        return (x, newCache)
    }
}
