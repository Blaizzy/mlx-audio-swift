import MLX
import MLXNN

public typealias KVCache = (keys: MLXArray, values: MLXArray)

public class T5Attention: Module {
    let numHeads: Int
    let dKv: Int

    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(config: T5Config) {
        self.numHeads = config.numHeads
        self.dKv = config.dKv
        let innerDim = config.innerDim

        self._queryProj.wrappedValue = Linear(config.dModel, innerDim, bias: false)
        self._keyProj.wrappedValue = Linear(config.dModel, innerDim, bias: false)
        self._valueProj.wrappedValue = Linear(config.dModel, innerDim, bias: false)
        self._outProj.wrappedValue = Linear(innerDim, config.dModel, bias: false)
    }

    public func callAsFunction(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXArray? = nil,
        cache: KVCache? = nil
    ) -> (MLXArray, KVCache) {
        let B = queries.shape[0]

        var q = queryProj(queries)
        var k = keyProj(keys)
        var v = valueProj(values)

        q = q.reshaped(B, -1, numHeads, dKv).transposed(0, 2, 1, 3)
        k = k.reshaped(B, -1, numHeads, dKv).transposed(0, 2, 1, 3)
        v = v.reshaped(B, -1, numHeads, dKv).transposed(0, 2, 1, 3)

        if let cache = cache {
            k = concatenated([cache.keys, k], axis: 2)
            v = concatenated([cache.values, v], axis: 2)
        }
        let newCache: KVCache = (keys: k, values: v)

        var scores = matmul(q, k.transposed(0, 1, 3, 2))

        if let mask = mask {
            scores = scores + mask
        }

        let weights = softmax(scores, axis: -1)
        var output = matmul(weights, v)

        output = output.transposed(0, 2, 1, 3).reshaped(B, -1, numHeads * dKv)
        return (outProj(output), newCache)
    }
}
