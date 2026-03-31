import MLX
import MLXNN

public class T5Decoder: Module {
    @ModuleInfo var layers: [T5DecoderLayer]
    let ln: RMSNorm
    @ModuleInfo(key: "relative_attention_bias") var positionBias: RelativePositionBias

    public init(config: T5Config) {
        self._layers.wrappedValue = (0 ..< config.numDecoderLayers).map { _ in
            T5DecoderLayer(config: config)
        }
        self.ln = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._positionBias.wrappedValue = RelativePositionBias(
            numHeads: config.numHeads,
            numBuckets: config.relativeAttentionNumBuckets,
            maxDistance: config.relativeAttentionMaxDistance,
            bidirectional: false
        )
    }

    public func callAsFunction(
        _ x: MLXArray,
        memory: MLXArray,
        cache: [KVCache?]? = nil
    ) -> (MLXArray, [KVCache]) {
        let T = x.shape[1]

        let offset: Int
        if let firstCache = cache?.first, let c = firstCache {
            offset = c.keys.shape[2]
        } else {
            offset = 0
        }

        var mask = positionBias(
            queryLength: T, keyLength: T + offset, offset: offset
        )

        if T > 1 {
            let causalMask = MultiHeadAttention.createAdditiveCausalMask(T)
            mask = mask + causalMask
        }

        var h = x
        var newCaches = [KVCache]()
        newCaches.reserveCapacity(layers.count)

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[safe: i] ?? nil
            let (out, newCache) = layer(
                h, memory: memory, selfAttnMask: mask, cache: layerCache
            )
            h = out
            newCaches.append(newCache)
        }

        return (ln(h), newCaches)
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
