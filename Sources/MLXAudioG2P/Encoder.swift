import MLX
import MLXNN

public class T5Encoder: Module {
    @ModuleInfo var layers: [T5EncoderLayer]
    let ln: RMSNorm
    @ModuleInfo(key: "relative_attention_bias") var positionBias: RelativePositionBias

    public init(config: T5Config) {
        self._layers.wrappedValue = (0 ..< config.numLayers).map { _ in
            T5EncoderLayer(config: config)
        }
        self.ln = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self._positionBias.wrappedValue = RelativePositionBias(
            numHeads: config.numHeads,
            numBuckets: config.relativeAttentionNumBuckets,
            maxDistance: config.relativeAttentionMaxDistance,
            bidirectional: true
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let seqLen = x.shape[1]
        let bias = positionBias(queryLength: seqLen, keyLength: seqLen)
        var h = x
        for layer in layers {
            h = layer(h, mask: bias)
        }
        return ln(h)
    }
}
