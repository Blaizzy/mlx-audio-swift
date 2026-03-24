import MLX
import MLXNN

public class T5EncoderLayer: Module {
    @ModuleInfo var attention: T5Attention
    @ModuleInfo var dense: T5DenseActivation
    let ln1: RMSNorm
    let ln2: RMSNorm

    public init(config: T5Config) {
        self._attention.wrappedValue = T5Attention(config: config)
        self._dense.wrappedValue = T5DenseActivation(config: config)
        self.ln1 = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
        self.ln2 = RMSNorm(dimensions: config.dModel, eps: config.layerNormEpsilon)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var x = x
        let y = ln1(x)
        let (attnOut, _) = attention(queries: y, keys: y, values: y, mask: mask)
        x = x + attnOut
        let z = ln2(x)
        x = x + dense(z)
        return x
    }
}
