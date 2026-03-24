import MLX
import MLXNN

public class T5DenseActivation: Module {
    @ModuleInfo(key: "wi_0") var wi0: Linear
    @ModuleInfo(key: "wi_1") var wi1: Linear
    @ModuleInfo var wo: Linear

    public init(config: T5Config) {
        self._wi0.wrappedValue = Linear(config.dModel, config.dFf, bias: false)
        self._wi1.wrappedValue = Linear(config.dModel, config.dFf, bias: false)
        self._wo.wrappedValue = Linear(config.dFf, config.dModel, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        wo(gelu(wi0(x)) * wi1(x))
    }
}
