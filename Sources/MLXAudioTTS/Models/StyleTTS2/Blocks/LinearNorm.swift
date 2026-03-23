import Foundation
@preconcurrency import MLX
import MLXNN

public class LinearNorm: Module {
    @ModuleInfo(key: "linear_layer") public var linearLayer: Linear

    public init(inDim: Int, outDim: Int) {
        _linearLayer = ModuleInfo(wrappedValue: Linear(inDim, outDim), key: "linear_layer")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearLayer(x)
    }
}
