import Foundation
@preconcurrency import MLX
import MLXNN

public class UpSample1d: Module {
    public let layerType: String

    public init(layerType: String = "none") {
        self.layerType = layerType
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if layerType == "none" { return x }
        return Upsample(scaleFactor: .float(2), mode: .nearest)(x)
    }
}
