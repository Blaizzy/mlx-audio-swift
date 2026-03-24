import Foundation
import MLX
import MLXNN

final class BARTLayerNorm : LayerNorm {
  public init(dimensions: Int, weight: MLXArray, bias: MLXArray) {
    super.init(dimensions: dimensions)
    
    for i in 0..<dimensions {
      self.weight![i] = weight[i]
      self.bias![i] = bias[i]
    }
  }
}
