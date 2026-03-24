import Foundation
import MLX
import MLXNN

nonisolated final class FeedForward: Module {
    let linear1: Linear
    let linear2: Linear
    
  init(weight1: MLXArray, bias1: MLXArray?, weight2: MLXArray, bias2: MLXArray?) {
    self.linear1 = Linear(weight: weight1, bias: bias1)
    self.linear2 = Linear(weight: weight2, bias: bias2)
    super.init()
  }
    
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var output = linear1(x)
    output = gelu(output)
    output = linear2(output)
    return output
  }
}
