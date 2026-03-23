import Foundation
import MLX
import MLXNN

nonisolated final class BARTDecoderLayer: Module {
  let selfAttn: BARTMultiHeadAttention
  let selfAttnNorm: LayerNorm
  let crossAttn: BARTMultiHeadAttention
  let crossAttnNorm: LayerNorm
  let ffn: FeedForward
  let ffnNorm: LayerNorm

  init(dModel: Int, numHeads: Int, dFF: Int, modelKey: String, weights: [String: MLXArray]) {
    self.selfAttn = BARTMultiHeadAttention(dModel: dModel, numHeads: numHeads, modelKey: modelKey + ".self_attn", weights: weights)
    self.selfAttnNorm = BARTLayerNorm(
      dimensions: dModel,
      weight: weights[modelKey + ".self_attn_layer_norm.weight"]!,
      bias: weights[modelKey + ".self_attn_layer_norm.bias"]!)
    
    self.crossAttn = BARTMultiHeadAttention(dModel: dModel, numHeads: numHeads, modelKey: modelKey + ".encoder_attn", weights: weights)
    self.crossAttnNorm = BARTLayerNorm(
      dimensions: dModel,
      weight: weights[modelKey + ".encoder_attn_layer_norm.weight"]!,
      bias: weights[modelKey + ".encoder_attn_layer_norm.bias"]!)
    
    self.ffn = FeedForward(
      weight1: weights[modelKey + ".fc1.weight"]!,
      bias1: weights[modelKey + ".fc1.bias"],
      weight2: weights[modelKey + ".fc2.weight"]!,
      bias2: weights[modelKey + ".fc2.bias"])
    
    self.ffnNorm = BARTLayerNorm(
      dimensions: dModel,
      weight: weights[modelKey + ".final_layer_norm.weight"]!,
      bias: weights[modelKey + ".final_layer_norm.bias"]!)
    
    super.init()
  }
    
    func callAsFunction(_ x: MLXArray, encoderOutput: MLXArray, selfMask: MLXArray? = nil, crossMask: MLXArray? = nil) -> MLXArray {
      // Self-attention with residual
      let attnOutput = selfAttn(x, mask: selfMask)
      var output = selfAttnNorm(x + attnOutput)
            
      // Cross-attention with residual
      let crossOutput = crossAttn(output, key: encoderOutput, value: encoderOutput, mask: crossMask)
      output = crossAttnNorm(output + crossOutput)
                  
      // Feed-forward with residual
      let ffnOutput = ffn(output)
      output = ffnNorm(output + ffnOutput)
      
      return output
    }
}
