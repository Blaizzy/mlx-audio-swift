import Foundation
import MLX
import MLXNN

nonisolated final class BARTMultiHeadAttention: Module {
    let numHeads: Int
    let dModel: Int
    let headDim: Int
    
    let qProj: Linear
    let kProj: Linear
    let vProj: Linear
    let outProj: Linear
    
    init(dModel: Int, numHeads: Int, modelKey: String, weights: [String: MLXArray]) {
      self.dModel = dModel
      self.numHeads = numHeads
      self.headDim = dModel / numHeads
      
      self.qProj = Linear(weight: weights[modelKey + ".q_proj.weight"]!, bias:  weights[modelKey + ".q_proj.bias"])
      self.kProj = Linear(weight: weights[modelKey + ".k_proj.weight"]!, bias:  weights[modelKey + ".k_proj.bias"])
      self.vProj = Linear(weight: weights[modelKey + ".v_proj.weight"]!, bias:  weights[modelKey + ".v_proj.bias"])
      self.outProj = Linear(weight: weights[modelKey + ".out_proj.weight"]!, bias:  weights[modelKey + ".out_proj.bias"])
        
      super.init()
    }
    
    func callAsFunction(_ query: MLXArray, key: MLXArray? = nil, value: MLXArray? = nil, mask: MLXArray? = nil) -> MLXArray {
      let key = key ?? query
      let value = value ?? query
            
      let batchSize = query.shape[0]
      let seqLen = query.shape[1]
        
      // Project and reshape
      let q = qProj(query).reshaped([batchSize, seqLen, numHeads, headDim]).transposed(0, 2, 1, 3)
      let k = kProj(key).reshaped([batchSize, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
      let v = vProj(value).reshaped([batchSize, -1, numHeads, headDim]).transposed(0, 2, 1, 3)
        
      // Scaled dot-product attention
      let scale = Float(1.0 / sqrt(Double(headDim)))
      var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        
      if let mask {
          scores = scores + mask
      }
        
      let attnWeights = softmax(scores, axis: -1)
      let attnOutput = matmul(attnWeights, v)
      
      // Reshape and project
      let output = attnOutput.transposed(0, 2, 1, 3).reshaped([batchSize, seqLen, dModel])
      return outProj(output)
    }
}
