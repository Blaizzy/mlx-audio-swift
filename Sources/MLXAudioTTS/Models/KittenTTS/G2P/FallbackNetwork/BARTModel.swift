import Foundation
import MLX
import MLXNN

nonisolated final class BARTModel: Module {
  let config: BARTConfig
  let sharedEmbedding: Embedding
  let encoderPositionalEmbedding: Embedding
  let decoderPositionalEmbedding: Embedding
  let encoderLayers: [BARTEncoderLayer]
  let decoderLayers: [BARTDecoderLayer]
  let encoderNorm: LayerNorm
  let decoderNorm: LayerNorm
  let lmHead: Linear
  let logitBias: MLXArray
    
  init(config: BARTConfig, weights: [String: MLXArray]) {
    self.config = config
    
    // Shared embedding for encoder and decoder
    self.sharedEmbedding = Embedding(weight: weights["model.shared.weight"]!)
    
    // Positional embeddings
    self.encoderPositionalEmbedding = Embedding(weight: weights["model.encoder.embed_positions.weight"]!)
    self.decoderPositionalEmbedding = Embedding(weight: weights["model.decoder.embed_positions.weight"]!)
      
    // Encoder layers
    self.encoderLayers = (0..<config.encoderLayers).map { index in
      BARTEncoderLayer(
        dModel: config.dModel,
        numHeads: config.encoderAttentionHeads,
        dFF: config.encoderFFNDim,
        modelKey: "model.encoder.layers.\(index)",
        weights: weights)
    }
      
    // Decoder layers
    self.decoderLayers = (0..<config.decoderLayers).map { index in
      BARTDecoderLayer(
        dModel: config.dModel,
        numHeads: config.decoderAttentionHeads,
        dFF: config.decoderFFNDim,
        modelKey: "model.decoder.layers.\(index)",
        weights: weights)
    }
      
    // Layer norms
    self.encoderNorm = BARTLayerNorm(
      dimensions: config.dModel,
      weight: weights["model.encoder.layernorm_embedding.weight"]!,
      bias: weights["model.encoder.layernorm_embedding.bias"]!
    )
    self.decoderNorm = BARTLayerNorm(
      dimensions: config.dModel,
      weight: weights["model.decoder.layernorm_embedding.weight"]!,
      bias: weights["model.decoder.layernorm_embedding.bias"]!
    )
      
    // Language model head
    self.lmHead = Linear(weight: weights["model.shared.weight"]!, bias: nil)
    
    // This is not used
    self.logitBias = weights["final_logits_bias"]!

    super.init()
  }
    
  func encode(_ inputIds: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let seqLen = inputIds.shape[1]
    let positions = MLXArray(0..<seqLen).reshaped([1, seqLen]) + 2
        
    // Embeddings
    var hidden = sharedEmbedding(inputIds)
    let embedPos = encoderPositionalEmbedding(positions)
            
    hidden = hidden + embedPos
    hidden = encoderNorm(hidden)
    
    // Encoder layers
    for layer in encoderLayers {
      hidden = layer(hidden, mask: mask)
    }
    
    return hidden
  }
    
  func decode(
    _ inputIds: MLXArray,
    encoderOutput: MLXArray,
    selfMask: MLXArray? = nil,
    crossMask: MLXArray? = nil) -> MLXArray
  {
    let seqLen = inputIds.shape[1]
    let positions = MLXArray(0..<seqLen).reshaped([1, seqLen]) + 2
    
    // Embeddings
    var hidden = sharedEmbedding(inputIds)
    let embedPositions = decoderPositionalEmbedding(positions)
    
    hidden = hidden + embedPositions
    hidden = decoderNorm(hidden)
    
    // Decoder layers
    for layer in decoderLayers {
      hidden = layer(hidden, encoderOutput: encoderOutput, selfMask: selfMask, crossMask: crossMask)
    }
    
    return lmHead(hidden) + logitBias
  }
    
  func generate(inputIds: MLXArray, maxLength: Int = 50, temperature: Float = 1.0) -> MLXArray {
    // Encode input
    let encoderOutput = encode(inputIds)
    
    // Start with BOS token
    var decoderInput = MLXArray([config.bosTokenId]).reshaped([1, 1])
    var generatedTokens: [Int32] = []
        
    for i in 0..<maxLength {
      if i == maxLength - 1 {
        generatedTokens.append(Int32(config.eosTokenId))
        break
      }
      
      // Decode next token
      let logits = decode(decoderInput, encoderOutput: encoderOutput)
      let nextTokenLogits = logits[0, logits.shape[1] - 1]
            
      // Apply temperature
      let scaledLogits = nextTokenLogits / temperature
      
      // Sample next token, take the max probability value
      let nextToken = scaledLogits.argMax().item(Int32.self)
      
      // Stop if EOS token
      if nextToken == config.eosTokenId {
          break
      }
      
      generatedTokens.append(nextToken)
      
      // Append to decoder input
      let newToken = MLXArray([nextToken]).reshaped([1, 1])
      decoderInput = concatenated([decoderInput, newToken], axis: 1)
    }
        
    return MLXArray(generatedTokens)
  }
}
