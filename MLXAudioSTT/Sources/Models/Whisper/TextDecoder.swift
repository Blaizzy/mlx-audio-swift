import Foundation
import MLX
import MLXNN

/// Whisper Text Decoder
/// Generates tokens autoregressively with cross-attention to encoder output
/// Captures cross-attention weights for AlignAtt streaming
public class TextDecoder: Module {
    @ModuleInfo public var tokenEmbedding: Embedding
    @ParameterInfo(key: "positional_embedding") public var positionalEmbedding: MLXArray
    @ModuleInfo(key: "blocks") public var blocks: [ResidualAttentionBlock]
    @ModuleInfo public var ln: LayerNorm

    let config: WhisperConfiguration

    public init(config: WhisperConfiguration) {
        self.config = config

        // Token embedding: nVocab -> nTextState
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.nVocab,
            dimensions: config.nTextState
        )

        // Learned positional embedding
        self._positionalEmbedding.wrappedValue = MLXArray.zeros([config.nTextCtx, config.nTextState])

        // Transformer blocks (decoder with cross-attention)
        self._blocks.wrappedValue = (0..<config.nTextLayer).map { _ in
            ResidualAttentionBlock(
                nState: config.nTextState,
                nHead: config.nTextHead,
                crossAttention: true
            )
        }

        // Final layer normalization
        self._ln.wrappedValue = LayerNorm(dimensions: config.nTextState)
    }

    /// Forward pass through the text decoder
    /// - Parameters:
    ///   - tokens: Token IDs [batch, seq]
    ///   - encoderOutput: Audio encoder output [batch, encoderSeq, nState]
    ///   - kvCache: Optional KV cache for incremental decoding
    /// - Returns: (logits, crossAttentionWeights) where logits is [batch, seq, nVocab]
    ///           and crossAttentionWeights is an array of weights from each layer
    public func callAsFunction(
        tokens: MLXArray,
        encoderOutput: MLXArray,
        kvCache: [KVCache]? = nil
    ) -> (MLXArray, [MLXArray]) {
        let seqLen = tokens.shape[1]
        let offset = kvCache?.first?.sequenceLength ?? 0

        // Token + positional embeddings
        var x = tokenEmbedding(tokens)
        x = x + positionalEmbedding[offset..<(offset + seqLen)]

        // Causal mask for autoregressive attention
        let mask = WhisperMultiHeadAttention.causalMask(size: offset + seqLen)
        let effectiveMask = offset > 0 ? mask[offset..., 0...] : mask

        // Pass through transformer blocks, collecting cross-attention weights
        var crossAttentionWeights: [MLXArray] = []

        for (i, block) in blocks.enumerated() {
            let cache = kvCache?[i]
            let (output, weights) = block(x: x, xa: encoderOutput, mask: effectiveMask, kvCache: cache)
            x = output

            // Collect cross-attention weights for AlignAtt
            if let weights = weights {
                crossAttentionWeights.append(weights)
            }
        }

        // Final layer norm
        x = ln(x)

        // Project to vocabulary logits using tied embeddings
        let logits = matmul(x, tokenEmbedding.weight.T)

        return (logits, crossAttentionWeights)
    }
}
