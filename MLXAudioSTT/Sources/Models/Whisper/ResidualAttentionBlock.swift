import Foundation
import MLX
import MLXNN

/// Transformer block with residual connections
/// Used in both AudioEncoder (self-attention only) and TextDecoder (self + cross attention)
public class ResidualAttentionBlock: Module {
    @ModuleInfo public var attnLn: LayerNorm
    @ModuleInfo public var attn: WhisperMultiHeadAttention

    @ModuleInfo public var crossAttnLn: LayerNorm?
    @ModuleInfo public var crossAttn: WhisperMultiHeadAttention?

    @ModuleInfo public var mlpLn: LayerNorm
    @ModuleInfo public var mlp1: Linear
    @ModuleInfo public var mlp2: Linear

    let hasCrossAttention: Bool

    public init(nState: Int, nHead: Int, crossAttention: Bool = false) {
        self.hasCrossAttention = crossAttention

        // Self-attention
        self._attnLn.wrappedValue = LayerNorm(dimensions: nState)
        self._attn.wrappedValue = WhisperMultiHeadAttention(nState: nState, nHead: nHead)

        // Cross-attention (only for decoder)
        if crossAttention {
            self._crossAttnLn.wrappedValue = LayerNorm(dimensions: nState)
            self._crossAttn.wrappedValue = WhisperMultiHeadAttention(nState: nState, nHead: nHead)
        }

        // MLP with 4x expansion
        self._mlpLn.wrappedValue = LayerNorm(dimensions: nState)
        self._mlp1.wrappedValue = Linear(nState, nState * 4)
        self._mlp2.wrappedValue = Linear(nState * 4, nState)
    }

    /// Forward pass through the block
    /// - Parameters:
    ///   - x: Input tensor [batch, seq, nState]
    ///   - xa: Optional encoder output for cross-attention [batch, enc_seq, nState]
    ///   - mask: Optional attention mask for causal attention
    ///   - kvCache: Optional KV cache for incremental decoding
    /// - Returns: (output, crossAttentionWeights) - weights returned only for decoder blocks
    public func callAsFunction(
        x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: KVCache? = nil
    ) -> (MLXArray, MLXArray?) {
        // Self-attention with residual (pre-norm architecture)
        let (attnOut, _) = attn(x: attnLn(x), mask: mask, kvCache: kvCache)
        var hidden = x + attnOut

        // Cross-attention with residual (decoder only)
        var crossAttnWeights: MLXArray?
        if hasCrossAttention,
           let crossAttnLayer = crossAttn,
           let crossAttnLnLayer = crossAttnLn,
           let encoderOutput = xa
        {
            let (crossOut, weights) = crossAttnLayer(x: crossAttnLnLayer(hidden), xa: encoderOutput)
            hidden = hidden + crossOut
            crossAttnWeights = weights
        }

        // MLP with residual (pre-norm architecture)
        let mlpInput = mlpLn(hidden)
        let mlpOut = mlp2(gelu(mlp1(mlpInput)))
        hidden = hidden + mlpOut

        return (hidden, crossAttnWeights)
    }
}
