//
//  Perceiver.swift
//  MLXAudio
//
//  Perceiver-style cross-attention resampler for T3 conditioning.
//  Ported from mlx-audio Python: chatterbox/t3/perceiver.py
//

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Attention with separate Q/K/V

/// Multi-head attention with separate Q, K, V tensors.
class AttentionQKV: Module {
    let nHeads: Int
    let headDim: Int
    let scale: Float

    init(nHeads: Int, headDim: Int, dropoutRate: Float = 0.1, scale: Float? = nil) {
        self.nHeads = nHeads
        self.headDim = headDim
        self.scale = scale ?? pow(Float(headDim), -0.5)
    }

    /// Scaled dot-product attention.
    ///
    /// - Parameters:
    ///   - q: Query (B, T_q, nHeads * headDim)
    ///   - k: Key (B, T_k, nHeads * headDim)
    ///   - v: Value (B, T_v, nHeads * headDim)
    ///   - mask: Optional attention mask
    /// - Returns: Output (B, T_q, nHeads * headDim)
    func callAsFunction(q: MLXArray, k: MLXArray, v: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let qH = splitHeads(q)
        let kH = splitHeads(k)
        let vH = splitHeads(v)

        let out = MLXFast.scaledDotProductAttention(
            queries: qH, keys: kH, values: vH,
            scale: scale, mask: mask
        )
        return combineHeads(out)
    }

    /// (B, T, D) → (B, nHeads, T, headDim)
    private func splitHeads(_ x: MLXArray) -> MLXArray {
        let (b, t, _) = (x.dim(0), x.dim(1), x.dim(2))
        let reshaped = x.reshaped([b, t, nHeads, headDim])
        return reshaped.transposed(0, 2, 1, 3)
    }

    /// (B, nHeads, T, headDim) → (B, T, D)
    private func combineHeads(_ x: MLXArray) -> MLXArray {
        let (b, _, t, _) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let transposed = x.transposed(0, 2, 1, 3)
        return transposed.reshaped([b, t, -1])
    }
}

// MARK: - Attention Block

/// Cross-attention block with separate Q, K, V projections + residual.
class AttentionBlock: Module {
    let channels: Int
    let numHeads: Int

    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    let attention: AttentionQKV
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(channels: Int, numHeads: Int = 1, dropoutRate: Float = 0.2, scale: Float? = nil) {
        self.channels = channels
        self.numHeads = numHeads

        self._norm.wrappedValue = LayerNorm(dimensions: channels)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self.attention = AttentionQKV(
            nHeads: numHeads,
            headDim: channels / numHeads,
            dropoutRate: dropoutRate,
            scale: scale
        )
        self._projOut.wrappedValue = Linear(channels, channels)
    }

    /// Cross-attention from x1 (query) to x2 (key/value).
    ///
    /// - Parameters:
    ///   - x1: Query source (B, T1, C)
    ///   - x2: Key/Value source (B, T2, C)
    ///   - mask: Optional attention mask
    /// - Returns: Output (B, T1, C) with residual connection.
    func callAsFunction(_ x1: MLXArray, _ x2: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let x1Norm = norm(x1)
        let x2Norm = norm(x2)

        let q = toQ(x1Norm)
        let k = toK(x2Norm)
        let v = toV(x2Norm)

        let h = attention.callAsFunction(q: q, k: k, v: v, mask: mask)
        let out = projOut(h)

        return x1 + out
    }
}

// MARK: - Perceiver Resampler

/// Perceiver-style resampler that reduces variable-length input to fixed-length latent.
///
/// Uses learnable query tokens and a shared attention block for both
/// cross-attention (query → input) and self-attention (query → query).
public class Perceiver: Module {
    @ParameterInfo(key: "pre_attention_query") var preAttentionQuery: MLXArray
    @ModuleInfo var attn: AttentionBlock

    public init(
        preAttentionQueryToken: Int = 32,
        preAttentionQuerySize: Int = 1024,
        embeddingDim: Int = 1024,
        numAttnHeads: Int = 4
    ) {
        // Learnable query tokens with uniform initialization
        let queryVariance = Float(sqrt(3.0) * sqrt(2.0 / Double(preAttentionQueryToken + preAttentionQueryToken)))
        self._preAttentionQuery.wrappedValue = MLXRandom.uniform(
            low: -queryVariance,
            high: queryVariance,
            [1, preAttentionQueryToken, preAttentionQuerySize]
        )
        self._attn.wrappedValue = AttentionBlock(channels: embeddingDim, numHeads: numAttnHeads)
    }

    /// Resample variable-length input to fixed-length output.
    ///
    /// - Parameter h: Input embeddings (B, T, D) — variable length T.
    /// - Returns: Fixed-length output (B, queryTokens, D).
    public func callAsFunction(_ h: MLXArray) -> MLXArray {
        let b = h.dim(0)

        // Expand query to batch size
        let query = MLX.broadcast(preAttentionQuery, to: [b, preAttentionQuery.dim(1), preAttentionQuery.dim(2)])

        // Cross-attention: query attends to input
        let preAtt = attn(query, h)

        // Self-attention: query attends to itself
        let attnOut = attn(preAtt, preAtt)

        return attnOut
    }
}
