//
//  TalkerAttention.swift
//  Qwen3TTS
//
//  Multi-head attention with MRoPE support for Qwen3-TTS Talker.
//  Ported from mlx_audio/tts/models/qwen3_tts/talker.py
//

import Foundation
import MLX
import MLXNN

// MARK: - RMSNorm

/// RMS Layer Normalization.
///
/// Used for QK normalization in TalkerAttention.
/// This is a Module-based version with learnable weight parameter.
public class RMSNorm: Module, UnaryLayer {
    public let dims: Int
    public let eps: Float

    @ParameterInfo(key: "weight") var weight: MLXArray

    public init(dims: Int, eps: Float = 1e-6) {
        self.dims = dims
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dims])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Cast to float32 for numerical stability
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat * xFloat, axis: -1, keepDims: true)
        let xNormed = xFloat * rsqrt(variance + eps)
        return (weight * xNormed).asType(x.dtype)
    }
}

// Note: rotateHalf, applyRotaryPosEmb, applyMultimodalRotaryPosEmb are in Qwen3TTSUtils.swift

// MARK: - TalkerAttention

/// Multi-head attention with MRoPE support for Qwen3-TTS Talker.
///
/// Features:
/// - Grouped Query Attention (GQA): 16 query heads, 8 key-value heads
/// - QK normalization using RMSNorm (like Qwen3)
/// - Multimodal Rotary Position Embeddings (MRoPE)
/// - KV cache support for autoregressive generation
public class TalkerAttention: Module {
    public let hiddenSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let numKVGroups: Int
    public let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    public init(config: Qwen3TTSTalkerConfig, layerIdx: Int) {
        self.hiddenSize = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.numKVGroups = numHeads / numKVHeads
        self.scale = pow(Float(headDim), -0.5)

        // Q, K, V projections
        self._qProj.wrappedValue = Linear(
            hiddenSize,
            numHeads * headDim,
            bias: config.attentionBias
        )
        self._kProj.wrappedValue = Linear(
            hiddenSize,
            numKVHeads * headDim,
            bias: config.attentionBias
        )
        self._vProj.wrappedValue = Linear(
            hiddenSize,
            numKVHeads * headDim,
            bias: config.attentionBias
        )
        self._oProj.wrappedValue = Linear(
            numHeads * headDim,
            hiddenSize,
            bias: config.attentionBias
        )

        // QK normalization (like Qwen3)
        self._qNorm.wrappedValue = RMSNorm(dims: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dims: headDim, eps: config.rmsNormEps)
    }

    /// Forward pass for TalkerAttention.
    ///
    /// - Parameters:
    ///   - x: Input tensor [batch, seq_len, hidden_size]
    ///   - positionEmbeddings: (cos, sin) from TalkerRotaryEmbedding
    ///   - mask: Optional attention mask [batch, 1, seq_len, kv_len]
    ///   - cache: Optional KV cache for autoregressive generation
    /// - Returns: Output tensor [batch, seq_len, hidden_size]
    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil,
        cache: TalkerKVCache? = nil
    ) -> MLXArray {
        let batch = x.shape[0]
        let seqLen = x.shape[1]

        // Project to Q, K, V
        var q = qProj(x)
        var k = kProj(x)
        var v = vProj(x)

        // Reshape to [batch, seq_len, num_heads, head_dim]
        q = q.reshaped([batch, seqLen, numHeads, headDim])
        k = k.reshaped([batch, seqLen, numKVHeads, headDim])
        v = v.reshaped([batch, seqLen, numKVHeads, headDim])

        // Apply QK normalization
        q = qNorm(q)
        k = kNorm(k)

        // Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // Apply rotary embeddings (interleaving already done in TalkerRotaryEmbedding)
        let (cos, sin) = positionEmbeddings
        (q, k) = applyMultimodalRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)

        // Handle KV cache
        if let cache = cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        // Use fast scaled dot product attention with GQA support
        var output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        output = output.transposed(0, 2, 1, 3)
        output = output.reshaped([batch, seqLen, -1])

        return oProj(output)
    }
}

// MARK: - TalkerKVCache Protocol

/// Protocol for KV cache used in Talker attention layers.
///
/// This is a simplified cache protocol specific to Qwen3-TTS Talker.
/// Note: Named TalkerKVCache to avoid conflict with MLXLMCommon.KVCache.
public protocol TalkerKVCache {
    /// Update the cache with new keys and values.
    /// Returns the full keys and values (including cached).
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
}

/// Simple KV cache implementation for Talker autoregressive generation.
public class TalkerSimpleKVCache: TalkerKVCache {
    private var keys: MLXArray?
    private var values: MLXArray?

    public init() {}

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            // Concatenate along sequence dimension (axis 2)
            let updatedKeys = concatenated([existingKeys, newKeys], axis: 2)
            let updatedValues = concatenated([existingValues, newValues], axis: 2)
            self.keys = updatedKeys
            self.values = updatedValues
            return (updatedKeys, updatedValues)
        } else {
            self.keys = newKeys
            self.values = newValues
            return (newKeys, newValues)
        }
    }

    /// Reset the cache
    public func reset() {
        keys = nil
        values = nil
    }

    /// Get current sequence length in cache
    public var sequenceLength: Int {
        keys?.shape[2] ?? 0
    }
}
