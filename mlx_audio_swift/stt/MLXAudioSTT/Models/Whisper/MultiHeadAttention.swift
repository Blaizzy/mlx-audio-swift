import Foundation
import MLX
import MLXNN

/// Multi-head attention with optional cross-attention weight capture for Whisper STT.
/// Used in both AudioEncoder (self-attention) and TextDecoder (self + cross attention).
///
/// Key differences from standard MLXNN.MultiHeadAttention:
/// - Returns cross-attention weights for AlignAtt streaming
/// - Key projection has no bias (matching OpenAI Whisper)
/// - Supports KV caching for incremental decoding
public class WhisperMultiHeadAttention: Module {
    let nHead: Int
    let nState: Int
    let headDim: Int

    @ModuleInfo public var query: Linear
    @ModuleInfo public var key: Linear
    @ModuleInfo public var value: Linear
    @ModuleInfo public var out: Linear

    public init(nState: Int, nHead: Int) {
        precondition(nState % nHead == 0, "nState must be divisible by nHead")

        self.nState = nState
        self.nHead = nHead
        self.headDim = nState / nHead

        self._query.wrappedValue = Linear(nState, nState)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState)
        self._out.wrappedValue = Linear(nState, nState)
    }

    /// Forward pass with optional cross-attention
    /// - Parameters:
    ///   - x: Query tensor, shape [batch, seq, nState]
    ///   - xa: Optional key/value tensor for cross-attention, shape [batch, kv_seq, nState]
    ///   - mask: Optional attention mask
    ///   - kvCache: Optional KV cache for incremental decoding
    /// - Returns: (output, crossAttentionWeights) where weights are returned only for cross-attention
    public func callAsFunction(
        x: MLXArray,
        xa: MLXArray? = nil,
        mask: MLXArray? = nil,
        kvCache: KVCache? = nil
    ) -> (MLXArray, MLXArray?) {
        let q = query(x)

        let k: MLXArray
        let v: MLXArray

        if let xa = xa {
            // Cross-attention: use xa for key/value
            k = key(xa)
            v = value(xa)
        } else if let cache = kvCache {
            // Self-attention with KV cache for incremental decoding
            let newK = key(x)
            let newV = value(x)
            (k, v) = cache.update(keys: newK, values: newV)
        } else {
            // Self-attention without cache
            k = key(x)
            v = value(x)
        }

        // Reshape for multi-head attention: [batch, seq, nState] -> [batch, nHead, seq, headDim]
        let batchSize = q.shape[0]
        let qSeq = q.shape[1]
        let kvSeq = k.shape[1]

        let qReshaped = q.reshaped([batchSize, qSeq, nHead, headDim]).transposed(0, 2, 1, 3)
        let kReshaped = k.reshaped([batchSize, kvSeq, nHead, headDim]).transposed(0, 2, 1, 3)
        let vReshaped = v.reshaped([batchSize, kvSeq, nHead, headDim]).transposed(0, 2, 1, 3)

        // Scaled dot-product attention
        let scale = Float(1.0 / sqrt(Double(headDim)))
        var scores = matmul(qReshaped, kReshaped.transposed(0, 1, 3, 2)) * scale

        // Apply mask if provided
        if let mask = mask {
            scores = scores + mask.asType(scores.dtype)
        }

        let weights = softmax(scores, axis: -1)
        let attended = matmul(weights, vReshaped)

        // Reshape back: [batch, nHead, seq, headDim] -> [batch, seq, nState]
        let outputReshaped = attended.transposed(0, 2, 1, 3).reshaped([batchSize, qSeq, nState])
        let result = out(outputReshaped)

        // Return cross-attention weights for AlignAtt streaming
        // Only cross-attention (xa != nil) returns weights
        let returnWeights = xa != nil ? weights : nil

        return (result, returnWeights)
    }

    /// Create a causal (lower triangular) mask for autoregressive attention
    /// - Parameter size: Sequence length
    /// - Returns: Mask tensor with -inf for positions to ignore
    public static func causalMask(size: Int) -> MLXArray {
        let mask = triu(MLXArray.full([size, size], values: MLXArray(-1e9)), k: 1)
        return mask
    }
}

/// Key-Value cache for incremental decoding in autoregressive generation
public class KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?

    public init() {}

    /// Update cache with new keys and values, concatenating with existing cache
    /// - Parameters:
    ///   - keys: New key tensor, shape [batch, new_seq, dim]
    ///   - values: New value tensor, shape [batch, new_seq, dim]
    /// - Returns: Concatenated (keys, values) including history
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existingKeys = keys, let existingValues = values {
            keys = concatenated([existingKeys, newKeys], axis: 1)
            values = concatenated([existingValues, newValues], axis: 1)
        } else {
            keys = newKeys
            values = newValues
        }
        return (keys!, values!)
    }

    /// Reset the cache, clearing all stored keys and values
    public func reset() {
        keys = nil
        values = nil
    }
}
