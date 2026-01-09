import Foundation
import MLX
import MLXFast
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

        let scale = Float(1.0 / sqrt(Double(headDim)))
        let attended: MLXArray
        let returnWeights: MLXArray?

        if xa != nil {
            // Cross-attention: use manual attention to capture weights for AlignAtt streaming
            var scores = matmul(qReshaped, kReshaped.transposed(0, 1, 3, 2)) * scale
            if let mask = mask {
                scores = scores + mask.asType(scores.dtype)
            }
            let weights = softmax(scores, axis: -1)
            attended = matmul(weights, vReshaped)
            returnWeights = weights
        } else {
            // Self-attention: use fused kernel (30-50% faster)
            attended = MLXFast.scaledDotProductAttention(
                queries: qReshaped,
                keys: kReshaped,
                values: vReshaped,
                scale: scale,
                mask: mask
            )
            returnWeights = nil
        }

        // Reshape back: [batch, nHead, seq, headDim] -> [batch, seq, nState]
        let outputReshaped = attended.transposed(0, 2, 1, 3).reshaped([batchSize, qSeq, nState])
        let result = out(outputReshaped)

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

/// Preallocated Key-Value cache for compile-friendly incremental decoding.
/// Uses fixed-shape arrays with offset tracking to avoid shape changes that would
/// trigger recompilation when using `mx.compile()`.
public class KVCache {
    private var keys: MLXArray
    private var values: MLXArray
    private var offset: Int = 0
    private let lock = NSLock()

    public let maxSequenceLength: Int
    public let dim: Int

    /// Current sequence length in the cache (0 if empty)
    public var sequenceLength: Int {
        lock.withLock { offset }
    }

    /// Initialize cache with preallocated arrays
    /// - Parameters:
    ///   - batchSize: Batch size (typically 1)
    ///   - maxSequenceLength: Maximum sequence length (Whisper: 448 tokens)
    ///   - dim: Hidden dimension (nTextState)
    public init(batchSize: Int = 1, maxSequenceLength: Int = 448, dim: Int) {
        self.maxSequenceLength = maxSequenceLength
        self.dim = dim
        self.keys = MLXArray.zeros([batchSize, maxSequenceLength, dim])
        self.values = MLXArray.zeros([batchSize, maxSequenceLength, dim])
    }

    /// Update cache using slice assignment (fixed shape, no concat)
    /// - Parameters:
    ///   - keys: New key tensor, shape [batch, new_seq, dim]
    ///   - values: New value tensor, shape [batch, new_seq, dim]
    /// - Returns: (keys, values) containing all cached entries up to current offset
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        lock.withLock {
            let newSeqLen = newKeys.shape[1]
            let end = offset + newSeqLen

            // Slice assignment - shape stays fixed
            keys[0..., offset..<end, 0...] = newKeys
            values[0..., offset..<end, 0...] = newValues
            offset = end

            // Return only the valid portion
            return (keys[0..., 0..<offset, 0...], values[0..., 0..<offset, 0...])
        }
    }

    /// Reset the cache for a new transcription
    public func reset() {
        lock.withLock {
            offset = 0
            // Zero out the cache arrays to prevent stale data leakage
            keys = MLXArray.zeros([keys.shape[0], maxSequenceLength, dim])
            values = MLXArray.zeros([values.shape[0], maxSequenceLength, dim])
        }
    }

    /// Debug: Get current cache state for diagnostics
    public var debugState: String {
        lock.withLock {
            let keysMax = keys.max().item(Float.self)
            let valuesMax = values.max().item(Float.self)
            return "offset=\(offset), keysMax=\(String(format: "%.4f", keysMax)), valuesMax=\(String(format: "%.4f", valuesMax))"
        }
    }
}
