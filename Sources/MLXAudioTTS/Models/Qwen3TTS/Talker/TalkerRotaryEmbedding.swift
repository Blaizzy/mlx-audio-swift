//
//  TalkerRotaryEmbedding.swift
//  Qwen3TTS
//
//  Multimodal Rotary Position Embedding for Qwen3-TTS Talker.
//  Ported from mlx_audio/tts/models/qwen3_tts/talker.py
//

import Foundation
import MLX
import MLXNN

/// Multimodal Rotary Embedding for 3D positions (temporal, height, width).
///
/// Uses interleaved MRoPE layout for better spatial-temporal modeling.
/// The interleaving pattern combines T/H/W frequencies as:
/// - H at indices 1, 4, 7, ... (mod 3 == 1)
/// - W at indices 2, 5, 8, ... (mod 3 == 2)
/// - T at indices 0, 3, 6, ... and beyond the interleave region
public class TalkerRotaryEmbedding: Module {
    public let dim: Int
    public let maxPositionEmbeddings: Int
    public let base: Float
    public let mropeSection: [Int]

    private var _invFreq: MLXArray

    /// Initialize the rotary embedding.
    ///
    /// - Parameters:
    ///   - dim: Head dimension (typically 128)
    ///   - maxPositionEmbeddings: Maximum sequence length (default 32768)
    ///   - base: RoPE base frequency (default 10000.0 for standard, 1000000.0 for Qwen3-TTS)
    ///   - mropeSection: Dimensions for [temporal, height, width] (default [24, 20, 20])
    public init(
        dim: Int,
        maxPositionEmbeddings: Int = 32768,
        base: Float = 10000.0,
        mropeSection: [Int]? = nil
    ) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        self.mropeSection = mropeSection ?? [24, 20, 20]

        // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        let exponents = indices / Float(dim)
        self._invFreq = 1.0 / pow(MLXArray(base), exponents)
    }

    /// Apply interleaved MRoPE to 3D rotary embeddings.
    ///
    /// Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    /// interleaved [THWTHWTHW...TT], preserving frequency continuity.
    ///
    /// - Parameters:
    ///   - freqs: Frequencies tensor [3, batch, seq_len, head_dim // 2]
    ///   - mropeSection: [temporal_dims, height_dims, width_dims]
    /// - Returns: Combined frequencies [batch, seq_len, head_dim // 2]
    public func applyInterleavedMRoPE(freqs: MLXArray, mropeSection: [Int]) -> MLXArray {
        let headDimHalf = freqs.shape[3]

        // Extract temporal, height, width frequencies
        // freqs[0], freqs[1], freqs[2] each have shape [batch, seq_len, head_dim // 2]
        let freqsT = freqs[0]
        let freqsH = freqs[1]
        let freqsW = freqs[2]

        // Create index array for mask computation
        let indices = MLXArray(Array(0..<headDimHalf).map { Int32($0) })

        // H mask: positions where index % 3 == 1, up to length mrope_section[1] * 3
        let hLength = mropeSection[1] * 3
        let mod3 = indices % 3
        let hMod3Eq1 = mod3 .== 1
        let hInRange = indices .< MLXArray(Int32(hLength))
        let hMask = hMod3Eq1 .&& hInRange

        // W mask: positions where index % 3 == 2, up to length mrope_section[2] * 3
        let wLength = mropeSection[2] * 3
        let wMod3Eq2 = mod3 .== 2
        let wInRange = indices .< MLXArray(Int32(wLength))
        let wMask = wMod3Eq2 .&& wInRange

        // Expand masks for broadcasting: [1, 1, head_dim // 2]
        let hMaskExpanded = hMask.reshaped([1, 1, headDimHalf])
        let wMaskExpanded = wMask.reshaped([1, 1, headDimHalf])

        // Apply interleaved combination:
        // Start with temporal, replace with H where h_mask, then with W where w_mask
        var freqsCombined = MLX.where(hMaskExpanded, freqsH, freqsT)
        freqsCombined = MLX.where(wMaskExpanded, freqsW, freqsCombined)

        return freqsCombined
    }

    /// Compute rotary position embeddings.
    ///
    /// - Parameters:
    ///   - x: Input tensor [batch, seq_len, hidden_size] (used for dtype)
    ///   - positionIds: Position indices, either [batch, seq_len] or [3, batch, seq_len]
    /// - Returns: (cos, sin) each with shape [batch, seq_len, head_dim]
    public func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        var posIds = positionIds

        // Ensure position_ids has 3D shape [3, batch, seq_len]
        if positionIds.ndim == 2 {
            // Broadcast [batch, seq_len] -> [3, batch, seq_len]
            let expanded = expandedDimensions(positionIds, axis: 0)
            posIds = broadcast(expanded, to: [3, positionIds.shape[0], positionIds.shape[1]])
        }

        let batchSize = posIds.shape[1]
        let headDimHalf = _invFreq.shape[0]

        // Expand inv_freq: [1, 1, head_dim/2, 1] -> [3, batch, head_dim/2, 1]
        let invFreqExpanded = broadcast(
            _invFreq.reshaped([1, 1, headDimHalf, 1]).asType(.float32),
            to: [3, batchSize, headDimHalf, 1]
        )

        // position_ids: [3, batch, seq_len] -> [3, batch, 1, seq_len]
        let posExpanded = expandedDimensions(posIds.asType(.float32), axis: 2)

        // Compute frequencies: [3, batch, head_dim/2, seq_len]
        // freqs = inv_freq @ pos (matrix multiply along last dims)
        var freqs = matmul(invFreqExpanded, posExpanded)

        // Transpose: [3, batch, seq_len, head_dim/2]
        freqs = swappedAxes(freqs, 2, 3)

        // Apply interleaved MRoPE to combine the 3 modalities
        freqs = applyInterleavedMRoPE(freqs: freqs, mropeSection: mropeSection)

        // Concatenate for full head_dim: [batch, seq_len, head_dim]
        let emb = concatenated([freqs, freqs], axis: -1)

        // Compute cos and sin, cast to input dtype
        let cos = MLX.cos(emb).asType(x.dtype)
        let sin = MLX.sin(emb).asType(x.dtype)

        return (cos, sin)
    }
}

// MARK: - Standard Rotary Embedding (for CodePredictor)

/// Standard Rotary Position Embedding (non-multimodal).
///
/// Used by CodePredictor which doesn't need 3D positions.
public class RotaryEmbedding: Module {
    public let dim: Int
    public let maxPositionEmbeddings: Int
    public let base: Float

    private var _invFreq: MLXArray

    public init(
        dim: Int,
        maxPositionEmbeddings: Int = 65536,
        base: Float = 1000000.0
    ) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base

        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        let exponents = indices / Float(dim)
        self._invFreq = 1.0 / pow(MLXArray(base), exponents)
    }

    /// Compute rotary position embeddings.
    ///
    /// - Parameters:
    ///   - x: Input tensor [batch, seq_len, hidden_size]
    ///   - positionIds: Position indices [batch, seq_len]
    /// - Returns: (cos, sin) each with shape [batch, seq_len, head_dim]
    public func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let batchSize = positionIds.shape[0]
        let headDimHalf = _invFreq.shape[0]

        // Expand inv_freq: [head_dim/2] -> [1, head_dim/2, 1]
        let invFreqExpanded = _invFreq.reshaped([1, headDimHalf, 1]).asType(.float32)

        // Broadcast to [batch, head_dim/2, 1]
        let invFreqBroadcast = broadcast(invFreqExpanded, to: [batchSize, headDimHalf, 1])

        // position_ids: [batch, seq_len] -> [batch, 1, seq_len]
        let posExpanded = expandedDimensions(positionIds.asType(.float32), axis: 1)

        // Compute frequencies: [batch, head_dim/2, seq_len]
        var freqs = matmul(invFreqBroadcast, posExpanded)

        // Transpose: [batch, seq_len, head_dim/2]
        freqs = swappedAxes(freqs, 1, 2)

        // Concatenate for full head_dim: [batch, seq_len, head_dim]
        let emb = concatenated([freqs, freqs], axis: -1)

        let cos = MLX.cos(emb).asType(x.dtype)
        let sin = MLX.sin(emb).asType(x.dtype)

        return (cos, sin)
    }
}
