//
//  LearnedPositionEmbeddings.swift
//  MLXAudio
//
//  Learned position embeddings for T3 model.
//  Ported from mlx-audio Python: chatterbox/t3/learned_pos_emb.py
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// Learned position embeddings (GPT-2 style initialization).
public class LearnedPositionEmbeddings: Module, UnaryLayer {
    @ModuleInfo var emb: Embedding

    public init(seqLen: Int, modelDim: Int, init initScale: Float = 0.02) {
        self._emb.wrappedValue = Embedding(embeddingCount: seqLen, dimensions: modelDim)
        // Note: GPT-2 style initialization would set weights to normal(0, initScale)
        // The actual weights will be loaded from the pretrained checkpoint, so
        // default initialization from Embedding is fine here.
    }

    /// Returns positional embeddings for positions 0..<seqLen.
    ///
    /// - Parameter x: Input tensor of shape (B, T, ...) — only `.shape[1]` is used.
    /// - Returns: Positional embeddings of shape (T, modelDim).
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let sl = x.dim(1)
        return emb(MLXArray(0 ..< Int32(sl)))
    }

    /// Get positional embedding for a specific index.
    ///
    /// - Parameter idx: Scalar position index.
    /// - Returns: Embedding of shape (1, 1, dim).
    public func getFixedEmbedding(_ idx: Int) -> MLXArray {
        let indices = MLXArray([Int32(idx)]).reshaped([1, 1])
        return emb(indices) // (1, 1, dim)
    }

    /// Get positional embeddings for an array of indices.
    ///
    /// - Parameter indices: Index array of shape (B, T).
    /// - Returns: Embeddings of shape (B, T, dim).
    public func getFixedEmbedding(_ indices: MLXArray) -> MLXArray {
        var idx = indices
        if idx.ndim == 1 {
            idx = idx.expandedDimensions(axis: 0)
        }
        return emb(idx)
    }
}
