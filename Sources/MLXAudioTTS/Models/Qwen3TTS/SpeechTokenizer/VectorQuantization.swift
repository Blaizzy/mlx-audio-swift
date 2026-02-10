//
//  VectorQuantization.swift
//  MLXAudio
//
//  Vector quantization components for Speech Tokenizer Decoder.
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
//

import Foundation
import MLX
import MLXNN

// MARK: - EuclideanCodebook

/// Euclidean codebook for vector quantization.
///
/// Uses nn.Embedding for the codebook to ensure proper weight loading.
/// The embedding is computed from cluster_usage and embedding_sum during weight sanitization.
public class EuclideanCodebook: Module {
    public let dim: Int
    public let codebookSize: Int
    public let eps: Float

    @ModuleInfo(key: "embed") var embed: Embedding

    public init(dim: Int, codebookSize: Int, eps: Float = 1e-5) {
        self.dim = dim
        self.codebookSize = codebookSize
        self.eps = eps

        // Use Embedding layer for proper weight loading
        self._embed.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: dim)
    }

    /// Decode indices to embeddings.
    ///
    /// - Parameter codes: Code indices [batch, time]
    /// - Returns: Embeddings [batch, time, dim]
    public func decode(_ codes: MLXArray) -> MLXArray {
        return embed(codes)
    }
}

// MARK: - VectorQuantization

/// Vector quantization layer with optional output projection.
public class VectorQuantization: Module {
    public let codebookSize: Int
    public let requiresProjection: Bool

    @ModuleInfo(key: "codebook") var codebook: EuclideanCodebook
    @ModuleInfo(key: "project_out") var projectOut: Linear?

    /// Initialize VectorQuantization.
    ///
    /// - Parameters:
    ///   - dim: Output dimension
    ///   - codebookSize: Number of codebook entries
    ///   - codebookDim: Codebook embedding dimension (defaults to dim)
    ///   - eps: Epsilon for numerical stability
    public init(
        dim: Int,
        codebookSize: Int,
        codebookDim: Int? = nil,
        eps: Float = 1e-5
    ) {
        let actualCodebookDim = codebookDim ?? dim
        self.codebookSize = codebookSize
        self.requiresProjection = actualCodebookDim != dim

        self._codebook.wrappedValue = EuclideanCodebook(
            dim: actualCodebookDim,
            codebookSize: codebookSize,
            eps: eps
        )

        if requiresProjection {
            self._projectOut.wrappedValue = Linear(actualCodebookDim, dim)
        } else {
            self._projectOut.wrappedValue = nil
        }
    }

    /// Decode codes to quantized vectors.
    ///
    /// - Parameter codes: Code indices [batch, time]
    /// - Returns: Quantized vectors [batch, dim, time] (NCL format)
    public func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, time]
        var quantized = codebook.decode(codes)  // [batch, time, codebook_dim]

        if let proj = projectOut {
            quantized = proj(quantized)  // [batch, time, dim]
        }

        // Transpose to NCL format: [batch, dim, time]
        return quantized.transposed(0, 2, 1)
    }
}

// MARK: - ResidualVectorQuantization

/// Residual vector quantization with multiple codebooks.
///
/// Each layer's output is summed to form the final quantized representation.
public class ResidualVectorQuantization: Module {
    @ModuleInfo(key: "layers") var layers: [VectorQuantization]

    /// Initialize ResidualVectorQuantization.
    ///
    /// - Parameters:
    ///   - numQuantizers: Number of VQ layers
    ///   - dim: Output dimension
    ///   - codebookSize: Number of codebook entries
    ///   - codebookDim: Codebook embedding dimension (defaults to dim)
    public init(
        numQuantizers: Int,
        dim: Int,
        codebookSize: Int,
        codebookDim: Int? = nil
    ) {
        var vqLayers: [VectorQuantization] = []
        for _ in 0..<numQuantizers {
            vqLayers.append(VectorQuantization(
                dim: dim,
                codebookSize: codebookSize,
                codebookDim: codebookDim
            ))
        }
        self._layers.wrappedValue = vqLayers
    }

    /// Decode codes to quantized vectors (summing all layers).
    ///
    /// - Parameter codes: Code indices [numQuantizers, batch, time]
    /// - Returns: Quantized vectors [batch, dim, time] (NCL format)
    public func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [num_quantizers, batch, time]
        // Build lazy computation graph — no intermediate eval() calls
        var quantized = layers[0].decode(codes[0])
        for idx in 1..<codes.shape[0] {
            quantized = quantized + layers[idx].decode(codes[idx])
        }
        return quantized
    }
}

// MARK: - ResidualVectorQuantizer

/// Residual vector quantizer with input/output Conv1d projections.
public class ResidualVectorQuantizer: Module {
    public let nQ: Int
    public let dimension: Int
    public let inputDimension: Int
    public let outputDimension: Int
    public let bins: Int

    @ModuleInfo(key: "input_proj") var inputProj: Conv1d?
    @ModuleInfo(key: "output_proj") var outputProj: Conv1d?
    @ModuleInfo(key: "vq") var vq: ResidualVectorQuantization

    /// Initialize ResidualVectorQuantizer.
    ///
    /// - Parameters:
    ///   - dimension: Internal quantization dimension
    ///   - inputDimension: Input dimension (defaults to dimension)
    ///   - outputDimension: Output dimension (defaults to dimension)
    ///   - nQ: Number of quantizers
    ///   - bins: Codebook size
    ///   - forceProjection: Force use of projection layers even if dimensions match
    public init(
        dimension: Int = 128,
        inputDimension: Int? = nil,
        outputDimension: Int? = nil,
        nQ: Int = 8,
        bins: Int = 1024,
        forceProjection: Bool = false
    ) {
        self.nQ = nQ
        self.dimension = dimension
        self.inputDimension = inputDimension ?? dimension
        self.outputDimension = outputDimension ?? dimension
        self.bins = bins

        // Input projection (Conv1d with kernel=1, no bias)
        if self.inputDimension != dimension || forceProjection {
            self._inputProj.wrappedValue = Conv1d(
                inputChannels: self.inputDimension,
                outputChannels: dimension,
                kernelSize: 1,
                bias: false
            )
        } else {
            self._inputProj.wrappedValue = nil
        }

        // Output projection (Conv1d with kernel=1, no bias)
        if self.outputDimension != dimension || forceProjection {
            self._outputProj.wrappedValue = Conv1d(
                inputChannels: dimension,
                outputChannels: self.outputDimension,
                kernelSize: 1,
                bias: false
            )
        } else {
            self._outputProj.wrappedValue = nil
        }

        self._vq.wrappedValue = ResidualVectorQuantization(
            numQuantizers: nQ,
            dim: dimension,
            codebookSize: bins
        )
    }

    /// Decode codes to quantized vectors.
    ///
    /// - Parameter codes: Code indices [batch, numQuantizers, time]
    /// - Returns: Quantized vectors [batch, outputDim, time] (NCL format)
    public func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, num_quantizers, time]
        // Transpose to [num_quantizers, batch, time] for RVQ
        let transposedCodes = codes.transposed(1, 0, 2)

        // Decode through VQ
        var quantized = vq.decode(transposedCodes)  // [batch, dim, time]

        // Apply output projection if present
        if let proj = outputProj {
            // Conv1d expects NLC format, we have NCL
            quantized = quantized.transposed(0, 2, 1)  // [batch, time, dim]
            quantized = proj(quantized)  // [batch, time, output_dim]
            quantized = quantized.transposed(0, 2, 1)  // [batch, output_dim, time]
        }

        return quantized
    }
}

// MARK: - SplitResidualVectorQuantizer

/// Split RVQ with separate quantizers for semantic and acoustic tokens.
///
/// Typically splits 16 quantizers into 1 semantic + 15 acoustic.
public class SplitResidualVectorQuantizer: Module {
    public let nQSemantic: Int
    public let nQAcoustic: Int

    @ModuleInfo(key: "rvq_first") var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: ResidualVectorQuantizer

    /// Initialize SplitResidualVectorQuantizer.
    ///
    /// - Parameters:
    ///   - nQ: Total number of quantizers
    ///   - nQSemantic: Number of semantic quantizers (default 1)
    ///   - dimension: Internal quantization dimension
    ///   - inputDimension: Input dimension
    ///   - outputDimension: Output dimension
    ///   - bins: Codebook size
    public init(
        nQ: Int = 8,
        nQSemantic: Int = 1,
        dimension: Int = 128,
        inputDimension: Int? = nil,
        outputDimension: Int? = nil,
        bins: Int = 1024
    ) {
        self.nQSemantic = nQSemantic
        self.nQAcoustic = nQ - nQSemantic

        // Semantic quantizer (first n_q_semantic)
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            nQ: nQSemantic,
            bins: bins,
            forceProjection: true
        )

        // Acoustic quantizer (remaining)
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            dimension: dimension,
            inputDimension: inputDimension,
            outputDimension: outputDimension,
            nQ: nQ - nQSemantic,
            bins: bins,
            forceProjection: true
        )
    }

    /// Decode codes to quantized vectors.
    ///
    /// - Parameter codes: Code indices [batch, numQuantizers, time]
    /// - Returns: Quantized vectors [batch, outputDim, time] (NCL format)
    public func decode(_ codes: MLXArray) -> MLXArray {
        // codes: [batch, num_quantizers, time]
        // Build lazy computation graph — no intermediate eval()/clearCache()
        var quantized = rvqFirst.decode(codes[0..., 0..<nQSemantic, 0...])
        if codes.shape[1] > nQSemantic {
            quantized = quantized + rvqRest.decode(codes[0..., nQSemantic..., 0...])
        }
        return quantized
    }
}
