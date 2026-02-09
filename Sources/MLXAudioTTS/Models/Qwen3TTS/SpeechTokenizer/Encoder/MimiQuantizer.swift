//
//  MimiQuantizer.swift
//  MLXAudio
//
//  Mimi vector quantization with encode support for the encoder pipeline.
//  Ported from mlx_audio/codec/models/mimi/modules/quantization.py
//
//  Separate from the decoder VQ to avoid breaking existing decoder code.
//  Key difference: includes encode() (nearest-neighbor argmin) for encoding audio to codes.
//

import Foundation
import MLX
import MLXNN

// MARK: - MimiEuclideanCodebook

/// Euclidean codebook with encode (nearest-neighbor lookup) and decode support.
///
/// Embeddings are derived from `embedding_sum / cluster_usage` during weight loading.
/// The `_embedding` and `_c2` fields are computed after weight update.
public class MimiEuclideanCodebook: Module {
    let dim: Int
    let codebookSize: Int
    let epsilon: Float

    // Raw weight storage (loaded from safetensors)
    var embeddingSum: MLXArray
    var clusterUsage: MLXArray

    public init(dim: Int, codebookSize: Int) {
        self.dim = dim
        self.codebookSize = codebookSize
        self.epsilon = 1e-5

        self.embeddingSum = MLXArray.zeros([codebookSize, dim])
        self.clusterUsage = MLXArray.zeros([codebookSize])
    }

    /// Compute the embedding from embeddingSum and clusterUsage.
    private func computeEmbedding() -> MLXArray {
        let usage = clip(clusterUsage, min: epsilon, max: Float.infinity).expandedDimensions(axis: 1)
        return embeddingSum / usage
    }

    /// Encode input vectors to nearest codebook indices.
    ///
    /// Uses argmin(c2 - dot(x, embed)) for efficient nearest-neighbor lookup.
    ///
    /// - Parameter x: Input vectors [..., dim]
    /// - Returns: Indices [...] (same shape minus last dim)
    public func encode(_ x: MLXArray) -> MLXArray {
        let targetShape = Array(x.shape.dropLast())
        let flat = x.reshaped([-1, dim])

        let embedding = computeEmbedding()
        let c2 = sum(embedding * embedding, axis: -1) / 2

        let xF32 = flat.asType(.float32)
        let embedF32 = embedding.asType(.float32)
        let c2F32 = c2.asType(.float32)
        let dotProd = matmul(xF32, embedF32.transposed())
        let distances = c2F32 - dotProd
        return argMin(distances, axis: -1).reshaped(targetShape)
    }

    /// Decode indices to embeddings.
    ///
    /// - Parameter codes: Indices [...]
    /// - Returns: Embeddings [..., dim]
    public func decode(_ codes: MLXArray) -> MLXArray {
        let embedding = computeEmbedding()
        let targetShape = codes.shape + [dim]
        let flat = codes.reshaped([-1])
        return embedding[flat].reshaped(targetShape)
    }
}

// MARK: - MimiVectorQuantization

/// Single VQ layer with optional input/output projections.
public class MimiVectorQuantization: Module {
    @ModuleInfo(key: "codebook") var codebook: MimiEuclideanCodebook
    @ModuleInfo(key: "project_in") var projectIn: Linear?
    @ModuleInfo(key: "project_out") var projectOut: Linear?

    public init(dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        let cbDim = codebookDim ?? dim
        self._codebook.wrappedValue = MimiEuclideanCodebook(dim: cbDim, codebookSize: codebookSize)

        if dim != cbDim {
            self._projectIn.wrappedValue = Linear(dim, cbDim)
            self._projectOut.wrappedValue = Linear(cbDim, dim)
        } else {
            self._projectIn.wrappedValue = nil
            self._projectOut.wrappedValue = nil
        }
    }

    /// Encode NCL input to codebook indices.
    ///
    /// - Parameter x: NCL format [batch, channels, time]
    /// - Returns: Indices [batch, time]
    public func encode(_ x: MLXArray) -> MLXArray {
        // NCL -> NLC for linear operations
        var xs = x.transposed(0, 2, 1)
        if let proj = projectIn {
            xs = proj(xs)
        }
        return codebook.encode(xs)
    }

    /// Decode indices to NCL vectors.
    ///
    /// - Parameter codes: Indices [batch, time]
    /// - Returns: Vectors NCL [batch, channels, time]
    public func decode(_ codes: MLXArray) -> MLXArray {
        var xs = codebook.decode(codes)  // [batch, time, cbDim]
        if let proj = projectOut {
            xs = proj(xs)
        }
        return xs.transposed(0, 2, 1)  // NLC -> NCL
    }
}

// MARK: - MimiResidualVectorQuantization

/// Multi-layer RVQ with residual subtraction for encoding.
public class MimiResidualVectorQuantization: Module {
    @ModuleInfo(key: "layers") var layers: [MimiVectorQuantization]

    public init(nq: Int, dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        var vqLayers: [MimiVectorQuantization] = []
        for _ in 0..<nq {
            vqLayers.append(MimiVectorQuantization(
                dim: dim,
                codebookSize: codebookSize,
                codebookDim: codebookDim
            ))
        }
        self._layers.wrappedValue = vqLayers
    }

    /// Encode input through all VQ layers with residual subtraction.
    ///
    /// - Parameter x: NCL format [batch, channels, time]
    /// - Returns: Codes [num_quantizers, batch, time]
    public func encode(_ x: MLXArray) -> MLXArray {
        var codes: [MLXArray] = []
        var residual = x
        for layer in layers {
            let indices = layer.encode(residual)
            let quantized = layer.decode(indices)
            residual = (residual.asType(.float32) - quantized.asType(.float32)).asType(x.dtype)
            codes.append(indices)
        }
        return stacked(codes, axis: 0)
    }

    /// Decode codes by summing all VQ layer outputs.
    ///
    /// - Parameter codes: [num_quantizers, batch, time]
    /// - Returns: NCL [batch, channels, time]
    public func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = layers[0].decode(codes[0])
        for i in 1..<codes.shape[0] {
            quantized = quantized + layers[i].decode(codes[i])
        }
        return quantized
    }
}

// MARK: - MimiResidualVectorQuantizer

/// RVQ with Conv1d input/output projections.
public class MimiResidualVectorQuantizer: Module {
    @ModuleInfo(key: "input_proj") var inputProj: MimiConv1d?
    @ModuleInfo(key: "output_proj") var outputProj: MimiConv1d?
    @ModuleInfo(key: "vq") var vq: MimiResidualVectorQuantization

    public init(
        dim: Int,
        inputDim: Int? = nil,
        outputDim: Int? = nil,
        nq: Int,
        bins: Int,
        forceProjection: Bool = false
    ) {
        let actualInputDim = inputDim ?? dim
        let actualOutputDim = outputDim ?? dim

        if actualInputDim != dim || forceProjection {
            self._inputProj.wrappedValue = MimiConv1d(
                inChannels: actualInputDim,
                outChannels: dim,
                kernelSize: 1,
                bias: false
            )
        } else {
            self._inputProj.wrappedValue = nil
        }

        if actualOutputDim != dim || forceProjection {
            self._outputProj.wrappedValue = MimiConv1d(
                inChannels: dim,
                outChannels: actualOutputDim,
                kernelSize: 1,
                bias: false
            )
        } else {
            self._outputProj.wrappedValue = nil
        }

        self._vq.wrappedValue = MimiResidualVectorQuantization(
            nq: nq,
            dim: dim,
            codebookSize: bins,
            codebookDim: nil
        )
    }

    /// Encode NCL input to codes.
    ///
    /// - Parameter x: NCL [batch, channels, time]
    /// - Returns: Codes [batch, num_quantizers, time]
    public func encode(_ x: MLXArray) -> MLXArray {
        var xs = x
        if let proj = inputProj {
            xs = proj(xs)
        }
        // vq.encode returns [num_quantizers, batch, time]
        // swap to [batch, num_quantizers, time]
        return vq.encode(xs).transposed(1, 0, 2)
    }

    /// Decode codes to NCL vectors.
    ///
    /// - Parameter codes: [batch, num_quantizers, time]
    /// - Returns: NCL [batch, channels, time]
    public func decode(_ codes: MLXArray) -> MLXArray {
        // swap to [num_quantizers, batch, time]
        let transposed = codes.transposed(1, 0, 2)
        var quantized = vq.decode(transposed)
        if let proj = outputProj {
            quantized = proj(quantized)
        }
        return quantized
    }
}

// MARK: - MimiSplitResidualVectorQuantizer

/// Split RVQ: 1 semantic (rvq_first) + N-1 acoustic (rvq_rest).
public class MimiSplitResidualVectorQuantizer: Module {
    let nq: Int

    @ModuleInfo(key: "rvq_first") var rvqFirst: MimiResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: MimiResidualVectorQuantizer

    public init(dim: Int, inputDim: Int, outputDim: Int, nq: Int, bins: Int) {
        self.nq = nq

        self._rvqFirst.wrappedValue = MimiResidualVectorQuantizer(
            dim: dim,
            inputDim: inputDim,
            outputDim: outputDim,
            nq: 1,
            bins: bins,
            forceProjection: true
        )

        self._rvqRest.wrappedValue = MimiResidualVectorQuantizer(
            dim: dim,
            inputDim: inputDim,
            outputDim: outputDim,
            nq: nq - 1,
            bins: bins,
            forceProjection: true
        )
    }

    /// Encode NCL input to codes.
    ///
    /// Note: The Python SplitRVQ encode passes the same input `xs` to both
    /// rvq_first and rvq_rest (NOT the residual). This matches the Mimi codec
    /// architecture where the split is for independent semantic + acoustic paths.
    ///
    /// - Parameter x: NCL [batch, channels, time]
    /// - Returns: Codes [batch, num_quantizers, time]
    public func encode(_ x: MLXArray) -> MLXArray {
        var codes = rvqFirst.encode(x)  // [batch, 1, time]
        if nq > 1 {
            let restCodes = rvqRest.encode(x)  // [batch, nq-1, time]
            codes = concatenated([codes, restCodes], axis: 1)
        }
        return codes
    }

    /// Decode codes to NCL vectors.
    ///
    /// - Parameter codes: [batch, num_quantizers, time]
    /// - Returns: NCL [batch, channels, time]
    public func decode(_ codes: MLXArray) -> MLXArray {
        var quantized = rvqFirst.decode(codes[0..., 0..<1, 0...])
        if nq > 1 {
            quantized = quantized + rvqRest.decode(codes[0..., 1..., 0...])
        }
        return quantized
    }
}
