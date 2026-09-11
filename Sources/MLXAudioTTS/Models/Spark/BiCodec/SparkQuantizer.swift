//
//  SparkQuantizer.swift
//  MLXAudio
//
//  Factorized vector quantizer for Spark-TTS BiCodec (ports
//  spark/modules/residual.py FactorizedVectorQuantize). Inference paths only:
//  `tokenize` (latents -> code ids) and `detokenize` (code ids -> latents).
//

import Foundation
@preconcurrency import MLX
import MLXNN

private func sparkL2Normalize(_ x: MLXArray, axis: Int) -> MLXArray {
    let norm = MLX.sqrt(MLX.sum(x * x, axis: axis, keepDims: true))
    return x / MLX.maximum(norm, MLXArray(1e-12))
}

/// Factorized VQ: projects `input_dim` latents into a low-dimensional codebook
/// space, matches against an L2-normalized codebook (cosine distance), and
/// projects the selected codes back to `input_dim`.
public final class SparkFactorizedVectorQuantize: Module {
    public let inputDim: Int
    public let codebookSize: Int
    public let codebookDim: Int

    @ModuleInfo(key: "in_project") public var inProject: WeightNormedConv
    @ModuleInfo(key: "out_project") public var outProject: WeightNormedConv
    @ModuleInfo(key: "codebook") public var codebook: Embedding

    public init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self.inputDim = inputDim
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self._inProject = ModuleInfo(
            wrappedValue: WeightNormedConv(
                inChannels: inputDim, outChannels: codebookDim,
                kernelSize: 1, padding: 0, bias: true))
        self._outProject = ModuleInfo(
            wrappedValue: WeightNormedConv(
                inChannels: codebookDim, outChannels: inputDim,
                kernelSize: 1, padding: 0, bias: true))
        self._codebook = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: codebookSize, dimensions: codebookDim))
    }

    /// `z`: [B, D=input_dim, T] -> code indices [B, T].
    public func tokenize(_ z: MLXArray) -> MLXArray {
        let zt = z.transposed(0, 2, 1)          // [B, T, input_dim]
        let ze = inProject(zt)                  // [B, T, codebook_dim]
        let b = ze.shape[0], t = ze.shape[1]
        var enc = ze.reshaped([b * t, codebookDim])
        enc = sparkL2Normalize(enc, axis: 1)
        let cb = sparkL2Normalize(codebook.weight, axis: 1)  // [codebook_size, codebook_dim]
        let dist =
            MLX.sum(enc * enc, axis: 1, keepDims: true)
            - 2 * MLX.matmul(enc, cb.transposed(1, 0))
            + MLX.sum(cb * cb, axis: 1, keepDims: true).transposed(1, 0)
        let indices = MLX.argMax(-dist, axis: 1)
        return indices.reshaped([b, t])
    }

    /// Code indices [B, T] -> latents [B, T, input_dim].
    public func detokenize(_ indices: MLXArray) -> MLXArray {
        let emb = codebook.weight[indices]      // [B, T, codebook_dim]
        return outProject(emb)                  // [B, T, input_dim]
    }
}
