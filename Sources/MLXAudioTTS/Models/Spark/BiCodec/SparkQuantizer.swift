import Foundation
@preconcurrency import MLX
import MLXNN

/// Maps semantic code ids to latents: gathers the codebook entry and projects it
/// back up to `input_dim`.
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
                kernelSize: 1, padding: 0, bias: true),
            key: "in_project")
        self._outProject = ModuleInfo(
            wrappedValue: WeightNormedConv(
                inChannels: codebookDim, outChannels: inputDim,
                kernelSize: 1, padding: 0, bias: true),
            key: "out_project")
        self._codebook = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: codebookSize, dimensions: codebookDim),
            key: "codebook")
    }

    /// Code indices [B, T] -> latents [B, T, input_dim].
    public func detokenize(_ indices: MLXArray) -> MLXArray {
        let emb = codebook.weight[indices]
        return outProject(emb)
    }

    /// Latents `z` [B, input_dim, T] -> code indices [B, T].
    public func tokenize(_ z: MLXArray) -> MLXArray {
        let ze = inProject(z.transposed(0, 2, 1))
        return decodeLatents(ze)
    }

    private func normalize(_ x: MLXArray) -> MLXArray {
        let norm = MLX.sqrt(MLX.sum(x * x, axis: 1, keepDims: true))
        return x / MLX.maximum(norm, MLXArray(Float(1e-12)))
    }

    private func decodeLatents(_ ze: MLXArray) -> MLXArray {
        let b = ze.shape[0], t = ze.shape[1], d = ze.shape[2]
        let encodings = normalize(ze.reshaped([b * t, d]))
        let cb = normalize(codebook.weight)
        let dist = MLX.sum(encodings * encodings, axis: 1, keepDims: true)
            - 2 * MLX.matmul(encodings, cb.transposed(1, 0))
            + MLX.sum(cb * cb, axis: 1, keepDims: true).transposed(1, 0)
        return MLX.argMax(-dist, axis: 1).reshaped([b, t])
    }
}
