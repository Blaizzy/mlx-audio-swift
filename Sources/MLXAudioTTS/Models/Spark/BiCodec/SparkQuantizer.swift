import Foundation
@preconcurrency import MLX
import MLXNN

/// Maps semantic code ids to latents: gathers the codebook entry and projects it
/// back up to `input_dim`.
public final class SparkFactorizedVectorQuantize: Module {
    public let inputDim: Int
    public let codebookSize: Int
    public let codebookDim: Int

    @ModuleInfo(key: "out_project") public var outProject: WeightNormedConv
    @ModuleInfo(key: "codebook") public var codebook: Embedding

    public init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self.inputDim = inputDim
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
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
}
