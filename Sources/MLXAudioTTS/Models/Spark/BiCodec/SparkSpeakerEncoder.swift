import Foundation
@preconcurrency import MLX
import MLXNN

/// Residual finite-scalar quantizer (single quantizer). Decodes code ids into
/// continuous codes via the implicit FSQ codebook, then projects to `dim`.
public final class SparkResidualFSQ: Module {
    @ModuleInfo(key: "project_in") public var projectIn: Linear
    @ModuleInfo(key: "project_out") public var projectOut: Linear

    private let basis: [Int32]
    private let halfWidth: Int
    private let levelSize: Int

    public init(dim: Int, levels: [Int]) {
        let codebookDim = levels.count
        self._projectIn = ModuleInfo(wrappedValue: Linear(dim, codebookDim), key: "project_in")
        self._projectOut = ModuleInfo(wrappedValue: Linear(codebookDim, dim), key: "project_out")
        self.levelSize = levels[0]
        self.halfWidth = levels[0] / 2
        var b = [Int32](); var acc: Int32 = 1
        for l in levels { b.append(acc); acc *= Int32(l) }
        self.basis = b
    }

    /// Continuous input `x` [B, dim, n] -> code indices [B, 1, n].
    public func tokenize(_ x: MLXArray) -> MLXArray {
        let z = projectIn(x.swappedAxes(1, 2))
        let eps: Float = 1e-3
        let level = Float(levelSize)
        let halfL = (level - 1) * (1 + eps) / 2
        let shift = atanhf(0.5 / halfL)
        let bounded = MLX.tanh(z + shift) * halfL - 0.5
        let zhat = MLX.round(bounded) + Float(halfWidth)
        let idx = MLX.sum(zhat * MLXArray(basis).asType(.float32), axis: -1)
        return idx.asType(.int32).expandedDimensions(axis: 1)
    }

    private func indicesToCodes(_ indices: MLXArray) -> MLXArray {
        let basisArr = MLXArray(basis)
        let levelIdx = MLX.floorDivide(indices[.ellipsis, .newAxis], basisArr) % levelSize
        return (levelIdx.asType(.float32) - Float(halfWidth)) / Float(halfWidth)
    }

    /// `indices`: [B, n, numQuantizers=1] -> [B, n, dim].
    public func getOutputFromIndices(_ indices: MLXArray) -> MLXArray {
        let q = indices[.ellipsis, 0]
        let codes = indicesToCodes(q)
        return projectOut(codes)
    }
}

/// Global token ids -> speaker d-vector.
public final class SparkSpeakerEncoder: Module {
    @ModuleInfo(key: "speaker_encoder") public var ecapa: SparkEcapaTDNN
    @ModuleInfo(key: "perceiver_sampler") public var perceiver: SparkPerceiverResampler
    @ModuleInfo(key: "quantizer") public var quantizer: SparkResidualFSQ
    @ModuleInfo(key: "project") public var project: Linear

    public init(inputDim: Int, latentDim: Int, outDim: Int, tokenNum: Int, fsqLevels: [Int]) {
        self._ecapa = ModuleInfo(wrappedValue: SparkEcapaTDNN(featDim: inputDim), key: "speaker_encoder")
        self._perceiver = ModuleInfo(
            wrappedValue: SparkPerceiverResampler(dim: latentDim, dimContext: 512 * 3, numLatents: tokenNum),
            key: "perceiver_sampler")
        self._quantizer = ModuleInfo(
            wrappedValue: SparkResidualFSQ(dim: latentDim, levels: fsqLevels), key: "quantizer")
        self._project = ModuleInfo(wrappedValue: Linear(latentDim * tokenNum, outDim), key: "project")
    }

    /// Mel features [B, T, feat_dim] -> global token ids [B, 1, tokenNum].
    public func tokenize(_ mel: MLXArray) -> MLXArray {
        let features = ecapa.latent(mel)
        let x = perceiver(features.transposed(0, 2, 1)).transposed(0, 2, 1)
        return quantizer.tokenize(x)
    }

    /// `globalTokens`: [B, 1, tokenNum] -> d-vector [B, outDim].
    public func detokenize(_ globalTokens: MLXArray) -> MLXArray {
        let idx = globalTokens.swappedAxes(-1, -2)
        let codes = quantizer.getOutputFromIndices(idx)
        let zq = codes.swappedAxes(-1, -2)
        let flat = zq.reshaped([zq.shape[0], -1])
        return project(flat)
    }
}
