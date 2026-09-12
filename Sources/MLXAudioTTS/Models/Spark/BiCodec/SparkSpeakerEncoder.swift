import Foundation
@preconcurrency import MLX
import MLXNN

/// Residual finite-scalar quantizer (single quantizer). Decodes code ids into
/// continuous codes via the implicit FSQ codebook, then projects to `dim`.
public final class SparkResidualFSQ: Module {
    @ModuleInfo(key: "project_out") public var projectOut: Linear

    private let basis: [Int32]
    private let halfWidth: Int
    private let levelSize: Int

    public init(dim: Int, levels: [Int]) {
        let codebookDim = levels.count
        self._projectOut = ModuleInfo(wrappedValue: Linear(codebookDim, dim), key: "project_out")
        self.levelSize = levels[0]
        self.halfWidth = levels[0] / 2
        var b = [Int32](); var acc: Int32 = 1
        for l in levels { b.append(acc); acc *= Int32(l) }
        self.basis = b
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
    @ModuleInfo(key: "quantizer") public var quantizer: SparkResidualFSQ
    @ModuleInfo(key: "project") public var project: Linear

    public init(latentDim: Int, outDim: Int, tokenNum: Int, fsqLevels: [Int]) {
        self._quantizer = ModuleInfo(
            wrappedValue: SparkResidualFSQ(dim: latentDim, levels: fsqLevels), key: "quantizer")
        self._project = ModuleInfo(wrappedValue: Linear(latentDim * tokenNum, outDim), key: "project")
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
