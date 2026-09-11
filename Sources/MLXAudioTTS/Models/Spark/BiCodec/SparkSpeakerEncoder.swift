//
//  SparkSpeakerEncoder.swift
//  MLXAudio
//
//  Speaker encoder for Spark-TTS BiCodec — DETOKENIZE path only (global token ids
//  -> speaker d-vector), which is all the controllable-TTS synthesis path needs.
//  The ECAPA/perceiver *encode* path (voice cloning) is intentionally omitted for
//  now; its checkpoint weights are dropped on load.
//
//  Ports the detokenize portions of spark/modules/residual_fsq.py (ResidualFSQ,
//  num_quantizers == 1, finite-scalar codebook) and speaker/speaker_encoder.py.
//

import Foundation
@preconcurrency import MLX
import MLXNN

/// Residual finite-scalar quantizer (single quantizer). Decodes code ids into
/// continuous codes via the implicit FSQ codebook, then projects to `dim`.
public final class SparkResidualFSQ: Module {
    @ModuleInfo(key: "project_out") public var projectOut: Linear

    private let basis: MLXArray   // cumprod([1, levels[:-1]])
    private let halfWidth: Int    // levels // 2 (levels are uniform == 4 here)
    private let levelSize: Int

    public init(dim: Int, levels: [Int]) {
        let codebookDim = levels.count
        self._projectOut = ModuleInfo(wrappedValue: Linear(codebookDim, dim), key: "project_out")
        self.levelSize = levels[0]
        self.halfWidth = levels[0] / 2
        var b = [Int](); var acc = 1
        for l in levels { b.append(acc); acc *= l }
        self.basis = MLXArray(b.map { Int32($0) })
    }

    /// FSQ code decode: index -> centered code in the codebook (matches
    /// `_indices_to_codes` / `_scale_and_shift_inverse`).
    private func indicesToCodes(_ indices: MLXArray) -> MLXArray {
        // indices: [...] -> codes [..., codebookDim]
        let levelIdx = MLX.floorDivide(indices[.ellipsis, .newAxis], basis) % levelSize
        return (levelIdx.asType(.float32) - Float(halfWidth)) / Float(halfWidth)
    }

    /// `indices`: [B, n, numQuantizers=1] -> [B, n, dim].
    public func getOutputFromIndices(_ indices: MLXArray) -> MLXArray {
        let q = indices[.ellipsis, 0]                 // [B, n]  (single quantizer)
        let codes = indicesToCodes(q)                 // [B, n, codebookDim]
        return projectOut(codes)                      // [B, n, dim]
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
        let idx = globalTokens.swappedAxes(-1, -2)        // [B, tokenNum, 1]
        let codes = quantizer.getOutputFromIndices(idx)   // [B, tokenNum, latentDim]
        let zq = codes.swappedAxes(-1, -2)                // [B, latentDim, tokenNum]
        let flat = zq.reshaped([zq.shape[0], -1])         // [B, latentDim*tokenNum]
        return project(flat)                              // [B, outDim]
    }
}
