//
//  VoxCPMEncoder.swift
//  MLXAudio
//
//  VoxCPM2 feature encoder: patches → CLS token embeddings via MiniCPM.
//  Ported from mlx-audio Python: voxcpm2/encoder.py
//

import Foundation
@preconcurrency import MLX
import MLXNN

class VoxCPMLocEnc: Module {
    let config: VoxCPM2LMConfig

    var specialToken: MLXArray
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo var encoder: VoxMiniCPMModel

    init(_ config: VoxCPM2LMConfig, inputDim: Int = 64) {
        self.config = config

        self.specialToken = MLXRandom.normal([1, 1, 1, config.hiddenSize])
        self._inProj.wrappedValue = Linear(inputDim, config.hiddenSize, bias: true)

        self._encoder.wrappedValue = VoxMiniCPMModel(config)

        super.init()
    }

    /// Encode audio patches to per-timestep embeddings.
    /// - Parameter x: (B, T, P, D) — batch, timesteps, patches, feature dim
    /// - Returns: (B, T, H) — CLS token output per timestep
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let P = x.dim(2)

        let projected = inProj(x) // (B, T, P, H)

        let specialTokens = MLX.broadcast(
            specialToken,
            to: [B, T, 1, config.hiddenSize]
        )

        let withCLS = MLX.concatenated([specialTokens, projected], axis: 2) // (B, T, P+1, H)

        let flat = withCLS.reshaped(B * T, P + 1, -1) // (B*T, P+1, H)

        let (outputs, _) = encoder(inputsEmbeds: flat, isCausal: false)

        let clsOutput = outputs[0..., 0, 0...] // (B*T, H)

        return clsOutput.reshaped(B, T, -1)
    }
}
