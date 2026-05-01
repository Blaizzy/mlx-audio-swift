//
//  VoxCPMDiT.swift
//  MLXAudio
//
//  VoxCPM2 Diffusion Transformer + Conditional Flow Matching (CFM) decoder.
//  Ported from mlx-audio Python: voxcpm2/dit.py
//

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Sinusoidal Position Embedding

class SinusoidalPosEmb: Module {
    let dim: Int

    init(dim: Int) {
        assert(dim % 2 == 0)
        self.dim = dim
        super.init()
    }

    func callAsFunction(_ x: MLXArray, scale: Float = 1000) -> MLXArray {
        var inp = x
        if inp.ndim < 1 {
            inp = inp.reshaped([1])
        }

        let halfDim = dim / 2
        let emb = log(Float(10000)) / Float(halfDim - 1)
        let freqs = MLX.exp(MLXArray(0 ..< halfDim).asType(.float32) * -emb)

        let scaled = scale * inp.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)

        return MLX.concatenated([MLX.sin(scaled), MLX.cos(scaled)], axis: -1)
    }
}

// MARK: - Timestep Embedding

class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inChannels: Int, timeEmbedDim: Int, outDim: Int? = nil) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
        self._linear2.wrappedValue = Linear(timeEmbedDim, outDim ?? timeEmbedDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

// MARK: - DiT V2

class VoxCPMLocDiTV2: Module {
    let config: VoxCPM2LMConfig
    let inChannels: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "time_embeddings") var timeEmbeddings: SinusoidalPosEmb
    @ModuleInfo(key: "time_mlp") var timeMlp: TimestepEmbedding
    @ModuleInfo(key: "delta_time_mlp") var deltaTimeMlp: TimestepEmbedding
    @ModuleInfo var decoder: VoxMiniCPMModel

    init(_ config: VoxCPM2LMConfig, inChannels: Int = 64) {
        self.config = config
        self.inChannels = inChannels

        self._inProj.wrappedValue = Linear(inChannels, config.hiddenSize)
        self._condProj.wrappedValue = Linear(inChannels, config.hiddenSize)
        self._outProj.wrappedValue = Linear(config.hiddenSize, inChannels)

        self._timeEmbeddings.wrappedValue = SinusoidalPosEmb(dim: config.hiddenSize)
        self._timeMlp.wrappedValue = TimestepEmbedding(inChannels: config.hiddenSize, timeEmbedDim: config.hiddenSize)
        self._deltaTimeMlp.wrappedValue = TimestepEmbedding(inChannels: config.hiddenSize, timeEmbedDim: config.hiddenSize)

        self._decoder.wrappedValue = VoxMiniCPMModel(config)

        super.init()
    }

    /// - Parameters:
    ///   - x: (N, C, T) noisy input
    ///   - mu: (N, 2*H) multi-token conditioning from LM+residual
    ///   - t: (N,) timestep
    ///   - cond: (N, C, T') conditioning signal
    ///   - dt: (N,) delta timestep
    /// - Returns: (N, C, T) velocity estimate
    func callAsFunction(
        _ x: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        cond: MLXArray,
        dt: MLXArray
    ) -> MLXArray {
        // x: (N, C, T) → (N, T, C)
        let xT = x.transposed(0, 2, 1)
        let xProj = inProj(xT)

        // cond: (N, C, T') → (N, T', C)
        let condT = cond.transposed(0, 2, 1)
        let condProjVal = condProj(condT)
        let prefix = condT.dim(1)

        let tEmb = timeMlp(timeEmbeddings(t))
        let dtEmb = deltaTimeMlp(timeEmbeddings(dt))
        let tComb = tEmb + dtEmb // (N, H)

        // V2: mu is (N, 2*H) → reshape to (N, numMuTokens, H)
        let H = xProj.dim(-1)
        let muTokens = mu.reshaped(x.dim(0), -1, H) // (N, numMuTokens, H)
        let numMuTokens = muTokens.dim(1)

        let hidden = MLX.concatenated(
            [muTokens, tComb[0..., .newAxis, 0...], condProjVal, xProj],
            axis: 1
        )

        let (decoded, _) = decoder(inputsEmbeds: hidden, isCausal: false)

        // Slice: skip mu_tokens + t_token + cond prefix
        let sliced = decoded[0..., (numMuTokens + 1 + prefix)..., 0...]

        let out = outProj(sliced)

        return out.transposed(0, 2, 1) // (N, C, T)
    }
}

// MARK: - Unified CFM

class VoxUnifiedCFM: Module {
    let inChannels: Int
    @ModuleInfo(key: "estimator") var estimator: VoxCPMLocDiTV2
    let cfmParams: VoxCPM2CFMConfig
    let meanMode: Bool

    init(
        inChannels: Int,
        cfmParams: VoxCPM2CFMConfig,
        estimator: VoxCPMLocDiTV2,
        meanMode: Bool = false
    ) {
        self.inChannels = inChannels
        self.cfmParams = cfmParams
        self.meanMode = meanMode
        self._estimator.wrappedValue = estimator
        super.init()
    }

    func solveEuler(
        _ x: MLXArray,
        tSpan: MLXArray,
        mu: MLXArray,
        cond: MLXArray,
        cfgValue: Float = 1.0,
        useCfgZeroStar: Bool = true
    ) -> MLXArray {
        var t = tSpan[0].item(Float.self)
        var dt = tSpan[0].item(Float.self) - tSpan[1].item(Float.self)

        var currentX = x

        let nSteps = tSpan.dim(0)
        let zeroInitSteps = max(1, Int(Float(nSteps) * 0.04))

        for step in 1 ..< nSteps {
            let dphiDt: MLXArray

            if useCfgZeroStar && step <= zeroInitSteps {
                dphiDt = MLXArray.zeros(like: currentX)
            } else {
                let b = currentX.dim(0)

                let xIn = MLX.concatenated([currentX, currentX], axis: 0)
                let muIn = MLX.concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)

                let n = xIn.dim(0)
                let tVal = MLXArray(Array(repeating: t, count: n))

                let dtValIn: MLXArray
                if meanMode {
                    dtValIn = MLXArray(Array(repeating: dt, count: n))
                } else {
                    dtValIn = MLXArray.zeros([n])
                }

                let condIn = MLX.concatenated([cond, cond], axis: 0)

                let out = estimator(xIn, mu: muIn, t: tVal, cond: condIn, dt: dtValIn)

                let chunkSize = b
                let posOut = out[..<chunkSize]
                let cfgOut = out[chunkSize...]

                let guided: MLXArray
                if useCfgZeroStar {
                    let posFlatOrig = posOut.reshaped(chunkSize, -1)
                    let negFlat = cfgOut.reshaped(chunkSize, -1)

                    let dotProd = MLX.sum(posFlatOrig * negFlat, axis: 1, keepDims: true)
                    let sqNorm = MLX.sum(negFlat * negFlat, axis: 1, keepDims: true) + MLXArray(Float(1e-8))
                    var stStar = dotProd / sqNorm
                    stStar = stStar.reshaped(chunkSize, 1, 1)

                    guided = cfgOut * stStar + MLXArray(cfgValue) * (posOut - cfgOut * stStar)
                } else {
                    guided = cfgOut + MLXArray(cfgValue) * (posOut - cfgOut)
                }

                dphiDt = guided
            }

            currentX = currentX - MLXArray(dt) * dphiDt
            t = t - dt

            if step < nSteps - 1 {
                dt = t - tSpan[step + 1].item(Float.self)
            }
        }

        return currentX
    }

    func sample(
        mu: MLXArray,
        nTimesteps: Int,
        patchSize: Int,
        cond: MLXArray,
        temperature: Float = 1.0,
        cfgValue: Float = 1.0
    ) -> MLXArray {
        let B = mu.dim(0)
        let T = patchSize

        let z = MLXRandom.normal([B, inChannels, T]) * MLXArray(temperature)

        var tSpan = MLX.linspace(Float(1), Float(0), count: nTimesteps + 1)
        // Sway scheduling: t + cos(pi/2 * t) - 1 + t = 2t + cos(pi/2 * t) - 1
        let sway = MLX.cos(MLXArray(Float.pi / 2) * tSpan) - 1 + tSpan
        tSpan = tSpan + sway

        return solveEuler(z, tSpan: tSpan, mu: mu, cond: cond, cfgValue: cfgValue)
    }
}
