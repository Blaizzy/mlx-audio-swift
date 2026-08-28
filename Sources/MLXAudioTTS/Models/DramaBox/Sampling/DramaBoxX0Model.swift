import Foundation
@preconcurrency import MLX

struct DramaBoxX0Model {
    let dit: DramaBoxLTXModel

    func callAsFunction(
        _ latent: MLXArray,
        aCtx: MLXArray,
        sigma: MLXArray,
        positions: MLXArray? = nil,
        ropeCosSin: (MLXArray, MLXArray)? = nil,
        attentionMask: MLXArray? = nil,
        denoiseMask: MLXArray? = nil,
        stgBlocks: Set<Int> = []
    ) -> MLXArray {
        let velocity = dit(
            latent,
            aCtx: aCtx,
            sigma: sigma,
            positions: positions,
            ropeCosSin: ropeCosSin,
            attentionMask: attentionMask,
            denoiseMask: denoiseMask,
            stgBlocks: stgBlocks
        )
        if denoiseMask == nil {
            return dramaBoxToDenoised(latent, velocity: velocity, sigma: sigma.asArray(Float.self)[0])
        }
        let timesteps = sigma.asType(.float32).reshaped([-1] + Array(repeating: 1, count: latent.ndim - 1))
            * denoiseMask!.asType(.float32)
        return (latent.asType(.float32) - velocity.asType(.float32) * timesteps).asType(latent.dtype)
    }
}

func dramaBoxSilencePriorFix(_ latent4d: MLXArray) -> MLXArray {
    if latent4d.dim(2) <= 513 {
        return latent4d
    }
    let a = latent4d[0..., 0..., 511, 0...]
    let b = latent4d[0..., 0..., 514, 0...]
    let pre = latent4d[0..., 0..., 0..<512, 0...]
    let f512 = (a * (2.0 / 3.0) + b * (1.0 / 3.0)).expandedDimensions(axis: 2)
    let f513 = (a * (1.0 / 3.0) + b * (2.0 / 3.0)).expandedDimensions(axis: 2)
    let post = latent4d[0..., 0..., 514..., 0...]
    return concatenated([pre, f512, f513, post], axis: 2)
}

func dramaBoxEulerDenoisingLoop(
    _ state: DramaBoxLatentState,
    sigmas: MLXArray,
    x0Model: DramaBoxX0Model,
    aCtx: MLXArray,
    aCtxNeg: MLXArray?,
    params: DramaBoxGuiderParams,
    positions: MLXArray? = nil,
    denoiseMask: MLXArray? = nil
) -> DramaBoxLatentState {
    let guider = DramaBoxMultiModalGuider(params: params)
    let nSteps = sigmas.dim(0) - 1
    var state = state
    for i in 0..<nSteps {
        let sigma = sigmas[i]
        let sigmaNext = sigmas[i + 1]
        let sigmaBatched = MLX.broadcast(sigma.reshaped([1]), to: [state.latent.dim(0)])
        let cond = x0Model(
            state.latent,
            aCtx: aCtx,
            sigma: sigmaBatched,
            positions: positions,
            attentionMask: state.attentionMask,
            denoiseMask: denoiseMask
        )
        let uncond: MLXArray? = if params.needsUncond, let aCtxNeg {
            x0Model(
                state.latent,
                aCtx: aCtxNeg,
                sigma: sigmaBatched,
                positions: positions,
                attentionMask: state.attentionMask,
                denoiseMask: denoiseMask
            )
        } else {
            nil
        }
        let ptb: MLXArray? = if params.needsPtb {
            x0Model(
                state.latent,
                aCtx: aCtx,
                sigma: sigmaBatched,
                positions: positions,
                attentionMask: state.attentionMask,
                denoiseMask: denoiseMask,
                stgBlocks: params.stgBlockSet
            )
        } else {
            nil
        }
        var pred = guider(cond: cond, uncond: uncond, ptb: ptb, modality: nil)
        pred = dramaBoxPostProcessLatent(
            denoised: pred,
            denoiseMask: state.denoiseMask,
            cleanLatent: state.cleanLatent
        )
        let sigmaVal = sigma.asArray(Float.self)[0]
        if sigmaVal == 0 { break }
        let velocity = dramaBoxToVelocity(state.latent, sigma: sigmaVal, denoised: pred)
        let dt = sigmaNext.asArray(Float.self)[0] - sigmaVal
        let newLatent = state.latent.asType(.float32) + velocity.asType(.float32) * dt
        state = state.replacing(latent: newLatent.asType(state.latent.dtype))
        eval(state.latent)
    }
    return state
}
