import Foundation
@preconcurrency import MLX

struct DramaBoxLatentState {
    var latent: MLXArray
    var denoiseMask: MLXArray
    var positions: MLXArray
    var cleanLatent: MLXArray
    var attentionMask: MLXArray?

    func replacing(
        latent: MLXArray? = nil,
        denoiseMask: MLXArray? = nil,
        positions: MLXArray? = nil,
        cleanLatent: MLXArray? = nil,
        attentionMask: MLXArray?? = nil
    ) -> DramaBoxLatentState {
        DramaBoxLatentState(
            latent: latent ?? self.latent,
            denoiseMask: denoiseMask ?? self.denoiseMask,
            positions: positions ?? self.positions,
            cleanLatent: cleanLatent ?? self.cleanLatent,
            attentionMask: attentionMask ?? self.attentionMask
        )
    }
}

struct DramaBoxAudioLatentTools {
    var patchifier: DramaBoxAudioPatchifier
    var targetShape: DramaBoxAudioLatentShape

    func createInitialState(dtype: DType = .bfloat16) -> DramaBoxLatentState {
        let shape = targetShape
        let zeros = MLXArray.zeros(shape.toTuple(), dtype: dtype)
        let mask = MLXArray.ones([shape.batch, 1, shape.frames, 1], dtype: .float32)
        let positions = patchifier.getPatchGridBounds(shape)
        return patchifyState(
            DramaBoxLatentState(
                latent: zeros,
                denoiseMask: mask,
                positions: positions,
                cleanLatent: zeros,
                attentionMask: nil
            )
        )
    }

    func patchifyState(_ state: DramaBoxLatentState) -> DramaBoxLatentState {
        state.replacing(
            latent: DramaBoxAudioPatchifier.patchify(state.latent),
            denoiseMask: DramaBoxAudioPatchifier.patchify(state.denoiseMask),
            cleanLatent: DramaBoxAudioPatchifier.patchify(state.cleanLatent)
        )
    }

    func unpatchifyState(_ state: DramaBoxLatentState) -> DramaBoxLatentState {
        let C = targetShape.channels
        let F = targetShape.melBins
        return state.replacing(
            latent: DramaBoxAudioPatchifier.unpatchify(state.latent, channels: C, melBins: F),
            cleanLatent: DramaBoxAudioPatchifier.unpatchify(state.cleanLatent, channels: C, melBins: F)
        )
    }

    func clearConditioning(_ state: DramaBoxLatentState) -> DramaBoxLatentState {
        let n = targetShape.tokenCount()
        return DramaBoxLatentState(
            latent: state.latent[0..., 0..<n],
            denoiseMask: MLXArray.ones(like: state.denoiseMask)[0..., 0..<n],
            positions: state.positions[0..., 0..., 0..<n],
            cleanLatent: state.cleanLatent[0..., 0..<n],
            attentionMask: nil
        )
    }
}

struct DramaBoxGaussianNoiser {
    let key: MLXArray

    init(seed: UInt64) {
        self.key = MLXRandom.key(seed)
    }

    func callAsFunction(_ state: DramaBoxLatentState, noiseScale: Float = 1.0) -> DramaBoxLatentState {
        var noise = MLXRandom.normal(state.latent.shape, dtype: .float32, key: key)
        noise = noise.asType(state.latent.dtype)
        var scaledMask = state.denoiseMask * noiseScale
        if scaledMask.shape != state.latent.shape {
            scaledMask = MLX.broadcast(scaledMask, to: state.latent.shape)
        }
        scaledMask = scaledMask.asType(state.latent.dtype)
        let noised = noise * scaledMask + state.latent * (1.0 - scaledMask)
        return state.replacing(latent: noised)
    }
}

func dramaBoxApplyReferenceLatent(
    _ state: DramaBoxLatentState,
    refLatent: MLXArray,
    patchifier: DramaBoxAudioPatchifier = DramaBoxAudioPatchifier(),
    positionOffsetS: Float = 0.5
) -> DramaBoxLatentState {
    let refTokens = DramaBoxAudioPatchifier.patchify(refLatent).asType(state.latent.dtype)
    let batch = state.latent.dim(0)
    let targetCount = state.latent.dim(1)
    let refCount = refTokens.dim(1)
    let refMask = MLXArray.zeros([batch, refCount, 1], dtype: state.denoiseMask.dtype)
    let refShape = DramaBoxAudioLatentShape(
        batch: batch,
        channels: refLatent.dim(1),
        frames: refLatent.dim(2),
        melBins: refLatent.dim(3)
    )
    let refPositions = patchifier.getPatchGridBounds(refShape) + positionOffsetS
    let attentionMask = dramaBoxAsymmetricAttentionMask(
        batch: batch,
        targetTokens: targetCount,
        refTokens: refCount,
        dtype: state.latent.dtype
    )
    return DramaBoxLatentState(
        latent: concatenated([state.latent, refTokens], axis: 1),
        denoiseMask: concatenated([state.denoiseMask, refMask], axis: 1),
        positions: concatenated([state.positions, refPositions], axis: 2),
        cleanLatent: concatenated([state.cleanLatent, refTokens], axis: 1),
        attentionMask: attentionMask
    )
}

func dramaBoxAsymmetricAttentionMask(
    batch: Int,
    targetTokens: Int,
    refTokens: Int,
    dtype: DType
) -> MLXArray {
    let total = targetTokens + refTokens
    let targetRows = MLXArray.zeros([batch, 1, targetTokens, total], dtype: dtype)
    let fill = MLXArray(dramaBoxFinfoMin(dtype), dtype: dtype)
    let refToTarget = MLXArray.full([batch, 1, refTokens, targetTokens], values: fill, dtype: dtype)
    let refToRef = MLXArray.zeros([batch, 1, refTokens, refTokens], dtype: dtype)
    let refRows = concatenated([refToTarget, refToRef], axis: -1)
    return concatenated([targetRows, refRows], axis: -2)
}

func dramaBoxToVelocity(_ sample: MLXArray, sigma: Float, denoised: MLXArray) -> MLXArray {
    precondition(sigma != 0)
    return ((sample.asType(.float32) - denoised.asType(.float32)) / sigma).asType(sample.dtype)
}

func dramaBoxToDenoised(_ sample: MLXArray, velocity: MLXArray, sigma: Float) -> MLXArray {
    (sample.asType(.float32) - velocity.asType(.float32) * sigma).asType(sample.dtype)
}

func dramaBoxPostProcessLatent(
    denoised: MLXArray,
    denoiseMask: MLXArray,
    cleanLatent: MLXArray
) -> MLXArray {
    denoiseMask * denoised + (1.0 - denoiseMask) * cleanLatent
}
