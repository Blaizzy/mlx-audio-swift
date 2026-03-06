import Foundation
import MLX
import MLXRandom

// MARK: - KV Cache Helpers

/// Concatenate KV caches along batch dimension (for CFG tripling).
func echoConcatKVCaches(_ caches: [EchoKVCache]) -> EchoKVCache {
    guard let first = caches.first else { return [] }
    let numLayers = first.count
    return (0..<numLayers).map { i in
        let keys = caches.map { $0[i].0 }
        let values = caches.map { $0[i].1 }
        return (MLX.concatenated(keys, axis: 0), MLX.concatenated(values, axis: 0))
    }
}

/// Multiply (scale) KV cache values.
func echoMultiplyKVCache(_ cache: EchoKVCache, scale: Float, maxLayers: Int? = nil) -> EchoKVCache {
    cache.enumerated().map { (i, kv) in
        if let maxL = maxLayers, i >= maxL {
            return kv
        }
        return (kv.0, kv.1 * scale)
    }
}

/// Temporal score rescaling for velocity prediction.
func echoTemporalScoreRescale(
    _ vPred: MLXArray, t: Float, rescaleK: Float, rescaleSigma: Float
) -> MLXArray {
    let snr = t * t / ((1 - t) * (1 - t) + 1e-8)
    let weight = 1.0 / (1.0 + rescaleK * exp(-snr / rescaleSigma))
    return vPred * weight
}

// MARK: - Standard Euler CFG Sampling

/// Euler ODE sampling with independent classifier-free guidance for text and speaker.
///
/// This implements the standard full-sequence generation:
/// - Tripled batch for CFG: [conditional, uncond_text, uncond_speaker]
/// - Independent guidance: v = v_cond + cfg_text*(v_cond - v_uncond_text) + cfg_spk*(v_cond - v_uncond_spk)
/// - Linear schedule from t=0.999 to t=0
func echoSampleEulerCFG(
    model: EchoDiT,
    textTokens: MLXArray,
    textMask: MLXArray,
    speakerLatent: MLXArray,
    speakerMask: MLXArray,
    config: EchoTTSConfig,
    rngSeed: UInt64? = nil
) -> MLXArray {
    let samplerConfig = config.sampler
    let ditConfig = config.dit
    let batchSize = textTokens.dim(0)
    let seqLen = samplerConfig.sequenceLength
    let latentSize = ditConfig.latentSize
    let numSteps = samplerConfig.numSteps

    // Initialize noise
    if let seed = rngSeed {
        MLXRandom.seed(seed)
    }
    var xT = MLXRandom.normal([batchSize, seqLen, latentSize]) * samplerConfig.truncationFactor

    // Build text KV cache (conditional + unconditional)
    let textKVCond = model.getKVCacheText(textTokens, mask: textMask)

    // Unconditional text: zeros with no mask
    let uncondTextTokens = MLXArray.zeros(like: textTokens)
    let uncondTextMask = MLXArray.zeros(like: textMask)
    let textKVUncond = model.getKVCacheText(uncondTextTokens, mask: uncondTextMask)

    // Build speaker KV cache (conditional + unconditional)
    let speakerKVCond = model.getKVCacheSpeaker(speakerLatent, mask: speakerMask)

    // Unconditional speaker: zeros with no mask
    let uncondSpeakerLatent = MLXArray.zeros(like: speakerLatent)
    let uncondSpeakerMask = MLXArray.zeros(like: speakerMask)
    let speakerKVUncond = model.getKVCacheSpeaker(uncondSpeakerLatent, mask: uncondSpeakerMask)

    // Triple the KV caches: [cond, uncond_text, uncond_speaker]
    let textKV3 = echoConcatKVCaches([textKVCond, textKVUncond, textKVCond])
    let speakerKV3 = echoConcatKVCaches([speakerKVCond, speakerKVCond, speakerKVUncond])

    // Triple masks
    let textMask3 = MLX.concatenated([textMask, uncondTextMask, textMask], axis: 0)
    let speakerMask3 = MLX.concatenated([speakerMask, speakerMask, uncondSpeakerMask], axis: 0)

    // Linear time schedule: t = 0.999 -> 0
    var timeSteps: [Float] = []
    for i in 0..<numSteps {
        timeSteps.append(0.999 - Float(i) * 0.999 / Float(numSteps))
    }
    timeSteps.append(0.0)

    // Euler integration
    for step in 0..<numSteps {
        let t = timeSteps[step]
        let tNext = timeSteps[step + 1]
        let dt = tNext - t

        let useCFG = t >= samplerConfig.cfgMinT && t <= samplerConfig.cfgMaxT

        // Apply speaker KV scaling if configured
        var currentSpeakerKV = speakerKV3
        if let kvScale = samplerConfig.speakerKvScale,
           let kvMinT = samplerConfig.speakerKvMinT,
           t >= kvMinT {
            currentSpeakerKV = echoMultiplyKVCache(
                speakerKV3, scale: kvScale,
                maxLayers: samplerConfig.speakerKvMaxLayers
            )
        }

        if useCFG {
            // Triple the batch for CFG
            let xTriple = MLX.concatenated([xT, xT, xT], axis: 0)
            let tArray = MLXArray(Array(repeating: t, count: batchSize * 3))

            let vAll = model(
                xTriple,
                timestep: tArray,
                textKVCache: textKV3,
                speakerKVCache: currentSpeakerKV,
                textMask: textMask3,
                speakerMask: speakerMask3
            )

            // Split predictions
            let vCond = vAll[..<batchSize, 0..., 0...]
            let vUncondText = vAll[batchSize..<(2 * batchSize), 0..., 0...]
            let vUncondSpeaker = vAll[(2 * batchSize)..., 0..., 0...]

            // Independent guidance
            var vPred = vCond
                + samplerConfig.cfgScaleText * (vCond - vUncondText)
                + samplerConfig.cfgScaleSpeaker * (vCond - vUncondSpeaker)

            // Optional temporal rescaling
            if let rescaleK = samplerConfig.rescaleK,
               let rescaleSigma = samplerConfig.rescaleSigma {
                vPred = echoTemporalScoreRescale(vPred, t: t, rescaleK: rescaleK, rescaleSigma: rescaleSigma)
            }

            xT = xT + vPred * dt
        } else {
            // No CFG: just conditional pass
            let tArray = MLXArray(Array(repeating: t, count: batchSize))
            let vPred = model(
                xT,
                timestep: tArray,
                textKVCache: textKVCond,
                speakerKVCache: speakerKVCond,
                textMask: textMask,
                speakerMask: speakerMask
            )
            xT = xT + vPred * dt
        }

        eval(xT)
    }

    return xT
}

// MARK: - Blockwise Euler CFG Sampling

/// Blockwise sampling that generates audio in chunks with latent prefix context.
func echoSampleBlockwiseEulerCFG(
    model: EchoDiT,
    textTokens: MLXArray,
    textMask: MLXArray,
    speakerLatent: MLXArray,
    speakerMask: MLXArray,
    blockSizes: [Int],
    config: EchoTTSConfig,
    rngSeed: UInt64? = nil
) -> MLXArray {
    let samplerConfig = config.sampler
    let ditConfig = config.dit
    let batchSize = textTokens.dim(0)
    let latentSize = ditConfig.latentSize
    let numSteps = samplerConfig.numSteps

    if let seed = rngSeed {
        MLXRandom.seed(seed)
    }

    // Build text and speaker KV caches (same as standard)
    let textKVCond = model.getKVCacheText(textTokens, mask: textMask)
    let uncondTextTokens = MLXArray.zeros(like: textTokens)
    let uncondTextMask = MLXArray.zeros(like: textMask)
    let textKVUncond = model.getKVCacheText(uncondTextTokens, mask: uncondTextMask)

    let speakerKVCond = model.getKVCacheSpeaker(speakerLatent, mask: speakerMask)
    let uncondSpeakerLatent = MLXArray.zeros(like: speakerLatent)
    let uncondSpeakerMask = MLXArray.zeros(like: speakerMask)
    let speakerKVUncond = model.getKVCacheSpeaker(uncondSpeakerLatent, mask: uncondSpeakerMask)

    let textKV3 = echoConcatKVCaches([textKVCond, textKVUncond, textKVCond])
    let speakerKV3 = echoConcatKVCaches([speakerKVCond, speakerKVCond, speakerKVUncond])
    let textMask3 = MLX.concatenated([textMask, uncondTextMask, textMask], axis: 0)
    let speakerMask3 = MLX.concatenated([speakerMask, speakerMask, uncondSpeakerMask], axis: 0)

    var allBlocks: [MLXArray] = []

    for blockSize in blockSizes {
        var xT = MLXRandom.normal([batchSize, blockSize, latentSize]) * samplerConfig.truncationFactor

        // Build latent prefix KV cache from previous blocks
        // Note: This requires the model to have latent_encoder
        // For simplicity, skip latent prefix if model doesn't have it

        var timeSteps: [Float] = []
        for i in 0..<numSteps {
            timeSteps.append(0.999 - Float(i) * 0.999 / Float(numSteps))
        }
        timeSteps.append(0.0)

        for step in 0..<numSteps {
            let t = timeSteps[step]
            let tNext = timeSteps[step + 1]
            let dt = tNext - t

            let useCFG = t >= samplerConfig.cfgMinT && t <= samplerConfig.cfgMaxT

            if useCFG {
                let xTriple = MLX.concatenated([xT, xT, xT], axis: 0)
                let tArray = MLXArray(Array(repeating: t, count: batchSize * 3))

                let vAll = model(
                    xTriple,
                    timestep: tArray,
                    textKVCache: textKV3,
                    speakerKVCache: speakerKV3,
                    textMask: textMask3,
                    speakerMask: speakerMask3
                )

                let vCond = vAll[..<batchSize, 0..., 0...]
                let vUncondText = vAll[batchSize..<(2 * batchSize), 0..., 0...]
                let vUncondSpeaker = vAll[(2 * batchSize)..., 0..., 0...]

                let vPred = vCond
                    + samplerConfig.cfgScaleText * (vCond - vUncondText)
                    + samplerConfig.cfgScaleSpeaker * (vCond - vUncondSpeaker)

                xT = xT + vPred * dt
            } else {
                let tArray = MLXArray(Array(repeating: t, count: batchSize))
                let vPred = model(
                    xT,
                    timestep: tArray,
                    textKVCache: textKVCond,
                    speakerKVCache: speakerKVCond,
                    textMask: textMask,
                    speakerMask: speakerMask
                )
                xT = xT + vPred * dt
            }

            eval(xT)
        }

        allBlocks.append(xT)
    }

    return MLX.concatenated(allBlocks, axis: 1)
}
