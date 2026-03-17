import Foundation
import MLX

private let echoTtsDefaultTruncationFactor: Float = 0.96

private func echoTtsConcatKVCaches(_ caches: [EchoTTSKVCache]...) -> [EchoTTSKVCache] {
    precondition(!caches.isEmpty, "Expected at least one cache collection")
    return (0 ..< caches[0].count).map { layer in
        (
            MLX.concatenated(caches.map { $0[layer].keys }, axis: 0),
            MLX.concatenated(caches.map { $0[layer].values }, axis: 0)
        )
    }
}

private func echoTtsScaleKVCache(
    _ cache: [EchoTTSKVCache],
    scale: Float,
    maxLayers: Int?
) -> [EchoTTSKVCache] {
    let limit = maxLayers.map { min($0, cache.count) } ?? cache.count
    return cache.enumerated().map { index, entry in
        guard index < limit else { return entry }
        return (entry.keys * scale, entry.values * scale)
    }
}

private func echoTtsTemporalScoreRescale(
    prediction: MLXArray,
    xT: MLXArray,
    t: Float,
    rescaleK: Float,
    rescaleSigma: Float
) -> MLXArray {
    guard t < 1 else { return prediction }
    let snr = pow(1 - t, 2) / pow(t, 2)
    let ratio = (snr * pow(rescaleSigma, 2) + 1) / ((snr * pow(rescaleSigma, 2) / rescaleK) + 1)
    return (1 / (1 - t)) * (ratio * ((1 - t) * prediction + xT) - xT)
}

func echoTtsSampleEulerCFGIndependentGuidances(
    model: EchoDiT,
    speakerLatent: MLXArray,
    speakerMask: MLXArray,
    textInputIDs: MLXArray,
    textMask: MLXArray,
    rngSeed: Int,
    numSteps: Int = 40,
    cfgScaleText: Float = 3,
    cfgScaleSpeaker: Float = 8,
    cfgMinT: Float = 0.5,
    cfgMaxT: Float = 1,
    truncationFactor: Float? = nil,
    rescaleK: Float? = nil,
    rescaleSigma: Float? = nil,
    speakerKVScale: Float? = nil,
    speakerKVMaxLayers: Int? = nil,
    speakerKVMinT: Float? = nil,
    sequenceLength: Int = 640
) -> MLXArray {
    let batchSize = textInputIDs.shape[0]
    let initScale: Float = 0.999

    seed(UInt64(truncatingIfNeeded: rngSeed))
    let schedule = MLX.linspace(initScale, Float(0), count: numSteps + 1)

    let textMaskUncond = MLXArray.zeros(textMask.shape, dtype: .bool)
    let speakerMaskUncond = MLXArray.zeros(speakerMask.shape, dtype: .bool)

    let kvTextCond = model.getKVCacheText(textInputIDs, textMask: textMask)
    var kvSpeakerCond = model.getKVCacheSpeaker(speakerLatent)
    if let speakerKVScale {
        kvSpeakerCond = echoTtsScaleKVCache(kvSpeakerCond, scale: speakerKVScale, maxLayers: speakerKVMaxLayers)
    }

    let kvTextFull = echoTtsConcatKVCaches(kvTextCond, kvTextCond, kvTextCond)
    var kvSpeakerFull = echoTtsConcatKVCaches(kvSpeakerCond, kvSpeakerCond, kvSpeakerCond)
    let fullTextMask = MLX.concatenated([textMask, textMaskUncond, textMask], axis: 0)
    let fullSpeakerMask = MLX.concatenated([speakerMask, speakerMask, speakerMaskUncond], axis: 0)

    let latentSize = model.outProj.weight.shape[0]
    var xT = MLXRandom.normal([batchSize, sequenceLength, latentSize])
    xT = xT * (truncationFactor ?? echoTtsDefaultTruncationFactor)

    for step in 0 ..< numSteps {
        let t = schedule[step].item(Float.self)
        let tNext = schedule[step + 1].item(Float.self)
        let hasCFG = cfgMinT <= t && t <= cfgMaxT

        let prediction: MLXArray
        if hasCFG {
            let xFull = MLX.concatenated([xT, xT, xT], axis: 0)
            let times = MLX.full([batchSize * 3], values: t)
            let output = model(
                x: xFull,
                t: times,
                textMask: fullTextMask,
                speakerMask: fullSpeakerMask,
                kvCacheText: kvTextFull,
                kvCacheSpeaker: kvSpeakerFull
            )
            let parts = output.split(parts: 3, axis: 0)
            prediction = parts[0]
                + cfgScaleText * (parts[0] - parts[1])
                + cfgScaleSpeaker * (parts[0] - parts[2])
        } else {
            prediction = model(
                x: xT,
                t: MLX.full([batchSize], values: t),
                textMask: textMask,
                speakerMask: speakerMask,
                kvCacheText: kvTextCond,
                kvCacheSpeaker: kvSpeakerCond
            )
        }

        let rescaled: MLXArray
        if let rescaleK, let rescaleSigma {
            rescaled = echoTtsTemporalScoreRescale(
                prediction: prediction,
                xT: xT,
                t: t,
                rescaleK: rescaleK,
                rescaleSigma: rescaleSigma
            )
        } else {
            rescaled = prediction
        }

        if let speakerKVScale, let speakerKVMinT, tNext < speakerKVMinT && speakerKVMinT <= t {
            kvSpeakerCond = echoTtsScaleKVCache(
                kvSpeakerCond,
                scale: 1 / speakerKVScale,
                maxLayers: speakerKVMaxLayers
            )
            kvSpeakerFull = echoTtsConcatKVCaches(kvSpeakerCond, kvSpeakerCond, kvSpeakerCond)
        }

        xT = xT + rescaled * (tNext - t)
    }

    return xT
}

func echoTtsSampleBlockwiseEulerCFGIndependentGuidances(
    model: EchoDiT,
    speakerLatent: MLXArray,
    speakerMask: MLXArray,
    textInputIDs: MLXArray,
    textMask: MLXArray,
    rngSeed: Int,
    blockSizes: [Int],
    numSteps: Int = 40,
    cfgScaleText: Float = 3,
    cfgScaleSpeaker: Float = 8,
    cfgMinT: Float = 0.5,
    cfgMaxT: Float = 1,
    truncationFactor: Float? = nil,
    rescaleK: Float? = nil,
    rescaleSigma: Float? = nil,
    speakerKVScale: Float? = nil,
    speakerKVMaxLayers: Int? = nil,
    speakerKVMinT: Float? = nil,
    continuationLatent: MLXArray? = nil
) throws -> MLXArray {
    let batchSize = textInputIDs.shape[0]
    let initScale: Float = 0.999

    seed(UInt64(truncatingIfNeeded: rngSeed))
    let schedule = MLX.linspace(initScale, Float(0), count: numSteps + 1)

    let textMaskUncond = MLXArray.zeros(textMask.shape, dtype: .bool)
    let speakerMaskUncond = MLXArray.zeros(speakerMask.shape, dtype: .bool)

    let kvTextCond = model.getKVCacheText(textInputIDs, textMask: textMask)
    var kvSpeakerCond = model.getKVCacheSpeaker(speakerLatent)
    let kvTextFull = echoTtsConcatKVCaches(kvTextCond, kvTextCond, kvTextCond)
    var kvSpeakerFull = echoTtsConcatKVCaches(kvSpeakerCond, kvSpeakerCond, kvSpeakerCond)
    let fullTextMask = MLX.concatenated([textMask, textMaskUncond, textMask], axis: 0)
    let fullSpeakerMask = MLX.concatenated([speakerMask, speakerMask, speakerMaskUncond], axis: 0)

    let latentSize = model.outProj.weight.shape[0]
    var generatedChunks: [MLXArray] = []
    var startPos = 0

    if let continuationLatent {
        generatedChunks.append(continuationLatent)
        startPos = continuationLatent.shape[1]
    }

    for blockSize in blockSizes {
        if let speakerKVScale {
            kvSpeakerCond = echoTtsScaleKVCache(kvSpeakerCond, scale: speakerKVScale, maxLayers: speakerKVMaxLayers)
            kvSpeakerFull = echoTtsConcatKVCaches(kvSpeakerCond, kvSpeakerCond, kvSpeakerCond)
        }

        let prefixLatent = generatedChunks.isEmpty
            ? MLXArray.zeros([batchSize, 0, latentSize], dtype: .float32)
            : MLX.concatenated(generatedChunks, axis: 1)
        let fullPrefixLatent = MLX.concatenated([prefixLatent, prefixLatent, prefixLatent], axis: 0)
        let kvLatentFull = try model.getKVCacheLatent(fullPrefixLatent)
        let kvLatentCond = kvLatentFull.map {
            (
                $0.keys[0..<batchSize, 0..., 0..., 0...],
                $0.values[0..<batchSize, 0..., 0..., 0...]
            )
        }

        var xT = MLXRandom.normal([batchSize, blockSize, latentSize])
        xT = xT * (truncationFactor ?? echoTtsDefaultTruncationFactor)

        for step in 0 ..< numSteps {
            let t = schedule[step].item(Float.self)
            let tNext = schedule[step + 1].item(Float.self)
            let hasCFG = cfgMinT <= t && t <= cfgMaxT

            let prediction: MLXArray
            if hasCFG {
                let output = model(
                    x: MLX.concatenated([xT, xT, xT], axis: 0),
                    t: MLX.full([batchSize * 3], values: t),
                    textMask: fullTextMask,
                    speakerMask: fullSpeakerMask,
                    kvCacheText: kvTextFull,
                    kvCacheSpeaker: kvSpeakerFull,
                    startPos: startPos,
                    kvCacheLatent: kvLatentFull
                )
                let parts = output.split(parts: 3, axis: 0)
                prediction = parts[0]
                    + cfgScaleText * (parts[0] - parts[1])
                    + cfgScaleSpeaker * (parts[0] - parts[2])
            } else {
                prediction = model(
                    x: xT,
                    t: MLX.full([batchSize], values: t),
                    textMask: textMask,
                    speakerMask: speakerMask,
                    kvCacheText: kvTextCond,
                    kvCacheSpeaker: kvSpeakerCond,
                    startPos: startPos,
                    kvCacheLatent: kvLatentCond
                )
            }

            let rescaled: MLXArray
            if let rescaleK, let rescaleSigma {
                rescaled = echoTtsTemporalScoreRescale(
                    prediction: prediction,
                    xT: xT,
                    t: t,
                    rescaleK: rescaleK,
                    rescaleSigma: rescaleSigma
                )
            } else {
                rescaled = prediction
            }

            if let speakerKVScale, let speakerKVMinT, tNext < speakerKVMinT && speakerKVMinT <= t {
                kvSpeakerCond = echoTtsScaleKVCache(
                    kvSpeakerCond,
                    scale: 1 / speakerKVScale,
                    maxLayers: speakerKVMaxLayers
                )
                kvSpeakerFull = echoTtsConcatKVCaches(kvSpeakerCond, kvSpeakerCond, kvSpeakerCond)
            }

            xT = xT + rescaled * (tNext - t)
        }

        generatedChunks.append(xT)
        startPos += blockSize
    }

    return MLX.concatenated(generatedChunks, axis: 1)
}
