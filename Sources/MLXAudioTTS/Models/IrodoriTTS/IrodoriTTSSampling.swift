import Foundation
import MLX

// MARK: - Rectified-Flow Euler sampler with Classifier-Free Guidance
// Faithful port of mlx_audio/tts/models/irodori_tts/sampling.py (sample_euler_cfg).
// Three CFG modes:
//   independent — text/speaker/caption guidance in a single batched forward (up to 4×).
//   joint       — single combined unconditional (two sequential passes).
//   alternating — text-uncond and context-uncond alternate per step (single-context only).

typealias IrodoriKVCacheList = [IrodoriKVCache]

/// Concatenate KV caches from multiple conditions along the batch axis.
private func irodoriConcatKVCaches(_ caches: [IrodoriKVCacheList]) -> IrodoriKVCacheList {
    var result = IrodoriKVCacheList()
    let layers = caches[0].count
    result.reserveCapacity(layers)
    for i in 0..<layers {
        let k = MLX.concatenated(caches.map { $0[i].keys }, axis: 0)
        let v = MLX.concatenated(caches.map { $0[i].values }, axis: 0)
        result.append((keys: k, values: v))
    }
    return result
}

/// Return a new KV cache with (the first `maxLayers`) speaker KVs scaled.
private func irodoriScaleKVCache(
    _ cache: IrodoriKVCacheList,
    scale: Float,
    maxLayers: Int? = nil
) -> IrodoriKVCacheList {
    let n = maxLayers.map { min($0, cache.count) } ?? cache.count
    var result = IrodoriKVCacheList()
    result.reserveCapacity(cache.count)
    for (i, kv) in cache.enumerated() {
        if i < n {
            result.append((keys: kv.keys * scale, values: kv.values * scale))
        } else {
            result.append(kv)
        }
    }
    return result
}

/// Temporal score rescaling from https://arxiv.org/pdf/2510.01184.
private func irodoriTemporalScoreRescale(
    vPred: MLXArray, xT: MLXArray, t: Float, rescaleK: Float, rescaleSigma: Float
) -> MLXArray {
    if t >= 1.0 { return vPred }
    let oneMinusT = 1.0 - t
    let snr = (oneMinusT * oneMinusT) / (t * t)
    let sigmaSq = rescaleSigma * rescaleSigma
    let ratio = (snr * sigmaSq + 1.0) / (snr * sigmaSq / rescaleK + 1.0)
    return (ratio * (oneMinusT * vPred + xT) - xT) / oneMinusT
}

/// Parameters for the Euler-CFG sampler (mirrors sampling.py kwargs).
struct IrodoriSamplerParams {
    var numSteps: Int = 40
    var cfgScaleText: Float = 3.0
    var cfgScaleSpeaker: Float = 5.0
    var cfgScaleCaption: Float = 3.0
    var cfgGuidanceMode: String = "independent"
    var cfgMinT: Float = 0.5
    var cfgMaxT: Float = 1.0
    var truncationFactor: Float? = nil
    var rescaleK: Float? = nil
    var rescaleSigma: Float? = nil
    var contextKvCache: Bool = true
    var speakerKvScale: Float? = nil
    var speakerKvMinT: Float? = nil
    var speakerKvMaxLayers: Int? = nil
    var tScheduleMode: String = "linear"
    var swayCoeff: Float = -1.0
}

/// Build the t-schedule (length numSteps+1) including optional Sway Sampling (v3).
private func irodoriTSchedule(numSteps: Int, initScale: Float, mode: String, swayCoeff: Float) -> [Float] {
    if mode.trimmingCharacters(in: .whitespaces).lowercased() == "sway" {
        var out = [Float]()
        out.reserveCapacity(numSteps + 1)
        for i in 0...numSteps {
            let u0 = Float(i) / Float(numSteps)
            var u = u0 + swayCoeff * (cos(0.5 * Float.pi * u0) + u0 - 1.0)
            u = min(max(u, 0.0), 1.0)
            out.append((1.0 - u) * initScale)
        }
        return out
    }
    // linear: linspace(1*initScale, 0, numSteps+1)
    var out = [Float]()
    out.reserveCapacity(numSteps + 1)
    for i in 0...numSteps {
        let frac = Float(i) / Float(numSteps)
        out.append(initScale * (1.0 - frac))
    }
    return out
}

/// Euler sampler for the Rectified-Flow ODE with CFG.
/// Returns latent of shape (batch, sequenceLength, latentDim).
func irodoriSampleEulerCFG(
    model: IrodoriDiT,
    textInputIDs: MLXArray,
    textMask: MLXArray,
    refLatent: MLXArray?,
    refMask: MLXArray?,
    captionInputIDs: MLXArray?,
    captionMask: MLXArray?,
    latentDim: Int,
    sequenceLength: Int,
    rngSeed: Int,
    params: IrodoriSamplerParams
) throws -> MLXArray {
    let useSpk = model.cfg.useSpeakerConditionResolved
    let useCap = model.cfg.useCaptionCondition
    let isDual = useSpk && useCap

    let cfgScaleText = params.cfgScaleText
    let cfgScaleSpeaker = params.cfgScaleSpeaker
    let cfgScaleCaption = params.cfgScaleCaption
    let cfgScaleContext = isDual ? cfgScaleSpeaker : (useCap ? cfgScaleCaption : cfgScaleSpeaker)

    let mode = params.cfgGuidanceMode.trimmingCharacters(in: .whitespaces).lowercased()
    precondition(["independent", "joint", "alternating"].contains(mode),
                 "Unknown cfgGuidanceMode=\(mode)")

    let batchSize = textInputIDs.shape[0]
    let hasTextCfg = cfgScaleText > 0
    let hasSpeakerCfg = cfgScaleSpeaker > 0 && useSpk
    let hasCaptionCfg = cfgScaleCaption > 0 && useCap
    let hasContextCfg = isDual ? false : (cfgScaleContext > 0)

    // ---- encode all conditions ----
    let enc = model.encodeConditionsFull(
        textInputIDs: textInputIDs,
        textMask: textMask,
        refLatent: refLatent,
        refMask: refMask,
        captionInputIDs: captionInputIDs,
        captionMask: captionMask
    )
    let textStateCond = enc.textState
    let textMaskCond = enc.textMask

    // For single-context caption models, alias caption into the "speaker" slot.
    var speakerStateCond: MLXArray?
    var speakerMaskCond: MLXArray?
    var captionStateCond: MLXArray?
    var captionMaskCond: MLXArray?
    if !isDual && useCap {
        speakerStateCond = enc.captionState
        speakerMaskCond = enc.captionMask
        captionStateCond = nil
        captionMaskCond = nil
    } else {
        speakerStateCond = enc.speakerState
        speakerMaskCond = enc.speakerMask
        captionStateCond = enc.captionState
        captionMaskCond = enc.captionMask
    }

    eval(textStateCond)
    if let s = speakerStateCond { eval(s) }
    if let c = captionStateCond { eval(c) }

    // unconditioned states
    let textStateUncond = MLXArray.zeros(like: textStateCond)
    let textMaskUncond = MLXArray.zeros(like: textMaskCond)
    let speakerStateUncond = speakerStateCond.map { MLXArray.zeros(like: $0) }
    let speakerMaskUncond = speakerMaskCond.map { MLXArray.zeros(like: $0) }
    let captionStateUncond = captionStateCond.map { MLXArray.zeros(like: $0) }
    let captionMaskUncond = captionMaskCond.map { MLXArray.zeros(like: $0) }

    // ---- build KV caches ----
    let useKVCache = params.contextKvCache || (params.speakerKvScale != nil)

    var kvTextCond: IrodoriKVCacheList?
    var kvSpeakerCond: IrodoriKVCacheList?
    var kvCaptionCond: IrodoriKVCacheList?
    var kvTextCfg: IrodoriKVCacheList?
    var kvSpeakerCfg: IrodoriKVCacheList?
    var kvCaptionCfg: IrodoriKVCacheList?
    var kvTextUncondJoint: IrodoriKVCacheList?
    var kvSpeakerUncondJoint: IrodoriKVCacheList?
    var kvCaptionUncondJoint: IrodoriKVCacheList?
    var kvTextUncondAlt: IrodoriKVCacheList?
    var kvSpeakerUncondAlt: IrodoriKVCacheList?

    if useKVCache {
        let c = try model.buildKVCache(
            textState: textStateCond, speakerState: speakerStateCond, captionState: captionStateCond)
        kvTextCond = c.kvText
        kvSpeakerCond = c.kvSpeaker
        kvCaptionCond = c.kvCaption
        if let scale = params.speakerKvScale, let kvs = kvSpeakerCond {
            kvSpeakerCond = irodoriScaleKVCache(kvs, scale: scale, maxLayers: params.speakerKvMaxLayers)
        }

        if mode == "independent" {
            if isDual {
                let nBundles = 1 + [hasTextCfg, hasSpeakerCfg, hasCaptionCfg].filter { $0 }.count
                if nBundles > 1 {
                    if let t = kvTextCond { kvTextCfg = irodoriConcatKVCaches(Array(repeating: t, count: nBundles)) }
                    if let s = kvSpeakerCond { kvSpeakerCfg = irodoriConcatKVCaches(Array(repeating: s, count: nBundles)) }
                    if let cp = kvCaptionCond { kvCaptionCfg = irodoriConcatKVCaches(Array(repeating: cp, count: nBundles)) }
                }
            } else {
                if hasTextCfg && hasContextCfg {
                    if let t = kvTextCond { kvTextCfg = irodoriConcatKVCaches([t, t, t]) }
                    if let s = kvSpeakerCond { kvSpeakerCfg = irodoriConcatKVCaches([s, s, s]) }
                } else if hasTextCfg || hasContextCfg {
                    if let t = kvTextCond { kvTextCfg = irodoriConcatKVCaches([t, t]) }
                    if let s = kvSpeakerCond { kvSpeakerCfg = irodoriConcatKVCaches([s, s]) }
                }
            }
        } else if mode == "joint" {
            if isDual {
                if hasTextCfg || hasSpeakerCfg || hasCaptionCfg {
                    let u = try model.buildKVCache(
                        textState: textStateUncond, speakerState: speakerStateUncond, captionState: captionStateUncond)
                    kvTextUncondJoint = u.kvText
                    kvSpeakerUncondJoint = u.kvSpeaker
                    kvCaptionUncondJoint = u.kvCaption
                }
            } else {
                if hasTextCfg || hasContextCfg {
                    let u = try model.buildKVCache(
                        textState: textStateUncond, speakerState: speakerStateUncond, captionState: nil)
                    kvTextUncondJoint = u.kvText
                    kvSpeakerUncondJoint = u.kvSpeaker
                }
            }
        } else if mode == "alternating" {
            if !isDual {
                if hasTextCfg {
                    let a = try model.buildKVCache(
                        textState: textStateUncond, speakerState: speakerStateCond, captionState: nil)
                    kvTextUncondAlt = a.kvText
                }
                if hasContextCfg {
                    let b = try model.buildKVCache(
                        textState: textStateCond, speakerState: speakerStateUncond, captionState: nil)
                    kvSpeakerUncondAlt = b.kvSpeaker
                    if let scale = params.speakerKvScale, let kvs = kvSpeakerUncondAlt {
                        kvSpeakerUncondAlt = irodoriScaleKVCache(kvs, scale: scale, maxLayers: params.speakerKvMaxLayers)
                    }
                }
            }
        }

        if let t = kvTextCond { for kv in t { eval(kv.keys, kv.values) } }
        if let s = kvSpeakerCond { for kv in s { eval(kv.keys, kv.values) } }
        if let cp = kvCaptionCond { for kv in cp { eval(kv.keys, kv.values) } }
    }

    // ---- initial noise ----
    MLXRandom.seed(UInt64(rngSeed))
    let initScale: Float = 0.999
    var xT = MLXRandom.normal([batchSize, sequenceLength, latentDim])
    if let tf = params.truncationFactor { xT = xT * tf }

    let tSchedule = irodoriTSchedule(
        numSteps: params.numSteps, initScale: initScale,
        mode: params.tScheduleMode, swayCoeff: params.swayCoeff)

    var speakerKvActive = params.speakerKvScale != nil

    // ---- Euler steps ----
    for i in 0..<params.numSteps {
        let t = tSchedule[i]
        let tNext = tSchedule[i + 1]
        let tArr = MLX.full([batchSize], values: t)
        let useCfg = (hasTextCfg || hasSpeakerCfg) && (params.cfgMinT <= t && t <= params.cfgMaxT)

        var vPred: MLXArray

        if useCfg && mode == "independent" && isDual {
            // Dual: build bundle list [cond, text-uncond?, spk-uncond?, cap-uncond?]
            var bx = [xT], bt = [textStateCond], btm = [textMaskCond]
            var bs = [speakerStateCond], bsm = [speakerMaskCond]
            var bc = [captionStateCond], bcm = [captionMaskCond]
            if hasTextCfg {
                bx.append(xT); bt.append(textStateUncond); btm.append(textMaskUncond)
                bs.append(speakerStateCond); bsm.append(speakerMaskCond)
                bc.append(captionStateCond); bcm.append(captionMaskCond)
            }
            if hasSpeakerCfg {
                bx.append(xT); bt.append(textStateCond); btm.append(textMaskCond)
                bs.append(speakerStateUncond); bsm.append(speakerMaskUncond)
                bc.append(captionStateCond); bcm.append(captionMaskCond)
            }
            if hasCaptionCfg {
                bx.append(xT); bt.append(textStateCond); btm.append(textMaskCond)
                bs.append(speakerStateCond); bsm.append(speakerMaskCond)
                bc.append(captionStateUncond); bcm.append(captionMaskUncond)
            }
            let nB = bx.count
            let vOut = try model.forwardWithConditions(
                xT: MLX.concatenated(bx, axis: 0),
                t: MLX.full([batchSize * nB], values: t),
                textState: MLX.concatenated(bt, axis: 0),
                textMask: MLX.concatenated(btm, axis: 0),
                speakerState: irodoriConcatOpt(bs),
                speakerMask: irodoriConcatOpt(bsm),
                kvText: kvTextCfg, kvSpeaker: kvSpeakerCfg,
                captionState: irodoriConcatOpt(bc), captionMask: irodoriConcatOpt(bcm),
                kvCaption: kvCaptionCfg)
            let splits = vOut.split(parts: nB, axis: 0)
            let vCond = splits[0]
            vPred = vCond
            var idx = 1
            if hasTextCfg { vPred = vPred + cfgScaleText * (vCond - splits[idx]); idx += 1 }
            if hasSpeakerCfg { vPred = vPred + cfgScaleSpeaker * (vCond - splits[idx]); idx += 1 }
            if hasCaptionCfg { vPred = vPred + cfgScaleCaption * (vCond - splits[idx]) }

        } else if useCfg && mode == "independent" && hasTextCfg && hasContextCfg {
            // single-context 3× batch: [cond, text-uncond, context-uncond]
            let vOut = try model.forwardWithConditions(
                xT: MLX.concatenated([xT, xT, xT], axis: 0),
                t: MLX.full([batchSize * 3], values: t),
                textState: MLX.concatenated([textStateCond, textStateUncond, textStateCond], axis: 0),
                textMask: MLX.concatenated([textMaskCond, textMaskUncond, textMaskCond], axis: 0),
                speakerState: irodoriConcatOpt([speakerStateCond, speakerStateCond, speakerStateUncond]),
                speakerMask: irodoriConcatOpt([speakerMaskCond, speakerMaskCond, speakerMaskUncond]),
                kvText: kvTextCfg, kvSpeaker: kvSpeakerCfg)
            let s = vOut.split(parts: 3, axis: 0)
            vPred = s[0] + cfgScaleText * (s[0] - s[1]) + cfgScaleContext * (s[0] - s[2])

        } else if useCfg && mode == "independent" && hasTextCfg {
            let vOut = try model.forwardWithConditions(
                xT: MLX.concatenated([xT, xT], axis: 0),
                t: MLX.full([batchSize * 2], values: t),
                textState: MLX.concatenated([textStateCond, textStateUncond], axis: 0),
                textMask: MLX.concatenated([textMaskCond, textMaskUncond], axis: 0),
                speakerState: irodoriConcatOpt([speakerStateCond, speakerStateCond]),
                speakerMask: irodoriConcatOpt([speakerMaskCond, speakerMaskCond]),
                kvText: kvTextCfg, kvSpeaker: kvSpeakerCfg)
            let s = vOut.split(parts: 2, axis: 0)
            vPred = s[0] + cfgScaleText * (s[0] - s[1])

        } else if useCfg && mode == "independent" {  // context-only
            let vOut = try model.forwardWithConditions(
                xT: MLX.concatenated([xT, xT], axis: 0),
                t: MLX.full([batchSize * 2], values: t),
                textState: MLX.concatenated([textStateCond, textStateCond], axis: 0),
                textMask: MLX.concatenated([textMaskCond, textMaskCond], axis: 0),
                speakerState: irodoriConcatOpt([speakerStateCond, speakerStateUncond]),
                speakerMask: irodoriConcatOpt([speakerMaskCond, speakerMaskUncond]),
                kvText: kvTextCfg, kvSpeaker: kvSpeakerCfg)
            let s = vOut.split(parts: 2, axis: 0)
            vPred = s[0] + cfgScaleContext * (s[0] - s[1])

        } else if useCfg && mode == "joint" {
            let jointScale: Float
            if isDual {
                let scales = [(cfgScaleText, hasTextCfg), (cfgScaleSpeaker, hasSpeakerCfg), (cfgScaleCaption, hasCaptionCfg)]
                    .filter { $0.1 }.map { $0.0 }
                jointScale = scales.first ?? cfgScaleText
            } else if hasTextCfg && hasContextCfg {
                precondition(abs(cfgScaleText - cfgScaleContext) <= 1e-6,
                             "joint mode requires equal text/context scales")
                jointScale = cfgScaleText
            } else {
                jointScale = hasTextCfg ? cfgScaleText : cfgScaleContext
            }
            let vCond = try model.forwardWithConditions(
                xT: xT, t: tArr,
                textState: textStateCond, textMask: textMaskCond,
                speakerState: speakerStateCond, speakerMask: speakerMaskCond,
                kvText: kvTextCond, kvSpeaker: kvSpeakerCond,
                captionState: captionStateCond, captionMask: captionMaskCond, kvCaption: kvCaptionCond)
            let vUncond = try model.forwardWithConditions(
                xT: xT, t: tArr,
                textState: textStateUncond, textMask: textMaskUncond,
                speakerState: speakerStateUncond, speakerMask: speakerMaskUncond,
                kvText: kvTextUncondJoint, kvSpeaker: kvSpeakerUncondJoint,
                captionState: captionStateUncond, captionMask: captionMaskUncond, kvCaption: kvCaptionUncondJoint)
            vPred = vCond + jointScale * (vCond - vUncond)

        } else if useCfg {  // alternating (single-context only)
            let vCond = try model.forwardWithConditions(
                xT: xT, t: tArr,
                textState: textStateCond, textMask: textMaskCond,
                speakerState: speakerStateCond, speakerMask: speakerMaskCond,
                kvText: kvTextCond, kvSpeaker: kvSpeakerCond)
            let useTextUncond = (hasTextCfg && hasContextCfg && i % 2 == 0) || (hasTextCfg && !hasContextCfg)
            if useTextUncond {
                let vUncond = try model.forwardWithConditions(
                    xT: xT, t: tArr,
                    textState: textStateUncond, textMask: textMaskUncond,
                    speakerState: speakerStateCond, speakerMask: speakerMaskCond,
                    kvText: kvTextUncondAlt, kvSpeaker: kvSpeakerCond)
                vPred = vCond + cfgScaleText * (vCond - vUncond)
            } else {
                let vUncond = try model.forwardWithConditions(
                    xT: xT, t: tArr,
                    textState: textStateCond, textMask: textMaskCond,
                    speakerState: speakerStateUncond, speakerMask: speakerMaskUncond,
                    kvText: kvTextCond, kvSpeaker: kvSpeakerUncondAlt)
                vPred = vCond + cfgScaleContext * (vCond - vUncond)
            }

        } else {
            // no CFG this step
            vPred = try model.forwardWithConditions(
                xT: xT, t: tArr,
                textState: textStateCond, textMask: textMaskCond,
                speakerState: speakerStateCond, speakerMask: speakerMaskCond,
                kvText: kvTextCond, kvSpeaker: kvSpeakerCond,
                captionState: captionStateCond, captionMask: captionMaskCond, kvCaption: kvCaptionCond)
        }

        if let rk = params.rescaleK, let rs = params.rescaleSigma {
            vPred = irodoriTemporalScoreRescale(vPred: vPred, xT: xT, t: t, rescaleK: rk, rescaleSigma: rs)
        }

        // speaker KV scale rollback at threshold
        if speakerKvActive, let minT = params.speakerKvMinT, let scale = params.speakerKvScale,
           tNext < minT && minT <= t, let kvs = kvSpeakerCond {
            let inv = 1.0 / scale
            kvSpeakerCond = irodoriScaleKVCache(kvs, scale: inv, maxLayers: params.speakerKvMaxLayers)
            if kvSpeakerCfg != nil, let s = kvSpeakerCond {
                let nRep = (!isDual && hasTextCfg && hasContextCfg) ? 3 : 2
                kvSpeakerCfg = irodoriConcatKVCaches(Array(repeating: s, count: nRep))
            }
            if let alt = kvSpeakerUncondAlt {
                kvSpeakerUncondAlt = irodoriScaleKVCache(alt, scale: inv, maxLayers: params.speakerKvMaxLayers)
            }
            speakerKvActive = false
        }

        // Euler update: x_{t-dt} = x_t + v * (t_next - t)
        xT = xT + vPred * (tNext - t)
        eval(xT)
    }

    return xT
}

/// Concatenate a list of optional MLXArrays along batch axis; nil if all nil.
private func irodoriConcatOpt(_ arrays: [MLXArray?]) -> MLXArray? {
    let present = arrays.compactMap { $0 }
    guard present.count == arrays.count, !present.isEmpty else { return nil }
    return MLX.concatenated(present, axis: 0)
}
