@preconcurrency import MLX
import MLXNN

public func mossTTSApplyRepetitionPenaltyDelayPattern(
    logits: MLXArray,
    previousTokens: MLXArray?,
    penalty: Float
) -> MLXArray {
    guard let previousTokens, penalty != 1.0, previousTokens.size > 0 else {
        return logits
    }

    let vocabSize = logits.dim(logits.ndim - 1)
    let previous = previousTokens.asType(.int32)
    let sanitized = MLX.where(
        previous .< MLXArray(Int32(0)),
        MLXArray(Int32(0)),
        MLX.where(previous .>= MLXArray(Int32(vocabSize)), MLXArray(Int32(0)), previous)
    )
    let penalized = MLX.where(logits .> MLXArray(0), logits / MLXArray(penalty), logits * MLXArray(penalty))

    if logits.ndim == 2 {
        let ids = Array(Set(sanitized.asArray(Int32.self).map(Int.init))).filter { $0 >= 0 && $0 < vocabSize }
        guard !ids.isEmpty else { return logits }
        let tokenIDs = MLXArray(ids.map(Int32.init)).reshaped([1, -1])
        let selected = takeAlong(penalized, tokenIDs, axis: -1)
        return putAlong(logits, tokenIDs, values: selected, axis: -1)
    }

    if logits.ndim == 3 {
        var heads: [MLXArray] = []
        let headCount = logits.dim(1)
        for head in 0 ..< headCount {
            let headLogits = logits[0..., head, 0...]
            let headPenalized = penalized[0..., head, 0...]
            let headPrevious = sanitized[0..., 0..., head]
            let ids = Array(Set(headPrevious.asArray(Int32.self).map(Int.init))).filter { $0 >= 0 && $0 < vocabSize }
            if ids.isEmpty {
                heads.append(headLogits)
            } else {
                let tokenIDs = MLXArray(ids.map(Int32.init)).reshaped([1, -1])
                let selected = takeAlong(headPenalized, tokenIDs, axis: -1)
                heads.append(putAlong(headLogits, tokenIDs, values: selected, axis: -1))
            }
        }
        return MLX.stacked(heads, axis: 1)
    }

    return logits
}

public func mossTTSSampleToken(
    logits: MLXArray,
    previousTokens: MLXArray? = nil,
    repetitionPenalty: Float = 1.0,
    topP: Float? = nil,
    topK: Int? = nil,
    doSample: Bool = true
) -> MLXArray {
    var scores = mossTTSApplyRepetitionPenaltyDelayPattern(
        logits: logits,
        previousTokens: previousTokens,
        penalty: repetitionPenalty
    )
    if !doSample {
        return argMax(scores, axis: -1).asType(.int32)
    }

    let originalShape = scores.shape
    let vocabSize = scores.dim(scores.ndim - 1)
    scores = scores.reshaped([-1, vocabSize])
    scores = mossApplyTopK(scores, topK: topK)
    scores = mossApplyTopP(scores, topP: topP)
    return MLXRandom.categorical(scores)
        .reshaped(Array(originalShape.dropLast()))
        .asType(.int32)
}
