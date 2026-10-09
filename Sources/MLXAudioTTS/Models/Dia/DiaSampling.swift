import MLX
import MLXNN
import MLXRandom

/// Normalize before nucleus filtering, then apply the CFG top-k filter.
func diaFilterLogits(_ logits: MLXArray, topP: Float, topK: Int) -> MLXArray {
    var filtered = logSoftmax(logits, axis: -1)
    if topP > 0 && topP < 1 {
        let indices = argSort(filtered, axis: -1)
        let sorted = takeAlong(filtered, indices, axis: -1)
        let cumulative = cumsum(exp(sorted), axis: -1)
        let nucleus = MLX.where(cumulative .> (1 - topP), sorted, MLXArray(-Float.infinity))
        filtered = putAlong(filtered, indices, values: nucleus, axis: -1)
    }
    let vocabSize = logits.shape[logits.ndim - 1]
    if topK > 0 && topK < vocabSize {
        let sorted = MLX.sorted(filtered, axis: -1)
        let threshold = sorted[.ellipsis, vocabSize - topK].expandedDimensions(axis: -1)
        filtered = MLX.where(filtered .< threshold, MLXArray(-Float.infinity), filtered)
    }
    return filtered
}

func diaSampleChannels(_ logits: MLXArray, temperature: Float, topP: Float, topK: Int) -> MLXArray {
    if temperature == 0 { return logits.argMax(axis: -1) }
    return MLXRandom.categorical(diaFilterLogits(logits, topP: topP, topK: topK) / temperature, axis: -1)
}
