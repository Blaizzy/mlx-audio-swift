import MLX
import MLXAudioCore
import MLXNN

func higgsV3Delay(_ codes: MLXArray, boc: Int, eoc: Int) -> MLXArray {
    let (t, n) = (codes.dim(0), codes.dim(1))
    let out = MLXArray.full([t + n - 1, n], values: MLXArray(eoc), dtype: .int32)
    for k in 0..<n {
        if k > 0 { out[0..<k, k] = MLXArray(boc) }
        out[k..<(k + t), k] = codes[0..., k]
    }
    return out
}

func higgsV3Undelay(_ codes: MLXArray) -> MLXArray {
    let n = codes.dim(1), t = codes.dim(0) - codes.dim(1) + 1
    return concatenated((0..<n).map { codes[$0..<($0 + t), $0..<($0 + 1)] }, axis: 1)
}

struct HiggsV3SamplerState {
    var delayCount = 0
    var eocCountdown: Int?
    var done = false

    mutating func step(_ sampled: MLXArray, codebooks: Int, boc: Int, eoc: Int) -> MLXArray {
        var codes = sampled.asType(.int32)
        if delayCount < codebooks {
            let next = delayCount + 1
            if next < codebooks {
                codes = MLX.where(MLXArray(0..<codebooks) .>= next, MLXArray(boc), codes)
            }
            delayCount += 1
        } else if let remaining = eocCountdown {
            eocCountdown = remaining - 1
            done = remaining <= 1
        } else if codes[0].item(Int.self) == eoc {
            if codebooks <= 2 { done = true } else { eocCountdown = codebooks - 2 }
        }
        return codes
    }
}

/// Higgs applies temperature before top-k and nucleus filtering.
struct HiggsV3Sampler {
    let parameters: GenerateParameters
    let random: MLXRandom.RandomState
    init(_ parameters: GenerateParameters) {
        self.parameters = parameters
        random = parameters.seed.map { .init(seed: $0) } ?? .init()
    }
    func sample(_ input: MLXArray) -> MLXArray {
        if parameters.temperature <= 1e-5 || parameters.topK == 1 {
            return argMax(input, axis: -1).asType(.int32)
        }
        var logits = input / parameters.temperature
        let vocab = logits.dim(-1)
        if parameters.topK > 0 && parameters.topK < vocab {
            let indices = argPartition(-logits, kth: parameters.topK - 1, axis: -1)[0..., parameters.topK...]
            logits = putAlong(logits, indices, values: MLXArray(-Float.infinity), axis: -1)
        }
        if parameters.topP > 0 && parameters.topP < 1 {
            let logprobs = logSoftmax(logits, axis: -1)
            let indices = argSort(logprobs, axis: -1)
            let sorted = takeAlong(logprobs, indices, axis: -1)
            let keep = cumsum(exp(sorted), axis: -1) .> (1 - parameters.topP)
            let masked = putAlong(logprobs, indices, values: MLX.where(keep, sorted, MLXArray(-Float.infinity)), axis: -1)
            logits = MLX.where(masked .== -Float.infinity, MLXArray(-Float.infinity), logits)
        }
        return MLXRandom.categorical(logits, axis: -1, key: random.next()).asType(.int32)
    }
}
