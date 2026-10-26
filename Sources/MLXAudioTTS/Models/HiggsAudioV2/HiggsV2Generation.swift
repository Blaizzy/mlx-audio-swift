import Foundation
import MLX

func higgsV2ApplyDelay(_ codes: MLXArray, bos: Int, eos: Int) -> MLXArray {
    let (k, count) = (codes.dim(0), codes.dim(1))
    return concatenated((0..<k).map { i in
        concatenated([MLXArray.full([1, i], values: MLXArray(Int32(bos)), dtype: .int32), codes[i..<(i + 1)],
                      MLXArray.full([1, k - 1 - i], values: MLXArray(Int32(eos)), dtype: .int32)], axis: 1)
    }, axis: 0).reshaped([k, count + k - 1])
}

func higgsV2RevertDelay(_ codes: MLXArray) -> MLXArray {
    let k = codes.dim(0), count = max(0, codes.dim(1) - codes.dim(0) + 1)
    guard count > 0 else { return MLXArray.zeros([k, 0], dtype: .int32) }
    return concatenated((0..<k).map { codes[$0..<($0 + 1), $0..<($0 + count)] }, axis: 0)
}

func higgsV2Sample(_ logits: MLXArray, temperature: Float, topP: Float, topK: Int,
                   key: MLXArray) -> MLXArray {
    if temperature <= 0 { return argMax(logits, axis: -1).asType(.int32) }
    var x = logits / temperature
    if topK > 0, topK < logits.dim(-1) {
        let sorted = sorted(x, axis: -1)
        let threshold = sorted[0..., (logits.dim(-1) - topK)..<(logits.dim(-1) - topK + 1)]
        x = MLX.where(x .< threshold, -Float.infinity, x)
    }
    if topP > 0, topP < 1 {
        let order = argSort(-x, axis: -1)
        let sorted = takeAlong(x, order, axis: -1)
        let cumulative = cumsum(softmax(sorted, axis: -1), axis: -1)
        let previous = concatenated([MLXArray.zeros([x.dim(0), 1], dtype: x.dtype), cumulative[0..., 0..<(x.dim(-1) - 1)]], axis: -1)
        let masked = MLX.where(previous .< topP, sorted, -Float.infinity)
        x = takeAlong(masked, argSort(order, axis: -1), axis: -1)
    }
    let u = MLXRandom.uniform(0..<1, x.shape, key: key)
    let gumbel = -log(-log(u + 1e-20) + 1e-20)
    return argMax(x + gumbel, axis: -1).asType(.int32)
}

func higgsV2Fade(_ pcm: inout [Float], milliseconds: Float, sampleRate: Int, fadeIn: Bool) {
    guard milliseconds.isFinite, milliseconds > 0 else { return }
    let n = min(pcm.count, Int(milliseconds * Float(sampleRate) / 1000))
    guard n > 1, pcm.count > n else { return }
    for i in 0..<n {
        let gain = Float(i) / Float(n - 1)
        if fadeIn { pcm[i] *= gain } else { pcm[pcm.count - n + i] *= 1 - gain }
    }
}

struct HiggsV2PCMStreamer {
    let overlap: Int
    private(set) var emitted = 0
    private var tail: [Float] = []

    init(overlap: Int) { self.overlap = max(0, overlap) }

    mutating func consume(_ pcm: [Float], final: Bool) -> [Float] {
        guard pcm.count >= emitted else { return [] }
        let end = final ? pcm.count : max(emitted, pcm.count - overlap)
        guard end > emitted else { return [] }
        var chunk = Array(pcm[emitted..<end])
        let count = min(tail.count, chunk.count)
        if count > 1 {
            for i in 0..<count {
                let gain = Float(i) / Float(count - 1)
                chunk[i] = tail[i] * (1 - gain) + chunk[i] * gain
            }
        }
        emitted = end
        tail = final ? [] : Array(pcm[end...])
        return chunk
    }
}
