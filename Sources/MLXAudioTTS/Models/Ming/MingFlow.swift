import Foundation
import MLX
import MLXFast
import MLXNN

/// Checkpoint-name remapping shared by Ming dense and the Tiny speech stack.
public func mingSanitizeSpeechWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]
    for (key, value) in weights {
        if key.hasSuffix("rotary_embed.inv_freq") { continue }
        let name = key.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.out.")
            .replacingOccurrences(of: ".mlp.ff.0.0.", with: ".mlp.fc1.")
            .replacingOccurrences(of: ".mlp.ff.2.", with: ".mlp.fc2.")
            .replacingOccurrences(of: ".t_embedder.time_mlp.0.", with: ".t_embedder.fc1.")
            .replacingOccurrences(of: ".t_embedder.time_mlp.2.", with: ".t_embedder.fc2.")
        result[name] = value
    }
    return result
}

final class MingFeedForward: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    init(_ dim: Int, ratio: Float) {
        _fc1.wrappedValue = Linear(dim, Int(Float(dim) * ratio))
        _fc2.wrappedValue = Linear(Int(Float(dim) * ratio), dim)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
}

final class MingAttention: Module {
    @ModuleInfo(key: "to_q") var q: Linear
    @ModuleInfo(key: "to_k") var k: Linear
    @ModuleInfo(key: "to_v") var v: Linear
    @ModuleInfo var out: Linear
    let heads: Int
    let headDim: Int
    init(_ dim: Int, heads: Int) {
        self.heads = heads; self.headDim = dim / heads
        _q.wrappedValue = Linear(dim, dim); _k.wrappedValue = Linear(dim, dim)
        _v.wrappedValue = Linear(dim, dim); _out.wrappedValue = Linear(dim, dim)
    }
    private func rotary(_ x: MLXArray) -> MLXArray {
        // The Ming DiT rotates adjacent pairs after FP32 phase generation.
        let frequencies = exp(-log(Float(10000)) * MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32) / Float(headDim))
        let phase = MLXArray(0..<x.dim(2)).asType(.float32)[0..., .newAxis] * frequencies[.newAxis]
        let c = cos(phase)[.newAxis, .newAxis]; let s = sin(phase)[.newAxis, .newAxis]
        let pairs = x.reshaped(x.dim(0), heads, x.dim(2), headDim / 2, 2)
        let even = pairs[0..., 0..., 0..., 0..., 0]
        let odd = pairs[0..., 0..., 0..., 0..., 1]
        return stacked([even * c - odd * s, even * s + odd * c], axis: -1).reshaped(x.shape).asType(x.dtype)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.dim(0), t = x.dim(1)
        let queries = rotary(q(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3))
        let keys = rotary(k(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3))
        let values = v(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        var output = out(MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values,
            scale: 1 / sqrt(Float(headDim)), mask: .none).transposed(0, 2, 1, 3).reshaped(b, t, -1))
        if let mask { output = MLX.where(mask[0..., 0..., .newAxis], output, MLXArray(0).asType(output.dtype)) }
        return output
    }
}

final class MingDiTBlock: Module {
    let norm1: RMSNorm
    let attn: MingAttention
    let norm2: RMSNorm
    let mlp: MingFeedForward
    init(_ config: MingTransformerConfiguration) {
        norm1 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        norm2 = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
        attn = MingAttention(config.hiddenSize, heads: config.numHeads)
        mlp = MingFeedForward(config.hiddenSize, ratio: config.mlpRatio)
    }
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let h = x + attn(norm1(x), mask: mask)
        return h + mlp(norm2(h))
    }
}
final class MingFinalLayer: Module {
    @ModuleInfo(key: "norm_final") var norm: RMSNorm
    @ModuleInfo var linear: Linear
    init(_ hidden: Int, output: Int) {
        _norm.wrappedValue = RMSNorm(dimensions: hidden, eps: 1e-6)
        _linear.wrappedValue = Linear(hidden, output)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(norm(x)) }
}
final class MingTimestepEmbedder: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    init(_ hidden: Int) { _fc1.wrappedValue = Linear(256, hidden); _fc2.wrappedValue = Linear(hidden, hidden) }
    func callAsFunction(_ time: MLXArray) -> MLXArray {
        let freq = exp(MLXArray(0..<128).asType(.float32) * (-log(Float(10000)) / 127))
        let phase = 1000 * time[0..., .newAxis] * freq[.newAxis]
        return fc2(silu(fc1(concatenated([sin(phase), cos(phase)], axis: -1).asType(time.dtype))))
    }
}
final class MingConditionEmbedder: Module {
    @ModuleInfo(key: "cond_embedder") var linear: Linear
    init(_ input: Int, hidden: Int) { _linear.wrappedValue = Linear(input, hidden) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(x) }
}

/// Ming's noncausal conditional speech transformer. Inputs use [batch, frames, channels].
public final class MingDiT: Module {
    @ModuleInfo(key: "t_embedder") var time: MingTimestepEmbedder
    @ModuleInfo(key: "x_embedder") var input: Linear
    @ModuleInfo(key: "c_embedder") var condition: MingConditionEmbedder
    let blocks: [MingDiTBlock]
    @ModuleInfo(key: "final_layer") var final: MingFinalLayer
    public var dtype: DType { (input as? QuantizedLinear)?.scales.dtype ?? input.weight.dtype }
    public init(latentDim: Int, conditionDim: Int, config: MingTransformerConfiguration) {
        _time.wrappedValue = MingTimestepEmbedder(config.hiddenSize)
        _input.wrappedValue = Linear(latentDim, config.hiddenSize)
        _condition.wrappedValue = MingConditionEmbedder(conditionDim, hidden: config.hiddenSize)
        blocks = (0..<config.depth).map { _ in MingDiTBlock(config) }
        _final.wrappedValue = MingFinalLayer(config.hiddenSize, output: latentDim)
    }
    public func callAsFunction(_ x: MLXArray, time t: MLXArray, condition c: MLXArray,
                               history: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let embedded = condition(c) + time(t)[0..., .newAxis, 0...]
        var full = concatenated([embedded, input(history), input(x)], axis: 1)
        var fullMask: MLXArray?
        if let mask {
            fullMask = concatenated([repeated(mask[0..., ..<1], count: history.dim(1) + c.dim(1), axis: 1), mask], axis: 1)
        }
        for block in blocks { full = block(full, mask: fullMask) }
        return final(full)
    }
    public func guided(_ x: MLXArray, time: MLXArray, condition: MLXArray,
                       history: MLXArray, patchSize: Int) -> (MLXArray, MLXArray) {
        let output = self(concatenated([x, x], axis: 0),
            time: time.ndim == 0 ? MLXArray.ones([x.dim(0) * 2], dtype: x.dtype) * time : time,
            condition: concatenated([condition, MLXArray.zeros(like: condition)], axis: 0),
            history: concatenated([history, history], axis: 0))
        let parts = output[0..., (output.dim(1) - patchSize)..., 0...].split(parts: 2, axis: 0)
        return (parts[0], parts[1])
    }
}

public final class MingLatentAggregator: Module {
    @ModuleInfo(key: "word_embedder") var cls: Embedding
    @ModuleInfo(key: "x_embedder") var input: Linear
    let blocks: [MingDiTBlock]
    @ModuleInfo(key: "final_layer") var final: MingFinalLayer
    public init(latentDim: Int, outputDim: Int, config: MingTransformerConfiguration) {
        _cls.wrappedValue = Embedding(embeddingCount: 1, dimensions: config.hiddenSize)
        _input.wrappedValue = Linear(latentDim, config.hiddenSize)
        blocks = (0..<config.depth).map { _ in MingDiTBlock(config) }
        _final.wrappedValue = MingFinalLayer(config.hiddenSize, output: outputDim)
    }
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = concatenated([cls(MLXArray.zeros([x.dim(0), 1], dtype: .int32)), input(x)], axis: 1)
        let fullMask = mask.map { concatenated([$0[0..., ..<1], $0], axis: 1) }
        for block in blocks { h = block(h, mask: fullMask) }
        return final(h)[0..., ..<1, 0...]
    }
}

public func mingFlowTimesteps(_ steps: Int, dtype: DType = .float32) -> MLXArray {
    let schedules: [Int: [Float]] = [
        5: [0,2,4,8,16,32], 6: [0,2,4,6,8,16,32], 7: [0,2,4,6,8,16,24,32],
        10: [0,2,4,6,8,12,16,20,24,28,32], 12: [0,2,4,6,8,10,12,14,16,20,24,28,32],
        16: [0,1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32]]
    return (schedules[steps].map { MLXArray($0) / 32 } ?? linspace(Float(0), Float(1), count: steps + 1)).asType(dtype)
}

public final class MingCFM: Module {
    public let model: MingDiT
    public init(_ model: MingDiT) { self.model = model }
}
public final class MingFlowLoss: Module {
    public let cfm: MingCFM
    public let latentDim: Int
    public let ming3: Bool
    public init(latentDim: Int, conditionDim: Int, config: MingTransformerConfiguration, ming3: Bool) {
        self.latentDim = latentDim; self.ming3 = ming3
        cfm = MingCFM(MingDiT(latentDim: latentDim, conditionDim: conditionDim, config: config))
    }
    public func sample(_ condition: MLXArray, history: MLXArray, patchSize: Int,
                       steps: Int = 10, cfg: Float = 2, sigma: Float = 0.25, temperature: Float = 0,
                       noise: MLXArray? = nil, sdeNoise: MLXArray? = nil) -> MLXArray {
        let dtype: DType = ming3 ? condition.dtype : .float32
        var state = noise ?? (ming3
            ? MLXRandom.normal([condition.dim(0), patchSize, latentDim], dtype: dtype)
            : MLXRandom.normal([condition.dim(0), latentDim, patchSize], dtype: dtype).transposed(0, 2, 1))
        state = state.asType(dtype)
        let allNoise = sdeNoise ?? (ming3 ? MLXRandom.normal([steps] + state.shape, dtype: dtype) : nil)
        var times = mingFlowTimesteps(steps, dtype: dtype)
        times = times - (cos(Float.pi / 2 * times) - 1 + times)
        for i in 0..<steps {
            let (pred, null) = cfm.model.guided(state, time: times[i], condition: condition, history: history, patchSize: patchSize)
            let derivative = (pred + (pred - null) * cfg).asType(dtype)
            let dt = times[i + 1] - times[i]
            state = state + dt * derivative
            let perturbation = allNoise?[i] ?? MLXRandom.normal(state.shape, dtype: dtype)
            state = state + sigma * sqrt(temperature) * sqrt(abs(dt)) * perturbation
            eval(state)
        }
        return state
    }
}
