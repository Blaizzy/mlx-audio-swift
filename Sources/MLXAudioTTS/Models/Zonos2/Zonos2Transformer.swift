import Foundation
import MLX
import MLXAudioCore
import MLXFast
import MLXNN

private let zonos2ExpertSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray =
    MLX.compile(shapeless: true) { gate, up in silu(gate) * up }

final class Zonos2MultiEmbedding: Module {
    @ModuleInfo var embedders: [Embedding]
    init(_ c: Zonos2Configuration) {
        _embedders.wrappedValue =
            (0..<c.nCodebooks).map { _ in Embedding(embeddingCount: c.audioVocabSize, dimensions: c.dim) }
            + [Embedding(embeddingCount: c.textVocab + 1, dimensions: c.dim)]
    }
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var x = embedders[0](ids[.ellipsis, 0])
        for i in 1..<embedders.count { x = x + embedders[i](ids[.ellipsis, i]) }
        return x
    }
}

final class Zonos2Attention: Module {
    let config: Zonos2Configuration
    @ModuleInfo var wq: Linear
    @ModuleInfo var wkv: Linear
    @ModuleInfo var wo: Linear
    @ModuleInfo var gater: Linear
    let temp: MLXArray
    let rope: RoPE
    init(_ c: Zonos2Configuration) {
        config = c
        _wq.wrappedValue = Linear(c.dim, c.numHeads * c.headDim, bias: false)
        _wkv.wrappedValue = Linear(c.dim, 2 * c.numKVHeads * c.headDim, bias: false)
        _wo.wrappedValue = Linear(c.numHeads * c.headDim, c.dim, bias: false)
        _gater.wrappedValue = Linear(c.dim, c.numHeads, bias: false)
        temp = .ones([1, c.numHeads, 1])
        rope = RoPE(dimensions: c.headDim, traditional: true, base: c.ropeTheta)
    }
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let c = config
        let b = x.dim(0)
        let t = x.dim(1)
        var q = wq(x).reshaped(b, t, c.numHeads, c.headDim)
        let kv = wkv(x).reshaped(b, t, 2, c.numKVHeads, c.headDim)
        var k = kv[0..., 0..., 0, 0..., 0...]
        var v = kv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        q =
            (MLXFast.rmsNorm(q, weight: MLXArray.ones([q.dim(-1)], dtype: q.dtype), eps: 1e-6)
            * abs(temp).asType(q.dtype)).transposed(0, 2, 1, 3)
        k = MLXFast.rmsNorm(k, weight: MLXArray.ones([k.dim(-1)], dtype: k.dtype), eps: 1e-6)
            .transposed(0, 2, 1, 3)
        q = rope(q, offset: cache?.offset ?? 0)
        k = rope(k, offset: cache?.offset ?? 0)
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: pow(Float(c.headDim), -0.5), mask: mask
        )
        .transposed(0, 2, 1, 3)
        return wo((output * sigmoid(gater(x)).expandedDimensions(axis: -1)).reshaped(b, t, -1))
    }
}

final class Zonos2RouterMLP: Module {
    @ModuleInfo var l0: Linear
    @ModuleInfo var l2: Linear
    @ModuleInfo var l4: Linear
    init(_ c: Zonos2Configuration) {
        _l0.wrappedValue = Linear(c.moeRouterDim, c.moeRouterDim)
        _l2.wrappedValue = Linear(c.moeRouterDim, c.moeRouterDim)
        _l4.wrappedValue = Linear(c.moeRouterDim, c.moeNExperts, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { l4(gelu(l2(gelu(l0(x))))) }
}

final class Zonos2Router: Module {
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "router_mlp") var mlp: Zonos2RouterMLP
    @ModuleInfo(key: "rmsnorm_eda") var norm: RMSNorm
    @ParameterInfo(key: "router_states_scale") var stateScale: MLXArray?
    @ParameterInfo(key: "balancing_biases") var balancingBiases: MLXArray
    let topK: Int
    let legacy: Bool
    init(_ c: Zonos2Configuration, index: Int) {
        topK = c.topK(index)
        legacy =
            c.moeBalancingStrategy.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            .replacingOccurrences(of: "-", with: "_") != "quantile"
        _down.wrappedValue = Linear(c.dim, c.moeRouterDim)
        _mlp.wrappedValue = Zonos2RouterMLP(c)
        _norm.wrappedValue = RMSNorm(dimensions: c.moeRouterDim, eps: c.normEps)
        _stateScale.wrappedValue = index != c.moeStartFromLayer ? .ones([c.moeRouterDim]) : nil
        _balancingBiases.wrappedValue = .zeros([c.moeNExperts])
    }
    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray, MLXArray) {
        var h = down(x)
        if let state, let stateScale { h = h + state * stateScale }
        let probability = softmax(mlp(norm(h)).asType(.float32), axis: -1)
        let scores =
            legacy
            ? probability + balancingBiases.asType(.float32)
            : probability - balancingBiases.asType(.float32)
        let ids = argSort(-scores, axis: -1)[.ellipsis, 0..<topK]
        return (takeAlong(probability, ids, axis: -1), ids, h)
    }
}

final class Zonos2Experts: Module {
    // The shared Ming expert projection supports native gathered BF16 and packed affine weights.
    @ModuleInfo(key: "gate_proj") var gate: MingTinySwitchLinear
    @ModuleInfo(key: "up_proj") var up: MingTinySwitchLinear
    @ModuleInfo(key: "down_proj") var down: MingTinySwitchLinear
    init(_ c: Zonos2Configuration) {
        _gate.wrappedValue = MingTinySwitchLinear(
            input: c.dim, output: c.intermediateSize, experts: c.moeNExperts)
        _up.wrappedValue = MingTinySwitchLinear(
            input: c.dim, output: c.intermediateSize, experts: c.moeNExperts)
        _down.wrappedValue = MingTinySwitchLinear(
            input: c.intermediateSize, output: c.dim, experts: c.moeNExperts)
    }
    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        var input = x.expandedDimensions(axes: [-2, -3])
        var selected = indices
        let sorted = indices.size >= 64
        var inverse: MLXArray?
        if sorted {
            let flat = indices.reshaped(-1)
            let order = argSort(flat)
            inverse = argSort(order)
            input = input.reshaped(-1, 1, x.dim(-1))[order.floorDivide(indices.dim(-1))]
            selected = flat[order]
        }
        let activated = zonos2ExpertSwiGLU(
            gate(input, indices: selected, sorted: sorted), up(input, indices: selected, sorted: sorted))
        var result = down(activated, indices: selected, sorted: sorted)
        if let inverse { result = result[inverse].reshaped(indices.shape + [1, x.dim(-1)]) }
        return result.squeezed(axis: -2)
    }
}

final class Zonos2FeedForward: Module {
    @ModuleInfo(key: "w_in") var input: Linear?
    @ModuleInfo(key: "w_out") var output: Linear?
    let router: Zonos2Router?
    let experts: Zonos2Experts?
    let normalize: Bool
    init(_ c: Zonos2Configuration, index: Int) {
        normalize = c.normTopKProbability
        if c.isMoE(index) {
            router = Zonos2Router(c, index: index)
            experts = Zonos2Experts(c)
        } else {
            router = nil
            experts = nil
            _input.wrappedValue = Linear(c.dim, c.intermediateSize * 2, bias: false)
            _output.wrappedValue = Linear(c.intermediateSize, c.dim, bias: false)
        }
    }
    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray?) {
        if let router, let experts {
            var (weights, indices, next) = router(x, state: state)
            if normalize { weights = weights / (sum(weights, axis: -1, keepDims: true) + 1e-8) }
            let values = experts(x, indices: indices).asType(.float32)
            return (sum(values * weights.expandedDimensions(axis: -1), axis: -2).asType(x.dtype), next)
        }
        let h = input!(x)
        let half = h.dim(-1) / 2
        return (output!(h[.ellipsis, 0..<half] * silu(h[.ellipsis, half...])), nil)
    }
}

final class Zonos2Block: Module {
    let attention: Zonos2Attention
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: Zonos2FeedForward
    init(_ c: Zonos2Configuration, index: Int) {
        attention = Zonos2Attention(c)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: c.dim, eps: c.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.dim, eps: c.normEps)
        _feedForward.wrappedValue = Zonos2FeedForward(c, index: index)
    }
    func callAsFunction(
        _ x: MLXArray, residual: MLXArray?, routerState: MLXArray?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> (MLXArray, MLXArray, MLXArray?) {
        let initial = residual.map { $0 + x } ?? x
        let attended = attention(attentionNorm(initial), mask: mask, cache: cache)
        let nextResidual = initial + attended
        let (out, state) = feedForward(ffnNorm(nextResidual), state: routerState)
        return (out, nextResidual, state)
    }
}
