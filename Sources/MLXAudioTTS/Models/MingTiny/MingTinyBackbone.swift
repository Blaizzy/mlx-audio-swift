// Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
// Adapted to Swift from inclusionAI's MingTiny V3 ASR and mlx-audio.

import Foundation
import MLX
import MLXAudioCore
import MLXFast
import MLXNN

final class MingTinyLayerState {
    var convolution: [MLXArray?] = [nil, nil, nil]
    var recurrent: MLXArray?
    var attention: KVCache = KVCacheSimple()
    var arrays: [MLXArray] {
        convolution.compactMap { $0 } + (recurrent.map { [$0] } ?? []) + attention.state
    }
}

class MingTinyAttention: Module {
    func callAsFunction(_ x: MLXArray, state: MingTinyLayerState) -> MLXArray { fatalError() }
}

final class MingTinyDeltaAttention: MingTinyAttention {
    let config: MingTinyTextConfiguration
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "f_proj") var fProj: Linear?
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "f_a_proj") var fA: Linear?
    @ModuleInfo(key: "f_b_proj") var fB: Linear?
    @ModuleInfo(key: "g_a_proj") var gA: Linear?
    @ModuleInfo(key: "g_b_proj") var gB: Linear?
    @ModuleInfo(key: "q_conv1d") var qConv: Conv1d
    @ModuleInfo(key: "k_conv1d") var kConv: Conv1d
    @ModuleInfo(key: "v_conv1d") var vConv: Conv1d
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ c: MingTinyTextConfiguration) {
        config = c
        let width = c.numAttentionHeads * c.headDim
        _qProj.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        _bProj.wrappedValue = Linear(c.hiddenSize, c.numAttentionHeads, bias: false)
        if c.noKdaLora {
            _fProj.wrappedValue = Linear(c.hiddenSize, width, bias: false)
            _gProj.wrappedValue = Linear(c.hiddenSize, width, bias: false)
        } else {
            _fA.wrappedValue = Linear(c.hiddenSize, c.headDim, bias: false)
            _gA.wrappedValue = Linear(c.hiddenSize, c.headDim, bias: false)
            _fB.wrappedValue = Linear(c.headDim, width, bias: false)
            _gB.wrappedValue = Linear(c.headDim, width, bias: false)
        }
        func conv() -> Conv1d {
            Conv1d(
                inputChannels: width, outputChannels: width, kernelSize: c.shortConvKernelSize,
                groups: width, bias: false)
        }
        _qConv.wrappedValue = conv()
        _kConv.wrappedValue = conv()
        _vConv.wrappedValue = conv()
        _aLog.wrappedValue = .zeros([c.numAttentionHeads])
        _dtBias.wrappedValue = .zeros([width])
        _oNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _oProj.wrappedValue = Linear(width, c.hiddenSize, bias: false)
    }

    override func callAsFunction(_ x: MLXArray, state: MingTinyLayerState) -> MLXArray {
        let c = config
        let b = x.dim(0)
        let length = x.dim(1)
        let keep = config.shortConvKernelSize - 1
        var projected: [MLXArray] = []
        for (i, pair) in zip([qProj, kProj, vProj], [qConv, kConv, vConv]).enumerated() {
            let value = pair.0(x)
            let previous =
                state.convolution[i] ?? .zeros([b, keep, value.dim(-1)], dtype: value.dtype)
            let joined = concatenated([previous, value], axis: 1)
            state.convolution[i] =
                keep > 0 ? joined[0..., (-keep)..., 0...] : joined[0..., 0 ..< 0, 0...]
            projected.append(
                silu(pair.1(joined)).reshaped(b, length, c.numAttentionHeads, c.headDim))
        }
        func normalize(_ value: MLXArray) -> MLXArray {
            let f = value.asType(.float32)
            return f * rsqrt((f * f).sum(axis: -1, keepDims: true) + 1e-6)
        }
        let q = normalize(projected[0]) * pow(Float(c.headDim), -0.5)
        let k = normalize(projected[1])
        let v = projected[2]
        let f = (fProj.map { $0(x) } ?? fB!(fA!(x))).asType(.float32)
        let gate = (f + dtBias).reshaped(b, length, c.numAttentionHeads, c.headDim)
        let rate = exp(aLog.asType(.float32)).expandedDimensions(axis: -1)
        let decay = exp(
            c.kdaSafeGate ? c.kdaLowerBound * sigmoid(rate * gate) : -rate * softplus(gate))
        let beta = sigmoid(bProj(x).asType(.float32))
        let initial = state.recurrent ?? .zeros([b, c.numAttentionHeads, c.headDim, c.headDim])
        let (out, next) = MingTinyDeltaKernel.callAsFunction(
            q: q, k: k, v: v, decay: decay, beta: beta, state: initial)
        state.recurrent = next
        // The reference rounds the recurrence to the value dtype before normalization.
        var normalized = MLXFast.rmsNorm(
            out.asType(v.dtype).asType(.float32),
            weight: oNorm.weight.asType(.float32), eps: c.rmsNormEps)
        let g = (gProj.map { $0(x) } ?? gB!(gA!(x))).reshaped(out.shape).asType(.float32)
        normalized = (normalized * sigmoid(g)).asType(x.dtype)
        return oProj(normalized.reshaped(b, length, -1))
    }
}

final class MingTinyLatentAttention: MingTinyAttention {
    let config: MingTinyTextConfiguration
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qA: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qB: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvA: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvB: Linear
    @ModuleInfo(key: "g_proj") var gate: Linear?
    @ModuleInfo var dense: Linear
    let rope: RoPE
    init(_ c: MingTinyTextConfiguration) {
        config = c
        let qDim = c.qkNopeHeadDim + c.qkRopeHeadDim
        if let rank = c.qLoraRank {
            _qA.wrappedValue = Linear(c.hiddenSize, rank, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: rank, eps: c.rmsNormEps)
            _qB.wrappedValue = Linear(rank, c.numAttentionHeads * qDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(c.hiddenSize, c.numAttentionHeads * qDim, bias: false)
        }
        _kvA.wrappedValue = Linear(c.hiddenSize, c.kvLoraRank + c.qkRopeHeadDim, bias: false)
        _kvNorm.wrappedValue = RMSNorm(dimensions: c.kvLoraRank, eps: c.rmsNormEps)
        _kvB.wrappedValue = Linear(
            c.kvLoraRank, c.numAttentionHeads * (c.qkNopeHeadDim + c.vHeadDim), bias: false)
        if let granularity = c.attentionGate {
            _gate.wrappedValue = Linear(
                c.hiddenSize,
                c.numAttentionHeads * (granularity == "element_wise" ? c.vHeadDim : 1), bias: false)
        }
        _dense.wrappedValue = Linear(c.numAttentionHeads * c.vHeadDim, c.hiddenSize, bias: false)
        rope = RoPE(dimensions: c.qkRopeHeadDim, traditional: c.ropeInterleave, base: c.ropeTheta)
    }
    override func callAsFunction(_ x: MLXArray, state: MingTinyLayerState) -> MLXArray {
        let c = config
        let b = x.dim(0)
        let t = x.dim(1)
        let q = (qProj.map { $0(x) } ?? qB!(qNorm!(qA!(x))))
            .reshaped(b, t, c.numAttentionHeads, -1).transposed(0, 2, 1, 3)
        let qParts = split(q, indices: [c.qkNopeHeadDim], axis: -1)
        let kvParts = split(kvA(x), indices: [c.kvLoraRank], axis: -1)
        let kv = kvB(kvNorm(kvParts[0])).reshaped(b, t, c.numAttentionHeads, -1).transposed(
            0, 2, 1, 3)
        let parts = split(kv, indices: [c.qkNopeHeadDim], axis: -1)
        let qr = rope(qParts[1], offset: state.attention.offset)
        let kr = rope(kvParts[1].expandedDimensions(axis: 1), offset: state.attention.offset)
        let keys = concatenated([parts[0], broadcast(kr, to: qr.shape)], axis: -1)
        let queries = concatenated([qParts[0], qr], axis: -1)
        let mask = state.attention.makeMask(n: t, windowSize: nil, returnArray: false)
        var out = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: parts[1],
            cache: state.attention, scale: pow(Float(c.qkNopeHeadDim + c.qkRopeHeadDim), -0.5),
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        if let gate {
            out =
                out
                * sigmoid(gate(x).asType(.float32)).asType(x.dtype).reshaped(
                    b, t, c.numAttentionHeads, -1)
        }
        return dense(out.reshaped(b, t, -1))
    }
}

func mingTinySwiGLU(up: MLXArray, gate: MLXArray, limit: Float?) -> MLXArray {
    var g = silu(gate)
    var u = up
    if let limit, limit > 0 {
        g = minimum(g, limit)
        u = clip(u, min: -limit, max: limit)
    }
    return g * u
}

class MingTinyFeedForward: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray { fatalError() }
}
final class MingTinyMLP: MingTinyFeedForward {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    let limit: Float?
    init(hidden: Int, intermediate: Int, limit: Float? = nil) {
        self.limit = limit
        _gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        _up.wrappedValue = Linear(hidden, intermediate, bias: false)
        _down.wrappedValue = Linear(intermediate, hidden, bias: false)
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(mingTinySwiGLU(up: up(x), gate: gate(x), limit: limit))
    }
}

// Copyright © 2023-2024 Apple Inc.
// Expert projections and batching adapted from mlx-lm (MIT).
class MingTinySwitchLinear: Module, Quantizable {
    let weight: MLXArray
    init(input: Int, output: Int, experts: Int) {
        weight = MLXRandom.normal([experts, output, input]) * sqrt(1 / Float(input))
    }
    init(weight: MLXArray) { self.weight = weight }
    func callAsFunction(_ x: MLXArray, indices: MLXArray, sorted: Bool = false) -> MLXArray {
        gatherMM(x, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
    }
    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        MingTinyQuantizedSwitchLinear(weight: weight, groupSize: groupSize, bits: bits, mode: mode)
    }
}
final class MingTinyQuantizedSwitchLinear: MingTinySwitchLinear, Quantized {
    let scales: MLXArray
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    init(weight: MLXArray, groupSize: Int, bits: Int, mode: QuantizationMode) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let packed = MLX.quantized(weight, groupSize: groupSize, bits: bits, mode: mode)
        scales = packed.1
        biases = packed.2
        super.init(weight: packed.0)
        freeze()
    }
    override func callAsFunction(_ x: MLXArray, indices: MLXArray, sorted: Bool = false) -> MLXArray
    {
        gatherQuantizedMM(
            x, weight, scales: scales, biases: biases, rhsIndices: indices,
            groupSize: groupSize, bits: bits, mode: mode, sortedIndices: sorted)
    }
}
final class MingTinySwitchMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: MingTinySwitchLinear
    @ModuleInfo(key: "up_proj") var up: MingTinySwitchLinear
    @ModuleInfo(key: "down_proj") var down: MingTinySwitchLinear
    let limit: Float?
    init(_ c: MingTinyTextConfiguration, limit: Float?) {
        self.limit = limit
        _gate.wrappedValue = MingTinySwitchLinear(
            input: c.hiddenSize, output: c.moeIntermediateSize, experts: c.numExperts)
        _up.wrappedValue = MingTinySwitchLinear(
            input: c.hiddenSize, output: c.moeIntermediateSize, experts: c.numExperts)
        _down.wrappedValue = MingTinySwitchLinear(
            input: c.moeIntermediateSize, output: c.hiddenSize, experts: c.numExperts)
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
        let activated = mingTinySwiGLU(
            up: up(input, indices: selected, sorted: sorted),
            gate: gate(input, indices: selected, sorted: sorted), limit: limit)
        var output = down(activated, indices: selected, sorted: sorted)
        if let inverse {
            output = output[inverse].reshaped(indices.shape + [1, x.dim(-1)])
        }
        return output.squeezed(axis: -2)
    }
}
final class MingTinyRouter: Module {
    let config: MingTinyTextConfiguration
    let weight: MLXArray
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
    init(_ c: MingTinyTextConfiguration) {
        config = c
        weight = Linear(c.hiddenSize, c.numExperts, bias: false).weight
        _expertBias.wrappedValue = .zeros([c.numExperts])
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let c = config
        let scores = sigmoid(matmul(x.asType(.float32), weight.asType(.float32).T))
        var adjusted = scores + expertBias
        if c.nGroup > 1 {
            let grouped = adjusted.reshaped(
                Array(scores.shape.dropLast()) + [c.nGroup, c.numExperts / c.nGroup])
            let groupScores = top(grouped, k: min(2, grouped.dim(-1)), axis: -1).sum(axis: -1)
            let chosen = argPartition(groupScores, kth: -c.topKGroup, axis: -1)[
                .ellipsis, (-c.topKGroup)...]
            let allowed = (MLXArray(0 ..< c.nGroup) .== chosen.expandedDimensions(axis: -1)).any(
                axis: -2)
            adjusted = MLX.where(
                allowed.expandedDimensions(axis: -1), grouped, MLXArray(-Float.infinity)
            ).reshaped(scores.shape)
        }
        let indices = argPartition(adjusted, kth: -c.numExpertsPerToken, axis: -1)[
            .ellipsis, (-c.numExpertsPerToken)...]
        var weights = takeAlong(scores, indices, axis: -1)
        if c.numExpertsPerToken > 1 {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * c.routedScalingFactor)
    }
}
final class MingTinySparseMoE: MingTinyFeedForward {
    @ModuleInfo var gate: MingTinyRouter
    @ModuleInfo(key: "switch_mlp") var experts: MingTinySwitchMLP
    @ModuleInfo(key: "shared_experts") var shared: MingTinyMLP?
    init(_ c: MingTinyTextConfiguration, index: Int) {
        func limit(_ list: [Float]?) -> Float? {
            list.flatMap { index < $0.count ? $0[index] : nil }
        }
        _gate.wrappedValue = MingTinyRouter(c)
        _experts.wrappedValue = MingTinySwitchMLP(c, limit: limit(c.expertSwiGLULimits))
        if c.numSharedExperts > 0 {
            _shared.wrappedValue = MingTinyMLP(
                hidden: c.hiddenSize,
                intermediate: c.moeSharedExpertIntermediateSize * c.numSharedExperts,
                limit: limit(c.sharedExpertSwiGLULimits))
        }
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x)
        let out =
            (experts(x, indices: indices).asType(.float32) * weights.expandedDimensions(axis: -1))
            .sum(axis: -2).asType(x.dtype)
        return shared.map { out + $0(x) } ?? out
    }
}
final class MingTinyDecoderLayer: Module {
    @ModuleInfo var attention: MingTinyAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo var mlp: MingTinyFeedForward
    init(_ c: MingTinyTextConfiguration, index: Int) {
        _attention.wrappedValue = c.usesMLA(index) ? MingTinyLatentAttention(c) : MingTinyDeltaAttention(c)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue =
            index >= c.firstKDenseReplace
            ? MingTinySparseMoE(c, index: index)
            : MingTinyMLP(hidden: c.hiddenSize, intermediate: c.intermediateSize)
    }
    func callAsFunction(_ x: MLXArray, state: MingTinyLayerState) -> MLXArray {
        let h = x + attention(inputNorm(x), state: state)
        return h + mlp(postNorm(h))
    }
}
final class MingTinyBackbone: Module {
    @ModuleInfo(key: "word_embeddings") var embeddings: Embedding
    @ModuleInfo var layers: [MingTinyDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    init(_ t: MingTinyTextConfiguration) {
        _embeddings.wrappedValue = Embedding(embeddingCount: t.vocabSize, dimensions: t.hiddenSize)
        _layers.wrappedValue = (0 ..< t.numHiddenLayers).map { MingTinyDecoderLayer(t, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: t.hiddenSize, eps: t.rmsNormEps)
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [MingTinyLayerState]) -> MLXArray {
        var h = embeddings
        for (layer, state) in zip(layers, cache) { h = layer(h, state: state) }
        return norm(h)
    }
    func makeCache() -> [MingTinyLayerState] { layers.map { _ in MingTinyLayerState() } }
}
