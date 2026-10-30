// Copyright © 2023-2024 Apple Inc.
// Bailing MoE architecture adapted from mlx-lm and mlx-audio.
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXAudioCore

/// Bailing's speech route uses the text gate, with routing and expert accumulation in FP32.
final class MingOmniRouter: Module {
    let config: MingOmniTextConfiguration
    @ModuleInfo(key: "gate_proj") var projection: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray?
    init(_ c: MingOmniTextConfiguration) {
        config = c
        _projection.wrappedValue = Linear(c.hiddenSize, c.numExperts, bias: false)
        if c.expertBiasEnabled { _expertBias.wrappedValue = .zeros([c.numExperts]) }
    }
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let c = config
        let logits = projection(x.asType(.float32))
        let scores = c.scoreFunction == "sigmoid" ? sigmoid(logits) : softmax(logits, axis: -1)
        var adjusted = expertBias.map { scores + $0 } ?? scores
        if c.numGroups > 1 {
            let grouped = adjusted.reshaped(Array(scores.shape.dropLast()) + [c.numGroups, c.numExperts / c.numGroups])
            let groupScores = top(grouped, k: min(2, grouped.dim(-1)), axis: -1).sum(axis: -1)
            let selected = argPartition(groupScores, kth: -c.topKGroups, axis: -1)[.ellipsis, (-c.topKGroups)...]
            let allowed = (MLXArray(0..<c.numGroups) .== selected.expandedDimensions(axis: -1)).any(axis: -2)
            adjusted = MLX.where(allowed.expandedDimensions(axis: -1), grouped, MLXArray(0)).reshaped(scores.shape)
        }
        var indices = argPartition(adjusted, kth: -c.numExpertsPerToken, axis: -1)[.ellipsis, (-c.numExpertsPerToken)...]
        let selectedScores = takeAlong(adjusted, indices, axis: -1)
        indices = takeAlong(indices, argSort(-selectedScores, axis: -1), axis: -1)
        var weights = takeAlong(scores, indices, axis: -1)
        if c.numExpertsPerToken > 1 && c.normTopKProbability {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * c.routedScalingFactor)
    }
}
class MingOmniFeedForward: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray { fatalError("Abstract Ming feed forward") }
}
final class MingOmniMLP: MingOmniFeedForward {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(hidden: Int, intermediate: Int) {
        _gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        _up.wrappedValue = Linear(hidden, intermediate, bias: false)
        _down.wrappedValue = Linear(intermediate, hidden, bias: false)
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32), g = gate(f)
        return down((g * sigmoid(g)) * up(f)).asType(x.dtype)
    }
}
final class MingOmniSparseMoE: MingOmniFeedForward {
    let gate: MingOmniRouter
    @ModuleInfo(key: "switch_mlp") var experts: MingSwitchMLP
    @ModuleInfo(key: "shared_experts") var shared: MingOmniMLP?
    init(_ c: MingOmniTextConfiguration) {
        gate = MingOmniRouter(c)
        _experts.wrappedValue = MingSwitchMLP(hidden: c.hiddenSize, intermediate: c.moeIntermediateSize, experts: c.numExperts)
        if c.numSharedExperts > 0 && c.sharedExpertEnabled {
            _shared.wrappedValue = MingOmniMLP(hidden: c.hiddenSize,
                intermediate: (c.sharedExpertIntermediateSize ?? c.moeIntermediateSize) * c.numSharedExperts)
        }
    }
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32), (indices, weights) = gate(f)
        var output = (experts(f, indices: indices).asType(.float32) * weights.expandedDimensions(axis: -1)).sum(axis: -2)
        if let shared { output = output + shared(f).asType(.float32) }
        return output.asType(x.dtype)
    }
}

/// Half-rotation RoPE, including Bailing's three position streams. Speech supplies equal streams.
func mingOmniRotary(_ x: MLXArray, positions: MLXArray, dimensions: Int, base: Float, sections: [Int]?) -> MLXArray {
    let inverse = 1 / pow(MLXArray(base), MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32) / Float(dimensions))
    let pos = positions.ndim == 2 ? stacked([positions, positions, positions]) : positions
    let frequencies = pos.asType(.float32).expandedDimensions(axis: -1) * inverse
    let phase = concatenated([frequencies, frequencies], axis: -1)
    var c = cos(phase).asType(x.dtype), s = sin(phase).asType(x.dtype)
    let widths = (sections ?? [16, 24, 24]).map { $0 * 2 }
    if widths.reduce(0, +) == dimensions {
        var start = 0
        var cosParts: [MLXArray] = [], sinParts: [MLXArray] = []
        for (i, width) in widths.enumerated() {
            cosParts.append(c[i % 3, 0..., 0..., start..<(start + width)])
            sinParts.append(s[i % 3, 0..., 0..., start..<(start + width)])
            start += width
        }
        c = concatenated(cosParts, axis: -1); s = concatenated(sinParts, axis: -1)
    } else { c = c[0]; s = s[0] }
    let part = x[.ellipsis, ..<dimensions]
    let half = dimensions / 2
    let rotated = concatenated([-part[.ellipsis, half...], part[.ellipsis, ..<half]], axis: -1)
    let result = part * c.expandedDimensions(axis: 1) + rotated * s.expandedDimensions(axis: 1)
    return dimensions < x.dim(-1) ? concatenated([result, x[.ellipsis, dimensions...]], axis: -1) : result
}
final class MingOmniAttention: Module {
    let config: MingOmniTextConfiguration
    @ModuleInfo(key: "query_key_value") var qkv: Linear
    @ModuleInfo var dense: Linear
    @ModuleInfo(key: "key_layernorm") var keyNorm: RMSNorm?
    @ModuleInfo(key: "query_layernorm") var queryNorm: RMSNorm?
    init(_ c: MingOmniTextConfiguration) {
        config = c
        _qkv.wrappedValue = Linear(c.hiddenSize, (c.numAttentionHeads + 2 * c.numKeyValueHeads) * c.headDim, bias: c.useQKVBias)
        _dense.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.useBias)
        if c.useQKNormalization {
            _keyNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
            _queryNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        }
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache?, positions: MLXArray?) -> MLXArray {
        let c = config, b = x.dim(0), length = x.dim(1)
        let qSize = c.numAttentionHeads * c.headDim, kvSize = c.numKeyValueHeads * c.headDim
        let parts = split(qkv(x), indices: [qSize, qSize + kvSize], axis: -1)
        var q = parts[0].reshaped(b, length, c.numAttentionHeads, c.headDim).transposed(0, 2, 1, 3)
        var k = parts[1].reshaped(b, length, c.numKeyValueHeads, c.headDim).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped(b, length, c.numKeyValueHeads, c.headDim).transposed(0, 2, 1, 3)
        if let queryNorm, let keyNorm { q = queryNorm(q); k = keyNorm(k) }
        let offset = cache?.offset ?? 0
        let pos = positions ?? broadcast(MLXArray(offset..<(offset + length))[.newAxis], to: [b, length])
        q = mingOmniRotary(q, positions: pos, dimensions: c.ropeDimensions, base: c.ropeTheta, sections: c.ropeScaling?.sections)
        k = mingOmniRotary(k, positions: pos, dimensions: c.ropeDimensions, base: c.ropeTheta, sections: c.ropeScaling?.sections)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if length > 1 {
            let allowed = MLXArray(0..<(offset + length))[.newAxis] .<= MLXArray(offset..<(offset + length))[0..., .newAxis]
            mask = .array(MLX.where(allowed, MLXArray(0).asType(q.dtype), MLXArray(-Float.greatestFiniteMagnitude).asType(q.dtype)))
        } else { mask = .none }
        let out = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache,
            scale: 1 / sqrt(Float(c.headDim)), mask: mask)
        return dense(out.transposed(0, 2, 1, 3).reshaped(b, length, -1))
    }
}
final class MingOmniDecoderLayer: Module {
    let attention: MingOmniAttention
    let mlp: MingOmniFeedForward
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(_ c: MingOmniTextConfiguration, index: Int) {
        attention = MingOmniAttention(c)
        mlp = index >= c.firstKDenseReplace ? MingOmniSparseMoE(c) : MingOmniMLP(hidden: c.hiddenSize, intermediate: c.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache?, positions: MLXArray?) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache, positions: positions)
        return h + mlp(postNorm(h))
    }
}
final class MingOmniBackbone: Module {
    @ModuleInfo(key: "word_embeddings") var embedding: Embedding
    let layers: [MingOmniDecoderLayer]
    let norm: RMSNorm
    init(_ c: MingOmniTextConfiguration) {
        _embedding.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        layers = (0..<c.numHiddenLayers).map { MingOmniDecoderLayer(c, index: $0) }
        norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]? = nil, positions: MLXArray? = nil) -> MLXArray {
        var h = embeddings
        for (i, layer) in layers.enumerated() { h = layer(h, cache: cache?[i], positions: positions) }
        return norm(h)
    }
}
final class MingOmniLanguageModel: Module {
    let model: MingOmniBackbone
    @ModuleInfo(key: "lm_head") var head: Linear?
    init(_ c: MingOmniTextConfiguration) {
        model = MingOmniBackbone(c)
        if !c.tieWordEmbeddings { _head.wrappedValue = Linear(c.hiddenSize, c.vocabSize, bias: false) }
    }
}
