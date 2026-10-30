// Copyright © 2025 Apple Inc.
// Falcon-H1 blocks adapted from mlx-lm under the MIT License.

import Foundation
import MLX
import MLXAudioCore
import MLXNN

private let audio8FalconSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = MLX.compile(shapeless: true) { gate, x in
    silu(gate) * x
}

final class Audio8FalconAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let rope: RoPE
    let heads: Int, kvHeads: Int, headDim: Int
    init(_ c: Audio8Config) {
        heads = c.nHead; kvHeads = c.nLocalHeads; headDim = c.headDim
        _qProj.wrappedValue = Linear(c.dim, heads * headDim, bias: c.attentionBias)
        _kProj.wrappedValue = Linear(c.dim, kvHeads * headDim, bias: c.attentionBias)
        _vProj.wrappedValue = Linear(c.dim, kvHeads * headDim, bias: c.attentionBias)
        _oProj.wrappedValue = Linear(heads * headDim, c.dim, bias: c.attentionBias)
        rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeBase)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), offset = cache?.offset ?? 0
        var q = qProj(x).reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([b, t, kvHeads, headDim]).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped([b, t, kvHeads, headDim]).transposed(0, 2, 1, 3)
        q = rope(q, offset: offset); k = rope(k, offset: offset)
        if let cache { (k, v) = cache.update(keys: k, values: v) }
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if t == 1 { mask = .none }
        else if offset == 0 { mask = .causal }
        else {
            let query = MLX.arange(offset, offset + t).expandedDimensions(axis: 1)
            let key = MLX.arange(offset + t).expandedDimensions(axis: 0)
            mask = .array((key .<= query).reshaped([1, 1, t, offset + t]))
        }
        let y = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
            scale: 1 / sqrt(Float(headDim)), mask: mask)
        return oProj(y.transposed(0, 2, 1, 3).reshaped([b, t, heads * headDim]))
    }
}

final class Audio8FalconMixer: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let conv1d: Conv1d
    var A_log: MLXArray
    var D: MLXArray
    var dt_bias: MLXArray
    let norm: RMSNorm?
    let inner: Int, convDim: Int, kernelSize: Int, heads: Int, headDim: Int, groups: Int, stateSize: Int
    let normBeforeGate: Bool
    init(_ c: Audio8Config) {
        inner = c.mambaDSSM; heads = c.mambaNHeads; headDim = c.mambaDHead
        groups = c.mambaNGroups; stateSize = c.mambaDState; kernelSize = c.mambaDConv
        convDim = inner + 2 * groups * stateSize
        normBeforeGate = c.mambaNormBeforeGate
        _inProj.wrappedValue = Linear(c.dim, inner + convDim + heads, bias: c.mambaProjBias)
        _outProj.wrappedValue = Linear(inner, c.dim, bias: c.projectorsBias)
        conv1d = Conv1d(inputChannels: convDim, outputChannels: convDim,
            kernelSize: kernelSize, groups: convDim, bias: c.mambaConvBias)
        A_log = log(MLX.arange(1, heads + 1, dtype: .float32))
        D = .ones([heads]); dt_bias = .ones([heads])
        norm = c.mambaRMSNorm ? RMSNorm(dimensions: inner, eps: c.normEps) : nil
    }
    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), projected = inProj(x)
        let gate = projected[0..., 0..., 0..<inner]
        let convInput = projected[0..., 0..., inner..<(inner + convDim)]
        let dt = projected[0..., 0..., (inner + convDim)...]
        let previous = cache?[0] ?? MLXArray.zeros([b, kernelSize - 1, convDim], dtype: x.dtype)
        let padded = concatenated([previous, convInput], axis: 1)
        if let cache { cache[0] = padded[0..., t..<(t + kernelSize - 1), 0...] }
        let convolved = silu(conv1d(padded))
        let input = convolved[0..., 0..., 0..<inner].reshaped([b, t, heads, headDim])
        let B = convolved[0..., 0..., inner..<(inner + groups * stateSize)].reshaped([b, t, groups, stateSize])
        let C = convolved[0..., 0..., (inner + groups * stateSize)...].reshaped([b, t, groups, stateSize])
        let (output, state) = Mamba2Scan.callAsFunction(x: input, aLog: A_log, b: B, c: C, d: D,
                                       dt: dt, dtBias: dt_bias, state: cache?[1])
        if let cache { cache[1] = state; cache.advance(t) }
        var y = output.reshaped([b, t, inner])
        if let norm, normBeforeGate { y = audio8FalconSwiGLU(gate, norm(y)) }
        else { y = audio8FalconSwiGLU(gate, y); if let norm { y = norm(y) } }
        return outProj(y)
    }
}

final class Audio8FalconMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    init(_ c: Audio8Config) {
        _gateProj.wrappedValue = Linear(c.dim, c.intermediateSize, bias: c.mlpBias)
        _upProj.wrappedValue = Linear(c.dim, c.intermediateSize, bias: c.mlpBias)
        _downProj.wrappedValue = Linear(c.intermediateSize, c.dim, bias: c.mlpBias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(audio8FalconSwiGLU(gateProj(x), upProj(x))) }
}

final class Audio8FalconBlock: Module {
    let mamba: Audio8FalconMixer
    @ModuleInfo(key: "self_attn") var selfAttn: Audio8FalconAttention
    @ModuleInfo(key: "feed_forward") var feedForward: Audio8FalconMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "pre_ff_layernorm") var preFFLayernorm: RMSNorm
    init(_ c: Audio8Config) {
        mamba = Audio8FalconMixer(c)
        _selfAttn.wrappedValue = Audio8FalconAttention(c)
        _feedForward.wrappedValue = Audio8FalconMLP(c)
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: c.dim, eps: c.normEps)
        _preFFLayernorm.wrappedValue = RMSNorm(dimensions: c.dim, eps: c.normEps)
    }
    func callAsFunction(_ x: MLXArray, cache: CacheList?) -> MLXArray {
        let n = inputLayernorm(x)
        let h = x + mamba(n, cache: cache?[0] as? ArraysCache) + selfAttn(n, cache: cache?[1])
        return h + feedForward(preFFLayernorm(h))
    }
}

final class Audio8FalconSlowStack: Module, UnaryLayer {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Audio8FalconBlock]
    @ModuleInfo(key: "final_layernorm") var finalLayernorm: RMSNorm
    let embeddingMultiplier: Float
    init(_ c: Audio8Config) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.dim)
        layers = (0..<c.nLayer).map { _ in Audio8FalconBlock(c) }
        _finalLayernorm.wrappedValue = RMSNorm(dimensions: c.dim, eps: c.normEps)
        embeddingMultiplier = c.embeddingMultiplier
    }
    func callAsFunction(_ hidden: MLXArray) -> MLXArray { callAsFunction(hidden, cache: nil) }
    func callAsFunction(_ hidden: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Scale the complete text + codebook embedding, as required by Audio8.
        var h = hidden * embeddingMultiplier
        for (i, layer) in layers.enumerated() { h = layer(h, cache: cache?[i] as? CacheList) }
        return finalLayernorm(h)
    }
}
