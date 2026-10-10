//
//  MiniCPM.swift
//  MLXAudio
//
//  MiniCPM transformer backbone for VoxCPM2.
//  Ported from mlx-audio Python: voxcpm2/minicpm.py
//

import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - RMS Norm

class VoxRMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dims: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dims])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - LongRoPE

class VoxCPMLongRoPE: Module {
    let config: VoxCPM2LMConfig
    let dim: Int
    let base: Float
    let maxPositionEmbeddings: Int
    let originalMaxPositionEmbeddings: Int
    let shortFactor: MLXArray
    let longFactor: MLXArray
    let scalingFactor: Float
    let invFreq: MLXArray

    init(_ config: VoxCPM2LMConfig) {
        self.config = config
        self.dim = config.kvChannels ?? (config.hiddenSize / config.numAttentionHeads)
        self.base = config.ropeTheta
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.originalMaxPositionEmbeddings = config.originalMaxPositionEmbeddings

        self.shortFactor = MLXArray(config.ropeShortFactor)
        self.longFactor = MLXArray(config.ropeLongFactor)

        let scale = Float(config.maxPositionEmbeddings) / Float(config.originalMaxPositionEmbeddings)
        self.scalingFactor = Foundation.sqrt(
            1.0 + log(max(scale, 1.0)) / log(Float(config.originalMaxPositionEmbeddings))
        )

        let halfDim = dim / 2
        let exponents = MLXArray(0 ..< halfDim).asType(.float32) / Float(halfDim)
        self.invFreq = 1.0 / MLX.pow(MLXArray(base), exponents)

        super.init()
    }

    func callAsFunction(_ positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let seqLen = positionIds.max().item(Int.self) + 1

        let factors = seqLen > originalMaxPositionEmbeddings ? longFactor : shortFactor

        let t = MLXArray(0 ..< seqLen).asType(.float32)

        let freqs = (t.expandedDimensions(axis: 1) * (1.0 / factors.expandedDimensions(axis: 0)))
            * invFreq.expandedDimensions(axis: 0)
        let emb = MLX.concatenated([freqs, freqs], axis: -1)

        let cos = MLX.cos(emb) * scalingFactor
        let sin = MLX.sin(emb) * scalingFactor

        return (cos[positionIds], sin[positionIds])
    }
}

// MARK: - Rotary Embedding Helpers

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return MLX.concatenated([-x2, x1], axis: -1)
}

private func applyRotaryPosEmb(
    _ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    // cos/sin: (B, L, D) -> (B, L, 1, D) for broadcasting over heads
    let cosE = cos[0..., 0..., .newAxis, 0...]
    let sinE = sin[0..., 0..., .newAxis, 0...]

    let qEmb = (q * cosE) + (rotateHalf(q) * sinE)
    let kEmb = (k * cosE) + (rotateHalf(k) * sinE)
    return (qEmb, kEmb)
}

// MARK: - Attention

class VoxAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: VoxCPM2LMConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.kvChannels ?? (config.hiddenSize / config.numAttentionHeads)

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray?,
        sin: MLXArray?,
        mask: MLXArray?,
        cache: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim)

        if let cos, let sin {
            (q, k) = applyRotaryPosEmb(q, k, cos: cos, sin: sin)
        }

        var kFull = k
        var vFull = v
        if let cache {
            kFull = MLX.concatenated([cache.0, k], axis: 1)
            vFull = MLX.concatenated([cache.1, v], axis: 1)
        }

        let newCache = (kFull, vFull)

        let qT = q.transposed(0, 2, 1, 3)
        let kT = kFull.transposed(0, 2, 1, 3)
        let vT = vFull.transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: qT, keys: kT, values: vT,
            scale: 1.0 / Foundation.sqrt(Float(headDim)),
            mask: mask
        )

        let outReshaped = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return (oProj(outReshaped), newCache)
    }
}

// MARK: - MLP (SwiGLU)

class VoxMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: VoxCPM2LMConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder Layer

class VoxMiniCPMDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VoxAttention
    @ModuleInfo var mlp: VoxMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: VoxRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: VoxRMSNorm

    let scaleDepth: Float
    let numHiddenLayers: Int
    let useMup: Bool

    init(_ config: VoxCPM2LMConfig) {
        self._selfAttn.wrappedValue = VoxAttention(config)
        self._mlp.wrappedValue = VoxMLP(config)
        self._inputLayernorm.wrappedValue = VoxRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = VoxRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)
        self.scaleDepth = config.scaleDepth
        self.numHiddenLayers = config.numHiddenLayers
        self.useMup = config.useMup
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray?,
        sin: MLXArray?,
        mask: MLXArray?,
        cache: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let scale = useMup ? scaleDepth / Foundation.sqrt(Float(numHiddenLayers)) : 1.0

        var r = x
        var h: MLXArray
        let newCache: (MLXArray, MLXArray)
        (h, newCache) = selfAttn(inputLayernorm(x), cos: cos, sin: sin, mask: mask, cache: cache)
        var out = r + h * scale

        r = out
        h = mlp(postAttentionLayernorm(out))
        out = r + h * scale

        return (out, newCache)
    }
}

// MARK: - MiniCPM Model

class VoxMiniCPMModel: Module {
    let config: VoxCPM2LMConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    let layers: [VoxMiniCPMDecoderLayer]
    @ModuleInfo var norm: VoxRMSNorm
    let rope: VoxCPMLongRoPE?

    init(_ config: VoxCPM2LMConfig) {
        self.config = config

        if config.vocabSize > 0 {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize
            )
        } else {
            self._embedTokens.wrappedValue = nil
        }

        self.layers = (0 ..< config.numHiddenLayers).map { _ in VoxMiniCPMDecoderLayer(config) }
        self._norm.wrappedValue = VoxRMSNorm(dims: config.hiddenSize, eps: config.rmsNormEps)

        self.rope = config.noRope ? nil : VoxCPMLongRoPE(config)

        super.init()
    }

    func callAsFunction(
        inputsEmbeds: MLXArray? = nil,
        inputIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil,
        isCausal: Bool = true
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var h: MLXArray
        if let inputsEmbeds {
            h = inputsEmbeds
        } else if let inputIds, let embedTokens {
            h = embedTokens(inputIds)
        } else {
            fatalError("MiniCPMModel requires either inputsEmbeds or inputIds")
        }

        let L = h.dim(1)

        var offset = 0
        if let cache, !cache.isEmpty {
            offset = cache[0].0.dim(1)
        }

        var cos: MLXArray?
        var sin: MLXArray?
        if let rope {
            let positionIds = MLXArray(offset ..< (offset + L)).asType(.int32)
            (cos, sin) = rope(positionIds)
            cos = cos!.expandedDimensions(axis: 0)
            sin = sin!.expandedDimensions(axis: 0)
        }

        var attnMask = mask
        if attnMask == nil && isCausal && L > 1 {
            let ones = MLXArray.ones([L, L])
            let causalMask = triu(ones, k: 1) * Float(-1e9)
            attnMask = causalMask.expandedDimensions(axes: [0, 1])
        }

        var newCaches: [(MLXArray, MLXArray)] = []
        newCaches.reserveCapacity(layers.count)
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let c: (MLXArray, MLXArray)
            (h, c) = layer(h, cos: cos, sin: sin, mask: attnMask, cache: layerCache)
            newCaches.append(c)
        }

        h = norm(h)
        return (h, newCaches)
    }
}
