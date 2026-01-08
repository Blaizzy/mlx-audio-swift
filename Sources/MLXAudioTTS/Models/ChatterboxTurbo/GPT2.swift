//
//  GPT2.swift
//  MLXAudio
//
//  Created by Prince Canuma on 09/01/2026.
//

import Foundation
@preconcurrency import MLX
import MLXNN
import MLXLMCommon
import MLXFast
import MLXAudioCore

private func geluNew(_ x: MLXArray) -> MLXArray {
    let coeff: Float = 0.044715
    let sqrtTwoOverPi = Float(2.0 / Float.pi).squareRoot()
    let inner = sqrtTwoOverPi * (x + coeff * MLX.pow(x, 3.0))
    return 0.5 * x * (1.0 + MLX.tanh(inner))
}

private final class GPT2Attention: Module {
    private let config: GPT2Config
    private let scale: Float

    @ModuleInfo(key: "c_attn") private var cAttn: Linear
    @ModuleInfo(key: "c_proj") private var cProj: Linear

    init(_ config: GPT2Config) {
        self.config = config
        self.scale = pow(Float(config.nEmbeddings / config.nHead), -0.5)
        self._cAttn.wrappedValue = Linear(config.nEmbeddings, 3 * config.nEmbeddings)
        self._cProj.wrappedValue = Linear(config.nEmbeddings, config.nEmbeddings)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (hiddenStates.dim(0), hiddenStates.dim(1))

        let qkv = cAttn(hiddenStates)
        let splits = qkv.split(parts: 3, axis: -1)
        var q = splits[0]
        var k = splits[1]
        var v = splits[2]

        let headDim = config.nEmbeddings / config.nHead
        q = q.reshaped(batch, length, config.nHead, headDim).transposed(0, 2, 1, 3)
        k = k.reshaped(batch, length, config.nHead, headDim).transposed(0, 2, 1, 3)
        v = v.reshaped(batch, length, config.nHead, headDim).transposed(0, 2, 1, 3)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        ).transposed(0, 2, 1, 3).reshaped(batch, length, -1)

        return cProj(attn)
    }
}

private final class GPT2MLP: Module {
    @ModuleInfo(key: "c_fc") private var cFc: Linear
    @ModuleInfo(key: "c_proj") private var cProj: Linear

    init(_ config: GPT2Config) {
        let innerDim = config.nInner ?? 4 * config.nEmbeddings
        self._cFc.wrappedValue = Linear(config.nEmbeddings, innerDim)
        self._cProj.wrappedValue = Linear(innerDim, config.nEmbeddings)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return cProj(geluNew(cFc(x)))
    }
}

private final class GPT2Block: Module {
    @ModuleInfo(key: "ln_1") private var ln1: LayerNorm
    @ModuleInfo(key: "attn") private var attn: GPT2Attention
    @ModuleInfo(key: "ln_2") private var ln2: LayerNorm
    @ModuleInfo(key: "mlp") private var mlp: GPT2MLP

    init(_ config: GPT2Config) {
        self._ln1.wrappedValue = LayerNorm(dimensions: config.nEmbeddings, eps: config.layerNormEpsilon)
        self._attn.wrappedValue = GPT2Attention(config)
        self._ln2.wrappedValue = LayerNorm(dimensions: config.nEmbeddings, eps: config.layerNormEpsilon)
        self._mlp.wrappedValue = GPT2MLP(config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var h = hiddenStates + attn(ln1(hiddenStates), mask: mask, cache: cache)
        h = h + mlp(ln2(h))
        return h
    }
}

final class GPT2Model: Module {
    private let config: GPT2Config

    @ModuleInfo(key: "wte") private var wte: Embedding
    @ModuleInfo(key: "wpe") private var wpe: Embedding
    @ModuleInfo(key: "h") private var blocks: [GPT2Block]
    @ModuleInfo(key: "ln_f") private var lnF: LayerNorm

    init(_ config: GPT2Config) {
        self.config = config

        self._wte.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.nEmbeddings)
        self._wpe.wrappedValue = Embedding(embeddingCount: config.nPositions, dimensions: config.nEmbeddings)
        self._blocks.wrappedValue = (0..<config.nLayer).map { _ in GPT2Block(config) }
        self._lnF.wrappedValue = LayerNorm(dimensions: config.nEmbeddings, eps: config.layerNormEpsilon)
    }

    func callAsFunction(
        inputIds: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> (MLXArray, [KVCache]) {
        let hiddenStates: MLXArray
        if let inputsEmbeds {
            hiddenStates = inputsEmbeds
        } else if let inputIds {
            hiddenStates = wte(inputIds)
        } else {
            fatalError("Either inputIds or inputsEmbeds must be provided")
        }

        let batch = hiddenStates.dim(0)
        let length = hiddenStates.dim(1)

        let cacheOffset = cache?.first?.offset ?? 0
        let positions = MLXArray.arange(cacheOffset, cacheOffset + length, dtype: .int32)
        let posEmbeds = wpe(positions).expandedDimensions(axis: 0)
        var h = hiddenStates + posEmbeds.broadcasted(to: [batch, length, config.nEmbeddings])

        let activeCache: [KVCache]
        if let cache {
            activeCache = cache
        } else {
            activeCache = (0..<config.nLayer).map { _ in KVCacheSimple() }
        }

        let mask = createAttentionMask(h: h, cache: activeCache.first)

        for (index, block) in blocks.enumerated() {
            h = block(h, mask: mask, cache: activeCache[index])
        }

        return (lnF(h), activeCache)
    }
}
