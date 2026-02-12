//
//  VoxtralRealtimeDecoder.swift
//  MLXAudioSTT
//

import Foundation
import MLX
import MLXNN

// MARK: - Time Embedding

func computeTimeEmbedding(tValue: Float, dim: Int, theta: Float = 10000.0) -> MLXArray {
    let halfDim = dim / 2
    let invFreq = MLX.exp(
        -log(theta) * MLX.arange(halfDim).asType(.float32) / MLXArray(Float(halfDim)))
    let emb = MLXArray(tValue) * invFreq
    return MLX.concatenated([MLX.cos(emb), MLX.sin(emb)])
}

// MARK: - AdaRMSNorm

class AdaRMSNorm: Module {
    @ModuleInfo(key: "ada_down") var adaDown: Linear
    @ModuleInfo(key: "ada_up") var adaUp: Linear

    init(dim: Int, bottleneckDim: Int) {
        self._adaDown.wrappedValue = Linear(dim, bottleneckDim, bias: false)
        self._adaUp.wrappedValue = Linear(bottleneckDim, dim, bias: false)
    }

    func computeScale(_ tCond: MLXArray) -> MLXArray {
        return adaUp(gelu(adaDown(tCond)))
    }

    func callAsFunction(_ x: MLXArray, adaScale: MLXArray) -> MLXArray {
        return x * (1.0 + adaScale)
    }
}

// MARK: - Decoder KV Cache

struct VoxtralKVCache {
    var k: MLXArray?
    var v: MLXArray?
    var posOffset: Int

    init() {
        self.k = nil
        self.v = nil
        self.posOffset = 0
    }

    init(k: MLXArray, v: MLXArray, posOffset: Int) {
        self.k = k
        self.v = v
        self.posOffset = posOffset
    }

    var isEmpty: Bool { k == nil }
}

// MARK: - Decoder Attention

class VoxtralDecoderAttention: Module {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let slidingWindow: Int
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    // Underscore prefix excludes from MLX Module parameter scanning
    var _ropeInvFreq: MLXArray

    init(_ config: VoxtralDecoderConfig) {
        self.nHeads = config.nHeads
        self.nKvHeads = config.nKvHeads
        self.headDim = config.headDim
        self.slidingWindow = config.slidingWindow
        self.scale = 1.0 / sqrt(Float(config.headDim))

        let qDim = config.nHeads * config.headDim
        let kvDim = config.nKvHeads * config.headDim

        self._wq.wrappedValue = Linear(config.dim, qDim, bias: false)
        self._wk.wrappedValue = Linear(config.dim, kvDim, bias: false)
        self._wv.wrappedValue = Linear(config.dim, kvDim, bias: false)
        self._wo.wrappedValue = Linear(qDim, config.dim, bias: false)

        self._ropeInvFreq =
            1.0
            / MLX.pow(
                MLXArray(config.ropeTheta),
                MLX.arange(0, config.headDim, step: 2).asType(.float32)
                    / MLXArray(Float(config.headDim)))
    }

    private func ropeFreqs(_ positions: MLXArray) -> (MLXArray, MLXArray) {
        let angles = positions[0..., .newAxis].asType(.float32) * _ropeInvFreq[.newAxis, 0...]
        return (MLX.cos(angles), MLX.sin(angles))
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, cache: VoxtralKVCache?
    ) -> (MLXArray, VoxtralKVCache) {
        let seqLen = x.dim(0)
        let q = wq(x)
        var k = wk(x)
        var v = wv(x)

        let (cos, sin) = ropeFreqs(positions)
        let qRoped = interleavedRope(q, cos: cos, sin: sin, nHeads: nHeads, headDim: headDim)
        k = interleavedRope(k, cos: cos, sin: sin, nHeads: nKvHeads, headDim: headDim)

        var cacheOffset = 0
        if let cache = cache, !cache.isEmpty {
            cacheOffset = cache.posOffset
            k = MLX.concatenated([cache.k!, k], axis: 0)
            v = MLX.concatenated([cache.v!, v], axis: 0)
        }

        var kvLen = k.dim(0)

        if kvLen > slidingWindow {
            let trim = kvLen - slidingWindow
            k = k[trim...]
            v = v[trim...]
            cacheOffset += trim
            kvLen = slidingWindow
        }

        let newCache = VoxtralKVCache(k: k, v: v, posOffset: cacheOffset)

        let qr = qRoped.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        let kr = k.reshaped(1, kvLen, nKvHeads, headDim).transposed(0, 2, 1, 3)
        let vr = v.reshaped(1, kvLen, nKvHeads, headDim).transposed(0, 2, 1, 3)

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if seqLen == 1 {
            mask = .none
        } else if seqLen <= slidingWindow && cache == nil {
            mask = .causal
        } else {
            let qPos = positions[0..., .newAxis]
            let kPos = MLX.arange(cacheOffset, cacheOffset + kvLen)[.newAxis, 0...]
            let causal = kPos .<= qPos
            let window = kPos .>= (qPos - MLXArray(Int32(slidingWindow - 1)))
            let combined = causal .&& window
            let maskArray = which(combined, MLXArray(Float(0.0)), MLXArray(Float(-1e9)))
            mask = .array(maskArray)
        }

        let attnOut = MLXFast.scaledDotProductAttention(
            queries: qr, keys: kr, values: vr, scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(seqLen, nHeads * headDim)

        return (wo(attnOut), newCache)
    }
}

// MARK: - Decoder Layer

class VoxtralDecoderLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralDecoderAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "ada_rms_norm_t_cond") var adaRmsNormTCond: AdaRMSNorm?
    // SwiGLU FFN (no biases in decoder)
    @ModuleInfo(key: "feed_forward_w1") var feedForwardW1: Linear
    @ModuleInfo(key: "feed_forward_w3") var feedForwardW3: Linear
    @ModuleInfo(key: "feed_forward_w2") var feedForwardW2: Linear

    init(_ config: VoxtralDecoderConfig) {
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._attention.wrappedValue = VoxtralDecoderAttention(config)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        if config.adaRmsNormTCond {
            self._adaRmsNormTCond.wrappedValue = AdaRMSNorm(
                dim: config.dim, bottleneckDim: config.adaRmsNormTCondDim)
        }

        self._feedForwardW1.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW3.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW2.wrappedValue = Linear(config.hiddenDim, config.dim, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, adaScale: MLXArray?, cache: VoxtralKVCache?
    ) -> (MLXArray, VoxtralKVCache) {
        var h = attentionNorm(x)
        let (attnOut, newCache) = attention(h, positions: positions, cache: cache)
        var out = x + attnOut

        h = ffnNorm(out)
        if let adaNorm = adaRmsNormTCond, let scale = adaScale {
            h = adaNorm(h, adaScale: scale)
        }
        let gate = silu(feedForwardW1(h))
        let up = feedForwardW3(h)
        out = out + feedForwardW2(gate * up)

        return (out, newCache)
    }
}

// MARK: - Decoder

public class VoxtralDecoder: Module {
    let config: VoxtralDecoderConfig

    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [VoxtralDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    var adaScales: [MLXArray?]?

    public init(_ config: VoxtralDecoderConfig) {
        self.config = config
        self._tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dim)
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in VoxtralDecoderLayer(config) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
    }

    func precomputeAdaScales(_ tCond: MLXArray) {
        var scales: [MLXArray?] = []
        for layer in layers {
            if let adaNorm = layer.adaRmsNormTCond {
                scales.append(adaNorm.computeScale(tCond))
            } else {
                scales.append(nil)
            }
        }
        self.adaScales = scales
    }

    func embedToken(_ tokenId: Int) -> MLXArray {
        return tokEmbeddings.weight[tokenId]
    }

    func embedTokens(_ tokenIds: MLXArray) -> MLXArray {
        return tokEmbeddings(tokenIds)
    }

    func forward(
        _ embeds: MLXArray, startPos: Int = 0, cache: [VoxtralKVCache]?
    ) -> (MLXArray, [VoxtralKVCache]) {
        var h = embeds
        let seqLen = h.dim(0)
        let positions = MLX.arange(startPos, startPos + seqLen)

        var newCaches: [VoxtralKVCache] = []
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let adaScale = adaScales?[i]
            let (out, kv) = layer(h, positions: positions, adaScale: adaScale, cache: layerCache)
            h = out
            newCaches.append(kv)
        }
        h = norm(h)
        return (h, newCaches)
    }

    func logits(_ h: MLXArray) -> MLXArray {
        return MLX.matmul(h, tokEmbeddings.weight.T)
    }
}
