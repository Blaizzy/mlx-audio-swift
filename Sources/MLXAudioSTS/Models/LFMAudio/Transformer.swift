import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - RoPE Utilities

func precomputeFreqsCis(dim: Int, maxSeqLen: Int, theta: Float = 10000.0) -> MLXArray {
    let freqs = 1.0 / MLX.pow(
        MLXArray(theta),
        MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
    )
    let t = MLXArray(0..<Int32(maxSeqLen)).asType(.float32)
    return MLX.outer(t, freqs)
}

func applyRotaryEmb(
    xq: MLXArray, xk: MLXArray, freqs: MLXArray, offset: Int = 0
) -> (MLXArray, MLXArray) {
    let seqLen = xq.dim(1)
    let f = freqs[offset..<(offset + seqLen)]
    let fExpanded = f.expandedDimensions(axes: [0, 2])

    let shape = xq.shape
    let lastDim = shape[shape.count - 1]
    let halfDim = lastDim / 2

    let xqR = xq[0..., 0..., 0..., .stride(by: 2)]
    let xqI = xq[0..., 0..., 0..., .stride(from: 1, by: 2)]
    let xkR = xk[0..., 0..., 0..., .stride(by: 2)]
    let xkI = xk[0..., 0..., 0..., .stride(from: 1, by: 2)]

    let cosF = MLX.cos(fExpanded)
    let sinF = MLX.sin(fExpanded)

    let xqOutR = xqR * cosF - xqI * sinF
    let xqOutI = xqR * sinF + xqI * cosF
    let xkOutR = xkR * cosF - xkI * sinF
    let xkOutI = xkR * sinF + xkI * cosF

    let xqOut = MLX.stacked([xqOutR, xqOutI], axis: -1).reshaped(xq.shape)
    let xkOut = MLX.stacked([xkOutR, xkOutI], axis: -1).reshaped(xk.shape)

    return (xqOut, xkOut)
}

// MARK: - SwiGLU (for Depthformer)

class DepthformerSwiGLU: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int, multipleOf: Int = 256) {
        var adjDim = Int(2.0 * Float(hiddenDim) / 3.0)
        adjDim = multipleOf * ((adjDim + multipleOf - 1) / multipleOf)
        self._w1.wrappedValue = Linear(dim, adjDim, bias: false)
        self._w2.wrappedValue = Linear(adjDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, adjDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

// MARK: - Depthformer Attention

class DepthformerAttention: Module {
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float
    let useQkNorm: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let freqs: MLXArray

    init(
        dim: Int, numHeads: Int, numKvHeads: Int,
        maxSeqLen: Int = 4096, ropeTheta: Float = 10000.0,
        useQkNorm: Bool = true
    ) {
        self.numHeads = numHeads
        self.numKvHeads = numKvHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(self.headDim), -0.5)
        self.useQkNorm = useQkNorm

        self._qProj.wrappedValue = Linear(dim, numHeads * self.headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, numKvHeads * self.headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, numKvHeads * self.headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * self.headDim, dim, bias: false)

        if useQkNorm {
            self._qNorm.wrappedValue = RMSNorm(dimensions: self.headDim)
            self._kNorm.wrappedValue = RMSNorm(dimensions: self.headDim)
        }

        self.freqs = precomputeFreqsCis(dim: self.headDim, maxSeqLen: maxSeqLen, theta: ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKvHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKvHeads, headDim)

        if useQkNorm, let qN = qNorm, let kN = kNorm {
            q = qN(q)
            k = kN(k)
        }

        let offset = cache?.0.dim(1) ?? 0
        let (qRot, kRot) = applyRotaryEmb(xq: q, xk: k, freqs: freqs, offset: offset)
        q = qRot
        k = kRot

        if let (kCache, vCache) = cache {
            k = concatenated([kCache, k], axis: 1)
            v = concatenated([vCache, v], axis: 1)
        }

        let newCache = (k, v)

        var qT = q.transposed(0, 2, 1, 3)
        var kT = k.transposed(0, 2, 1, 3)
        var vT = v.transposed(0, 2, 1, 3)

        if numKvHeads < numHeads {
            let nRep = numHeads / numKvHeads
            kT = MLX.repeated(kT, count: nRep, axis: 1)
            vT = MLX.repeated(vT, count: nRep, axis: 1)
        }

        var scores = MLX.matmul(qT, kT.transposed(0, 1, 3, 2)) * MLXArray(scale)
        if let mask = mask {
            scores = scores + mask
        }

        let attn = softmax(scores, axis: -1)
        let out = MLX.matmul(attn, vT).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return (oProj(out), newCache)
    }
}

// MARK: - Depthformer Block

class DepthformerBlock: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "attn") var attn: DepthformerAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "ffn") var ffn: DepthformerSwiGLU

    init(
        dim: Int, numHeads: Int, numKvHeads: Int, ffDim: Int,
        maxSeqLen: Int = 4096, ropeTheta: Float = 10000.0,
        normEps: Float = 1e-5, multipleOf: Int = 256, useQkNorm: Bool = true
    ) {
        self._attnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._attn.wrappedValue = DepthformerAttention(
            dim: dim, numHeads: numHeads, numKvHeads: numKvHeads,
            maxSeqLen: maxSeqLen, ropeTheta: ropeTheta, useQkNorm: useQkNorm
        )
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: dim, eps: normEps)
        self._ffn.wrappedValue = DepthformerSwiGLU(dim: dim, hiddenDim: ffDim, multipleOf: multipleOf)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (h, newCache) = attn(attnNorm(x), mask: mask, cache: cache)
        var out = x + h
        out = out + ffn(ffnNorm(out))
        return (out, newCache)
    }
}

// MARK: - Depthformer

public class Depthformer: Module {
    public let layersCount: Int
    let dim: Int
    let blocks: [DepthformerBlock]

    public init(layers: Int, dim: Int, numHeads: Int = 32, numKvHeads: Int = 8,
                ffDim: Int? = nil, tie: Bool = true) {
        self.layersCount = layers
        self.dim = dim
        let effectiveFFDim = ffDim ?? (dim * 4)

        self.blocks = (0..<layers).map { _ in
            DepthformerBlock(
                dim: dim, numHeads: numHeads, numKvHeads: numKvHeads,
                ffDim: effectiveFFDim, maxSeqLen: 4096, ropeTheta: 10000.0,
                useQkNorm: true
            )
        }
    }

    public func callAsFunction(
        _ x: MLXArray, cache: [(MLXArray, MLXArray)?]? = nil,
        useCache: Bool = false
    ) -> (MLXArray, [(MLXArray, MLXArray)]?) {
        var h = x
        var newCache: [(MLXArray, MLXArray)]? = useCache ? [] : nil

        for i in 0..<layersCount {
            let layerCache = cache?[i]
            let (out, lc) = blocks[i](h, cache: layerCache)
            h = out
            newCache?.append(lc)
        }

        return (h, newCache)
    }
}

// MARK: - LFM2 Backbone

class Lfm2Attention: Module {
    let args: LFM2BackboneConfig
    let scale: Float
    let headDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_layernorm") var qLayernorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kLayernorm: RMSNorm
    let rope: RoPE

    init(_ args: LFM2BackboneConfig) {
        self.args = args
        self.headDim = args.headDimensions
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        let heads = args.numAttentionHeads
        let kvHeads = args.numKeyValueHeads

        self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
        self._qLayernorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        self._kLayernorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(B, L, args.numAttentionHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        queries = qLayernorm(queries)
        keys = kLayernorm(keys)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
            (keys, values) = cache.update(keys: keys, values: values)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        ).transposed(0, 2, 1, 3).reshaped(B, L, -1)

        return outProj(output)
    }
}

class Lfm2ShortConv: Module {
    let lCache: Int
    let hiddenSize: Int
    let bias: Bool

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM2BackboneConfig, layerIdx: Int) {
        self.lCache = args.convLCache
        self.hiddenSize = args.hiddenSize
        self.bias = args.convBias

        self._conv.wrappedValue = Conv1d(
            inputChannels: args.hiddenSize, outputChannels: args.hiddenSize,
            kernelSize: lCache, groups: args.hiddenSize, bias: args.convBias
        )
        self._inProj.wrappedValue = Linear(args.hiddenSize, 3 * args.hiddenSize, bias: args.convBias)
        self._outProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: args.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let projected = inProj(x).split(parts: 3, axis: -1)
        let b = projected[0], c = projected[1], xIn = projected[2]
        let bx = b * xIn

        var state: MLXArray? = cache?[0]
        if state == nil {
            state = MLXArray.zeros([bx.dim(0), lCache - 1, hiddenSize], dtype: bx.dtype)
        }

        let xConv = concatenated([state!, bx], axis: -2)
        if let cache {
            cache[0] = xConv[0..., (xConv.dim(1) - (lCache - 1))..., 0...]
        }

        let convOut = conv(xConv)
        let y = c * convOut
        return outProj(y)
    }
}

class Lfm2MLP: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, ffDim: Int, multipleOf: Int, autoAdjust: Bool, multiplier: Float?) {
        var adjDim = ffDim
        if autoAdjust {
            adjDim = Int(Float(2 * ffDim) / 3.0)
            if let m = multiplier { adjDim = Int(m * Float(adjDim)) }
            adjDim = multipleOf * ((adjDim + multipleOf - 1) / multipleOf)
        }
        self._w1.wrappedValue = Linear(dim, adjDim, bias: false)
        self._w2.wrappedValue = Linear(adjDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, adjDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

class Lfm2DecoderLayer: Module {
    let isAttentionLayer: Bool

    @ModuleInfo(key: "self_attn") var attention: Lfm2Attention?
    @ModuleInfo(key: "conv") var conv: Lfm2ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: Lfm2MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    init(_ args: LFM2BackboneConfig, layerIdx: Int) {
        self.isAttentionLayer = args.resolvedFullAttnIdxs.contains(layerIdx)

        if isAttentionLayer {
            self._attention.wrappedValue = Lfm2Attention(args)
        } else {
            self._conv.wrappedValue = Lfm2ShortConv(args, layerIdx: layerIdx)
        }

        self._feedForward.wrappedValue = Lfm2MLP(
            dim: args.effectiveBlockDim, ffDim: args.effectiveBlockFFDim,
            multipleOf: args.blockMultipleOf, autoAdjust: args.blockAutoAdjustFFDim,
            multiplier: args.blockFFNDimMultiplier
        )
        self._operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r: MLXArray
        if isAttentionLayer {
            r = attention!(operatorNorm(x), mask: mask, cache: cache)
        } else {
            r = conv!(operatorNorm(x), cache: cache as? MambaCache)
        }
        let h = x + r
        return h + feedForward(ffnNorm(h))
    }
}

public class Lfm2Model: Module {
    let args: LFM2BackboneConfig
    let layers: [Lfm2DecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

    public init(_ args: LFM2BackboneConfig) {
        self.args = args

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize
        )
        self.layers = (0..<args.numHiddenLayers).map { Lfm2DecoderLayer(args, layerIdx: $0) }
        self._embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    public func callAsFunction(
        _ inputs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let emb = inputEmbeddings {
            h = emb
        } else {
            h = embedTokens(inputs!)
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode = {
            let firstAttnIdx = args.resolvedFullAttnIdxs.first ?? 0
            let c = (cache != nil && firstAttnIdx < cache!.count) ? cache![firstAttnIdx] : nil
            return createAttentionMask(h: h, cache: c)
        }()

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return embeddingNorm(h)
    }

    public func makeCache() -> [KVCache] {
        (0..<args.numHiddenLayers).map { layerIdx in
            if args.resolvedFullAttnIdxs.contains(layerIdx) {
                KVCacheSimple()
            } else {
                MambaCache()
            }
        }
    }
}
