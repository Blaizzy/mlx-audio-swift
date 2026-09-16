import Foundation
import MLX
import MLXNN

/// Index arithmetic for one `VoxtralRealtimeDecoderKVCache.append`, testable without
/// Metal. Rows older than the sliding window stay in place until a block of them has
/// piled up or the storage is full; then the window is copied down to row zero.
struct VoxtralRealtimeDecoderKVCacheAppendPlan: Equatable {
    static let capacityBlock = 256

    /// Rows to copy down to row zero before appending, or nil when nothing moves.
    let compactionRange: Range<Int>?
    /// Where the new rows land, after any compaction.
    let appendRange: Range<Int>
    let capacity: Int
    let count: Int
    /// Absolute position of storage row zero after this append.
    let positionOffset: Int
    /// The rows attention sees: the last `slidingWindow` stored rows.
    let windowRange: Range<Int>
    /// Absolute position of the first row in `windowRange`.
    let windowPositionOffset: Int
    let requiresGrowth: Bool

    static func make(
        count: Int,
        capacity: Int,
        positionOffset: Int,
        appendCount: Int,
        slidingWindow: Int
    ) -> Self {
        precondition(count >= 0 && count <= capacity)
        precondition(appendCount >= 0)
        precondition(slidingWindow > 0)

        let projectedCount = count + appendCount
        let excessAfterAppend = max(0, projectedCount - slidingWindow)
        // Let one block of invisible rows accumulate. If the storage fills first,
        // reclaim that prefix instead of growing it again.
        let shouldCompact = count > slidingWindow
            && (projectedCount > capacity || excessAfterAppend > capacityBlock)
        let compactionRange: Range<Int>? = shouldCompact
            ? ((count - slidingWindow)..<count)
            : nil
        let droppedCount = compactionRange?.lowerBound ?? 0
        let compactedCount = compactionRange?.count ?? count
        let appendRange = compactedCount..<(compactedCount + appendCount)
        let requiredCapacity = appendRange.upperBound
        let requiresGrowth = requiredCapacity > capacity
        let capacityAfterGrowth: Int
        if requiresGrowth {
            capacityAfterGrowth = max(
                capacityBlock,
                ((requiredCapacity + capacityBlock - 1) / capacityBlock) * capacityBlock
            )
        } else {
            capacityAfterGrowth = capacity
        }

        let newCount = appendRange.upperBound
        let windowStart = max(0, newCount - slidingWindow)
        let newPositionOffset = positionOffset + droppedCount
        return Self(
            compactionRange: compactionRange,
            appendRange: appendRange,
            capacity: capacityAfterGrowth,
            count: newCount,
            positionOffset: newPositionOffset,
            windowRange: windowStart..<newCount,
            windowPositionOffset: newPositionOffset + windowStart,
            requiresGrowth: requiresGrowth
        )
    }
}

/// Decoder key/value cache that writes each new row into preallocated storage.
///
/// A token's rows go in through a slice update, which MLX runs in place only while
/// nothing else references the storage buffer. Holding `keys` or `values` across an
/// append makes MLX copy the whole storage instead.
///
/// It is a class, so `append` changes it for every holder.
final class VoxtralRealtimeDecoderKVCache {
    private(set) var keys: MLXArray   // [capacity, n_kv_heads * head_dim]
    private(set) var values: MLXArray // [capacity, n_kv_heads * head_dim]
    private(set) var count = 0
    private(set) var positionOffset = 0 // absolute position of storage row zero

    private let slidingWindow: Int

    init(width: Int, dtype: DType, slidingWindow: Int) {
        self.slidingWindow = slidingWindow
        let capacity = VoxtralRealtimeDecoderKVCacheAppendPlan.capacityBlock
        keys = MLXArray.zeros([capacity, width], dtype: dtype)
        values = MLXArray.zeros([capacity, width], dtype: dtype)
    }

    /// Append rows and return what attention should read: the last `slidingWindow`
    /// stored rows, and the absolute position of the first of them.
    func append(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        keys: MLXArray, values: MLXArray, positionOffset: Int
    ) {
        precondition(newKeys.ndim == 2 && newValues.ndim == 2)
        precondition(newKeys.shape == newValues.shape)
        precondition(newKeys.shape[1] == keys.shape[1])
        precondition(newKeys.dtype == keys.dtype && newValues.dtype == values.dtype)

        let plan = VoxtralRealtimeDecoderKVCacheAppendPlan.make(
            count: count,
            capacity: keys.shape[0],
            positionOffset: positionOffset,
            appendCount: newKeys.shape[0],
            slidingWindow: slidingWindow
        )

        if let retained = plan.compactionRange {
            // The ranges may overlap. That is safe: the setter is functional, so the
            // right-hand side reads the pre-assignment values.
            keys[0..<retained.count] = keys[retained]
            values[0..<retained.count] = values[retained]
            count = retained.count  // the growth copy below reads this
        }

        if plan.requiresGrowth {
            var grownKeys = MLXArray.zeros([plan.capacity, keys.shape[1]], dtype: keys.dtype)
            var grownValues = MLXArray.zeros([plan.capacity, values.shape[1]], dtype: values.dtype)
            if count > 0 {
                grownKeys[0..<count] = keys[0..<count]
                grownValues[0..<count] = values[0..<count]
            }
            keys = grownKeys
            values = grownValues
        }

        keys[plan.appendRange] = newKeys
        values[plan.appendRange] = newValues
        count = plan.count
        positionOffset = plan.positionOffset

        return (
            keys[plan.windowRange],
            values[plan.windowRange],
            plan.windowPositionOffset
        )
    }
}

func voxtralComputeTimeEmbedding(
    tValue: Float,
    dim: Int,
    theta: Float = 10000.0
) -> MLXArray {
    let halfDim = dim / 2
    let invFreq = MLX.exp(
        -log(theta) * MLXArray(0..<halfDim).asType(.float32) / Float(halfDim)
    )
    let emb = tValue * invFreq
    return MLX.concatenated([MLX.cos(emb), MLX.sin(emb)], axis: 0)
}

final class VoxtralRealtimeAdaRMSNorm: Module {
    @ModuleInfo(key: "ada_down") var adaDown: Linear
    @ModuleInfo(key: "ada_up") var adaUp: Linear

    init(dim: Int, bottleneckDim: Int) {
        self._adaDown.wrappedValue = Linear(dim, bottleneckDim, bias: false)
        self._adaUp.wrappedValue = Linear(bottleneckDim, dim, bias: false)
    }

    func computeScale(tCond: MLXArray) -> MLXArray {
        let hidden = gelu(adaDown(tCond))
        return adaUp(hidden)
    }

    func callAsFunction(_ x: MLXArray, adaScale: MLXArray) -> MLXArray {
        // Cast the float32 adaScale down so it doesn't promote the fp16 hidden state.
        x * (1.0 + adaScale.asType(x.dtype))
    }
}

final class VoxtralRealtimeDecoderAttention: Module {
    let nHeads: Int
    let nKvHeads: Int
    let headDim: Int
    let slidingWindow: Int
    let ropeTheta: Float
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(_ config: VoxtralRealtimeDecoderConfig) {
        nHeads = config.nHeads
        nKvHeads = config.nKvHeads
        headDim = config.headDim
        slidingWindow = config.slidingWindow
        ropeTheta = config.ropeTheta
        scale = pow(Float(config.headDim), -0.5)

        let qDim = config.nHeads * config.headDim
        let kvDim = config.nKvHeads * config.headDim

        self._wq.wrappedValue = Linear(config.dim, qDim, bias: false)
        self._wk.wrappedValue = Linear(config.dim, kvDim, bias: false)
        self._wv.wrappedValue = Linear(config.dim, kvDim, bias: false)
        self._wo.wrappedValue = Linear(qDim, config.dim, bias: false)

    }

    /// Interleaved-RoPE cos/sin tables for `positions`. Every decoder layer rotates
    /// by the same tables, so the decoder builds them once per forward pass instead
    /// of once per layer — same operations, bit-identical outputs, `nLayers`× fewer
    /// tiny kernel launches per decoded token.
    fileprivate static func ropeFrequencies(
        positions: MLXArray,
        headDim: Int,
        ropeTheta: Float
    ) -> (MLXArray, MLXArray) {
        let idx = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
        let ropeInvFreq = 1.0 / MLX.pow(MLXArray(ropeTheta), idx / Float(headDim))
        let angles = positions.asType(.float32).expandedDimensions(axis: 1) * ropeInvFreq.expandedDimensions(axis: 0)
        return (MLX.cos(angles), MLX.sin(angles))
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray,
        cache: VoxtralRealtimeDecoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeDecoderKVCache) {
        let seqLen = x.shape[0]

        var q = wq(x)
        var k = wk(x)
        var v = wv(x)

        q = voxtralApplyInterleavedRoPE(q, cos: ropeCos, sin: ropeSin, nHeads: nHeads, headDim: headDim)
        k = voxtralApplyInterleavedRoPE(k, cos: ropeCos, sin: ropeSin, nHeads: nKvHeads, headDim: headDim)

        let newCache = cache ?? VoxtralRealtimeDecoderKVCache(
            width: k.shape[1], dtype: k.dtype, slidingWindow: slidingWindow
        )
        let window = newCache.append(keys: k, values: v)
        k = window.keys
        v = window.values
        let positionOffset = window.positionOffset
        let kvLen = k.shape[0]

        let q4 = q.reshaped(1, seqLen, nHeads, headDim).transposed(0, 2, 1, 3)
        let k4 = k.reshaped(1, kvLen, nKvHeads, headDim).transposed(0, 2, 1, 3)
        let v4 = v.reshaped(1, kvLen, nKvHeads, headDim).transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if seqLen == 1 {
            maskMode = .none
        } else if seqLen <= slidingWindow && cache == nil {
            maskMode = .causal
        } else {
            let qPos = positions.expandedDimensions(axis: 1)
            let kPos = MLXArray(positionOffset..<(positionOffset + kvLen)).asType(.int32).expandedDimensions(axis: 0)
            let causal = kPos .<= qPos
            let window = kPos .>= (qPos - MLXArray(Int32(slidingWindow - 1)))
            let allowed = logicalAnd(causal, window)
            // Match the activation dtype: a float32 mask over fp16 q/k aborts SDPA.
            let mask = MLX.where(allowed, MLXArray(0.0), MLXArray(-1e9)).asType(q.dtype)
            maskMode = .array(mask)
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: q4,
            keys: k4,
            values: v4,
            scale: scale,
            mask: maskMode
        )

        let out = attn.transposed(0, 2, 1, 3).reshaped(seqLen, nHeads * headDim)
        return (wo(out), newCache)
    }
}

final class VoxtralRealtimeDecoderLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: VoxtralRealtimeDecoderAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    @ModuleInfo(key: "ada_rms_norm_t_cond") var adaRmsNormTCond: VoxtralRealtimeAdaRMSNorm?

    @ModuleInfo(key: "feed_forward_w1") var feedForwardW1: Linear
    @ModuleInfo(key: "feed_forward_w3") var feedForwardW3: Linear
    @ModuleInfo(key: "feed_forward_w2") var feedForwardW2: Linear

    init(_ config: VoxtralRealtimeDecoderConfig) {
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        self._attention.wrappedValue = VoxtralRealtimeDecoderAttention(config)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)

        if config.adaRmsNormTCond {
            self._adaRmsNormTCond.wrappedValue = VoxtralRealtimeAdaRMSNorm(
                dim: config.dim,
                bottleneckDim: config.adaRmsNormTCondDim
            )
        } else {
            self._adaRmsNormTCond.wrappedValue = nil
        }

        self._feedForwardW1.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW3.wrappedValue = Linear(config.dim, config.hiddenDim, bias: false)
        self._feedForwardW2.wrappedValue = Linear(config.hiddenDim, config.dim, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray,
        positions: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray,
        adaScale: MLXArray?,
        cache: VoxtralRealtimeDecoderKVCache?
    ) -> (MLXArray, VoxtralRealtimeDecoderKVCache) {
        var out = x

        var h = attentionNorm(out)
        let attn = attention(h, positions: positions, ropeCos: ropeCos, ropeSin: ropeSin, cache: cache)
        h = attn.0
        out = out + h

        h = ffnNorm(out)
        if let adaScale, let ada = adaRmsNormTCond {
            h = ada(h, adaScale: adaScale)
        }

        let gate = silu(feedForwardW1(h))
        let up = feedForwardW3(h)
        out = out + feedForwardW2(gate * up)

        return (out, attn.1)
    }
}

final class VoxtralRealtimeDecoder: Module {
    let config: VoxtralRealtimeDecoderConfig

    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [VoxtralRealtimeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    var adaScales: [MLXArray?]?

    init(_ config: VoxtralRealtimeDecoderConfig) {
        self.config = config
        self._tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.dim
        )
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            VoxtralRealtimeDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
    }

    func precomputeAdaScales(_ tCond: MLXArray) {
        var scales: [MLXArray?] = []
        scales.reserveCapacity(layers.count)

        for layer in layers {
            if let ada = layer.adaRmsNormTCond {
                scales.append(ada.computeScale(tCond: tCond))
            } else {
                scales.append(nil)
            }
        }

        adaScales = scales
    }

    func embedToken(tokenId: Int) -> MLXArray {
        // Module lookup rather than a raw `weight` row: for a QuantizedEmbedding the
        // weight is bit-packed and only the module's gather dequantizes it.
        tokEmbeddings(MLXArray([Int32(tokenId)])).squeezed(axis: 0)
    }

    func embedTokens(_ tokenIds: MLXArray) -> MLXArray {
        tokEmbeddings(tokenIds)
    }

    func callAsFunction(
        _ embeds: MLXArray,
        startPos: Int,
        cache: [VoxtralRealtimeDecoderKVCache?]? = nil
    ) -> (MLXArray, [VoxtralRealtimeDecoderKVCache?]) {
        var h = embeds
        let seqLen = h.shape[0]
        let positions = MLXArray(startPos..<(startPos + seqLen)).asType(.int32)
        // Shared by every layer — see `VoxtralRealtimeDecoderAttention.ropeFrequencies`.
        let (ropeCos, ropeSin) = VoxtralRealtimeDecoderAttention.ropeFrequencies(
            positions: positions,
            headDim: config.headDim,
            ropeTheta: config.ropeTheta
        )

        var newCache: [VoxtralRealtimeDecoderKVCache?] = []
        newCache.reserveCapacity(layers.count)

        for i in layers.indices {
            let layerCache = cache?[i]
            let adaScale = adaScales?[i]
            let next = layers[i](
                h, positions: positions, ropeCos: ropeCos, ropeSin: ropeSin,
                adaScale: adaScale, cache: layerCache)
            h = next.0
            newCache.append(next.1)
        }

        h = norm(h)
        return (h, newCache)
    }

    func logits(_ h: MLXArray) -> MLXArray {
        // `asLinear` dispatches to the module's tied projection: a plain embedding
        // computes the same contraction as the previous matmul(h, weight.T), and a
        // QuantizedEmbedding takes the quantized-matmul path. Callers pass a single
        // hidden row, so lift to rank 2 for the projection and drop the batch axis.
        tokEmbeddings.asLinear(h.expandedDimensions(axis: 0)).squeezed(axis: 0)
    }
}
