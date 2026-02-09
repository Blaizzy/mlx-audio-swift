import Foundation
@preconcurrency import MLX
import MLXNN
import MLXLMCommon

@inline(__always)
private func pocketCreateAdditiveCausalMask(_ n: Int, offset: Int = 0) -> MLXArray {
    let total = offset + n
    let rinds = MLXArray(stride(from: 0, to: total, by: 1))
    let linds = offset > 0 ? MLXArray(stride(from: offset, to: total, by: 1)) : rinds
    let li = linds.reshaped([n, 1])
    let ri = rinds.reshaped([1, total])
    let mask = li .< ri
    return mask.asType(.float32) * MLXArray(-1e9)
}

public final class PocketLayerScale: Module {
    public var scale: MLXArray

    public init(channels: Int, initValue: Float) {
        self.scale = MLXArray.ones([channels]) * MLXArray(initValue)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { x * scale }
}

public final class PocketRotaryEmbedding: Module {
    public let maxPeriod: Double

    public init(maxPeriod: Double = 10000.0) {
        self.maxPeriod = maxPeriod
        super.init()
    }

    public func callAsFunction(_ q: MLXArray, _ k: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        return pocketApplyRoPE(q: q, k: k, offset: offset, maxPeriod: maxPeriod)
    }
}

@inline(__always)
private func pocketApplyRoPE(q: MLXArray, k: MLXArray, offset: Int, maxPeriod: Double) -> (MLXArray, MLXArray) {
    let b = q.shape[0]
    let t = q.shape[1]
    let h = q.shape[2]
    let d = q.shape[3]
    precondition(d % 2 == 0, "RoPE requires an even head dimension")
    let half = d / 2

    let indices = MLXArray(stride(from: 0, to: half, by: 1)).asType(.float32)
    let freqScale = -Float(log(maxPeriod)) * 2.0 / Float(d)
    let freqs = MLX.exp(indices * MLXArray(freqScale))

    let ts = (MLXArray(stride(from: 0, to: t, by: 1)).asType(.float32) + MLXArray(Float(offset)))
        .reshaped([1, t, 1, 1])
    let freq4d = freqs.reshaped([1, 1, 1, half])

    let q2 = q.reshaped([b, t, h, half, 2])
    let k2 = k.reshaped([b, t, h, half, 2])

    let qParts = split(q2, indices: [1], axis: 4)
    let kParts = split(k2, indices: [1], axis: 4)

    let qr = qParts[0].squeezed(axis: 4).asType(.float32)
    let qi = qParts[1].squeezed(axis: 4).asType(.float32)
    let kr = kParts[0].squeezed(axis: 4).asType(.float32)
    let ki = kParts[1].squeezed(axis: 4).asType(.float32)

    let rotr = MLX.cos(freq4d * ts)
    let roti = MLX.sin(freq4d * ts)

    let qor = qr * rotr - qi * roti
    let qoi = qr * roti + qi * rotr
    let kor = kr * rotr - ki * roti
    let koi = kr * roti + ki * rotr

    let qOut = concatenated([
        qor.asType(q.dtype).expandedDimensions(axis: 4),
        qoi.asType(q.dtype).expandedDimensions(axis: 4)
    ], axis: 4).reshaped([b, t, h, d])

    let kOut = concatenated([
        kor.asType(k.dtype).expandedDimensions(axis: 4),
        koi.asType(k.dtype).expandedDimensions(axis: 4)
    ], axis: 4).reshaped([b, t, h, d])

    return (qOut, kOut)
}

public final class PocketStreamingMultiheadAttention: Module {
    public let embedDim: Int
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float
    public let rope: PocketRotaryEmbedding

    @ModuleInfo(key: "in_proj") public var in_proj: Linear
    @ModuleInfo(key: "out_proj") public var out_proj: Linear

    public init(embedDim: Int, numHeads: Int, rope: PocketRotaryEmbedding) {
        precondition(embedDim % numHeads == 0, "embed_dim must be divisible by num_heads")
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self.rope = rope
        self._in_proj = ModuleInfo(wrappedValue: Linear(embedDim, 3 * embedDim, bias: false))
        self._out_proj = ModuleInfo(wrappedValue: Linear(embedDim, embedDim, bias: false))
        super.init()
    }

    public func callAsFunction(_ query: MLXArray, cache: KVCacheSimple?) -> MLXArray {
        let b = query.shape[0]
        let t = query.shape[1]

        var projected = in_proj(query)
        projected = projected.reshaped([b, t, 3, numHeads, headDim])

        var q = projected[0..<b, 0..<t, 0, 0..<numHeads, 0..<headDim]
        var k = projected[0..<b, 0..<t, 1, 0..<numHeads, 0..<headDim]
        var v = projected[0..<b, 0..<t, 2, 0..<numHeads, 0..<headDim]

        let offset = cache?.offset ?? 0
        let (qR, kR) = rope(q, k, offset: offset)
        q = qR
        k = kR

        q = swappedAxes(q, 1, 2)
        k = swappedAxes(k, 1, 2)
        v = swappedAxes(v, 1, 2)

        let kFull: MLXArray
        let vFull: MLXArray
        if let cache {
            (kFull, vFull) = cache.update(keys: k, values: v)
        } else {
            kFull = k
            vFull = v
        }

        let mask = pocketCreateAdditiveCausalMask(t, offset: offset).asType(query.dtype)
        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: kFull,
            values: vFull,
            scale: scale,
            mask: mask
        )
        let merged = swappedAxes(out, 1, 2).reshaped([b, t, embedDim])
        return out_proj(merged)
    }
}

public final class PocketStreamingTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") public var self_attn: PocketStreamingMultiheadAttention
    @ModuleInfo(key: "norm1") public var norm1: PocketLayerNorm
    @ModuleInfo(key: "norm2") public var norm2: PocketLayerNorm
    @ModuleInfo(key: "linear1") public var linear1: Linear
    @ModuleInfo(key: "linear2") public var linear2: Linear
    @ModuleInfo(key: "layer_scale_1") public var layer_scale_1: PocketLayerScale?
    @ModuleInfo(key: "layer_scale_2") public var layer_scale_2: PocketLayerScale?

    public init(dModel: Int, numHeads: Int, dimFeedforward: Int, rope: PocketRotaryEmbedding, layerScale: Float?) {
        self._self_attn = ModuleInfo(wrappedValue: PocketStreamingMultiheadAttention(embedDim: dModel, numHeads: numHeads, rope: rope))
        self._norm1 = ModuleInfo(wrappedValue: PocketLayerNorm(channels: dModel, eps: 1e-5))
        self._norm2 = ModuleInfo(wrappedValue: PocketLayerNorm(channels: dModel, eps: 1e-5))
        self._linear1 = ModuleInfo(wrappedValue: Linear(dModel, dimFeedforward, bias: false))
        self._linear2 = ModuleInfo(wrappedValue: Linear(dimFeedforward, dModel, bias: false))
        if let layerScale {
            self._layer_scale_1 = ModuleInfo(wrappedValue: PocketLayerScale(channels: dModel, initValue: layerScale))
            self._layer_scale_2 = ModuleInfo(wrappedValue: PocketLayerScale(channels: dModel, initValue: layerScale))
        } else {
            self._layer_scale_1 = ModuleInfo(wrappedValue: nil)
            self._layer_scale_2 = ModuleInfo(wrappedValue: nil)
        }
        super.init()
    }

    private func applyScale(_ x: MLXArray, _ scale: PocketLayerScale?) -> MLXArray {
        guard let scale else { return x }
        return scale(x)
    }

    public func callAsFunction(_ x: MLXArray, cache: KVCacheSimple?) -> MLXArray {
        let attnOut = self_attn(norm1(x), cache: cache)
        var h = x + applyScale(attnOut, layer_scale_1)
        let ff = linear2(gelu(linear1(norm2(h))))
        h = h + applyScale(ff, layer_scale_2)
        return h
    }
}

public final class PocketStreamingTransformer: Module {
    public let dModel: Int
    public let numHeads: Int
    public let numLayers: Int
    public let headDim: Int
    public let rope: PocketRotaryEmbedding

    @ModuleInfo(key: "layers") public var layers: [PocketStreamingTransformerLayer]

    public init(dModel: Int, numHeads: Int, numLayers: Int, dimFeedforward: Int, maxPeriod: Double, layerScale: Float? = nil) {
        self.dModel = dModel
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.headDim = dModel / numHeads
        let rope = PocketRotaryEmbedding(maxPeriod: maxPeriod)
        self.rope = rope
        self._layers = ModuleInfo(wrappedValue: (0..<numLayers).map { _ in
            PocketStreamingTransformerLayer(
                dModel: dModel,
                numHeads: numHeads,
                dimFeedforward: dimFeedforward,
                rope: rope,
                layerScale: layerScale
            )
        })
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cache: [KVCacheSimple]?) -> MLXArray {
        var h = x
        let caches = cache ?? (0..<layers.count).map { _ in KVCacheSimple() }
        for (layer, layerCache) in zip(layers, caches) {
            h = layer(h, cache: layerCache)
        }
        return h
    }

    public func makeCache() -> [KVCacheSimple] {
        return (0..<layers.count).map { _ in KVCacheSimple() }
    }
}
