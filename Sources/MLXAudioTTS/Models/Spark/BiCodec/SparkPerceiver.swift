import Foundation
@preconcurrency import MLX
import MLXNN

private final class SparkPerceiverRMSNorm: Module {
    var gamma: MLXArray
    private let scale: Float

    init(dim: Int) {
        self.scale = powf(Float(dim), 0.5)
        self.gamma = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let norm = MLX.sqrt(MLX.sum(x * x, axis: -1, keepDims: true))
        let normalized = x / MLX.maximum(norm, MLXArray(Float(1e-12)))
        return normalized * scale * gamma
    }
}

private final class SparkPerceiverAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_kv") var toKV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    private let heads: Int
    private let dimHead: Int

    init(dim: Int, dimHead: Int = 64, heads: Int = 8) {
        self.heads = heads
        self.dimHead = dimHead
        let inner = dimHead * heads
        self._toQ = ModuleInfo(wrappedValue: Linear(dim, inner, bias: false), key: "to_q")
        self._toKV = ModuleInfo(wrappedValue: Linear(dim, inner * 2, bias: false), key: "to_kv")
        self._toOut = ModuleInfo(wrappedValue: Linear(inner, dim, bias: false), key: "to_out")
    }

    private func splitHeads(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0], n = x.shape[1]
        return x.reshaped([b, n, heads, dimHead]).transposed(0, 2, 1, 3)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let ctx = MLX.concatenated([x, context], axis: -2)
        let q = splitHeads(toQ(x))
        let kv = toKV(ctx)
        let parts = MLX.split(kv, parts: 2, axis: -1)
        let k = splitHeads(parts[0])
        let v = splitHeads(parts[1])
        let scale = powf(Float(dimHead), -0.5)
        let sim = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        let attn = MLX.softmax(sim, axis: -1)
        let out = MLX.matmul(attn, v).transposed(0, 2, 1, 3)
        let merged = out.reshaped([out.shape[0], out.shape[1], heads * dimHead])
        return toOut(merged)
    }
}

private final class SparkPerceiverFeedForward: Module {
    @ModuleInfo(key: "lin_in") var linearIn: Linear
    @ModuleInfo(key: "lin_out") var linearOut: Linear

    init(dim: Int, mult: Int = 4) {
        let inner = Int(Double(dim) * Double(mult) * 2.0 / 3.0)
        self._linearIn = ModuleInfo(wrappedValue: Linear(dim, inner * 2), key: "lin_in")
        self._linearOut = ModuleInfo(wrappedValue: Linear(inner, dim), key: "lin_out")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = linearIn(x)
        let parts = MLX.split(projected, parts: 2, axis: -1)
        let gated = MLXNN.gelu(parts[1]) * parts[0]
        return linearOut(gated)
    }
}

private final class SparkPerceiverLayer: Module {
    @ModuleInfo(key: "attn") var attn: SparkPerceiverAttention
    @ModuleInfo(key: "ff") var ff: SparkPerceiverFeedForward

    init(dim: Int, dimHead: Int, heads: Int, ffMult: Int) {
        self._attn = ModuleInfo(wrappedValue: SparkPerceiverAttention(dim: dim, dimHead: dimHead, heads: heads), key: "attn")
        self._ff = ModuleInfo(wrappedValue: SparkPerceiverFeedForward(dim: dim, mult: ffMult), key: "ff")
    }
}

/// Perceiver resampler: cross-attends `num_latents` learned queries to the ECAPA
/// feature map to produce a fixed-size speaker summary.
public final class SparkPerceiverResampler: Module {
    @ModuleInfo(key: "proj_context") var projContext: Linear
    @ModuleInfo(key: "layers") fileprivate var layers: [SparkPerceiverLayer]
    @ModuleInfo(key: "norm") fileprivate var norm: SparkPerceiverRMSNorm
    var latents: MLXArray

    public init(dim: Int, dimContext: Int, numLatents: Int, depth: Int = 2, dimHead: Int = 64, heads: Int = 8, ffMult: Int = 4) {
        self._projContext = ModuleInfo(wrappedValue: Linear(dimContext, dim), key: "proj_context")
        self.latents = MLXArray.zeros([numLatents, dim])
        self._layers = ModuleInfo(
            wrappedValue: (0 ..< depth).map { _ in
                SparkPerceiverLayer(dim: dim, dimHead: dimHead, heads: heads, ffMult: ffMult)
            }, key: "layers")
        self._norm = ModuleInfo(wrappedValue: SparkPerceiverRMSNorm(dim: dim), key: "norm")
    }

    /// Context [B, T, dimContext] -> latents [B, numLatents, dim].
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let context = projContext(x)
        var l = MLX.broadcast(latents.expandedDimensions(axis: 0), to: [x.shape[0]] + latents.shape)
        for layer in layers {
            l = layer.attn(l, context: context) + l
            l = layer.ff(l) + l
        }
        return norm(l)
    }
}
