import Foundation
import MLX
import MLXNN

private func samActivation(_ x: MLXArray, kind: String) -> MLXArray {
    switch kind {
    case "relu":
        return relu(x)
    case "gelu", "approx_gelu":
        return gelu(x)
    case "silu":
        return silu(x)
    default:
        return silu(x)
    }
}

/// RMSNorm used by SAMAudio blocks.
public final class SAMRMSNorm: Module {
    let eps: Float
    @ModuleInfo(key: "weight") var weight: MLXArray

    public init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat = x.asType(.float32)
        let norm = xFloat * rsqrt(mean(xFloat * xFloat, axis: -1, keepDims: true) + MLXArray(eps))
        return (norm * weight).asType(x.dtype)
    }
}

/// Projection layer with optional SwiGLU behavior.
public final class ProjectionLayer: Module {
    let swiglu: Bool
    let nonLinearity: String

    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear?

    public init(
        inDim: Int,
        outDim: Int,
        nonLinearity: String,
        dropout _: Float = 0,
        fcBias: Bool = false
    ) {
        self.swiglu = nonLinearity == "swiglu"
        self.nonLinearity = nonLinearity

        self._w1.wrappedValue = Linear(inDim, outDim, bias: fcBias)
        self._w2.wrappedValue = Linear(outDim, outDim, bias: fcBias)
        self._w3.wrappedValue = swiglu ? Linear(inDim, outDim, bias: fcBias) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden1 = w1(x)
        let hidden: MLXArray
        if swiglu {
            hidden = silu(hidden1) * (w3?(x) ?? MLXArray.zeros(hidden1.shape))
        } else {
            hidden = samActivation(hidden1, kind: nonLinearity)
        }
        return w2(hidden)
    }
}

/// Multi-head attention with SAM-Audio head layout for compatibility.
public final class SAMAttention: Module {
    let headDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let useQKNorm: Bool
    let scale: Float

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SAMRMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: SAMRMSNorm?

    public init(
        dim: Int,
        headDim: Int,
        nHeads: Int,
        nKVHeads: Int,
        normEps: Float = 1e-5,
        useQKNorm: Bool = false,
        fcBias: Bool = false
    ) {
        precondition(nHeads % nKVHeads == 0, "nHeads must be divisible by nKVHeads")

        self.headDim = headDim
        self.nHeads = nHeads
        self.nKVHeads = nKVHeads
        self.useQKNorm = useQKNorm
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: fcBias)
        self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: fcBias)
        self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: fcBias)
        self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: fcBias)

        self._qNorm.wrappedValue = useQKNorm ? SAMRMSNorm(dim: headDim, eps: normEps) : nil
        self._kNorm.wrappedValue = useQKNorm ? SAMRMSNorm(dim: headDim, eps: normEps) : nil
    }

    private func reshapeHeads(_ x: MLXArray, nHeads: Int) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let c = x.shape[2]
        let reshaped = x.reshaped([b, t, c / nHeads, nHeads]) // (B, T, headDim, H)
        return reshaped.transposed(0, 3, 1, 2) // (B, H, T, headDim)
    }

    private func repeatKV(_ x: MLXArray, repeats: Int) -> MLXArray {
        guard repeats > 1 else { return x }
        let b = x.shape[0]
        let kv = x.shape[1]
        let t = x.shape[2]
        let d = x.shape[3]
        let expanded = x.expandedDimensions(axis: 2) // (B, KV, 1, T, D)
        let broadcasted = MLX.broadcast(expanded, to: [b, kv, repeats, t, d])
        return broadcasted.reshaped([b, kv * repeats, t, d])
    }

    public func callAsFunction(
        _ x: MLXArray,
        crossX: MLXArray? = nil,
        keyPaddingMask: MLXArray? = nil,
        rope: RotaryEmbedding? = nil
    ) -> MLXArray {
        let batchSize = x.shape[0]
        let seqLen = x.shape[1]

        var xq = wq(x)
        let keyValueSource = crossX ?? x
        var xk = wk(keyValueSource)
        var xv = wv(keyValueSource)

        xq = reshapeHeads(xq, nHeads: nHeads)
        xk = reshapeHeads(xk, nHeads: nKVHeads)
        xv = reshapeHeads(xv, nHeads: nKVHeads)

        if useQKNorm {
            if let qNorm { xq = qNorm(xq) }
            if let kNorm { xk = kNorm(xk) }
        }

        if rope != nil, crossX == nil {
            xq = rope!(xq, bhle: true)
            xk = rope!(xk, bhle: true)
        }

        if nKVHeads < nHeads {
            let repeats = nHeads / nKVHeads
            xk = repeatKV(xk, repeats: repeats)
            xv = repeatKV(xv, repeats: repeats)
        }

        var scores = matmul(xq, xk.transposed(0, 1, 3, 2)) * scale // (B, H, T, KV_T)

        if let keyPaddingMask {
            let mask = keyPaddingMask.expandedDimensions(axis: 1).expandedDimensions(axis: 1)
            let negInf = MLXArray(-Float.infinity).asType(scores.dtype)
            scores = MLX.where(mask, scores, negInf)
        }

        let weights = softmax(scores, axis: -1)
        var output = matmul(weights, xv) // (B, H, T, D)
        output = output.transposed(0, 2, 1, 3).reshaped([batchSize, seqLen, -1])
        return wo(output)
    }
}

/// Feed-forward sublayer for DiT blocks.
public final class FeedForward: Module {
    let swiglu: Bool
    let nonLinearity: String

    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear?

    public init(
        dim: Int,
        hiddenDim: Int,
        ffnDimMultiplier: Float = 1,
        multipleOf: Int = 64,
        dropout _: Float = 0,
        nonLinearity: String = "swiglu",
        fcBias: Bool = false
    ) {
        self.swiglu = nonLinearity == "swiglu"
        self.nonLinearity = nonLinearity

        var adjustedHidden = hiddenDim
        if swiglu {
            adjustedHidden = Int(2 * adjustedHidden / 3)
        }
        adjustedHidden = Int(Float(adjustedHidden) * ffnDimMultiplier)
        adjustedHidden = multipleOf * ((adjustedHidden + multipleOf - 1) / multipleOf)

        self._w1.wrappedValue = Linear(dim, adjustedHidden, bias: fcBias)
        self._w2.wrappedValue = Linear(adjustedHidden, dim, bias: fcBias)
        self._w3.wrappedValue = swiglu ? Linear(dim, adjustedHidden, bias: fcBias) : nil
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden1 = w1(x)
        let hidden: MLXArray
        if swiglu {
            hidden = silu(hidden1) * (w3?(x) ?? MLXArray.zeros(hidden1.shape))
        } else {
            hidden = samActivation(hidden1, kind: nonLinearity)
        }
        return w2(hidden)
    }
}

/// Sinusoidal timestep embedder used by DiT.
public final class TimestepEmbedder: Module {
    let frequencyEmbeddingSize: Int
    let freqs: MLXArray

    @ModuleInfo(key: "projection") var projection: ProjectionLayer

    public init(
        dim: Int,
        frequencyEmbeddingDim: Int,
        nonLinearity: String,
        dropout: Float = 0,
        fcBias: Bool = false,
        maxPeriod: Int = 10000
    ) {
        self.frequencyEmbeddingSize = frequencyEmbeddingDim
        self._projection.wrappedValue = ProjectionLayer(
            inDim: frequencyEmbeddingDim,
            outDim: dim,
            nonLinearity: nonLinearity,
            dropout: dropout,
            fcBias: fcBias
        )

        let half = frequencyEmbeddingDim / 2
        self.freqs = exp(
            -Foundation.log(Float(maxPeriod))
                * MLXArray(Array(0..<half).map(Float.init)).asType(.float32)
                / Float(half)
        )
    }

    private func timestepEmbedding(_ t: MLXArray, dim: Int) -> MLXArray {
        let args = t.expandedDimensions(axis: 1).asType(.float32) * freqs.expandedDimensions(axis: 0)
        var embedding = concatenated([cos(args), sin(args)], axis: -1)
        if dim % 2 == 1 {
            let zeros = MLXArray.zeros([embedding.shape[0], 1]).asType(embedding.dtype)
            embedding = concatenated([embedding, zeros], axis: -1)
        }
        return embedding.asType(t.dtype)
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let x = timestepEmbedding(t, dim: frequencyEmbeddingSize)
        return projection(x)
    }
}

/// Projects memory/context features into transformer space.
public final class ContextEmbedder: Module {
    let contextNormEnabled: Bool

    @ModuleInfo(key: "norm") var norm: SAMRMSNorm?
    @ModuleInfo(key: "projection") var projection: ProjectionLayer

    public init(
        inDim: Int,
        outDim: Int,
        nonLinearity: String,
        dropout: Float = 0,
        fcBias: Bool = false,
        normEps: Float = 1e-5,
        contextNorm: Bool = false
    ) {
        self.contextNormEnabled = contextNorm
        self._norm.wrappedValue = contextNorm ? SAMRMSNorm(dim: inDim, eps: normEps) : nil
        self._projection.wrappedValue = ProjectionLayer(
            inDim: inDim,
            outDim: outDim,
            nonLinearity: nonLinearity,
            dropout: dropout,
            fcBias: fcBias
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = contextNormEnabled ? (norm?(x) ?? x) : x
        return projection(normed)
    }
}

/// Single DiT block with adaptive layer modulations.
public final class DiTBlock: Module {
    @ModuleInfo(key: "attention") var attention: SAMAttention
    @ModuleInfo(key: "feed_forward") var feedForward: FeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: SAMRMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: SAMRMSNorm
    @ModuleInfo(key: "cross_attention") var crossAttention: SAMAttention?
    @ModuleInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    public init(
        dim: Int,
        nHeads: Int,
        nKVHeads: Int? = nil,
        dropout: Float = 0,
        normEps: Float = 1e-5,
        qkNorm: Bool = false,
        fcBias: Bool = false,
        ffnExp: Int = 1,
        ffnDimMultiplier: Float = 4,
        multipleOf: Int = 64,
        nonLinearity: String = "silu",
        noCrossAttention: Bool = false
    ) {
        precondition(dim % nHeads == 0, "dim must be divisible by nHeads")
        let nKV = nKVHeads ?? nHeads
        let headDim = dim / nHeads

        self._attention.wrappedValue = SAMAttention(
            dim: dim,
            headDim: headDim,
            nHeads: nHeads,
            nKVHeads: nKV,
            normEps: normEps,
            useQKNorm: qkNorm,
            fcBias: fcBias
        )
        self._feedForward.wrappedValue = FeedForward(
            dim: dim,
            hiddenDim: ffnExp * dim,
            ffnDimMultiplier: ffnDimMultiplier,
            multipleOf: multipleOf,
            dropout: dropout,
            nonLinearity: nonLinearity,
            fcBias: fcBias
        )
        self._attentionNorm.wrappedValue = SAMRMSNorm(dim: dim, eps: normEps)
        self._ffnNorm.wrappedValue = SAMRMSNorm(dim: dim, eps: normEps)
        self._crossAttention.wrappedValue = noCrossAttention ? nil : SAMAttention(
            dim: dim,
            headDim: headDim,
            nHeads: nHeads,
            nKVHeads: nHeads,
            normEps: normEps,
            useQKNorm: qkNorm,
            fcBias: fcBias
        )
        self._scaleShiftTable.wrappedValue = MLXRandom.normal([6, dim]) / MLXArray(Float(sqrt(Float(dim))))
    }

    public func callAsFunction(
        _ x: MLXArray,
        crossX: MLXArray?,
        t: MLXArray,
        paddingMask: MLXArray?,
        memoryPaddingMask: MLXArray?,
        rope: RotaryEmbedding? = nil
    ) -> MLXArray {
        let biases = scaleShiftTable.expandedDimensions(axis: 0) + t.reshaped([x.shape[0], 6, -1])
        let parts = biases.split(parts: 6, axis: 1)
        let shiftMSA = parts[0]
        let scaleMSA = parts[1]
        let gateMSA = parts[2]
        let shiftMLP = parts[3]
        let scaleMLP = parts[4]
        let gateMLP = parts[5]

        let hNormed = attentionNorm(x)
        let hModulated = hNormed * (1 + scaleMSA) + shiftMSA
        let hAttn = attention(hModulated, keyPaddingMask: paddingMask, rope: rope)
        var h = x + hAttn * gateMSA

        if let crossAttention, let crossX {
            let hCross = crossAttention(h, crossX: crossX, keyPaddingMask: memoryPaddingMask)
            h = h + hCross
        }

        let ffNormed = ffnNorm(h)
        let ffModulated = ffNormed * (1 + scaleMLP) + shiftMLP
        let hFF = feedForward(ffModulated)
        return h + hFF * gateMLP
    }
}

/// Diffusion Transformer (DiT) backbone for SAMAudio.
public final class DiT: Module {
    @ModuleInfo(key: "data_proj") var dataProj: Linear?
    @ModuleInfo(key: "layers") var layers: [DiTBlock]
    @ModuleInfo(key: "norm") var norm: SAMRMSNorm
    @ModuleInfo(key: "output") var output: Linear
    @ModuleInfo(key: "x_embedder") var xEmbedder: Patcher
    @ModuleInfo(key: "y_embedder") var yEmbedder: ContextEmbedder
    @ModuleInfo(key: "t_embedder") var tEmbedder: TimestepEmbedder
    @ModuleInfo(key: "t_block") var tBlock: Linear
    @ModuleInfo(key: "final_layer_scale_shift_table") var finalLayerScaleShiftTable: MLXArray

    let tBlockNonLinearity: String
    let ropeEmbeddings: RotaryEmbedding?

    public init(config: TransformerConfig) {
        self._dataProj.wrappedValue = config.inChannels == nil ? nil : Linear(config.inChannels!, config.dim)
        self.ropeEmbeddings = config.useRope
            ? RotaryEmbedding(
                theta: Float(max(10000, 2 * config.maxPositions)),
                headDim: config.dim / config.nHeads,
                maxSequenceLength: config.maxPositions
            )
            : nil

        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            DiTBlock(
                dim: config.dim,
                nHeads: config.nHeads,
                dropout: config.dropout,
                normEps: config.normEps,
                qkNorm: config.qkNorm,
                fcBias: config.fcBias,
                ffnExp: config.ffnExp,
                ffnDimMultiplier: Float(config.ffnDimMultiplier),
                multipleOf: config.multipleOf,
                nonLinearity: config.nonLinearity
            )
        }

        self._norm.wrappedValue = SAMRMSNorm(dim: config.dim, eps: config.normEps)
        self._output.wrappedValue = Linear(config.dim, config.outChannels, bias: config.fcBias)
        self._xEmbedder.wrappedValue = Patcher(
            inChannels: config.dim,
            outChannels: config.dim,
            patchSize: 1
        )
        self._yEmbedder.wrappedValue = ContextEmbedder(
            inDim: config.contextDim,
            outDim: config.dim,
            nonLinearity: config.contextNonLinearity,
            dropout: config.contextEmbedderDropout,
            fcBias: config.fcBias,
            normEps: config.normEps,
            contextNorm: config.contextNorm
        )
        self._tEmbedder.wrappedValue = TimestepEmbedder(
            dim: config.dim,
            frequencyEmbeddingDim: config.frequencyEmbeddingDim,
            nonLinearity: config.timestepNonLinearity,
            dropout: config.dropout,
            fcBias: config.fcBias
        )
        self.tBlockNonLinearity = config.tBlockNonLinearity
        self._tBlock.wrappedValue = Linear(config.dim, config.dim * 6, bias: config.tBlockBias)
        self._finalLayerScaleShiftTable.wrappedValue = MLXRandom.normal([2, config.dim]) / MLXArray(Float(sqrt(Float(config.dim))))
    }

    public func callAsFunction(
        _ x: MLXArray,
        time: MLXArray,
        paddingMask: MLXArray? = nil,
        memory: MLXArray? = nil,
        memoryPaddingMask: MLXArray? = nil
    ) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        h = xEmbedder(h)
        h = h.transposed(0, 2, 1)
        let originalN = h.shape[1]

        let t = tEmbedder(time)
        var t0 = samActivation(t, kind: tBlockNonLinearity)
        t0 = tBlock(t0)

        let y = memory == nil ? nil : yEmbedder(memory!)

        for layer in layers {
            h = layer(
                h,
                crossX: y,
                t: t0,
                paddingMask: paddingMask,
                memoryPaddingMask: memoryPaddingMask,
                rope: ropeEmbeddings
            )
        }

        let finalBiases = finalLayerScaleShiftTable.expandedDimensions(axis: 0) + t.expandedDimensions(axis: 1)
        let finalParts = finalBiases.split(parts: 2, axis: 1)
        let shift = finalParts[0]
        let scale = finalParts[1]

        h = norm(h)
        h = h * (1 + scale) + shift
        var out = output(h)

        if originalN != out.shape[1] {
            out = out[0..., (out.shape[1] - originalN)..<out.shape[1], 0...]
        }

        return out
    }
}
