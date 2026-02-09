//
//  MimiTransformer.swift
//  MLXAudio
//
//  Mimi transformer for the encoder pipeline.
//  Ported from mlx_audio/codec/models/mimi/modules/transformer.py
//
//  Architecture: Stack of TransformerLayers with RoPE, LayerNorm, and GeLU MLP.
//  Used as ProjectedTransformer with NCL<->NLC conversion (conv_layout=true).
//

import Foundation
import MLX
import MLXNN

// MARK: - MimiLayerScale

/// Learnable per-channel scale for residual connections.
public class MimiLayerScale: Module {
    @ModuleInfo(key: "scale") var scale: MLXArray

    public init(dim: Int) {
        self._scale.wrappedValue = MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x * scale
    }
}

// MARK: - MimiKVCache

/// Simple KV cache for the Mimi encoder transformer.
public class MimiKVCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset: Int = 0

    public init() {}

    public func reset() {
        keys = nil
        values = nil
        offset = 0
    }

    /// Update cache with new keys/values and return full K, V.
    public func updateAndFetch(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        if let existingK = keys, let existingV = values {
            let k = concatenated([existingK, newK], axis: 2)
            let v = concatenated([existingV, newV], axis: 2)
            self.keys = k
            self.values = v
            offset = k.shape[2]
            return (k, v)
        } else {
            self.keys = newK
            self.values = newV
            offset = newK.shape[2]
            return (newK, newV)
        }
    }
}

// MARK: - MimiAttention

/// Multi-head attention with RoPE and sliding window support.
///
/// Uses a combined in_proj (q+k+v concatenated) and separate out_proj.
public class MimiAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let context: Int  // sliding window size (0 = unlimited)

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let rope: MLXNN.RoPE?

    public init(
        dModel: Int,
        numHeads: Int,
        biasAttn: Bool = false,
        context: Int = 0,
        ropeTheta: Float = 10000.0,
        useRope: Bool = true
    ) {
        self.numHeads = numHeads
        self.headDim = dModel / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self.context = context

        // Combined Q+K+V projection (kv_repeat=1 so all same size)
        let outDim = dModel * 3
        self._inProj.wrappedValue = Linear(dModel, outDim, bias: biasAttn)
        self._outProj.wrappedValue = Linear(dModel, dModel, bias: biasAttn)

        if useRope {
            self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: ropeTheta)
        } else {
            self.rope = nil
        }
    }

    public func callAsFunction(
        _ x: MLXArray,
        cache: MimiKVCache,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let (b, t, hd) = (x.shape[0], x.shape[1], x.shape[2])
        let offset = cache.offset

        // Project to Q, K, V
        let qkv = inProj(x).reshaped([b, t, 3, numHeads, headDim])
        var q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)  // [b, heads, t, head_dim]
        var k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)

        // Apply RoPE
        if let rope = rope {
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)
        }

        // Update KV cache
        let (fullK, fullV) = cache.updateAndFetch(keys: k, values: v)

        // Use provided mask or build one from scratch
        let attnMask: MLXArray
        if let m = mask {
            attnMask = m
        } else {
            // Build causal + sliding window mask
            let kLen = fullK.shape[2]
            let posK = MLXArray(Array(0..<kLen).map { Int32($0) }) + MLXArray(Int32(offset - kLen))
            let posQ = MLXArray(Array(0..<t).map { Int32($0) }) + MLXArray(Int32(offset))
            // delta[i,j] = posQ[i] - posK[j]
            let delta = expandedDimensions(posQ, axis: 1) - expandedDimensions(posK, axis: 0)
            let posKCond = expandedDimensions(posK, axis: 0) .>= MLXArray(Int32(0))
            let deltaCond = delta .>= MLXArray(Int32(0))
            var allowed = posKCond * deltaCond  // element-wise AND via multiplication
            if context > 0 {
                let windowCond = delta .< MLXArray(Int32(context))
                allowed = allowed * windowCond
            }
            attnMask = MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9)))
                .reshaped([1, 1, t, kLen]).asType(x.dtype)
        }

        // Scaled dot-product attention
        var xs = MLXFast.scaledDotProductAttention(
            queries: q, keys: fullK, values: fullV,
            scale: scale, mask: attnMask
        )
        xs = xs.transposed(0, 2, 1, 3).reshaped([b, t, hd])
        return outProj(xs)
    }
}

// MARK: - MimiMlpNoGating

/// Feed-forward MLP with GeLU activation (no gating).
public class MimiMlpNoGating: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    public init(dModel: Int, dimFeedforward: Int, biasFF: Bool = false) {
        self._linear1.wrappedValue = Linear(dModel, dimFeedforward, bias: biasFF)
        self._linear2.wrappedValue = Linear(dimFeedforward, dModel, bias: biasFF)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear2(geluApproximate(linear1(x)))
    }
}

// MARK: - MimiTransformerLayer

/// Single transformer layer: LayerNorm + Attention + LayerScale + MLP.
public class MimiTransformerLayer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: MimiAttention
    @ModuleInfo(key: "gating") var gating: MimiMlpNoGating
    @ModuleInfo(key: "layer_scale_1") var layerScale1: MimiLayerScale
    @ModuleInfo(key: "layer_scale_2") var layerScale2: MimiLayerScale

    public init(
        dModel: Int,
        numHeads: Int,
        dimFeedforward: Int,
        biasAttn: Bool = false,
        biasFF: Bool = false,
        context: Int = 0,
        ropeTheta: Float = 10000.0,
        useRope: Bool = true,
        layerScale: Float? = 0.01
    ) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)
        self._norm2.wrappedValue = LayerNorm(dimensions: dModel, eps: 1e-5)

        self._selfAttn.wrappedValue = MimiAttention(
            dModel: dModel,
            numHeads: numHeads,
            biasAttn: biasAttn,
            context: context,
            ropeTheta: ropeTheta,
            useRope: useRope
        )

        self._gating.wrappedValue = MimiMlpNoGating(
            dModel: dModel,
            dimFeedforward: dimFeedforward,
            biasFF: biasFF
        )

        self._layerScale1.wrappedValue = MimiLayerScale(dim: dModel)
        self._layerScale2.wrappedValue = MimiLayerScale(dim: dModel)
    }

    public func callAsFunction(
        _ x: MLXArray,
        cache: MimiKVCache,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let n1 = selfAttn(norm1(x), cache: cache, mask: mask)
        var xs = x + layerScale1(n1)
        xs = xs + layerScale2(gating(norm2(xs)))
        return xs
    }
}

// MARK: - MimiTransformer

/// Stack of transformer layers with KV cache management.
public class MimiTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [MimiTransformerLayer]

    public init(
        dModel: Int,
        numHeads: Int,
        numLayers: Int,
        dimFeedforward: Int,
        biasAttn: Bool = false,
        biasFF: Bool = false,
        context: Int = 0,
        ropeTheta: Float = 10000.0,
        useRope: Bool = true,
        layerScale: Float? = 0.01
    ) {
        var transformerLayers: [MimiTransformerLayer] = []
        for _ in 0..<numLayers {
            transformerLayers.append(MimiTransformerLayer(
                dModel: dModel,
                numHeads: numHeads,
                dimFeedforward: dimFeedforward,
                biasAttn: biasAttn,
                biasFF: biasFF,
                context: context,
                ropeTheta: ropeTheta,
                useRope: useRope,
                layerScale: layerScale
            ))
        }
        self._layers.wrappedValue = transformerLayers
    }

    public func callAsFunction(
        _ x: MLXArray,
        cache: [MimiKVCache],
        mask: MLXArray? = nil
    ) -> MLXArray {
        var xs = x
        for (layer, c) in zip(layers, cache) {
            xs = layer(xs, cache: c, mask: mask)
        }
        return xs
    }

    public func makeCache() -> [MimiKVCache] {
        return layers.map { _ in MimiKVCache() }
    }
}

// MARK: - MimiProjectedTransformer

/// Transformer with input/output projections and NCL<->NLC conversion.
///
/// When conv_layout=true (default for encoder), swaps axes before/after processing.
public class MimiProjectedTransformer: Module {
    let convLayout: Bool

    @ModuleInfo(key: "input_proj") var inputProj: Linear?
    @ModuleInfo(key: "transformer") var transformer: MimiTransformer
    // Single output projection (output_projs[0] in Python)
    @ModuleInfo(key: "output_projs") var outputProjs: [Linear?]

    public init(config: Qwen3TTSTokenizerEncoderConfig) {
        self.convLayout = true  // Always true for encoder

        self._transformer.wrappedValue = MimiTransformer(
            dModel: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numLayers: config.numHiddenLayers,
            dimFeedforward: config.intermediateSize,
            biasAttn: config.attentionBias,
            biasFF: false,
            context: config.slidingWindow,
            ropeTheta: config.ropeTheta,
            useRope: true,
            layerScale: config.layerScaleInitialScale
        )

        // Input projection: hiddenSize -> hiddenSize (identity, so nil)
        self._inputProj.wrappedValue = nil

        // Output projection: hiddenSize -> hiddenSize (identity, so nil)
        self._outputProjs.wrappedValue = [nil]
    }

    public func callAsFunction(
        _ x: MLXArray,
        cache: [MimiKVCache],
        mask: MLXArray? = nil
    ) -> [MLXArray] {
        var xs = x
        if convLayout {
            xs = xs.transposed(0, 2, 1)  // NCL -> NLC
        }
        if let proj = inputProj {
            xs = proj(xs)
        }
        xs = transformer(xs, cache: cache, mask: mask)

        var outs: [MLXArray] = []
        for proj in outputProjs {
            var out = xs
            if let p = proj {
                out = p(out)
            }
            if convLayout {
                out = out.transposed(0, 2, 1)  // NLC -> NCL
            }
            outs.append(out)
        }
        return outs
    }

    public func makeCache() -> [MimiKVCache] {
        return transformer.makeCache()
    }
}

// MARK: - GELU Approximate

/// Approximate GELU activation (matches Python nn.gelu_approx).
func geluApproximate(_ x: MLXArray) -> MLXArray {
    return x * (1 + MLX.tanh(sqrt(MLXArray(Float(2.0 / Float.pi))) * (x + MLXArray(Float(0.044715)) * x * x * x))) / 2
}
