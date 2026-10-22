import Foundation
import MLX
import MLXNN

// MARK: - ModernBERT encoder (Swift port of mlx_audio/tts/models/irodori_tts/modernbert.py)
//
// Irodori-TTS v4 conditions on a pretrained encoder (sbintuitions/modernbert-ja-310m)
// whose weights ship inside the checkpoint under `pretrained_text_backbone.backbone.*`,
// so only the bidirectional encoder stack is needed. Property names are the camelCase
// form of the HuggingFace module tree, which is what `IrodoriTTSModel.sanitize` emits.

/// LayerNorm with optional bias that accumulates in float32 and casts back,
/// matching the Python port bit-for-bit rather than the fused kernel's rounding.
final class IrodoriModernBertLayerNorm: Module, UnaryLayer {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    init(dimensions: Int, eps: Float, bias: Bool) {
        self.eps = eps
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([dimensions]))
        self._bias = ParameterInfo(wrappedValue: bias ? MLXArray.zeros([dimensions]) : nil)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xDtype = x.dtype
        let xf = x.asType(.float32)
        let mean = xf.mean(axis: -1, keepDims: true)
        let variance = xf.variance(axis: -1, keepDims: true)
        var y = (xf - mean) * MLX.rsqrt(variance + eps)
        y = y * weight.asType(.float32)
        if let bias { y = y + bias.asType(.float32) }
        return y.asType(xDtype)
    }
}

/// HuggingFace-style (rotate_half) RoPE tables of shape (seqLen, headDim).
func irodoriModernBertRopeCosSin(headDim: Int, seqLen: Int, theta: Float) -> (cos: MLXArray, sin: MLXArray) {
    let exponents = MLX.arange(0, headDim, step: 2, dtype: .float32) / Float(headDim)
    let invFreq = 1 / MLX.pow(MLXArray(theta), exponents)
    let positions = MLX.arange(seqLen, dtype: .float32)
    let freqs = MLX.outer(positions, invFreq)
    let emb = MLX.concatenated([freqs, freqs], axis: -1)
    return (MLX.cos(emb), MLX.sin(emb))
}

/// x: (B, H, S, D); cos/sin: (S, D).
private func irodoriModernBertApplyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let half = x.shape[3] / 2
    let rotated = MLX.concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    return x * cos.asType(x.dtype) + rotated * sin.asType(x.dtype)
}

/// Additive attention mask in the compute dtype. Uses the dtype's finite minimum
/// (as transformers does with `finfo.min`) rather than -inf.
private func irodoriModernBertAdditiveMask(_ boolMask: MLXArray, dtype: DType) -> MLXArray {
    let floor: Float
    switch dtype {
    case .float16: floor = -65_504
    case .bfloat16: floor = -3.3895314e38
    default: floor = -Float.greatestFiniteMagnitude
    }
    return MLX.where(
        boolMask,
        MLXArray.zeros(boolMask.shape, dtype: dtype),
        MLXArray.full(boolMask.shape, values: MLXArray(floor), dtype: dtype)
    )
}

final class IrodoriModernBertAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(config: IrodoriModernBertConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.scale = 1 / sqrt(Float(config.headDim))
        self._wqkv = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, 3 * config.hiddenSize, bias: config.attentionBias),
            key: "Wqkv")
        self._wo = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.hiddenSize, bias: config.attentionBias),
            key: "Wo")
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        let bsz = x.shape[0]
        let seqLen = x.shape[1]
        // (B, S, 3, H, D) -> (3, B, H, S, D)
        let qkv = wqkv(x).reshaped([bsz, seqLen, 3, numHeads, headDim]).transposed(2, 0, 3, 1, 4)
        let q = irodoriModernBertApplyRope(qkv[0], cos: cos, sin: sin)
        let k = irodoriModernBertApplyRope(qkv[1], cos: cos, sin: sin)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: qkv[2], scale: scale, mask: mask)
        return wo(out.transposed(0, 2, 1, 3).reshaped([bsz, seqLen, numHeads * headDim]))
    }
}

/// GeGLU MLP with a single fused input projection (Wi -> 2 * intermediate).
final class IrodoriModernBertMLP: Module {
    let exactGelu: Bool

    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(config: IrodoriModernBertConfig) throws {
        let activation = config.hiddenActivation.lowercased()
        guard ["gelu", "gelu_new", "gelu_pytorch_tanh"].contains(activation) else {
            throw IrodoriTTSError.generation(
                "Unsupported ModernBERT hidden_activation=\(config.hiddenActivation)")
        }
        self.exactGelu = activation == "gelu"
        self._wi = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, 2 * config.intermediateSize, bias: config.mlpBias),
            key: "Wi")
        self._wo = ModuleInfo(
            wrappedValue: Linear(config.intermediateSize, config.hiddenSize, bias: config.mlpBias),
            key: "Wo")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = wi(x)
        let half = gated.shape[gated.ndim - 1] / 2
        let input = gated[.ellipsis, ..<half]
        let act = exactGelu ? gelu(input) : geluApproximate(input)
        return wo(act * gated[.ellipsis, half...])
    }
}

final class IrodoriModernBertEncoderLayer: Module {
    /// nil for layer 0, which consumes the already-normalised embedding output.
    @ModuleInfo(key: "attn_norm") var attnNorm: IrodoriModernBertLayerNorm?
    @ModuleInfo(key: "attn") var attn: IrodoriModernBertAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: IrodoriModernBertLayerNorm
    @ModuleInfo(key: "mlp") var mlp: IrodoriModernBertMLP

    init(config: IrodoriModernBertConfig, layerIdx: Int) throws {
        self._attnNorm = ModuleInfo(wrappedValue: layerIdx == 0 ? nil : IrodoriModernBertLayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias))
        self._attn = ModuleInfo(wrappedValue: IrodoriModernBertAttention(config: config))
        self._mlpNorm = ModuleInfo(wrappedValue: IrodoriModernBertLayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias))
        self._mlp = ModuleInfo(wrappedValue: try IrodoriModernBertMLP(config: config))
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        let h = x + attn(attnNorm.map { $0(x) } ?? x, cos: cos, sin: sin, mask: mask)
        return h + mlp(mlpNorm(h))
    }
}

final class IrodoriModernBertEmbeddings: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: IrodoriModernBertLayerNorm

    init(config: IrodoriModernBertConfig) {
        self._tokEmbeddings = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize))
        self._norm = ModuleInfo(wrappedValue: IrodoriModernBertLayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias))
    }

    func callAsFunction(_ inputIDs: MLXArray) -> MLXArray {
        norm(tokEmbeddings(inputIDs))
    }
}

/// Bidirectional ModernBERT encoder returning last_hidden_state.
final class IrodoriModernBertEncoder: Module {
    let config: IrodoriModernBertConfig

    @ModuleInfo(key: "embeddings") var embeddings: IrodoriModernBertEmbeddings
    @ModuleInfo(key: "layers") var layers: [IrodoriModernBertEncoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: IrodoriModernBertLayerNorm

    init(config: IrodoriModernBertConfig) throws {
        self.config = config
        self._embeddings = ModuleInfo(wrappedValue: IrodoriModernBertEmbeddings(config: config))
        self._layers = ModuleInfo(wrappedValue: try (0 ..< config.numHiddenLayers).map {
            try IrodoriModernBertEncoderLayer(config: config, layerIdx: $0)
        })
        self._finalNorm = ModuleInfo(wrappedValue: IrodoriModernBertLayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, bias: config.normBias))
    }

    /// Returns (global, local) additive (B|1, 1, S, S) masks; global is nil without a key mask.
    private func attentionMasks(_ mask: MLXArray?, seqLen: Int, dtype: DType) -> (MLXArray?, MLXArray) {
        let window = config.localAttention / 2
        let positions = MLX.arange(seqLen)
        let distance = MLX.abs(positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0))
        let withinWindow = (distance .<= window).expandedDimensions(axes: [0, 1])
        guard let mask else {
            return (nil, irodoriModernBertAdditiveMask(withinWindow, dtype: dtype))
        }
        let keyMask = mask.asType(.bool).expandedDimensions(axes: [1, 2])
        let globalBool = MLX.broadcast(keyMask, to: [mask.shape[0], 1, seqLen, seqLen])
        // A padding query far from every real token would otherwise have an all-masked
        // row, whose softmax is NaN and would poison later layers. Leaving the diagonal
        // open keeps those rows finite without changing any non-padding output.
        let localBool = (globalBool .&& withinWindow) .|| (distance .== 0).expandedDimensions(axes: [0, 1])
        return (
            irodoriModernBertAdditiveMask(globalBool, dtype: dtype),
            irodoriModernBertAdditiveMask(localBool, dtype: dtype)
        )
    }

    func callAsFunction(_ inputIDs: MLXArray, mask: MLXArray?) -> MLXArray {
        var x = embeddings(inputIDs)
        let seqLen = inputIDs.shape[1]
        let global = irodoriModernBertRopeCosSin(
            headDim: config.headDim, seqLen: seqLen, theta: config.globalRopeTheta)
        let local = irodoriModernBertRopeCosSin(
            headDim: config.headDim, seqLen: seqLen, theta: config.localRopeTheta)
        let (globalMask, localMask) = attentionMasks(mask, seqLen: seqLen, dtype: x.dtype)

        for (i, layer) in layers.enumerated() {
            if config.isGlobalLayer(i) {
                x = layer(x, cos: global.cos, sin: global.sin, mask: globalMask)
            } else {
                x = layer(x, cos: local.cos, sin: local.sin, mask: localMask)
            }
        }
        return finalNorm(x)
    }
}
