import Foundation
@preconcurrency import MLX
@preconcurrency import MLXFast
@preconcurrency import MLXLMCommon
import MLXNN

/// Gemma 3 RMSNorm: fp32 rsqrt, then multiply by `(1 + weight)`.
func dramaBoxGemmaRMSNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
    let origDtype = x.dtype
    let x32 = x.asType(.float32)
    let variance = MLX.mean(x32 * x32, axis: -1, keepDims: true)
    let normed = x32 * MLX.rsqrt(variance + eps)
    let out = normed * (1.0 + weight.asType(.float32))
    return out.asType(origDtype)
}

final class DramaBoxGemmaRMSNorm: Module {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float = 1e-6) {
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        dramaBoxGemmaRMSNorm(x, weight: weight, eps: eps)
    }
}

func dramaBoxGemmaRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let first = x[.ellipsis, 0..<half]
    let second = x[.ellipsis, half...]
    return concatenated([-second, first], axis: -1)
}

func dramaBoxGemmaRopeCosSin(
    seqLen: Int,
    headDim: Int,
    base: Float,
    scalingFactor: Float = 1.0
) -> (MLXArray, MLXArray) {
    let idx = MLXArray(Array(stride(from: 0, to: headDim, by: 2)).map { Float($0) })
    var inv = 1.0 / MLX.pow(MLXArray(base), idx / Float(headDim))
    if scalingFactor != 1.0 {
        inv = inv / scalingFactor
    }
    let pos = MLXArray(Array(0..<seqLen).map { Float($0) })
    let freqs = pos.reshaped([seqLen, 1]) * inv.reshaped([1, headDim / 2])
    let emb = concatenated([freqs, freqs], axis: -1)
    return (MLX.cos(emb), MLX.sin(emb))
}

func dramaBoxGemmaApplyRope(
    _ q: MLXArray,
    _ k: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {
    let cosB = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    let sinB = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    let q32 = q.asType(.float32)
    let k32 = k.asType(.float32)
    let qRot = q32 * cosB + dramaBoxGemmaRotateHalf(q32) * sinB
    let kRot = k32 * cosB + dramaBoxGemmaRotateHalf(k32) * sinB
    return (qRot.asType(q.dtype), kRot.asType(k.dtype))
}

final class DramaBoxGemma3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let useTanhGELU: Bool

    init(_ config: DramaBoxGemmaTextConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self.useTanhGELU = config.hiddenActivation != "gelu"
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gated = useTanhGELU ? MLXNN.geluApproximate(gateProj(x)) : MLXNN.gelu(gateProj(x))
        return downProj(gated * upProj(x))
    }
}

final class DramaBoxGemma3Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: DramaBoxGemmaRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: DramaBoxGemmaRMSNorm

    init(_ config: DramaBoxGemmaTextConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = config.attentionScale
        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = DramaBoxGemmaRMSNorm(headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = DramaBoxGemmaRMSNorm(headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        q = qNorm(q)
        k = kNorm(k)
        (q, k) = dramaBoxGemmaApplyRope(q, k, cos: cos, sin: sin)
        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        out = out.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(out)
    }
}

final class DramaBoxGemma3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DramaBoxGemma3Attention
    @ModuleInfo var mlp: DramaBoxGemma3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: DramaBoxGemmaRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: DramaBoxGemmaRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: DramaBoxGemmaRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: DramaBoxGemmaRMSNorm

    init(_ config: DramaBoxGemmaTextConfig) {
        self._selfAttn.wrappedValue = DramaBoxGemma3Attention(config)
        self._mlp.wrappedValue = DramaBoxGemma3MLP(config)
        self._inputLayernorm.wrappedValue = DramaBoxGemmaRMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = DramaBoxGemmaRMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = DramaBoxGemmaRMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = DramaBoxGemmaRMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?) -> MLXArray {
        var hidden = x
        var residual = hidden
        hidden = selfAttn(inputLayernorm(hidden), cos: cos, sin: sin, mask: mask)
        hidden = residual + postAttentionLayernorm(hidden)

        residual = hidden
        hidden = mlp(preFeedforwardLayernorm(hidden))
        hidden = residual + postFeedforwardLayernorm(hidden)
        return hidden
    }
}

struct DramaBoxGemma3Output {
    var lastHiddenState: MLXArray
    /// Embedding output + each decoder layer: `numHiddenLayers + 1` tensors.
    var hiddenStates: [MLXArray]
}

/// Headless Gemma 3 text backbone. Returns the full hidden-state stack
/// (49 tensors for Gemma 3 12B). There is no LM head.
final class DramaBoxGemma3TextBackbone: Module {
    let config: DramaBoxGemmaTextConfig
    let embedScale: Float
    let layerTypes: [String]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [DramaBoxGemma3DecoderLayer]
    @ModuleInfo var norm: DramaBoxGemmaRMSNorm

    init(_ config: DramaBoxGemmaTextConfig) {
        self.config = config
        self.embedScale = Float(sqrt(Double(config.hiddenSize)))
        self.layerTypes = config.layerTypes()
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in DramaBoxGemma3DecoderLayer(config) }
        self._norm.wrappedValue = DramaBoxGemmaRMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func buildCausalMask(attentionMask: MLXArray?, seqLen: Int, dtype: DType) -> MLXArray {
        let qIdx = MLXArray(Array(0..<seqLen).map { Int32($0) }).reshaped([seqLen, 1])
        let kIdx = MLXArray(Array(0..<seqLen).map { Int32($0) }).reshaped([1, seqLen])
        let causal = qIdx .>= kIdx

        let full: MLXArray
        if let attentionMask {
            let keep = attentionMask.asType(.bool).expandedDimensions(axis: 1)
            full = causal.expandedDimensions(axis: 0) .&& keep
        } else {
            full = broadcast(causal.expandedDimensions(axis: 0), to: [1, seqLen, seqLen])
        }

        let largeNeg = MLXArray(dramaBoxFinfoMin(dtype), dtype: dtype)
        let zero = MLXArray(0 as Float, dtype: dtype)
        return which(full, zero, largeNeg).expandedDimensions(axis: 1)
    }

    func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> DramaBoxGemma3Output {
        let L = inputIds.dim(1)
        var x = embedTokens(inputIds)
        x = x * MLXArray(embedScale, dtype: x.dtype)
        var hiddenStates: [MLXArray] = [x]

        let (cosFull, sinFull) = dramaBoxGemmaRopeCosSin(
            seqLen: L,
            headDim: config.headDim,
            base: config.ropeTheta,
            scalingFactor: config.ropeScalingFactor
        )
        let (cosSliding, sinSliding) = dramaBoxGemmaRopeCosSin(
            seqLen: L,
            headDim: config.headDim,
            base: config.ropeLocalBaseFreq,
            scalingFactor: 1.0
        )
        let mask = buildCausalMask(attentionMask: attentionMask, seqLen: L, dtype: x.dtype)

        for (layer, layerType) in zip(layers, layerTypes) {
            let (cos, sin) = layerType == "full_attention" ? (cosFull, sinFull) : (cosSliding, sinSliding)
            x = layer(x, cos: cos, sin: sin, mask: mask)
            hiddenStates.append(x)
        }

        return DramaBoxGemma3Output(lastHiddenState: norm(x), hiddenStates: hiddenStates)
    }

    static func load(from directory: URL) throws -> (DramaBoxGemma3TextBackbone, DramaBoxGemmaTextConfig) {
        let config = try DramaBoxGemmaTextConfig.load(from: directory)
        let model = DramaBoxGemma3TextBackbone(config)
        let weights = try DramaBoxWeights.loadGemmaShards(from: directory)

        if let quantization = config.quantization {
            quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits) { path, _ in
                weights["\(path).scales"] != nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        return (model, config)
    }
}
