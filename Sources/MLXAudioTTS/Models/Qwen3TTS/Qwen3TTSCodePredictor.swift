// Port of code predictor classes from mlx_audio/tts/models/qwen3_tts/talker.py
// Multi-codebook token prediction sub-model

@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import Foundation

// MARK: - Code Predictor Attention

final class CodePredictorAttention: Module {
    let numHeads: Int
    let numKvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKvHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Foundation.sqrt(Float(headDim))

        self._qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, numKvHeads * headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: config.attentionBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray),
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qProj(x).reshaped(batch, seqLen, numHeads, headDim)
        var k = kProj(x).reshaped(batch, seqLen, numKvHeads, headDim)
        var v = vProj(x).reshaped(batch, seqLen, numKvHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let (cosVal, sinVal) = positionEmbeddings
        let cosE = expandedDimensions(cosVal, axis: 1)
        let sinE = expandedDimensions(sinVal, axis: 1)
        q = q * cosE + cpRotateHalf(q) * sinE
        k = k * cosE + cpRotateHalf(k) * sinE

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1))
    }
}

func cpRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

// MARK: - Code Predictor MLP

final class CodePredictorMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3TTSTalkerCodePredictorConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Code Predictor Decoder Layer

final class CodePredictorDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: CodePredictorAttention
    @ModuleInfo var mlp: CodePredictorMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(config: Qwen3TTSTalkerCodePredictorConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = CodePredictorAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = CodePredictorMLP(config: config)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (MLXArray, MLXArray),
        mask: MLXArray? = nil,
        cache: (any KVCache)? = nil
    ) -> MLXArray {
        var out = x + selfAttn(inputLayernorm(x), positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
        out = out + mlp(postAttentionLayernorm(out))
        return out
    }
}

// MARK: - Code Predictor Model (inner)

final class CodePredictorModel: Module {
    let config: Qwen3TTSTalkerCodePredictorConfig
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]
    let layers: [CodePredictorDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    let rotaryEmb: Qwen3TTSRotaryEmbedding

    init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self._codecEmbedding.wrappedValue = (0 ..< config.numCodeGroups - 1).map { _ in
            Embedding(embeddingCount: config.vocabSize, dimensions: talkerHiddenSize)
        }
        self.layers = (0 ..< config.numHiddenLayers).map { CodePredictorDecoderLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmb = Qwen3TTSRotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [any KVCache]? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (inputsEmbeds.dim(0), inputsEmbeds.dim(1), inputsEmbeds.dim(2))

        let offset: Int
        if let firstCache = cache?.first {
            offset = firstCache.offset
        } else {
            offset = 0
        }

        let posIds: MLXArray
        if let positionIds {
            posIds = positionIds
        } else {
            let pos = MLXArray(Int32(offset) ..< Int32(offset + seqLen)).reshaped(1, seqLen)
            posIds = broadcast(pos, to: [batch, seqLen])
        }

        let posEmbeddings = rotaryEmb(inputsEmbeds, positionIds: posIds)

        var causalMask = mask
        if causalMask == nil && seqLen > 1 {
            causalMask = MultiHeadAttention.createAdditiveCausalMask(seqLen).asType(inputsEmbeds.dtype)
        }

        var x = inputsEmbeds
        for (i, layer) in layers.enumerated() {
            x = layer(x, positionEmbeddings: posEmbeddings, mask: causalMask, cache: cache?[i])
        }
        return norm(x)
    }

    func makeCache() -> [any KVCache] {
        layers.map { _ in KVCacheSimple() }
    }
}

// MARK: - Code Predictor (public)

final class Qwen3TTSCodePredictor: Module {
    let config: Qwen3TTSTalkerCodePredictorConfig
    let numCodeGroups: Int
    let talkerHiddenSize: Int

    @ModuleInfo(key: "small_to_mtp_projection") var projection: Linear?
    @ModuleInfo var model: CodePredictorModel
    @ModuleInfo(key: "lm_head") var lmHead: [Linear]

    var codecEmbedding: [Embedding] { model.codecEmbedding }

    init(config: Qwen3TTSTalkerCodePredictorConfig, talkerHiddenSize: Int) {
        self.config = config
        self.numCodeGroups = config.numCodeGroups
        self.talkerHiddenSize = talkerHiddenSize

        if config.hiddenSize != talkerHiddenSize {
            self._projection.wrappedValue = Linear(talkerHiddenSize, config.hiddenSize, bias: true)
        } else {
            self._projection.wrappedValue = nil
        }

        self._model.wrappedValue = CodePredictorModel(config: config, talkerHiddenSize: talkerHiddenSize)
        self._lmHead.wrappedValue = (0 ..< config.numCodeGroups - 1).map { _ in
            Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [any KVCache]? = nil,
        generationStep: Int = 0
    ) -> (MLXArray, [any KVCache]?, Int) {
        var embeds = inputsEmbeds
        if let proj = projection {
            embeds = proj(embeds)
        }

        let x = model(embeds, positionIds: positionIds, mask: mask, cache: cache)
        let logits = lmHead[generationStep](x)
        return (logits, cache, generationStep + 1)
    }

    func makeCache() -> [any KVCache] {
        model.makeCache()
    }
}
