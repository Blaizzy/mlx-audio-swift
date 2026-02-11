// Port of mlx_audio/tts/models/qwen3_tts/talker.py
// Talker transformer with multimodal 3D RoPE for Qwen3-TTS

@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import Foundation

// MARK: - RoPE helpers

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

private func applyRotaryPosEmb(
    _ q: MLXArray, _ k: MLXArray, cos cosVal: MLXArray, sin sinVal: MLXArray
) -> (MLXArray, MLXArray) {
    let cosE = expandedDimensions(cosVal, axis: 1)
    let sinE = expandedDimensions(sinVal, axis: 1)
    let qEmbed = q * cosE + rotateHalf(q) * sinE
    let kEmbed = k * cosE + rotateHalf(k) * sinE
    return (qEmbed, kEmbed)
}

// MARK: - Compute inv_freq for RoPE

private func computeInvFreq(dim: Int, base: Float) -> MLXArray {
    let arange = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)
    let exponent = arange / Float(dim)
    return 1.0 / MLXArray(base).pow(exponent)
}

// MARK: - Multimodal Rotary Embedding (3D MRoPE)

final class TalkerRotaryEmbedding: Module {
    let dim: Int
    let maxPositionEmbeddings: Int
    let base: Float
    let mropeSection: [Int]
    let invFreq: MLXArray

    init(dim: Int, maxPositionEmbeddings: Int = 32768, base: Float = 10000.0, mropeSection: [Int]? = nil) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base
        self.mropeSection = mropeSection ?? [24, 20, 20]
        self.invFreq = computeInvFreq(dim: dim, base: base)
    }

    func applyInterleavedMrope(_ freqs: MLXArray, mropeSection sec: [Int]) -> MLXArray {
        let headDimHalf = freqs.dim(-1)
        let freqsT = freqs[0]
        let freqsH = freqs[1]
        let freqsW = freqs[2]

        let indices = MLXArray(0 ..< headDimHalf)
        let hLength = sec[1] * 3
        let wLength = sec[2] * 3

        let mod3 = indices % 3
        let isH: MLXArray = mod3 .== 1
        let isW: MLXArray = mod3 .== 2
        let ltH: MLXArray = indices .< MLXArray(hLength)
        let ltW: MLXArray = indices .< MLXArray(wLength)
        let hMask = isH .&& ltH
        let wMask = isW .&& ltW

        let hMaskR = hMask.reshaped(1, 1, headDimHalf)
        let wMaskR = wMask.reshaped(1, 1, headDimHalf)

        var combined = which(hMaskR, freqsH, freqsT)
        combined = which(wMaskR, freqsW, combined)
        return combined
    }

    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        var posIds = positionIds
        if posIds.ndim == 2 {
            posIds = broadcast(expandedDimensions(posIds, axis: 0), to: [3, posIds.dim(0), posIds.dim(1)])
        }

        let invFreqExpanded = broadcast(
            invFreq.reshaped(1, 1, invFreq.dim(0), 1).asType(.float32),
            to: [3, posIds.dim(1), invFreq.dim(0), 1]
        )
        let pos = expandedDimensions(posIds.asType(.float32), axis: 2)

        let freqsRaw = matmul(invFreqExpanded, pos)
        let freqs = swappedAxes(freqsRaw, 2, 3)

        let combined = applyInterleavedMrope(freqs, mropeSection: mropeSection)
        let emb = concatenated([combined, combined], axis: -1)
        let cosVal = MLX.cos(emb).asType(x.dtype)
        let sinVal = MLX.sin(emb).asType(x.dtype)
        return (cosVal, sinVal)
    }
}

// MARK: - Standard Rotary Embedding (for Code Predictor)

final class Qwen3TTSRotaryEmbedding: Module {
    let dim: Int
    let invFreq: MLXArray

    init(dim: Int, maxPositionEmbeddings: Int = 32768, base: Float = 10000.0) {
        self.dim = dim
        self.invFreq = computeInvFreq(dim: dim, base: base)
    }

    func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let inv = expandedDimensions(invFreq, axes: [0, 2])
        let pos = expandedDimensions(positionIds.asType(.float32), axis: 1)
        let freqs = swappedAxes(matmul(inv, pos), 1, 2)
        let emb = concatenated([freqs, freqs], axis: -1)
        return (MLX.cos(emb).asType(x.dtype), MLX.sin(emb).asType(x.dtype))
    }
}

// MARK: - Talker Attention

final class TalkerAttention: Module {
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

    init(config: Qwen3TTSTalkerConfig, layerIdx: Int) {
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
        (q, k) = applyRotaryPosEmb(q, k, cos: cosVal, sin: sinVal)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )

        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1))
    }
}

// MARK: - Talker MLP (SwiGLU)

final class TalkerMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: Qwen3TTSTalkerConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - ResizeMLP (text projection)

final class ResizeMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(inputSize: Int, intermediateSize: Int, outputSize: Int, bias: Bool = false) {
        self._fc1.wrappedValue = Linear(inputSize, intermediateSize, bias: bias)
        self._fc2.wrappedValue = Linear(intermediateSize, outputSize, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(silu(fc1(x)))
    }
}

// MARK: - Talker Decoder Layer

final class TalkerDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: TalkerAttention
    @ModuleInfo var mlp: TalkerMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(config: Qwen3TTSTalkerConfig, layerIdx: Int) {
        self._selfAttn.wrappedValue = TalkerAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = TalkerMLP(config: config)
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

// MARK: - Talker Model (inner)

final class Qwen3TTSTalkerModel: Module {
    let config: Qwen3TTSTalkerConfig

    @ModuleInfo(key: "codec_embedding") var codecEmbedding: Embedding
    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding
    let layers: [TalkerDecoderLayer]
    @ModuleInfo var norm: RMSNorm
    let rotaryEmb: TalkerRotaryEmbedding

    init(config: Qwen3TTSTalkerConfig) {
        self.config = config
        self._codecEmbedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._textEmbedding.wrappedValue = Embedding(embeddingCount: config.textVocabSize, dimensions: config.textHiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { TalkerDecoderLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.rotaryEmb = TalkerRotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta,
            mropeSection: config.mropeSection
        )
    }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [any KVCache]? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (inputsEmbeds.dim(0), inputsEmbeds.dim(1), inputsEmbeds.dim(2))

        let offset: Int = cache?.first?.offset ?? 0

        let posIds: MLXArray
        if let positionIds {
            posIds = positionIds
        } else {
            let pos = MLXArray(Int32(offset) ..< Int32(offset + seqLen)).reshaped(1, seqLen)
            let bpos = broadcast(pos, to: [batch, seqLen])
            posIds = stacked([bpos, bpos, bpos], axis: 0)
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

// MARK: - Talker for Conditional Generation (full model)

final class Qwen3TTSTalkerForConditionalGeneration: Module {
    let config: Qwen3TTSTalkerConfig
    @ModuleInfo var model: Qwen3TTSTalkerModel
    @ModuleInfo(key: "text_projection") var textProjection: ResizeMLP
    @ModuleInfo(key: "codec_head") var codecHead: Linear
    @ModuleInfo(key: "code_predictor") var codePredictor: Qwen3TTSCodePredictor

    init(config: Qwen3TTSTalkerConfig) {
        self.config = config
        self._model.wrappedValue = Qwen3TTSTalkerModel(config: config)
        self._textProjection.wrappedValue = ResizeMLP(
            inputSize: config.textHiddenSize,
            intermediateSize: config.textHiddenSize,
            outputSize: config.hiddenSize,
            bias: true
        )
        self._codecHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)

        let cpConfig = config.codePredictorConfig ?? {
            let json = "{}".data(using: .utf8)!
            return try! JSONDecoder().decode(Qwen3TTSTalkerCodePredictorConfig.self, from: json)
        }()
        self._codePredictor.wrappedValue = Qwen3TTSCodePredictor(config: cpConfig, talkerHiddenSize: config.hiddenSize)
    }

    func getInputEmbeddings() -> Embedding { model.codecEmbedding }
    func getTextEmbeddings() -> Embedding { model.textEmbedding }

    func callAsFunction(
        _ inputsEmbeds: MLXArray,
        positionIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [any KVCache]? = nil
    ) -> (MLXArray, MLXArray) {
        let hiddenStates = model(inputsEmbeds, positionIds: positionIds, mask: mask, cache: cache)
        let logits = codecHead(hiddenStates)
        return (logits, hiddenStates)
    }

    func makeCache() -> [any KVCache] {
        model.makeCache()
    }

    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (k, v) in weights {
            guard k.hasPrefix("talker.") else { continue }
            let newKey = String(k.dropFirst("talker.".count))
            sanitized[newKey] = v
        }
        return sanitized
    }
}
