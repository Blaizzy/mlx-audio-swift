import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Attention

/// Multi-head attention shared by encoder self-attention and decoder
/// self/cross-attention. Whisper omits the bias on `k_proj`; all other
/// projections carry bias.
final class WhisperAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let headDim: Int
    let scaling: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(embedDim: Int, numHeads: Int) {
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scaling = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._kProj.wrappedValue = Linear(embedDim, embedDim, bias: false)
        self._vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        self._outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
    }

    /// Run attention with optional cached K/V (for the decoder's autoregressive
    /// path). Returns `(output, newK, newV)`.
    func callAsFunction(
        _ hidden: MLXArray,
        keyValueInput: MLXArray,
        cachedKeys: MLXArray? = nil,
        cachedValues: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
    ) -> (MLXArray, MLXArray, MLXArray) {
        let B = hidden.shape[0]
        let Tq = hidden.shape[1]

        let q = qProj(hidden).reshaped([B, Tq, numHeads, headDim]).transposed(0, 2, 1, 3)

        var keys: MLXArray
        var values: MLXArray
        if let cachedKeys, let cachedValues {
            let Tnew = keyValueInput.shape[1]
            let newK = kProj(keyValueInput).reshaped([B, Tnew, numHeads, headDim]).transposed(0, 2, 1, 3)
            let newV = vProj(keyValueInput).reshaped([B, Tnew, numHeads, headDim]).transposed(0, 2, 1, 3)
            keys = MLX.concatenated([cachedKeys, newK], axis: 2)
            values = MLX.concatenated([cachedValues, newV], axis: 2)
        } else {
            let Tk = keyValueInput.shape[1]
            keys = kProj(keyValueInput).reshaped([B, Tk, numHeads, headDim]).transposed(0, 2, 1, 3)
            values = vProj(keyValueInput).reshaped([B, Tk, numHeads, headDim]).transposed(0, 2, 1, 3)
        }

        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: keys,
            values: values,
            scale: scaling,
            mask: mask
        )

        let merged = attn.transposed(0, 2, 1, 3).reshaped([B, Tq, embedDim])
        return (outProj(merged), keys, values)
    }
}

// MARK: - Encoder

final class WhisperEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: WhisperAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: WhisperConfig) {
        let d = config.dModel
        self._selfAttn.wrappedValue = WhisperAttention(
            embedDim: d,
            numHeads: config.encoderAttentionHeads
        )
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: d)
        self._fc1.wrappedValue = Linear(d, config.encoderFfnDim, bias: true)
        self._fc2.wrappedValue = Linear(config.encoderFfnDim, d, bias: true)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: d)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x
        var h = selfAttnLayerNorm(x)
        let (attnOut, _, _) = selfAttn(h, keyValueInput: h)
        h = residual + attnOut

        residual = h
        h = finalLayerNorm(h)
        h = gelu(fc1(h))
        h = fc2(h)
        return residual + h
    }
}

final class WhisperEncoder: Module {
    let config: WhisperConfig

    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    @ModuleInfo(key: "layers") var layers: [WhisperEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(config: WhisperConfig) {
        self.config = config
        let d = config.dModel
        self._conv1.wrappedValue = Conv1d(
            inputChannels: config.numMelBins,
            outputChannels: d,
            kernelSize: 3,
            stride: 1,
            padding: 1
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: d,
            outputChannels: d,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        self._embedPositions.wrappedValue = Embedding(
            embeddingCount: config.maxSourcePositions,
            dimensions: d
        )
        self._layers.wrappedValue = (0..<config.encoderLayers).map { _ in
            WhisperEncoderLayer(config: config)
        }
        self._layerNorm.wrappedValue = LayerNorm(dimensions: d)
    }

    func callAsFunction(_ inputFeatures: MLXArray) -> MLXArray {
        var h = gelu(conv1(inputFeatures))
        h = gelu(conv2(h))
        let seqLen = h.shape[1]
        h = h + embedPositions.weight[0..<seqLen]
        for layer in layers {
            h = layer(h)
        }
        return layerNorm(h)
    }
}

// MARK: - Decoder

/// Per-layer KV cache: self-attn caches grow with each new token; cross-attn
/// caches are computed once per chunk and reused for every subsequent step.
struct WhisperLayerCache {
    var selfKeys: MLXArray? = nil
    var selfValues: MLXArray? = nil
    var crossKeys: MLXArray? = nil
    var crossValues: MLXArray? = nil
}

final class WhisperDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: WhisperAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: WhisperAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(config: WhisperConfig) {
        let d = config.dModel
        self._selfAttn.wrappedValue = WhisperAttention(
            embedDim: d,
            numHeads: config.decoderAttentionHeads
        )
        self._selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: d)
        self._encoderAttn.wrappedValue = WhisperAttention(
            embedDim: d,
            numHeads: config.decoderAttentionHeads
        )
        self._encoderAttnLayerNorm.wrappedValue = LayerNorm(dimensions: d)
        self._fc1.wrappedValue = Linear(d, config.decoderFfnDim, bias: true)
        self._fc2.wrappedValue = Linear(config.decoderFfnDim, d, bias: true)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: d)
    }

    func callAsFunction(
        _ x: MLXArray,
        encoderHidden: MLXArray,
        selfMask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: inout WhisperLayerCache
    ) -> MLXArray {
        var residual = x
        var h = selfAttnLayerNorm(x)
        let (selfOut, newSelfK, newSelfV) = selfAttn(
            h,
            keyValueInput: h,
            cachedKeys: cache.selfKeys,
            cachedValues: cache.selfValues,
            mask: selfMask
        )
        cache.selfKeys = newSelfK
        cache.selfValues = newSelfV
        h = residual + selfOut

        residual = h
        h = encoderAttnLayerNorm(h)
        let (crossOut, crossK, crossV): (MLXArray, MLXArray, MLXArray)
        if let cachedCrossK = cache.crossKeys, let cachedCrossV = cache.crossValues {
            // Cross K/V already projected once; skip the linear and run SDPA directly.
            let B = h.shape[0]
            let Tq = h.shape[1]
            let q = encoderAttn.qProj(h)
                .reshaped([B, Tq, encoderAttn.numHeads, encoderAttn.headDim])
                .transposed(0, 2, 1, 3)
            let attn = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: cachedCrossK,
                values: cachedCrossV,
                scale: encoderAttn.scaling,
                mask: .none
            )
            let merged = attn.transposed(0, 2, 1, 3).reshaped([B, Tq, encoderAttn.embedDim])
            crossOut = encoderAttn.outProj(merged)
            crossK = cachedCrossK
            crossV = cachedCrossV
        } else {
            (crossOut, crossK, crossV) = encoderAttn(
                h,
                keyValueInput: encoderHidden,
                cachedKeys: nil,
                cachedValues: nil,
                mask: .none
            )
        }
        cache.crossKeys = crossK
        cache.crossValues = crossV
        h = residual + crossOut

        residual = h
        h = finalLayerNorm(h)
        h = gelu(fc1(h))
        h = fc2(h)
        return residual + h
    }
}

final class WhisperDecoder: Module {
    let config: WhisperConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
    @ModuleInfo(key: "layers") var layers: [WhisperDecoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(config: WhisperConfig) {
        self.config = config
        let d = config.dModel
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: d
        )
        self._embedPositions.wrappedValue = Embedding(
            embeddingCount: config.maxTargetPositions,
            dimensions: d
        )
        self._layers.wrappedValue = (0..<config.decoderLayers).map { _ in
            WhisperDecoderLayer(config: config)
        }
        self._layerNorm.wrappedValue = LayerNorm(dimensions: d)
    }

    /// Run the decoder for either a prefill (multi-token) or a single step.
    /// `startPosition` is the position-embedding offset for the first new token.
    func callAsFunction(
        tokens: MLXArray,
        startPosition: Int,
        encoderHidden: MLXArray,
        caches: inout [WhisperLayerCache]
    ) -> MLXArray {
        let Tnew = tokens.shape[1]
        let posIndices = MLXArray((startPosition..<(startPosition + Tnew)).map { Int32($0) })
        let positions = embedPositions(posIndices).expandedDimensions(axis: 0)

        var h = embedTokens(tokens) + positions

        let selfMask: MLXFast.ScaledDotProductAttentionMaskMode
        if Tnew > 1 {
            let total = startPosition + Tnew
            selfMask = .array(causalMask(Tnew: Tnew, Ttotal: total, dtype: h.dtype))
        } else {
            selfMask = .none
        }

        for index in layers.indices {
            h = layers[index](
                h,
                encoderHidden: encoderHidden,
                selfMask: selfMask,
                cache: &caches[index]
            )
        }
        return layerNorm(h)
    }

    private func causalMask(Tnew: Int, Ttotal: Int, dtype: DType) -> MLXArray {
        let startPosition = Ttotal - Tnew
        let rows = MLXArray((0..<Tnew).map { Int32($0 + startPosition) }).expandedDimensions(axis: 1)
        let cols = MLXArray((0..<Ttotal).map(Int32.init)).expandedDimensions(axis: 0)
        let allowed = cols .<= rows
        let zero = MLXArray.zeros([Tnew, Ttotal], dtype: dtype)
        let negInf = MLXArray.full([Tnew, Ttotal], values: MLXArray(Float(-1e9)), dtype: dtype)
        return MLX.where(allowed, zero, negInf)
    }

    /// Project hidden states to vocab logits using the tied embedding matrix.
    func projectToVocab(_ hidden: MLXArray) -> MLXArray {
        return hidden.matmul(embedTokens.weight.transposed(1, 0))
    }
}

// MARK: - Container

/// Mirrors HuggingFace's `WhisperModel` so safetensors prefixed with
/// `model.encoder.*` / `model.decoder.*` load directly via Module update.
final class WhisperSubmodels: Module {
    @ModuleInfo(key: "encoder") var encoder: WhisperEncoder
    @ModuleInfo(key: "decoder") var decoder: WhisperDecoder

    init(config: WhisperConfig) {
        self._encoder.wrappedValue = WhisperEncoder(config: config)
        self._decoder.wrappedValue = WhisperDecoder(config: config)
    }
}
