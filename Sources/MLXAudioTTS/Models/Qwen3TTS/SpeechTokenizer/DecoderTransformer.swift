//
//  DecoderTransformer.swift
//  MLXAudio
//
//  8-layer Transformer for Speech Tokenizer Decoder.
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
//

import Foundation
import MLX
import MLXNN

// MARK: - DecoderRMSNorm

/// RMS normalization for decoder.
public class DecoderRMSNorm: Module, UnaryLayer {
    public let hiddenSize: Int
    public let eps: Float

    @ModuleInfo(key: "weight") var weight: MLXArray

    public init(hiddenSize: Int, eps: Float = 1e-5) {
        self.hiddenSize = hiddenSize
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([hiddenSize])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Cast to float32 for numerical stability
        let xFloat = x.asType(.float32)
        let variance = mean(xFloat * xFloat, axis: -1, keepDims: true)
        let xNormed = xFloat * rsqrt(variance + eps)
        return (weight * xNormed).asType(x.dtype)
    }
}

// MARK: - LayerScale

/// Layer scale for residual connections.
///
/// Applies a learnable per-channel scale to the input.
public class LayerScale: Module, UnaryLayer {
    public let channels: Int

    @ModuleInfo(key: "scale") var scale: MLXArray

    public init(channels: Int, initialScale: Float = 0.01) {
        self.channels = channels
        self._scale.wrappedValue = MLXArray.ones([channels]) * initialScale
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return scale * x
    }
}

// MARK: - DecoderRotaryEmbedding

/// Rotary position embedding for decoder transformer.
///
/// Standard RoPE implementation (not multimodal like Talker).
public class DecoderRotaryEmbedding: Module {
    public let dim: Int
    public let maxPositionEmbeddings: Int
    public let base: Float

    private let invFreq: MLXArray

    public init(dim: Int, maxPositionEmbeddings: Int = 8000, base: Float = 10000.0) {
        self.dim = dim
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.base = base

        // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        let indices = MLXArray(Array(stride(from: 0, to: dim, by: 2)).map { Float($0) })
        self.invFreq = 1.0 / pow(MLXArray(base), indices / Float(dim))
    }

    /// Compute cos and sin embeddings for given positions.
    ///
    /// - Parameters:
    ///   - x: Input tensor (used only for dtype)
    ///   - positionIds: Position indices [batch, seq_len]
    /// - Returns: (cos, sin) embeddings each of shape [batch, seq_len, dim]
    public func callAsFunction(_ x: MLXArray, positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        // inv_freq: [dim/2] -> [1, dim/2, 1]
        let invFreqExpanded = invFreq.reshaped([1, -1, 1]).asType(.float32)

        // position_ids: [batch, seq_len] -> [batch, 1, seq_len]
        let posExpanded = positionIds.expandedDimensions(axis: 1).asType(.float32)

        // freqs: [batch, dim/2, seq_len] -> transpose -> [batch, seq_len, dim/2]
        let freqs = (invFreqExpanded * posExpanded).transposed(0, 2, 1)

        // Concatenate to get full dimension: [batch, seq_len, dim]
        let emb = concatenated([freqs, freqs], axis: -1)

        let cosEmb = cos(emb).asType(x.dtype)
        let sinEmb = sin(emb).asType(x.dtype)

        return (cosEmb, sinEmb)
    }
}

// MARK: - Decoder Rotary Position Embedding Application

/// Apply rotary position embeddings to query and key tensors for decoder.
///
/// Uses the existing rotateHalf from Qwen3TTSUtils.
///
/// - Parameters:
///   - q: Query tensor [batch, heads, seq_len, head_dim]
///   - k: Key tensor [batch, heads, seq_len, head_dim]
///   - cos: Cosine embeddings [batch, seq_len, head_dim]
///   - sin: Sine embeddings [batch, seq_len, head_dim]
/// - Returns: (rotated_q, rotated_k) with same shapes as input
public func applyDecoderRotaryPosEmb(
    q: MLXArray,
    k: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> (MLXArray, MLXArray) {
    // Expand for heads dimension: [batch, 1, seq_len, head_dim]
    let cosExpanded = cos.expandedDimensions(axis: 1)
    let sinExpanded = sin.expandedDimensions(axis: 1)

    let qEmbed = (q * cosExpanded) + (rotateHalf(q) * sinExpanded)
    let kEmbed = (k * cosExpanded) + (rotateHalf(k) * sinExpanded)

    return (qEmbed, kEmbed)
}

// MARK: - DecoderAttention

/// Multi-head attention for decoder transformer.
public class DecoderAttention: Module {
    public let config: Qwen3TTSTokenizerDecoderConfig
    public let layerIdx: Int
    public let headDim: Int
    public let numHeads: Int
    public let numKvHeads: Int
    public let numKvGroups: Int
    public let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    public init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        self.headDim = config.headDim
        self.numHeads = config.numAttentionHeads
        self.numKvHeads = config.numKeyValueHeads
        self.numKvGroups = numHeads / numKvHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qProj.wrappedValue = Linear(
            config.hiddenSize,
            numHeads * headDim,
            bias: config.attentionBias
        )
        self._kProj.wrappedValue = Linear(
            config.hiddenSize,
            numKvHeads * headDim,
            bias: config.attentionBias
        )
        self._vProj.wrappedValue = Linear(
            config.hiddenSize,
            numKvHeads * headDim,
            bias: config.attentionBias
        )
        self._oProj.wrappedValue = Linear(
            numHeads * headDim,
            config.hiddenSize,
            bias: config.attentionBias
        )
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil,
        cache: TalkerKVCache? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (x.shape[0], x.shape[1], x.shape[2])

        // Project to Q, K, V
        var q = qProj(x)
            .reshaped([batch, seqLen, numHeads, headDim])
            .transposed(0, 2, 1, 3)  // [batch, heads, seq_len, head_dim]

        var k = kProj(x)
            .reshaped([batch, seqLen, numKvHeads, headDim])
            .transposed(0, 2, 1, 3)

        var v = vProj(x)
            .reshaped([batch, seqLen, numKvHeads, headDim])
            .transposed(0, 2, 1, 3)

        // Apply rotary embeddings
        let (cos, sin) = positionEmbeddings
        (q, k) = applyDecoderRotaryPosEmb(q: q, k: k, cos: cos, sin: sin)

        // Handle KV cache
        if let cache = cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        // Scaled dot product attention with GQA support
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        // Reshape back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        let outputReshaped = output
            .transposed(0, 2, 1, 3)
            .reshaped([batch, seqLen, -1])

        return oProj(outputReshaped)
    }
}

// MARK: - DecoderMLP

/// MLP for decoder transformer using SwiGLU activation.
public class DecoderMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(config: Qwen3TTSTokenizerDecoderConfig) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        self._upProj.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        self._downProj.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: down(silu(gate(x)) * up(x))
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - DecoderTransformerLayer

/// Transformer layer for decoder with pre-norm and LayerScale.
public class DecoderTransformerLayer: Module {
    public let hiddenSize: Int

    @ModuleInfo(key: "self_attn") var selfAttn: DecoderAttention
    @ModuleInfo(key: "mlp") var mlp: DecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: DecoderRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: DecoderRMSNorm
    @ModuleInfo(key: "self_attn_layer_scale") var selfAttnLayerScale: LayerScale
    @ModuleInfo(key: "mlp_layer_scale") var mlpLayerScale: LayerScale

    public init(config: Qwen3TTSTokenizerDecoderConfig, layerIdx: Int) {
        self.hiddenSize = config.hiddenSize

        self._selfAttn.wrappedValue = DecoderAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = DecoderMLP(config: config)
        self._inputLayernorm.wrappedValue = DecoderRMSNorm(
            hiddenSize: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayernorm.wrappedValue = DecoderRMSNorm(
            hiddenSize: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._selfAttnLayerScale.wrappedValue = LayerScale(
            channels: config.hiddenSize,
            initialScale: config.layerScaleInitialScale
        )
        self._mlpLayerScale.wrappedValue = LayerScale(
            channels: config.hiddenSize,
            initialScale: config.layerScaleInitialScale
        )
    }

    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil,
        cache: TalkerKVCache? = nil
    ) -> MLXArray {
        // Self attention with LayerScale
        var residual = x
        var hidden = inputLayernorm(x)
        hidden = selfAttn(hidden, positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
        hidden = residual + selfAttnLayerScale(hidden)

        // MLP with LayerScale
        residual = hidden
        hidden = postAttentionLayernorm(hidden)
        hidden = mlp(hidden)
        hidden = residual + mlpLayerScale(hidden)

        return hidden
    }
}

// MARK: - DecoderTransformer

/// Transformer model for decoder.
///
/// 8-layer transformer with input/output projections:
/// - Input: [batch, seq_len, latent_dim] (1024)
/// - Hidden: [batch, seq_len, hidden_size] (512)
/// - Output: [batch, seq_len, latent_dim] (1024)
public class DecoderTransformer: Module {
    public let config: Qwen3TTSTokenizerDecoderConfig

    @ModuleInfo(key: "layers") var layers: [DecoderTransformerLayer]
    @ModuleInfo(key: "norm") var norm: DecoderRMSNorm
    @ModuleInfo(key: "rotary_emb") var rotaryEmb: DecoderRotaryEmbedding
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    public init(config: Qwen3TTSTokenizerDecoderConfig) {
        self.config = config

        // Build transformer layers
        var layerList: [DecoderTransformerLayer] = []
        for i in 0..<config.numHiddenLayers {
            layerList.append(DecoderTransformerLayer(config: config, layerIdx: i))
        }
        self._layers.wrappedValue = layerList

        // Final norm
        self._norm.wrappedValue = DecoderRMSNorm(
            hiddenSize: config.hiddenSize,
            eps: config.rmsNormEps
        )

        // Rotary embeddings
        self._rotaryEmb.wrappedValue = DecoderRotaryEmbedding(
            dim: config.headDim,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            base: config.ropeTheta
        )

        // Input projection: latent_dim (1024) -> hidden_size (512)
        self._inputProj.wrappedValue = Linear(config.latentDim, config.hiddenSize)

        // Output projection: hidden_size (512) -> latent_dim (1024)
        self._outputProj.wrappedValue = Linear(config.hiddenSize, config.latentDim)
    }

    /// Create KV caches for all layers.
    public func makeCache() -> [TalkerSimpleKVCache] {
        return layers.map { _ in TalkerSimpleKVCache() }
    }

    public func callAsFunction(
        _ inputsEmbeds: MLXArray,
        mask: MLXArray? = nil,
        cache: [TalkerSimpleKVCache]? = nil
    ) -> MLXArray {
        let (batch, seqLen, _) = (inputsEmbeds.shape[0], inputsEmbeds.shape[1], inputsEmbeds.shape[2])

        // Input projection
        var x = inputProj(inputsEmbeds)

        // Position ids - use cache offset for position tracking
        let offset = cache?.first?.sequenceLength ?? 0
        let positionIdsArray = MLXArray(Array(offset..<(offset + seqLen)))
            .expandedDimensions(axis: 0)
        let positionIds = broadcast(positionIdsArray, to: [batch, seqLen])

        // Compute position embeddings
        let positionEmbeddings = rotaryEmb(x, positionIds: positionIds)

        // Create causal mask if needed
        var attnMask = mask
        if attnMask == nil && seqLen > 1 {
            attnMask = createCausalMask(seqLen: seqLen, dtype: x.dtype)
        }

        // Apply transformer layers
        for (i, layer) in layers.enumerated() {
            let layerCache: TalkerSimpleKVCache? = cache?[i]
            x = layer(x, positionEmbeddings: positionEmbeddings, mask: attnMask, cache: layerCache)
        }

        // Final norm and output projection
        x = norm(x)
        x = outputProj(x)

        return x
    }
}
