//
//  TalkerLayers.swift
//  Qwen3TTS
//
//  MLP and Decoder Layer for Qwen3-TTS Talker.
//  Ported from mlx_audio/tts/models/qwen3_tts/talker.py
//

import Foundation
import MLX
import MLXNN

// MARK: - TalkerMLP

/// MLP with SwiGLU activation for Qwen3-TTS Talker.
///
/// Implements the gated linear unit: `down_proj(silu(gate_proj(x)) * up_proj(x))`
public class TalkerMLP: Module {
    public let hiddenSize: Int
    public let intermediateSize: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(config: Qwen3TTSTalkerConfig) {
        self.hiddenSize = config.hiddenSize
        self.intermediateSize = config.intermediateSize

        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - ResizeMLP

/// MLP for resizing hidden dimensions.
///
/// Used to project between different hidden sizes (e.g., text to talker embedding).
public class ResizeMLP: Module {
    public let inputSize: Int
    public let intermediateSize: Int
    public let outputSize: Int

    @ModuleInfo(key: "linear_fc1") var linearFc1: Linear
    @ModuleInfo(key: "linear_fc2") var linearFc2: Linear

    private let actFn: (MLXArray) -> MLXArray

    public init(
        inputSize: Int,
        intermediateSize: Int,
        outputSize: Int,
        hiddenAct: String = "silu",
        bias: Bool = false
    ) {
        self.inputSize = inputSize
        self.intermediateSize = intermediateSize
        self.outputSize = outputSize

        self._linearFc1.wrappedValue = Linear(inputSize, intermediateSize, bias: bias)
        self._linearFc2.wrappedValue = Linear(intermediateSize, outputSize, bias: bias)

        self.actFn = getActivation(hiddenAct)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linearFc2(actFn(linearFc1(x)))
    }
}

// MARK: - TalkerDecoderLayer

/// Transformer decoder layer for Qwen3-TTS Talker.
///
/// Pre-norm architecture with:
/// - Self-attention with MRoPE
/// - SwiGLU MLP
/// - Residual connections
public class TalkerDecoderLayer: Module {
    public let hiddenSize: Int

    @ModuleInfo(key: "self_attn") var selfAttn: TalkerAttention
    @ModuleInfo(key: "mlp") var mlp: TalkerMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    public init(config: Qwen3TTSTalkerConfig, layerIdx: Int) {
        self.hiddenSize = config.hiddenSize

        self._selfAttn.wrappedValue = TalkerAttention(config: config, layerIdx: layerIdx)
        self._mlp.wrappedValue = TalkerMLP(config: config)
        self._inputLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Forward pass for the decoder layer.
    ///
    /// - Parameters:
    ///   - x: Input tensor [batch, seq_len, hidden_size]
    ///   - positionEmbeddings: (cos, sin) from TalkerRotaryEmbedding
    ///   - mask: Optional attention mask
    ///   - cache: Optional KV cache
    /// - Returns: Output tensor [batch, seq_len, hidden_size]
    public func callAsFunction(
        _ x: MLXArray,
        positionEmbeddings: (cos: MLXArray, sin: MLXArray),
        mask: MLXArray? = nil,
        cache: TalkerKVCache? = nil
    ) -> MLXArray {
        // Self attention with pre-norm
        var residual = x
        var hidden = inputLayernorm(x)
        hidden = selfAttn(hidden, positionEmbeddings: positionEmbeddings, mask: mask, cache: cache)
        hidden = residual + hidden

        // MLP with pre-norm
        residual = hidden
        hidden = postAttentionLayernorm(hidden)
        hidden = mlp(hidden)
        hidden = residual + hidden

        return hidden
    }
}
