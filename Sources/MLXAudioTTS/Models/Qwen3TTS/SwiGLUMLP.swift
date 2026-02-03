//
//  SwiGLUMLP.swift
//  MLXAudioTTS
//
//  Shared SwiGLU MLP implementation used across TTS models.
//

import Foundation
import MLX
import MLXNN

// MARK: - SwiGLU MLP

/// SwiGLU MLP layer used in transformer models.
///
/// Implements: `down_proj(silu(gate_proj(x)) * up_proj(x))`
///
/// This gated linear unit with SiLU activation is used in:
/// - Qwen3TTS Talker
/// - Qwen3 TTS
/// - Soprano TTS
/// - LlamaTTS
public class SwiGLUMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    /// Initialize SwiGLU MLP.
    ///
    /// - Parameters:
    ///   - inputSize: Input dimension (hidden_size)
    ///   - hiddenSize: Intermediate dimension (intermediate_size)
    ///   - bias: Whether to include bias in linear layers (default: false)
    public init(inputSize: Int, hiddenSize: Int, bias: Bool = false) {
        self._gateProj.wrappedValue = Linear(inputSize, hiddenSize, bias: bias)
        self._downProj.wrappedValue = Linear(hiddenSize, inputSize, bias: bias)
        self._upProj.wrappedValue = Linear(inputSize, hiddenSize, bias: bias)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(silu(gateProj(x)) * upProj(x))
    }
}
