//
//  Activations.swift
//  MLXAudio
//
//  Activation functions for Speech Tokenizer Decoder.
//  Ported from mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py
//

import Foundation
import MLX
import MLXNN

// MARK: - SnakeBeta Activation

/// Snake activation with learnable alpha and beta parameters.
///
/// Formula: SnakeBeta(x) = x + (1/beta) * sin²(x * alpha)
///
/// Where alpha and beta are stored in log-space and exponentiated during forward pass.
/// This learnable periodic activation helps with audio synthesis by allowing the network
/// to learn frequency-dependent nonlinearities.
///
/// Input format: NCL (batch, channels, time)
/// Output format: NCL (batch, channels, time)
public class SnakeBeta: Module, UnaryLayer {
    public let channels: Int
    private let eps: Float = 1e-9

    /// Learnable log-alpha parameter (one per channel)
    var alpha: MLXArray

    /// Learnable log-beta parameter (one per channel)
    var beta: MLXArray

    /// Initialize SnakeBeta activation.
    ///
    /// - Parameter channels: Number of input channels
    public init(channels: Int) {
        self.channels = channels
        // Initialize with zeros: exp(0) = 1, so initial alpha=1, beta=1
        self.alpha = MLXArray.zeros([channels])
        self.beta = MLXArray.zeros([channels])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [batch, channels, time] (NCL format)

        // Exponentiate and reshape for broadcasting: [1, channels, 1]
        let alphaExp = exp(alpha).reshaped([1, channels, 1])
        let betaExp = exp(beta).reshaped([1, channels, 1])

        // SnakeBeta formula: x + (1/(beta + eps)) * sin²(x * alpha)
        let sinPart = pow(sin(x * alphaExp), 2)
        return x + (1.0 / (betaExp + eps)) * sinPart
    }
}
