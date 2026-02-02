import Foundation
import MLX
import MLXNN

/// Whisper Audio Encoder
/// Processes mel spectrograms through Conv1D layers and transformer blocks
public class AudioEncoder: Module {
    @ModuleInfo public var conv1: Conv1d
    @ModuleInfo public var conv2: Conv1d
    // Whisper encoder uses computed sinusoidal positional embeddings (not learned from weights)
    // See: https://github.com/openai/whisper/discussions/697
    public var positionalEmbedding: MLXArray
    @ModuleInfo(key: "blocks") public var blocks: [ResidualAttentionBlock]
    @ModuleInfo(key: "ln_post") public var ln: LayerNorm

    let config: WhisperConfiguration

    public init(config: WhisperConfiguration) {
        self.config = config

        // Conv1d: nMels -> nAudioState, kernel=3, padding=1
        // Input format for MLXNN.Conv1d: [batch, seq, channels] (NLC)
        self._conv1.wrappedValue = Conv1d(
            inputChannels: config.nMels,
            outputChannels: config.nAudioState,
            kernelSize: 3,
            padding: 1
        )

        // Conv1d: nAudioState -> nAudioState, kernel=3, stride=2, padding=1
        // stride=2 downsamples the sequence length by half
        self._conv2.wrappedValue = Conv1d(
            inputChannels: config.nAudioState,
            outputChannels: config.nAudioState,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )

        // Sinusoidal positional embedding (computed, not loaded from weights)
        self.positionalEmbedding = Self.sinusoidalPositionalEmbedding(
            maxLen: config.nAudioCtx,
            dims: config.nAudioState
        )

        // Transformer blocks (encoder-only, no cross-attention)
        self._blocks.wrappedValue = (0..<config.nAudioLayer).map { _ in
            ResidualAttentionBlock(
                nState: config.nAudioState,
                nHead: config.nAudioHead,
                crossAttention: false
            )
        }

        // Final layer normalization
        self._ln.wrappedValue = LayerNorm(dimensions: config.nAudioState)
    }

    /// Forward pass through the audio encoder
    /// - Parameter mel: Mel spectrogram [batch, nMels, nFrames]
    /// - Returns: Encoder output [batch, seqLen, nAudioState]
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // Input: [batch, nMels, nFrames]
        // Conv1d expects: [batch, seq, channels] (NLC format)
        // Transpose to: [batch, nFrames, nMels]
        var x = mel.transposed(0, 2, 1)

        // Conv layers with GELU activation
        x = gelu(conv1(x))
        x = gelu(conv2(x))

        // After conv2 with stride=2, sequence length is halved
        // x shape: [batch, nFrames/2, nAudioState]

        // Add positional embedding
        let seqLen = x.shape[1]
        x = x + positionalEmbedding[0..<seqLen]

        // Pass through transformer blocks
        for block in blocks {
            let (output, _) = block(x: x)
            x = output
        }

        // Final layer norm
        x = ln(x)

        return x
    }

    /// Generate sinusoidal positional embeddings matching Whisper/mlx-examples implementation
    /// Uses log-spaced timescales for better position encoding across different frequencies
    private static func sinusoidalPositionalEmbedding(maxLen: Int, dims: Int, maxTimescale: Float = 10000.0) -> MLXArray {
        precondition(dims % 2 == 0, "dims must be even for sinusoidal positional embedding")

        let halfDims = dims / 2
        let logTimescaleIncrement = log(maxTimescale) / Float(halfDims - 1)

        var embedding = [[Float]](repeating: [Float](repeating: 0, count: dims), count: maxLen)

        for pos in 0..<maxLen {
            for i in 0..<halfDims {
                let invTimescale = exp(-logTimescaleIncrement * Float(i))
                let scaledTime = Float(pos) * invTimescale
                embedding[pos][i] = sin(scaledTime)
                embedding[pos][i + halfDims] = cos(scaledTime)
            }
        }

        return MLXArray(embedding.flatMap { $0 }).reshaped([maxLen, dims])
    }
}
