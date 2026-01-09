import Foundation
import MLX

/// Core AlignAtt algorithm functions for streaming STT.
/// Determines which audio frames the decoder is attending to and whether tokens are stable.
public enum StreamingDecoder {
    /// Find the audio frame that received the most attention from the last decoded token.
    /// - Parameters:
    ///   - crossQK: Cross-attention weights from each decoder layer, shape [batch, heads, tokens, frames]
    ///   - alignmentHeads: List of (layer, head) tuples to use for alignment
    /// - Returns: Frame index with highest average attention across specified heads
    public static func getMostAttendedFrame(
        crossQK: [MLXArray?],
        alignmentHeads: [(layer: Int, head: Int)]
    ) -> Int {
        var weights: [MLXArray] = []

        for (layer, head) in alignmentHeads {
            guard layer < crossQK.count, let layerQK = crossQK[layer] else { continue }
            // Extract last token's attention: [batch, head, -1, frames] -> [frames]
            // Ensure float32 to avoid GPU float64 errors
            let attention = layerQK[0, head, -1, 0...].asType(.float32)
            weights.append(attention)
        }

        guard !weights.isEmpty else { return 0 }

        // Average across heads (ensure float32 to avoid GPU float64 error)
        let stacked = MLX.stacked(weights, axis: 0).asType(.float32)
        let avgAttention = stacked.mean(axis: 0)

        // Find max
        return Int(MLX.argMax(avgAttention).item(Int32.self))
    }

    /// Determine if the current token should be emitted based on attention stability.
    /// A token is stable if the decoder's attention is far enough from the audio end.
    /// - Parameters:
    ///   - mostAttendedFrame: Frame index with highest attention
    ///   - totalContentFrames: Total number of audio content frames
    ///   - threshold: Minimum distance from end before emitting
    /// - Returns: True if token should be emitted (attention far from audio end)
    public static func shouldEmit(
        mostAttendedFrame: Int,
        totalContentFrames: Int,
        threshold: Int
    ) -> Bool {
        let distanceToEnd = totalContentFrames - mostAttendedFrame
        return distanceToEnd >= threshold
    }
}
