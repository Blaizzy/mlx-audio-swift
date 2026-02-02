import Foundation
import MLX
import Testing

@testable import MLXAudioSTT

struct StreamingDecoderTests {
    @Test func getMostAttendedFrame_findsMaxAttention() throws {
        // Given: Cross-attention weights with known peak
        // Shape: [batch, heads, tokens, frames]
        var weights = MLXArray.zeros([1, 8, 1, 100], dtype: .float32)
        // Set peak at frame 42
        weights[0, 0, 0, 42] = MLXArray([Float(1.0)])

        let crossQK: [MLXArray?] = [weights]
        let alignmentHeads = [(layer: 0, head: 0)]

        // When: Finding most attended frame
        let frame = StreamingDecoder.getMostAttendedFrame(
            crossQK: crossQK,
            alignmentHeads: alignmentHeads
        )

        // Then: Returns frame 42
        #expect(frame == 42)
    }

    @Test func getMostAttendedFrame_averagesAcrossHeads() throws {
        // Given: Two heads with different peaks
        var weights = MLXArray.zeros([1, 8, 1, 100], dtype: .float32)
        // Head 0 peak at frame 40
        weights[0, 0, 0, 40] = MLXArray([Float(1.0)])
        // Head 1 peak at frame 60
        weights[0, 1, 0, 60] = MLXArray([Float(1.0)])

        let crossQK: [MLXArray?] = [weights]
        let alignmentHeads = [(layer: 0, head: 0), (layer: 0, head: 1)]

        // When: Finding most attended frame
        let frame = StreamingDecoder.getMostAttendedFrame(
            crossQK: crossQK,
            alignmentHeads: alignmentHeads
        )

        // Then: Returns average of 40 and 60 -> max attention at either 40 or 60
        // Since they have equal max weights, the first one (40) wins in argMax
        #expect(frame == 40 || frame == 60)
    }

    @Test func getMostAttendedFrame_handlesEmptyAlignmentHeads() throws {
        // Given: Empty alignment heads
        let weights = MLXArray.zeros([1, 8, 1, 100], dtype: .float32)
        let crossQK: [MLXArray?] = [weights]
        let alignmentHeads: [(layer: Int, head: Int)] = []

        // When: Finding most attended frame
        let frame = StreamingDecoder.getMostAttendedFrame(
            crossQK: crossQK,
            alignmentHeads: alignmentHeads
        )

        // Then: Returns 0 as fallback
        #expect(frame == 0)
    }

    @Test func getMostAttendedFrame_handlesNilLayers() throws {
        // Given: Some layers are nil
        var weights = MLXArray.zeros([1, 8, 1, 100], dtype: .float32)
        weights[0, 0, 0, 50] = MLXArray([Float(1.0)])

        let crossQK: [MLXArray?] = [nil, weights, nil]
        let alignmentHeads = [(layer: 0, head: 0), (layer: 1, head: 0), (layer: 2, head: 0)]

        // When: Finding most attended frame
        let frame = StreamingDecoder.getMostAttendedFrame(
            crossQK: crossQK,
            alignmentHeads: alignmentHeads
        )

        // Then: Uses only valid layer (layer 1)
        #expect(frame == 50)
    }

    @Test func getMostAttendedFrame_handlesOutOfBoundsLayer() throws {
        // Given: Layer index beyond crossQK array
        var weights = MLXArray.zeros([1, 8, 1, 100], dtype: .float32)
        weights[0, 0, 0, 30] = MLXArray([Float(1.0)])

        let crossQK: [MLXArray?] = [weights]
        let alignmentHeads = [(layer: 0, head: 0), (layer: 5, head: 0)]

        // When: Finding most attended frame
        let frame = StreamingDecoder.getMostAttendedFrame(
            crossQK: crossQK,
            alignmentHeads: alignmentHeads
        )

        // Then: Skips invalid layer, uses valid one
        #expect(frame == 30)
    }

    @Test func shouldEmit_nearEnd_returnsFalse() throws {
        // Given: Most attended frame near end of audio
        let mostAttendedFrame = 95
        let totalFrames = 100
        let threshold = 25

        // When: Checking if should emit
        let result = StreamingDecoder.shouldEmit(
            mostAttendedFrame: mostAttendedFrame,
            totalContentFrames: totalFrames,
            threshold: threshold
        )

        // Then: Should NOT emit (only 5 frames from end < threshold)
        #expect(!result)
    }

    @Test func shouldEmit_farFromEnd_returnsTrue() throws {
        // Given: Most attended frame far from end
        let mostAttendedFrame = 50
        let totalFrames = 100
        let threshold = 25

        // When: Checking if should emit
        let result = StreamingDecoder.shouldEmit(
            mostAttendedFrame: mostAttendedFrame,
            totalContentFrames: totalFrames,
            threshold: threshold
        )

        // Then: Should emit (50 frames from end >= threshold)
        #expect(result)
    }

    @Test func shouldEmit_exactlyAtThreshold_returnsTrue() throws {
        // Given: Distance exactly equals threshold
        let mostAttendedFrame = 75
        let totalFrames = 100
        let threshold = 25

        // When: Checking if should emit
        let result = StreamingDecoder.shouldEmit(
            mostAttendedFrame: mostAttendedFrame,
            totalContentFrames: totalFrames,
            threshold: threshold
        )

        // Then: Should emit (25 frames from end == threshold)
        #expect(result)
    }

    @Test func shouldEmit_oneBelowThreshold_returnsFalse() throws {
        // Given: Distance one below threshold
        let mostAttendedFrame = 76
        let totalFrames = 100
        let threshold = 25

        // When: Checking if should emit
        let result = StreamingDecoder.shouldEmit(
            mostAttendedFrame: mostAttendedFrame,
            totalContentFrames: totalFrames,
            threshold: threshold
        )

        // Then: Should NOT emit (24 frames from end < threshold)
        #expect(!result)
    }

    @Test func shouldEmit_atFrameZero_alwaysEmits() throws {
        // Given: Attention at frame 0 (very far from end)
        let mostAttendedFrame = 0
        let totalFrames = 100
        let threshold = 25

        // When: Checking if should emit
        let result = StreamingDecoder.shouldEmit(
            mostAttendedFrame: mostAttendedFrame,
            totalContentFrames: totalFrames,
            threshold: threshold
        )

        // Then: Should emit (100 frames from end >= threshold)
        #expect(result)
    }
}
