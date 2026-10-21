import Foundation
import MLX
import Testing

@testable import MLXAudioSTT

/// The streaming encoder cache must give attention what concatenation gave: the same
/// rows, in the same order, across irregular appends and window resets.
struct VoxtralRealtimeEncoderStreamKVCacheTests {
    @Test func appendsMatchConcatenationAcrossResets() {
        let width = 3
        // Past two growth blocks, so the storage grows twice within the first window.
        let capacity = 600
        let appendSizes = [1, 100, 200, 7, 1, 291]
        var nextValue = 0
        func rows(_ count: Int) -> MLXArray {
            let values = (nextValue..<(nextValue + count * width)).map(Float.init)
            nextValue += count * width
            return MLXArray(values, [count, width])
        }

        let cache = VoxtralRealtimeEncoderStreamKVCache(capacity: capacity)
        // Each window is filled by irregular appends that sum to `capacity`.
        for _ in 0..<3 {
            var reference: MLXArray?
            for n in appendSizes {
                let newKeys = rows(n)
                let window = cache.append(keys: newKeys, values: newKeys + 0.5)
                reference = reference.map { MLX.concatenated([$0, newKeys], axis: 0) } ?? newKeys

                #expect(cache.count == reference!.shape[0])
                #expect(window.keys.shape == reference!.shape)
                #expect(MLX.arrayEqual(window.keys, reference!).item(Bool.self))
                #expect(MLX.arrayEqual(window.values, reference! + 0.5).item(Bool.self))
            }
            #expect(cache.count == capacity)
            #expect(cache.keys?.shape[0] == capacity)
            cache.reset()
            #expect(cache.count == 0)
        }
    }
}
