import MLX
import Testing

@testable import MLXAudioSTT

struct VoxtralRealtimeDecoderKVCachePlanTests {
    @Test func growsAtExactlyOneCapacityBlock() {
        let plan = VoxtralRealtimeDecoderKVCacheAppendPlan.make(
            count: 256,
            capacity: 256,
            positionOffset: 0,
            appendCount: 1,
            slidingWindow: 512
        )

        #expect(plan.compactionRange == nil)
        #expect(plan.appendRange == (256..<257))
        #expect(plan.requiresGrowth)
        #expect(plan.capacity == 512)
        #expect(plan.windowRange == (0..<257))
        #expect(plan.windowPositionOffset == 0)
    }

    @Test func compactsOnlyAfterExcessExceedsOneBlock() {
        let atBlock = VoxtralRealtimeDecoderKVCacheAppendPlan.make(
            count: 767,
            capacity: 1024,
            positionOffset: 0,
            appendCount: 1,
            slidingWindow: 512
        )
        #expect(atBlock.compactionRange == nil)
        #expect(atBlock.windowRange == (256..<768))
        #expect(atBlock.windowPositionOffset == 256)

        let beyondBlock = VoxtralRealtimeDecoderKVCacheAppendPlan.make(
            count: 768,
            capacity: 1024,
            positionOffset: 0,
            appendCount: 1,
            slidingWindow: 512
        )
        #expect(beyondBlock.compactionRange == (256..<768))
        #expect(beyondBlock.appendRange == (512..<513))
        #expect(!beyondBlock.requiresGrowth)
        #expect(beyondBlock.count == 513)
        #expect(beyondBlock.positionOffset == 256)
        #expect(beyondBlock.windowRange == (1..<513))
        #expect(beyondBlock.windowPositionOffset == 257)
    }

    @Test func compactsBeforeGrowingWhenAnInvisiblePrefixExists() {
        let plan = VoxtralRealtimeDecoderKVCacheAppendPlan.make(
            count: 512,
            capacity: 512,
            positionOffset: 30,
            appendCount: 1,
            slidingWindow: 400
        )

        #expect(plan.compactionRange == (112..<512))
        #expect(plan.appendRange == (400..<401))
        #expect(!plan.requiresGrowth)
        #expect(plan.capacity == 512)
        #expect(plan.positionOffset == 142)
        #expect(plan.windowRange == (1..<401))
        #expect(plan.windowPositionOffset == 143)
    }

    @Test func positionOffsetMatchesEagerTrimAcrossLongGeneration() {
        let slidingWindow = 512
        var count = 0
        var capacity = 256
        var positionOffset = 0

        var referenceCount = 0
        var referencePositionOffset = 0

        for _ in 0..<2000 {
            let plan = VoxtralRealtimeDecoderKVCacheAppendPlan.make(
                count: count,
                capacity: capacity,
                positionOffset: positionOffset,
                appendCount: 1,
                slidingWindow: slidingWindow
            )
            count = plan.count
            capacity = plan.capacity
            positionOffset = plan.positionOffset

            referenceCount += 1
            if referenceCount > slidingWindow {
                let trim = referenceCount - slidingWindow
                referenceCount = slidingWindow
                referencePositionOffset += trim
            }

            #expect(plan.windowRange.count == referenceCount)
            #expect(plan.windowPositionOffset == referencePositionOffset)
            #expect(positionOffset + plan.windowRange.lowerBound == referencePositionOffset)
        }
    }
}

/// The cache must give attention what concatenate-then-trim gives: the same rows, in the
/// same order, at the same position offset.
struct VoxtralRealtimeDecoderKVCacheTests {
    @Test func windowsMatchConcatenateAndTrimAcrossGrowthAndCompaction() {
        let width = 2
        // Wider than one 256-row block, so the storage grows before it compacts.
        let slidingWindow = 300
        var nextValue = 0
        func rows(_ count: Int) -> MLXArray {
            let values = (nextValue..<(nextValue + count * width)).map(Float.init)
            nextValue += count * width
            return MLXArray(values, [count, width])
        }

        // A prefill-sized first append, then one row per decoded token.
        let prefill = rows(5)
        let cache = VoxtralRealtimeDecoderKVCache(
            width: width, dtype: .float32, slidingWindow: slidingWindow
        )
        _ = cache.append(keys: prefill, values: prefill + 0.5)
        var referenceKeys = prefill
        var referencePositionOffset = 0
        var largestCapacity = cache.keys.shape[0]

        for _ in 0..<1_500 {
            let newKeys = rows(1)
            let window = cache.append(keys: newKeys, values: newKeys + 0.5)
            largestCapacity = max(largestCapacity, cache.keys.shape[0])

            referenceKeys = MLX.concatenated([referenceKeys, newKeys], axis: 0)
            if referenceKeys.shape[0] > slidingWindow {
                let trim = referenceKeys.shape[0] - slidingWindow
                referenceKeys = referenceKeys[trim...]
                referencePositionOffset += trim
            }

            #expect(window.keys.shape == referenceKeys.shape)
            #expect(MLX.arrayEqual(window.keys, referenceKeys).item(Bool.self))
            #expect(MLX.arrayEqual(window.values, referenceKeys + 0.5).item(Bool.self))
            #expect(window.positionOffset == referencePositionOffset)
        }

        // The storage grew once, from one block to two, and compaction kept it there.
        #expect(largestCapacity == 512)
    }
}
