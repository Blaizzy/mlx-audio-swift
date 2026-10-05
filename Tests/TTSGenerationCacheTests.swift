import Foundation
import MLX
import MLXLMCommon
@testable import MLXAudioTTS
import Testing

@Suite("Reusable TTS cache", .serialized, .timeLimit(.minutes(1)))
struct TTSGenerationCacheTests {
    private final class Owner: Sendable {}
    private enum Failure: Error { case expected }

    @Test func serializesAcrossSuspensionAndPreservesCache() async throws {
        let cache = TTSGenerationCache()
        let owner = Owner()
        let activity = Activity()
        let offsets = try await withThrowingTaskGroup(of: Int.self) { group in
            for _ in 0..<16 {
                group.addTask {
                    try await cache.withCache(model: owner, makeCache: { [KVCacheSimple()] }) { entries in
                        await activity.enter()
                        await Task.yield()
                        let tensor = MLXArray.zeros([1, 1, 1, 4])
                        _ = entries[0].update(keys: tensor, values: tensor)
                        await Task.yield()
                        let offset = entries[0].offset
                        await activity.leave()
                        return offset
                    }
                }
            }
            var offsets = [Int]()
            for try await offset in group { offsets.append(offset) }
            return offsets
        }
        #expect(offsets.sorted() == Array(1...16))
        #expect(await activity.maximum == 1)
    }

    @Test func cancellationKeepsLeaseUntilWorkerExits() async throws {
        let cache = TTSGenerationCache()
        let owner = Owner()
        let started = Signal()
        let release = Signal()
        let activity = Activity()
        let holder = Task {
            try await cache.withCache(model: owner, makeCache: { [KVCacheSimple()] }) { entries in
                await activity.enter()
                await started.signal()
                // Deliberately ignore cancellation until cleanup is complete.
                await release.wait()
                let tensor = MLXArray.zeros([1, 1, 1, 4])
                _ = entries[0].update(keys: tensor, values: tensor)
                await activity.leave()
                try Task.checkCancellation()
            }
        }
        await started.wait()

        let waiterStarted = Signal()
        let cancelledWaiter = Task {
            await waiterStarted.signal()
            try await cache.withCache(model: owner, makeCache: { [KVCacheSimple()] }) { _ in
                _ = Issue.record("Cancelled waiter accessed the cache")
            }
        }
        await waiterStarted.wait()
        cancelledWaiter.cancel()
        // Must finish even while the holder has not released its lease.
        do {
            try await cancelledWaiter.value
            Issue.record("Cancelled waiter succeeded")
        } catch { #expect(error is CancellationError) }

        holder.cancel()
        let next = Task {
            try await cache.withCache(model: owner, makeCache: { [KVCacheSimple()] }) { entries in
                await activity.enter()
                let offset = entries[0].offset
                await activity.leave()
                return offset
            }
        }
        await release.signal()
        do {
            try await holder.value
            Issue.record("Cancelled holder succeeded")
        } catch { #expect(error is CancellationError) }
        #expect(try await next.value == 1)
        #expect(await activity.maximum == 1)
    }

    @Test func adoptsExistingCacheRecoversFromErrorsAndResets() async throws {
        let original = KVCacheSimple()
        let identity = ObjectIdentifier(original)
        let tensor = MLXArray.zeros([1, 1, 1, 4])
        _ = original.update(keys: tensor, values: tensor)
        let cache = TTSGenerationCache([original])
        let owner = Owner()

        do {
            try await cache.withCache(model: owner, makeCache: { [KVCacheSimple()] }) { entries in
                #expect(ObjectIdentifier(entries[0] as! KVCacheSimple) == identity)
                #expect(entries[0].offset == 1)
                throw Failure.expected
            }
            Issue.record("Expected worker error")
        } catch { #expect(error is Failure) }

        let offset = try await cache.withCache(model: owner, makeCache: { [KVCacheSimple()] }) {
            $0[0].offset
        }
        #expect(offset == 1)
        try await cache.reset()
        let resetOffset = try await cache.withCache(model: owner, makeCache: { [KVCacheSimple()] }) {
            $0[0].offset
        }
        #expect(resetOffset == 0)

        do {
            try await cache.withCache(model: Owner(), makeCache: { [KVCacheSimple()] }) { _ in
                _ = Issue.record("Cache accepted a different model")
            }
            Issue.record("Expected model mismatch")
        } catch { #expect(error as? TTSGenerationCache.CacheError == .differentModel) }
    }

    @Test func bothModelsUseContainerForStreamingAndNonStreaming() async throws {
        let (llama, qwen) = tinyModels()
        let cache = TTSGenerationCache()
        try await cache.withCache(model: Owner(), makeCache: { [KVCacheSimple()] }) { _ in }

        // A mismatch must be detected before either model touches tokenizer/SNAC
        // state. Tiny uninitialized models keep this regression entirely local.
        do {
            _ = try await llama.generate(text: "test", cache: cache)
            Issue.record("Llama bypassed the container")
        } catch { #expect(error as? TTSGenerationCache.CacheError == .differentModel) }
        do {
            _ = try await qwen.generate(text: "test", cache: cache)
            Issue.record("Qwen bypassed the container")
        } catch { #expect(error as? TTSGenerationCache.CacheError == .differentModel) }
        do {
            for try await _ in llama.generateStream(text: "test", cache: cache) {}
            Issue.record("Llama stream bypassed the container")
        } catch { #expect(error as? TTSGenerationCache.CacheError == .differentModel) }
        do {
            for try await _ in qwen.generateStream(text: "test", cache: cache) {}
            Issue.record("Qwen stream bypassed the container")
        } catch { #expect(error as? TTSGenerationCache.CacheError == .differentModel) }
    }

    @Test func cachedLlamaAndQwenForwardsMatchDirectCacheUse() async throws {
        let (llama, qwen) = tinyModels()
        try await verifyContinuation(
            model: llama, makeCache: { llama.makeCache() },
            forward: { llama($0, cache: $1) })
        try await verifyContinuation(
            model: qwen, makeCache: { qwen.makeCache() },
            forward: { qwen($0, cache: $1) })
    }

    private func verifyContinuation(
        model: any AnyObject & Sendable,
        makeCache: @Sendable () -> sending [KVCache],
        forward: @Sendable (MLXArray, [KVCache]?) -> MLXArray
    ) async throws {
        let directCache = makeCache()
        let prefix = forward(MLXArray([Int32(1), 2]).reshaped([1, 2]), directCache)
        eval(prefix)
        let expected = forward(MLXArray([Int32(3)]).reshaped([1, 1]), directCache)
        let expectedLast = expected[0, -1].asArray(Float.self)
        let cache = TTSGenerationCache()
        try await cache.withCache(model: model, makeCache: makeCache) { entries in
            let logits = forward(MLXArray([Int32(1), 2]).reshaped([1, 2]), entries)
            eval(logits)
            #expect(entries.allSatisfy { $0.offset == 2 })
        }
        let actual = try await cache.withCache(model: model, makeCache: makeCache) { entries in
            let logits = forward(MLXArray([Int32(3)]).reshaped([1, 1]), entries)
            eval(logits)
            #expect(entries.allSatisfy { $0.offset == 3 })
            return logits[0, -1].asArray(Float.self)
        }
        let maxDifference = zip(actual, expectedLast).map { abs($0 - $1) }.max() ?? 0
        #expect(maxDifference < 1e-4)
    }

    private func tinyModels() -> (LlamaTTSModel, Qwen3Model) {
        let llama = LlamaTTSModel(LlamaTTSConfiguration(
            hiddenSize: 8, hiddenLayers: 1, intermediateSize: 16, attentionHeads: 2,
            rmsNormEps: 1e-6, vocabularySize: 32, kvHeads: 2))
        let qwen = Qwen3Model(Qwen3Configuration(
            hiddenSize: 8, hiddenLayers: 1, intermediateSize: 16, attentionHeads: 2,
            kvHeads: 2, headDim: 4, vocabularySize: 32, rmsNormEps: 1e-6, ropeTheta: 10_000))
        return (llama, qwen)
    }
}

private actor Activity {
    private var active = 0
    private(set) var maximum = 0
    func enter() { active += 1; maximum = max(maximum, active) }
    func leave() { active -= 1 }
}

private actor Signal {
    private var signalled = false
    private var waiters: [CheckedContinuation<Void, Never>] = []
    func wait() async {
        if signalled { return }
        await withCheckedContinuation { waiters.append($0) }
    }
    func signal() {
        signalled = true
        for waiter in waiters { waiter.resume() }
        waiters.removeAll()
    }
}
