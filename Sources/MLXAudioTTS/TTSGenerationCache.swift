import Foundation
import MLX
import MLXLMCommon

/// An opaque, reusable KV cache for Llama and Qwen TTS generation.
///
/// Generations using the same container run sequentially, including across
/// suspension points. A container binds to and retains its first model instance. It does not
/// serialize calls using other containers or calls to the model's low-level API.
public final class TTSGenerationCache: Sendable {
    public enum CacheError: Error, Equatable {
        case differentModel
    }

    private let storage: Storage

    /// Allocate the model's cache lazily on the first generation.
    public init() {
        storage = Storage(nil)
    }

    /// Transfer an existing cache into this container. Do not access its entries
    /// or tensor aliases afterward. Use it with the model that created it.
    public init(_ cache: sending [KVCache]) {
        for entry in cache { eval(entry) }
        storage = Storage(cache)
    }

    /// Discard cached context after any active generation finishes. The next
    /// generation starts with an empty cache for the same model instance.
    /// Failed or cancelled generations can leave a partial prefix; reset before
    /// starting an unrelated request.
    public func reset() async throws {
        try await storage.reset()
    }

    // The closure is internal so callers cannot retain or expose raw cache
    // objects. Generation must finish all cache work before this closure returns.
    func withCache<R: Sendable>(
        model: any AnyObject & Sendable,
        makeCache: @Sendable () -> sending [KVCache],
        _ body: @Sendable ([KVCache]) async throws -> R
    ) async throws -> R {
        try await storage.withCache(model: model, makeCache: makeCache, body)
    }

    // All access to these non-Sendable objects is protected for the complete
    // async operation by the gate, rather than by actor isolation alone.
    private final class Storage: @unchecked Sendable {
        private var cache: [KVCache]?
        private var model: AnyObject?
        private let gate = GenerationGate()

        init(_ cache: sending [KVCache]?) { self.cache = cache }

        func withCache<R: Sendable>(
            model: any AnyObject & Sendable,
            makeCache: @Sendable () -> sending [KVCache],
            _ body: @Sendable ([KVCache]) async throws -> R
        ) async throws -> R {
            try await gate.withLock {
                if let owner = self.model, owner !== model {
                    throw CacheError.differentModel
                }
                self.model = model
                if self.cache == nil { self.cache = makeCache() }
                let cache = self.cache!
                // Materialize any final lazy cache updates on both success and
                // failure before another task can acquire the container.
                defer { for entry in cache { eval(entry) } }
                return try await body(cache)
            }
        }

        func reset() async throws {
            try await gate.withLock { self.cache = nil }
        }
    }
}

/// An async mutex that keeps its lease across awaits and removes cancelled
/// waiters. Cancellation of the holder never releases an active worker's lease;
/// the worker must actually return or throw first.
private actor GenerationGate {
    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, Error>
    }

    private var locked = false
    private var waiters: [Waiter] = []

    func withLock<R: Sendable>(
        _ body: @Sendable () async throws -> R
    ) async throws -> R {
        try await acquire()
        defer { release() }
        try Task.checkCancellation()
        return try await body()
    }

    private func acquire() async throws {
        try Task.checkCancellation()
        if !locked {
            locked = true
            return
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                if Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                } else {
                    waiters.append(Waiter(id: id, continuation: continuation))
                }
            }
        } onCancel: {
            Task { await self.cancel(id) }
        }
    }

    private func cancel(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: index).continuation.resume(throwing: CancellationError())
    }

    private func release() {
        if waiters.isEmpty {
            locked = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }
}
