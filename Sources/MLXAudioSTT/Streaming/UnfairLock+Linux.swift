// Linux shim for `os.OSAllocatedUnfairLock`, which the streaming sessions use to guard shared
// state across escaping tasks. Apple platforms use the real `os` type; on Linux we provide a
// `Mutex`-backed reference type with the same `init(initialState:)` / `withLock` surface, so the
// call sites compile unchanged and no macOS/iOS deployment-target bump is needed.
#if !canImport(os)
import Synchronization

final class OSAllocatedUnfairLock<State: Sendable>: @unchecked Sendable {
    private let mutex: Mutex<State>

    init(initialState: State) {
        self.mutex = Mutex(initialState)
    }

    // Mirror Mutex's `sending` closure signature so the body forwards without a conversion.
    @discardableResult
    func withLock<Result, E>(
        _ body: (inout sending State) throws(E) -> sending Result
    ) throws(E) -> sending Result where E: Error {
        try mutex.withLock(body)
    }
}
#endif
