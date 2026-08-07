import Synchronization

/// A `Sendable` reference-type lock wrapping a noncopyable `Mutex`.
///
/// The streaming sessions share a single lock instance across escaping `Task`s (copying the
/// reference out of `self` and capturing it). A bare `Mutex` is noncopyable and can't be shared
/// that way; wrapping it in a `final class` restores reference semantics while keeping `Mutex`'s
/// guarantees. (Mirrors the `LockedState` pattern used in swift-foundation.)
final class LockedState<State: Sendable>: Sendable {
    private let mutex: Mutex<State>

    init(_ initialState: State) {
        self.mutex = Mutex(initialState)
    }

    // Mirror `Mutex.withLock`'s `sending` signature so the closure forwards without a conversion.
    @discardableResult
    func withLock<Result, E>(_ body: (inout sending State) throws(E) -> sending Result) throws(E) -> sending Result where E: Error {
        try mutex.withLock(body)
    }
}
