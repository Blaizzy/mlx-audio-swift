import Foundation

/// Result from deduplication processing
public struct DeduplicationResult: Sendable, Equatable {
    /// The deduplicated text
    public let text: String
    /// Count of removed words
    public let wordsRemoved: Int
    /// Which strategy was used
    public let method: String

    public init(text: String, wordsRemoved: Int, method: String) {
        self.text = text
        self.wordsRemoved = wordsRemoved
        self.method = method
    }
}

/// Protocol for deduplication strategies that handle overlapping audio chunks
public protocol DeduplicationStrategy: Sendable {
    /// Deduplicate text from overlapping chunks
    /// - Parameters:
    ///   - currentText: The text from the current chunk
    ///   - previousEndWords: Words from the end of the previous chunk (for overlap detection)
    ///   - currentWords: Optional word timestamps from the current chunk
    /// - Returns: Deduplicated result
    func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult

    /// Strategy identifier for logging/debugging
    var name: String { get }
}

/// No-op deduplication strategy that returns text unchanged
/// Used when VAD creates non-overlapping chunks
public struct NoOpDeduplicationStrategy: DeduplicationStrategy {
    public var name: String { "noop" }

    public init() {}

    public func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult {
        DeduplicationResult(
            text: currentText,
            wordsRemoved: 0,
            method: name
        )
    }
}
