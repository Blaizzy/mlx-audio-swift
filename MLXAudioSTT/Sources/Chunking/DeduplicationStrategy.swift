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

/// Levenshtein distance-based deduplication strategy
/// Uses edit distance to detect overlapping text between chunks
/// Suitable when word timestamps are unavailable (~1ms computation time)
public struct LevenshteinDeduplicationStrategy: DeduplicationStrategy {
    public var name: String { "levenshtein" }

    private let maxLookback: Int

    public init(maxLookback: Int = 10) {
        self.maxLookback = maxLookback
    }

    public func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult {
        guard !currentText.isEmpty, !previousEndWords.isEmpty else {
            return DeduplicationResult(text: currentText, wordsRemoved: 0, method: name)
        }

        let currentWords = currentText.split(separator: " ").map(String.init)
        guard !currentWords.isEmpty else {
            return DeduplicationResult(text: currentText, wordsRemoved: 0, method: name)
        }

        let lookbackWords = Array(previousEndWords.suffix(maxLookback))

        var bestMatchLength = 0

        for prefixLength in 1...min(currentWords.count, lookbackWords.count) {
            let currentPrefix = Array(currentWords.prefix(prefixLength))
            let previousSuffix = Array(lookbackWords.suffix(prefixLength))

            let distance = editDistance(previousSuffix, currentPrefix)
            // 20% threshold: for sequences < 5 words, require exact match (distance 0)
            // unless words are character-similar
            let threshold = prefixLength / 5

            if distance <= threshold {
                bestMatchLength = prefixLength
            } else if distance == 1 && prefixLength <= 2 {
                // For short sequences with 1 word difference, check character similarity
                if areWordsSimilar(previousSuffix, currentPrefix) {
                    bestMatchLength = prefixLength
                }
            }
        }

        if bestMatchLength > 0 {
            let remainingWords = Array(currentWords.dropFirst(bestMatchLength))
            let deduplicatedText = remainingWords.joined(separator: " ")
            return DeduplicationResult(
                text: deduplicatedText,
                wordsRemoved: bestMatchLength,
                method: name
            )
        }

        return DeduplicationResult(text: currentText, wordsRemoved: 0, method: name)
    }

    private func editDistance(_ a: [String], _ b: [String]) -> Int {
        let m = a.count
        let n = b.count
        if m == 0 { return n }
        if n == 0 { return m }

        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)

        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                let cost = a[i - 1].lowercased() == b[j - 1].lowercased() ? 0 : 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            }
            swap(&prev, &curr)
        }
        return prev[n]
    }

    private func areWordsSimilar(_ a: [String], _ b: [String]) -> Bool {
        guard a.count == b.count else { return false }
        for (wordA, wordB) in zip(a, b) {
            if !isWordSimilar(wordA, wordB) {
                return false
            }
        }
        return true
    }

    private func isWordSimilar(_ a: String, _ b: String) -> Bool {
        let aLower = a.lowercased()
        let bLower = b.lowercased()
        if aLower == bLower { return true }

        let distance = charEditDistance(Array(aLower), Array(bLower))
        let maxLen = max(aLower.count, bLower.count)
        // Allow up to 20% character difference
        let threshold = max(1, maxLen / 5)
        return distance <= threshold
    }

    private func charEditDistance(_ a: [Character], _ b: [Character]) -> Int {
        let m = a.count
        let n = b.count
        if m == 0 { return n }
        if n == 0 { return m }

        var prev = Array(0...n)
        var curr = [Int](repeating: 0, count: n + 1)

        for i in 1...m {
            curr[0] = i
            for j in 1...n {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            }
            swap(&prev, &curr)
        }
        return prev[n]
    }
}

/// Timestamp-based deduplication strategy
/// Filters words based on their end timestamp relative to the overlap boundary
public struct TimestampDeduplicationStrategy: DeduplicationStrategy {
    public var name: String { "timestamp" }

    private let overlapEnd: TimeInterval

    public init(overlapEnd: TimeInterval) {
        self.overlapEnd = overlapEnd
    }

    public func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult {
        guard let words = currentWords, !words.isEmpty else {
            return DeduplicationResult(
                text: currentText,
                wordsRemoved: 0,
                method: "timestamp-fallback"
            )
        }

        let filteredWords = words.filter { $0.end > overlapEnd }
        let wordsRemoved = words.count - filteredWords.count
        let deduplicatedText = filteredWords.map(\.word).joined(separator: " ")

        return DeduplicationResult(
            text: deduplicatedText,
            wordsRemoved: wordsRemoved,
            method: name
        )
    }
}

/// Composite deduplication strategy that selects the best approach based on available data
/// Priority order:
/// 1. TimestampDeduplicationStrategy if timestamps available AND overlapEnd configured
/// 2. LevenshteinDeduplicationStrategy if previousEndWords available
/// 3. NoOpDeduplicationStrategy otherwise
public struct CompositeDeduplicationStrategy: DeduplicationStrategy {
    public var name: String { "composite" }

    private let overlapEnd: TimeInterval?
    private let maxLookback: Int

    public init(overlapEnd: TimeInterval? = nil, maxLookback: Int = 10) {
        self.overlapEnd = overlapEnd
        self.maxLookback = maxLookback
    }

    public func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult {
        // Priority 1: Use timestamps if available AND overlapEnd configured
        if let overlapEnd = overlapEnd,
           let words = currentWords,
           !words.isEmpty {
            let timestampStrategy = TimestampDeduplicationStrategy(overlapEnd: overlapEnd)
            return timestampStrategy.deduplicate(
                currentText: currentText,
                previousEndWords: previousEndWords,
                currentWords: words
            )
        }

        // Priority 2: Use Levenshtein if previousEndWords available
        if !previousEndWords.isEmpty {
            let levenshteinStrategy = LevenshteinDeduplicationStrategy(maxLookback: maxLookback)
            return levenshteinStrategy.deduplicate(
                currentText: currentText,
                previousEndWords: previousEndWords,
                currentWords: nil
            )
        }

        // Priority 3: No deduplication needed
        let noopStrategy = NoOpDeduplicationStrategy()
        return noopStrategy.deduplicate(
            currentText: currentText,
            previousEndWords: previousEndWords,
            currentWords: currentWords
        )
    }
}
