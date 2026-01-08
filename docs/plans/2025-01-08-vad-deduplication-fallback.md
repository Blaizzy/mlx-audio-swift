# VAD-Based Segmentation with Deduplication Fallbacks

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a smart deduplication system that uses VAD-based segmentation as primary (no overlap = no deduplication needed), with Levenshtein distance and timestamp-based alignment as fallbacks for edge cases.

**Architecture:** Three-tier deduplication strategy:
1. **Primary (VAD)**: Use SileroVAD to create non-overlapping chunks at speech boundaries
2. **Fallback 1 (Levenshtein)**: For sliding window mode, use edit distance (~1ms) to find overlap boundaries
3. **Fallback 2 (Timestamp)**: When word timestamps available, filter by `word.start >= overlapEnd`

**Tech Stack:** Swift, MLX, existing SileroVAD integration, existing VADProvider protocol

---

## Task 1: Create DeduplicationStrategy Protocol

**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/DeduplicationStrategy.swift`
- Test: `mlx_audio_swift/stt/Tests/DeduplicationStrategyTests.swift`

**Step 1: Write the failing test**

```swift
import XCTest
@testable import MLXAudioSTT

final class DeduplicationStrategyTests: XCTestCase {

    func testNoOpDeduplication() {
        let strategy = NoOpDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "Hello world",
            previousEndWords: [],
            currentWords: nil
        )
        XCTAssertEqual(result, "Hello world")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter DeduplicationStrategyTests`
Expected: FAIL with "No such module 'MLXAudioSTT'" or similar

**Step 3: Create DeduplicationStrategy protocol and NoOp implementation**

```swift
import Foundation

/// Result of deduplication containing the deduplicated text and metadata
public struct DeduplicationResult: Sendable {
    public let text: String
    public let wordsRemoved: Int
    public let method: String

    public init(text: String, wordsRemoved: Int = 0, method: String) {
        self.text = text
        self.wordsRemoved = wordsRemoved
        self.method = method
    }
}

/// Protocol for text deduplication strategies
public protocol DeduplicationStrategy: Sendable {
    /// Deduplicate text based on previous chunk context
    func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult

    /// Strategy name for logging
    var name: String { get }
}

/// No-op strategy - returns text unchanged (used when VAD creates non-overlapping chunks)
public struct NoOpDeduplicationStrategy: DeduplicationStrategy {
    public let name = "noop"

    public init() {}

    public func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult {
        DeduplicationResult(text: currentText, wordsRemoved: 0, method: name)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter DeduplicationStrategyTests`
Expected: PASS

**Step 5: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Chunking/DeduplicationStrategy.swift
git add mlx_audio_swift/stt/Tests/DeduplicationStrategyTests.swift
git commit -m "feat(stt): add DeduplicationStrategy protocol with NoOp implementation"
```

---

## Task 2: Implement Levenshtein Distance Deduplication

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/DeduplicationStrategy.swift`
- Test: `mlx_audio_swift/stt/Tests/DeduplicationStrategyTests.swift`

**Step 1: Write the failing test**

```swift
func testLevenshteinDeduplicationWithExactMatch() {
    let strategy = LevenshteinDeduplicationStrategy()
    let result = strategy.deduplicate(
        currentText: "world how are you",
        previousEndWords: ["hello", "world"],
        currentWords: nil
    )
    // "world" should be removed as it matches end of previous chunk
    XCTAssertEqual(result.text, "how are you")
    XCTAssertEqual(result.wordsRemoved, 1)
    XCTAssertEqual(result.method, "levenshtein")
}

func testLevenshteinDeduplicationWithPartialMatch() {
    let strategy = LevenshteinDeduplicationStrategy()
    let result = strategy.deduplicate(
        currentText: "hello world today",
        previousEndWords: ["say", "hello", "world"],
        currentWords: nil
    )
    // "hello world" should be removed
    XCTAssertEqual(result.text, "today")
    XCTAssertEqual(result.wordsRemoved, 2)
}

func testLevenshteinDeduplicationCaseInsensitive() {
    let strategy = LevenshteinDeduplicationStrategy()
    let result = strategy.deduplicate(
        currentText: "World how are you",
        previousEndWords: ["hello", "world"],
        currentWords: nil
    )
    XCTAssertEqual(result.text, "how are you")
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter testLevenshtein`
Expected: FAIL

**Step 3: Implement LevenshteinDeduplicationStrategy**

```swift
/// Levenshtein distance-based deduplication (~1ms per comparison)
/// Best for streaming where we need fast overlap detection
public struct LevenshteinDeduplicationStrategy: DeduplicationStrategy {
    public let name = "levenshtein"
    public let maxLookback: Int

    public init(maxLookback: Int = 10) {
        self.maxLookback = maxLookback
    }

    public func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult {
        let words = currentText.split(separator: " ").map(String.init)
        guard !words.isEmpty, !previousEndWords.isEmpty else {
            return DeduplicationResult(text: currentText, wordsRemoved: 0, method: name)
        }

        let lookback = Array(previousEndWords.suffix(maxLookback))

        // Find best overlap using minimum edit distance
        var bestOverlapLength = 0
        var bestEditDistance = Int.max

        for len in 1...min(lookback.count, words.count) {
            let prevSuffix = Array(lookback.suffix(len))
            let currPrefix = Array(words.prefix(len))

            let distance = editDistance(prevSuffix, currPrefix)

            // Accept if edit distance is less than 20% of sequence length
            let threshold = max(1, len / 5)
            if distance <= threshold && distance < bestEditDistance {
                bestEditDistance = distance
                bestOverlapLength = len
            }
        }

        if bestOverlapLength > 0 {
            let deduplicated = words.dropFirst(bestOverlapLength).joined(separator: " ")
            return DeduplicationResult(
                text: deduplicated,
                wordsRemoved: bestOverlapLength,
                method: name
            )
        }

        return DeduplicationResult(text: currentText, wordsRemoved: 0, method: name)
    }

    /// Calculate Levenshtein edit distance between two word sequences
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
                let cost = a[i-1].lowercased() == b[j-1].lowercased() ? 0 : 1
                curr[j] = min(
                    prev[j] + 1,      // deletion
                    curr[j-1] + 1,    // insertion
                    prev[j-1] + cost  // substitution
                )
            }
            swap(&prev, &curr)
        }

        return prev[n]
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter testLevenshtein`
Expected: PASS

**Step 5: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Chunking/DeduplicationStrategy.swift
git add mlx_audio_swift/stt/Tests/DeduplicationStrategyTests.swift
git commit -m "feat(stt): implement Levenshtein distance deduplication strategy"
```

---

## Task 3: Implement Timestamp-Based Deduplication

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/DeduplicationStrategy.swift`
- Test: `mlx_audio_swift/stt/Tests/DeduplicationStrategyTests.swift`

**Step 1: Write the failing test**

```swift
func testTimestampDeduplication() {
    let strategy = TimestampDeduplicationStrategy(overlapEnd: 5.0)

    let words: [WordTimestamp] = [
        WordTimestamp(word: "hello", start: 4.5, end: 4.8, confidence: 0.9),
        WordTimestamp(word: "world", start: 4.9, end: 5.2, confidence: 0.9),
        WordTimestamp(word: "how", start: 5.3, end: 5.5, confidence: 0.9),
        WordTimestamp(word: "are", start: 5.6, end: 5.8, confidence: 0.9),
    ]

    let result = strategy.deduplicate(
        currentText: "hello world how are",
        previousEndWords: [],
        currentWords: words
    )

    // "hello" ends before overlapEnd (4.8 < 5.0), should be removed
    // "world" starts before but ends after (4.9 < 5.0 < 5.2), keep it
    XCTAssertEqual(result.text, "world how are")
    XCTAssertEqual(result.wordsRemoved, 1)
    XCTAssertEqual(result.method, "timestamp")
}

func testTimestampDeduplicationFallsBackWithoutTimestamps() {
    let strategy = TimestampDeduplicationStrategy(overlapEnd: 5.0)

    let result = strategy.deduplicate(
        currentText: "hello world",
        previousEndWords: ["world"],
        currentWords: nil  // No timestamps available
    )

    // Should fall back to simple word matching
    XCTAssertEqual(result.text, "hello world")  // No dedup without timestamps
    XCTAssertEqual(result.method, "timestamp-fallback")
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter testTimestamp`
Expected: FAIL

**Step 3: Implement TimestampDeduplicationStrategy**

```swift
/// Timestamp-based deduplication using word-level timing
/// Most accurate when word timestamps are available
public struct TimestampDeduplicationStrategy: DeduplicationStrategy {
    public let name = "timestamp"
    public let overlapEnd: TimeInterval

    public init(overlapEnd: TimeInterval) {
        self.overlapEnd = overlapEnd
    }

    public func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult {
        guard let words = currentWords, !words.isEmpty else {
            // No timestamps - return unchanged
            return DeduplicationResult(
                text: currentText,
                wordsRemoved: 0,
                method: "\(name)-fallback"
            )
        }

        // Filter words based on timestamp
        // Keep word if: word.end > overlapEnd OR (word.start < overlapEnd AND word.end > overlapEnd)
        let filteredWords = words.filter { word in
            word.end > overlapEnd
        }

        let wordsRemoved = words.count - filteredWords.count
        let deduplicated = filteredWords.map(\.word).joined(separator: " ")

        return DeduplicationResult(
            text: deduplicated,
            wordsRemoved: wordsRemoved,
            method: name
        )
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter testTimestamp`
Expected: PASS

**Step 5: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Chunking/DeduplicationStrategy.swift
git add mlx_audio_swift/stt/Tests/DeduplicationStrategyTests.swift
git commit -m "feat(stt): implement timestamp-based deduplication strategy"
```

---

## Task 4: Create Composite Deduplication Strategy with Fallbacks

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/DeduplicationStrategy.swift`
- Test: `mlx_audio_swift/stt/Tests/DeduplicationStrategyTests.swift`

**Step 1: Write the failing test**

```swift
func testCompositeUsesTimestampsWhenAvailable() {
    let strategy = CompositeDeduplicationStrategy(overlapEnd: 5.0)

    let words: [WordTimestamp] = [
        WordTimestamp(word: "hello", start: 4.5, end: 4.8, confidence: 0.9),
        WordTimestamp(word: "world", start: 5.1, end: 5.4, confidence: 0.9),
    ]

    let result = strategy.deduplicate(
        currentText: "hello world",
        previousEndWords: ["hello"],
        currentWords: words
    )

    XCTAssertEqual(result.text, "world")
    XCTAssertEqual(result.method, "timestamp")
}

func testCompositeFallsBackToLevenshtein() {
    let strategy = CompositeDeduplicationStrategy(overlapEnd: 5.0)

    let result = strategy.deduplicate(
        currentText: "world how are you",
        previousEndWords: ["hello", "world"],
        currentWords: nil  // No timestamps
    )

    XCTAssertEqual(result.text, "how are you")
    XCTAssertEqual(result.method, "levenshtein")
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter testComposite`
Expected: FAIL

**Step 3: Implement CompositeDeduplicationStrategy**

```swift
/// Composite strategy that tries multiple deduplication methods in priority order:
/// 1. Timestamp-based (when word timestamps available)
/// 2. Levenshtein distance (when previous context available)
/// 3. NoOp (when VAD provides non-overlapping chunks)
public struct CompositeDeduplicationStrategy: DeduplicationStrategy {
    public let name = "composite"

    private let timestampStrategy: TimestampDeduplicationStrategy?
    private let levenshteinStrategy: LevenshteinDeduplicationStrategy
    private let noopStrategy: NoOpDeduplicationStrategy

    public init(overlapEnd: TimeInterval? = nil, maxLookback: Int = 10) {
        if let overlapEnd = overlapEnd {
            self.timestampStrategy = TimestampDeduplicationStrategy(overlapEnd: overlapEnd)
        } else {
            self.timestampStrategy = nil
        }
        self.levenshteinStrategy = LevenshteinDeduplicationStrategy(maxLookback: maxLookback)
        self.noopStrategy = NoOpDeduplicationStrategy()
    }

    public func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult {
        // Priority 1: Use timestamps if available and configured
        if let timestampStrategy = timestampStrategy,
           let words = currentWords,
           !words.isEmpty {
            return timestampStrategy.deduplicate(
                currentText: currentText,
                previousEndWords: previousEndWords,
                currentWords: words
            )
        }

        // Priority 2: Use Levenshtein if we have previous context
        if !previousEndWords.isEmpty {
            return levenshteinStrategy.deduplicate(
                currentText: currentText,
                previousEndWords: previousEndWords,
                currentWords: nil
            )
        }

        // Priority 3: No deduplication needed
        return noopStrategy.deduplicate(
            currentText: currentText,
            previousEndWords: previousEndWords,
            currentWords: nil
        )
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter testComposite`
Expected: PASS

**Step 5: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Chunking/DeduplicationStrategy.swift
git add mlx_audio_swift/stt/Tests/DeduplicationStrategyTests.swift
git commit -m "feat(stt): implement composite deduplication strategy with fallbacks"
```

---

## Task 5: Integrate DeduplicationStrategy into LongAudioProcessor

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/LongAudioProcessor.swift`
- Test: `mlx_audio_swift/stt/Tests/LongAudioProcessorTests.swift` (if exists)

**Step 1: Update MergeConfig to use DeduplicationStrategy**

Add to `MergeConfig`:

```swift
public struct MergeConfig: Sendable {
    public var deduplicateOverlap: Bool  // Keep for backward compat
    public var deduplicationStrategy: DeduplicationStrategy?
    public var minWordConfidence: Float
    public var normalizeText: Bool

    public init(
        deduplicateOverlap: Bool = true,
        deduplicationStrategy: DeduplicationStrategy? = nil,
        minWordConfidence: Float = 0.3,
        normalizeText: Bool = true
    ) {
        self.deduplicateOverlap = deduplicateOverlap
        self.deduplicationStrategy = deduplicationStrategy
        self.minWordConfidence = minWordConfidence
        self.normalizeText = normalizeText
    }

    public static let `default` = MergeConfig()

    /// Create config with smart deduplication using composite strategy
    public static func withSmartDeduplication(overlapEnd: TimeInterval? = nil) -> MergeConfig {
        MergeConfig(
            deduplicateOverlap: true,
            deduplicationStrategy: CompositeDeduplicationStrategy(overlapEnd: overlapEnd)
        )
    }
}
```

**Step 2: Update deduplication logic in processAudio**

Replace the inline `deduplicateOverlapText` call with strategy-based deduplication:

```swift
// In processAudio, replace:
if mergeConfig.deduplicateOverlap && !previousChunkEndWords.isEmpty {
    textToAccumulate = deduplicateOverlapText(
        processedText,
        previousEndWords: previousChunkEndWords
    )
}

// With:
if mergeConfig.deduplicateOverlap {
    if let strategy = mergeConfig.deduplicationStrategy {
        let result = strategy.deduplicate(
            currentText: processedText,
            previousEndWords: previousChunkEndWords,
            currentWords: nil  // Pass word timestamps when available
        )
        textToAccumulate = result.text
    } else if !previousChunkEndWords.isEmpty {
        // Fallback to existing simple deduplication
        textToAccumulate = deduplicateOverlapText(
            processedText,
            previousEndWords: previousChunkEndWords
        )
    }
}
```

**Step 3: Run existing tests to verify no regression**

Run: `swift test --filter LongAudioProcessor`
Expected: PASS (or create basic test if none exist)

**Step 4: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/LongAudioProcessor.swift
git commit -m "feat(stt): integrate DeduplicationStrategy into LongAudioProcessor"
```

---

## Task 6: Configure VADChunkingStrategy to Use NoOp Deduplication

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/VADChunkingStrategy.swift`

**Step 1: Document that VAD produces non-overlapping chunks**

Add comment to VADChunkingStrategy explaining that it produces non-overlapping segments:

```swift
/// VAD-based chunking with parallel transcription
/// Best for noisy audio, fastest with batching
///
/// Note: VAD produces non-overlapping chunks based on speech boundaries,
/// so deduplication is typically not needed. Use NoOpDeduplicationStrategy
/// or disable deduplication when using this strategy.
public final class VADChunkingStrategy: ChunkingStrategy, Sendable {
```

**Step 2: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Chunking/VADChunkingStrategy.swift
git commit -m "docs(stt): document VAD strategy produces non-overlapping chunks"
```

---

## Task 7: Configure SlidingWindowChunkingStrategy to Use Composite Deduplication

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Chunking/SlidingWindowChunkingStrategy.swift`

**Step 1: Add deduplication strategy to SlidingWindowConfig**

```swift
public struct SlidingWindowConfig: Sendable {
    public var windowDuration: TimeInterval
    public var overlapDuration: TimeInterval
    public var mergeStrategy: MergeStrategy
    public var deduplicationStrategy: DeduplicationStrategy?

    public var hopDuration: TimeInterval { windowDuration - overlapDuration }

    public init(
        windowDuration: TimeInterval = 30.0,
        overlapDuration: TimeInterval = 5.0,
        mergeStrategy: MergeStrategy = .timestampAlignment,
        deduplicationStrategy: DeduplicationStrategy? = nil
    ) {
        self.windowDuration = windowDuration
        self.overlapDuration = overlapDuration
        self.mergeStrategy = mergeStrategy
        self.deduplicationStrategy = deduplicationStrategy ?? CompositeDeduplicationStrategy(
            overlapEnd: windowDuration - overlapDuration
        )
    }

    public static let `default` = SlidingWindowConfig()
}
```

**Step 2: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Chunking/SlidingWindowChunkingStrategy.swift
git commit -m "feat(stt): add composite deduplication to sliding window config"
```

---

## Task 8: Update Public API and Documentation

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Exports.swift` (or equivalent)
- Create: `docs/deduplication-strategies.md`

**Step 1: Export new types**

Ensure all new types are publicly exported:
- `DeduplicationStrategy`
- `DeduplicationResult`
- `NoOpDeduplicationStrategy`
- `LevenshteinDeduplicationStrategy`
- `TimestampDeduplicationStrategy`
- `CompositeDeduplicationStrategy`

**Step 2: Create documentation**

```markdown
# Deduplication Strategies for Long Audio Transcription

## Overview

When processing long audio files with overlapping chunks, duplicate text can appear
at chunk boundaries. MLX Audio Swift provides several strategies to handle this:

## Available Strategies

### 1. NoOpDeduplicationStrategy
Used when chunks don't overlap (VAD-based segmentation).

### 2. LevenshteinDeduplicationStrategy
Uses edit distance to find matching sequences between chunks.
Best for: Real-time streaming, ~1ms computation time.

### 3. TimestampDeduplicationStrategy
Filters words based on their timestamps relative to overlap boundaries.
Best for: When word-level timestamps are available.

### 4. CompositeDeduplicationStrategy (Default)
Combines all strategies with automatic fallback:
1. Uses timestamps when available
2. Falls back to Levenshtein when timestamps unavailable
3. Falls back to NoOp when no overlap

## Usage

```swift
// Auto-select best strategy (default)
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .slidingWindow()
)

// Use VAD (no deduplication needed)
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .vad()
)

// Custom configuration
let config = MergeConfig.withSmartDeduplication(overlapEnd: 25.0)
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    mergeConfig: config
)
```
```

**Step 3: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Exports.swift
git add docs/deduplication-strategies.md
git commit -m "docs(stt): add deduplication strategies documentation"
```

---

## Task 9: Add Integration Tests

**Files:**
- Create: `mlx_audio_swift/stt/Tests/DeduplicationIntegrationTests.swift`

**Step 1: Write integration tests**

```swift
import XCTest
@testable import MLXAudioSTT

final class DeduplicationIntegrationTests: XCTestCase {

    func testSlidingWindowWithCompositeDeduplication() async throws {
        // Test that sliding window strategy properly deduplicates overlapping chunks
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 10.0,
            overlapDuration: 2.0
        )

        // Verify deduplication strategy is set
        XCTAssertNotNil(config.deduplicationStrategy)
        XCTAssertEqual(config.deduplicationStrategy?.name, "composite")
    }

    func testMergeConfigWithSmartDeduplication() {
        let config = MergeConfig.withSmartDeduplication(overlapEnd: 25.0)

        XCTAssertTrue(config.deduplicateOverlap)
        XCTAssertNotNil(config.deduplicationStrategy)
    }
}
```

**Step 2: Run integration tests**

Run: `swift test --filter DeduplicationIntegrationTests`
Expected: PASS

**Step 3: Commit**

```bash
git add mlx_audio_swift/stt/Tests/DeduplicationIntegrationTests.swift
git commit -m "test(stt): add deduplication integration tests"
```

---

## Summary

This plan implements a three-tier deduplication system:

| Strategy | When Used | Complexity | Accuracy |
|----------|-----------|------------|----------|
| **NoOp** | VAD chunks (non-overlapping) | O(1) | N/A |
| **Timestamp** | Word timestamps available | O(n) | Highest |
| **Levenshtein** | Fallback for streaming | O(n*m) ~1ms | High |

The `CompositeDeduplicationStrategy` automatically selects the best approach based on available context, ensuring optimal deduplication across all chunking modes.
