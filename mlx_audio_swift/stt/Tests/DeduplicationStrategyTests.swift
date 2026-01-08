import Foundation
import Testing
@testable import MLXAudioSTT

struct DeduplicationStrategyTests {

    // MARK: - NoOpDeduplicationStrategy Tests

    @Test func testNoOpDeduplication() {
        let strategy = NoOpDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "Hello world",
            previousEndWords: [],
            currentWords: nil
        )
        #expect(result.text == "Hello world")
        #expect(result.wordsRemoved == 0)
        #expect(result.method == "noop")
    }

    @Test func testNoOpDeduplicationWithPreviousWords() {
        let strategy = NoOpDeduplicationStrategy()
        let previousWords = ["previous"]
        let result = strategy.deduplicate(
            currentText: "Hello world",
            previousEndWords: previousWords,
            currentWords: nil
        )
        #expect(result.text == "Hello world")
        #expect(result.wordsRemoved == 0)
        #expect(result.method == "noop")
    }

    @Test func testNoOpDeduplicationWithCurrentWords() {
        let strategy = NoOpDeduplicationStrategy()
        let currentWords = [
            WordTimestamp(word: "Hello", start: 0.0, end: 0.3, confidence: 0.9),
            WordTimestamp(word: "world", start: 0.4, end: 0.8, confidence: 0.9)
        ]
        let result = strategy.deduplicate(
            currentText: "Hello world",
            previousEndWords: [],
            currentWords: currentWords
        )
        #expect(result.text == "Hello world")
        #expect(result.wordsRemoved == 0)
        #expect(result.method == "noop")
    }

    @Test func testNoOpStrategyName() {
        let strategy = NoOpDeduplicationStrategy()
        #expect(strategy.name == "noop")
    }

    @Test func testNoOpDeduplicationWithEmptyText() {
        let strategy = NoOpDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "",
            previousEndWords: [],
            currentWords: nil
        )
        #expect(result.text == "")
        #expect(result.wordsRemoved == 0)
        #expect(result.method == "noop")
    }

    // MARK: - DeduplicationResult Tests

    @Test func testDeduplicationResultEquatable() {
        let result1 = DeduplicationResult(text: "Hello", wordsRemoved: 1, method: "test")
        let result2 = DeduplicationResult(text: "Hello", wordsRemoved: 1, method: "test")
        let result3 = DeduplicationResult(text: "World", wordsRemoved: 1, method: "test")

        #expect(result1 == result2)
        #expect(result1 != result3)
    }

    // MARK: - LevenshteinDeduplicationStrategy Tests

    @Test func testLevenshteinDeduplicationWithExactMatch() {
        let strategy = LevenshteinDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "world how are you",
            previousEndWords: ["hello", "world"],
            currentWords: nil
        )
        // "world" should be removed as it matches end of previous chunk
        #expect(result.text == "how are you")
        #expect(result.wordsRemoved == 1)
        #expect(result.method == "levenshtein")
    }

    @Test func testLevenshteinDeduplicationWithPartialMatch() {
        let strategy = LevenshteinDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "hello world today",
            previousEndWords: ["say", "hello", "world"],
            currentWords: nil
        )
        // "hello world" should be removed
        #expect(result.text == "today")
        #expect(result.wordsRemoved == 2)
    }

    @Test func testLevenshteinDeduplicationCaseInsensitive() {
        let strategy = LevenshteinDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "World how are you",
            previousEndWords: ["hello", "world"],
            currentWords: nil
        )
        #expect(result.text == "how are you")
    }

    @Test func testLevenshteinDeduplicationNoMatch() {
        let strategy = LevenshteinDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "completely different text",
            previousEndWords: ["hello", "world"],
            currentWords: nil
        )
        #expect(result.text == "completely different text")
        #expect(result.wordsRemoved == 0)
    }

    @Test func testLevenshteinDeduplicationEmptyPreviousWords() {
        let strategy = LevenshteinDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: [],
            currentWords: nil
        )
        #expect(result.text == "hello world")
        #expect(result.wordsRemoved == 0)
    }

    @Test func testLevenshteinDeduplicationEmptyCurrentText() {
        let strategy = LevenshteinDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "",
            previousEndWords: ["hello", "world"],
            currentWords: nil
        )
        #expect(result.text == "")
        #expect(result.wordsRemoved == 0)
    }

    @Test func testLevenshteinStrategyName() {
        let strategy = LevenshteinDeduplicationStrategy()
        #expect(strategy.name == "levenshtein")
    }

    @Test func testLevenshteinDeduplicationWithMaxLookback() {
        let strategy = LevenshteinDeduplicationStrategy(maxLookback: 2)
        let result = strategy.deduplicate(
            currentText: "world today is nice",
            previousEndWords: ["one", "two", "three", "hello", "world"],
            currentWords: nil
        )
        // Only last 2 words ["hello", "world"] should be considered
        #expect(result.text == "today is nice")
        #expect(result.wordsRemoved == 1)
    }

    @Test func testLevenshteinDeduplicationWithFuzzyMatch() {
        let strategy = LevenshteinDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "wrld how are you",  // typo: "wrld" instead of "world"
            previousEndWords: ["hello", "world"],
            currentWords: nil
        )
        // Edit distance of 1 for single word, threshold = max(1, 1/5) = 1
        // Should still match since distance <= threshold
        #expect(result.text == "how are you")
        #expect(result.wordsRemoved == 1)
    }

    // MARK: - TimestampDeduplicationStrategy Tests

    @Test func testTimestampDeduplication() {
        let strategy = TimestampDeduplicationStrategy(overlapEnd: 5.0)
        let words: [WordTimestamp] = [
            WordTimestamp(word: "hello", start: 4.5, end: 4.8, confidence: 0.9),
            WordTimestamp(word: "world", start: 4.9, end: 5.2, confidence: 0.9),
            WordTimestamp(word: "how", start: 5.3, end: 5.5, confidence: 0.9),
        ]
        let result = strategy.deduplicate(
            currentText: "hello world how",
            previousEndWords: [],
            currentWords: words
        )
        #expect(result.text == "world how")
        #expect(result.wordsRemoved == 1)
        #expect(result.method == "timestamp")
    }

    @Test func testTimestampDeduplicationNoWordsRemoved() {
        let strategy = TimestampDeduplicationStrategy(overlapEnd: 4.0)
        let words: [WordTimestamp] = [
            WordTimestamp(word: "hello", start: 4.5, end: 4.8, confidence: 0.9),
            WordTimestamp(word: "world", start: 4.9, end: 5.2, confidence: 0.9),
        ]
        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: [],
            currentWords: words
        )
        #expect(result.text == "hello world")
        #expect(result.wordsRemoved == 0)
        #expect(result.method == "timestamp")
    }

    @Test func testTimestampDeduplicationAllWordsRemoved() {
        let strategy = TimestampDeduplicationStrategy(overlapEnd: 6.0)
        let words: [WordTimestamp] = [
            WordTimestamp(word: "hello", start: 4.5, end: 4.8, confidence: 0.9),
            WordTimestamp(word: "world", start: 4.9, end: 5.2, confidence: 0.9),
        ]
        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: [],
            currentWords: words
        )
        #expect(result.text == "")
        #expect(result.wordsRemoved == 2)
        #expect(result.method == "timestamp")
    }

    @Test func testTimestampDeduplicationFallbackWhenNoTimestamps() {
        let strategy = TimestampDeduplicationStrategy(overlapEnd: 5.0)
        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: [],
            currentWords: nil
        )
        #expect(result.text == "hello world")
        #expect(result.wordsRemoved == 0)
        #expect(result.method == "timestamp-fallback")
    }

    @Test func testTimestampDeduplicationFallbackWhenEmptyTimestamps() {
        let strategy = TimestampDeduplicationStrategy(overlapEnd: 5.0)
        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: [],
            currentWords: []
        )
        #expect(result.text == "hello world")
        #expect(result.wordsRemoved == 0)
        #expect(result.method == "timestamp-fallback")
    }

    @Test func testTimestampStrategyName() {
        let strategy = TimestampDeduplicationStrategy(overlapEnd: 5.0)
        #expect(strategy.name == "timestamp")
    }

    @Test func testTimestampDeduplicationExactBoundary() {
        let strategy = TimestampDeduplicationStrategy(overlapEnd: 5.0)
        let words: [WordTimestamp] = [
            WordTimestamp(word: "hello", start: 4.5, end: 5.0, confidence: 0.9),
            WordTimestamp(word: "world", start: 5.0, end: 5.5, confidence: 0.9),
        ]
        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: [],
            currentWords: words
        )
        // word.end > overlapEnd means end=5.0 is NOT > 5.0, so "hello" is removed
        #expect(result.text == "world")
        #expect(result.wordsRemoved == 1)
    }

    // MARK: - CompositeDeduplicationStrategy Tests

    @Test func testCompositeUsesTimestampsWhenAvailable() {
        let strategy = CompositeDeduplicationStrategy(overlapEnd: 5.0)
        let words = [
            WordTimestamp(word: "hello", start: 4.5, end: 4.8, confidence: 0.9),
            WordTimestamp(word: "world", start: 5.1, end: 5.4, confidence: 0.9)
        ]
        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: ["hello"],
            currentWords: words
        )
        #expect(result.method == "timestamp")
    }

    @Test func testCompositeFallsBackToLevenshtein() {
        let strategy = CompositeDeduplicationStrategy(overlapEnd: 5.0)
        let result = strategy.deduplicate(
            currentText: "world how are you",
            previousEndWords: ["hello", "world"],
            currentWords: nil
        )
        #expect(result.text == "how are you")
        #expect(result.method == "levenshtein")
    }

    @Test func testCompositeStrategyName() {
        let strategy = CompositeDeduplicationStrategy()
        #expect(strategy.name == "composite")
    }

    @Test func testCompositeFallsBackToNoOpWhenNoPreviousWords() {
        let strategy = CompositeDeduplicationStrategy()
        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: [],
            currentWords: nil
        )
        #expect(result.text == "hello world")
        #expect(result.method == "noop")
    }

    @Test func testCompositeWithoutOverlapEndUsesLevenshtein() {
        let strategy = CompositeDeduplicationStrategy(overlapEnd: nil)
        let words = [
            WordTimestamp(word: "hello", start: 4.5, end: 4.8, confidence: 0.9),
            WordTimestamp(word: "world", start: 5.1, end: 5.4, confidence: 0.9)
        ]
        let result = strategy.deduplicate(
            currentText: "world how are you",
            previousEndWords: ["hello", "world"],
            currentWords: words
        )
        #expect(result.method == "levenshtein")
    }

    @Test func testCompositeWithEmptyTimestampsUsesLevenshtein() {
        let strategy = CompositeDeduplicationStrategy(overlapEnd: 5.0)
        let result = strategy.deduplicate(
            currentText: "world how are you",
            previousEndWords: ["hello", "world"],
            currentWords: []
        )
        #expect(result.method == "levenshtein")
    }

    @Test func testCompositeMaxLookbackParameter() {
        let strategy = CompositeDeduplicationStrategy(overlapEnd: nil, maxLookback: 2)
        let result = strategy.deduplicate(
            currentText: "world today",
            previousEndWords: ["one", "two", "three", "hello", "world"],
            currentWords: nil
        )
        #expect(result.text == "today")
        #expect(result.wordsRemoved == 1)
    }
}
