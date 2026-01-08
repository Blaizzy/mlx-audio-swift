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
}
