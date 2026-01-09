import Foundation
import Testing
@testable import MLXAudioSTT

struct DeduplicationIntegrationTests {

    // MARK: - SlidingWindowConfig Tests

    @Test("SlidingWindowConfig has composite deduplication by default")
    func testSlidingWindowWithCompositeDeduplication() throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig()

        #expect(config.deduplicationStrategy != nil)
        #expect(config.deduplicationStrategy?.name == "composite")
    }

    @Test("SlidingWindowConfig computes correct overlap end")
    func testSlidingWindowOverlapEndCalculation() throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 30.0,
            overlapDuration: 5.0
        )

        // hopDuration should be windowDuration - overlapDuration = 25.0
        #expect(config.hopDuration == 25.0)
        #expect(config.deduplicationStrategy != nil)
    }

    @Test("SlidingWindowConfig with custom deduplication strategy")
    func testSlidingWindowWithCustomStrategy() throws {
        let customStrategy = LevenshteinDeduplicationStrategy(maxLookback: 3)
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig(
            windowDuration: 20.0,
            overlapDuration: 4.0,
            deduplicationStrategy: customStrategy
        )

        #expect(config.hopDuration == 16.0)
        #expect(config.deduplicationStrategy?.name == "levenshtein")
    }

    @Test("SlidingWindowConfig default static property")
    func testSlidingWindowConfigDefault() throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig.default

        #expect(config.windowDuration == 30.0)
        #expect(config.overlapDuration == 5.0)
        #expect(config.hopDuration == 25.0)
        #expect(config.deduplicationStrategy != nil)
    }

    // MARK: - MergeConfig Tests

    @Test("MergeConfig.withSmartDeduplication creates composite strategy")
    func testMergeConfigWithSmartDeduplication() throws {
        let config = LongAudioProcessor.MergeConfig.withSmartDeduplication(overlapEnd: 25.0)

        #expect(config.deduplicateOverlap == true)
        #expect(config.deduplicationStrategy != nil)
        #expect(config.deduplicationStrategy?.name == "composite")
    }

    @Test("MergeConfig.withSmartDeduplication without overlapEnd")
    func testMergeConfigWithSmartDeduplicationNoOverlapEnd() throws {
        let config = LongAudioProcessor.MergeConfig.withSmartDeduplication()

        #expect(config.deduplicateOverlap == true)
        #expect(config.deduplicationStrategy != nil)
        #expect(config.deduplicationStrategy?.name == "composite")
    }

    @Test("Default MergeConfig has composite deduplication strategy")
    func testDefaultMergeConfigHasCompositeStrategy() throws {
        let config = LongAudioProcessor.MergeConfig.default

        #expect(config.deduplicateOverlap == true)
        #expect(config.deduplicationStrategy != nil)
        #expect(config.deduplicationStrategy?.name == "composite")
    }

    @Test("Custom deduplication strategy can be set on MergeConfig")
    func testCustomDeduplicationStrategy() throws {
        let strategy = LevenshteinDeduplicationStrategy(maxLookback: 5)
        let config = LongAudioProcessor.MergeConfig(
            deduplicateOverlap: true,
            deduplicationStrategy: strategy
        )

        #expect(config.deduplicationStrategy != nil)
        #expect(config.deduplicationStrategy?.name == "levenshtein")
    }

    @Test("MergeConfig with timestamp strategy")
    func testMergeConfigWithTimestampStrategy() throws {
        let strategy = TimestampDeduplicationStrategy(overlapEnd: 10.0)
        let config = LongAudioProcessor.MergeConfig(
            deduplicateOverlap: true,
            deduplicationStrategy: strategy
        )

        #expect(config.deduplicationStrategy != nil)
        #expect(config.deduplicationStrategy?.name == "timestamp")
    }

    @Test("MergeConfig with noop strategy")
    func testMergeConfigWithNoOpStrategy() throws {
        let strategy = NoOpDeduplicationStrategy()
        let config = LongAudioProcessor.MergeConfig(
            deduplicateOverlap: false,
            deduplicationStrategy: strategy
        )

        #expect(config.deduplicateOverlap == false)
        #expect(config.deduplicationStrategy?.name == "noop")
    }

    // MARK: - Strategy Integration Tests

    @Test("CompositeDeduplicationStrategy uses timestamps when configured")
    func testCompositeStrategyUsesTimestamps() throws {
        let strategy = CompositeDeduplicationStrategy(overlapEnd: 5.0)
        let words = [
            WordTimestamp(word: "hello", start: 4.5, end: 4.9, confidence: 0.9),
            WordTimestamp(word: "world", start: 5.1, end: 5.5, confidence: 0.9)
        ]

        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: ["prev"],
            currentWords: words
        )

        #expect(result.method == "timestamp")
        #expect(result.text == "world")
        #expect(result.wordsRemoved == 1)
    }

    @Test("CompositeDeduplicationStrategy falls back to Levenshtein without timestamps")
    func testCompositeStrategyFallsBackToLevenshtein() throws {
        let strategy = CompositeDeduplicationStrategy(overlapEnd: 5.0)

        let result = strategy.deduplicate(
            currentText: "world is great",
            previousEndWords: ["hello", "world"],
            currentWords: nil
        )

        #expect(result.method == "levenshtein")
        #expect(result.text == "is great")
    }

    @Test("CompositeDeduplicationStrategy uses noop when no context available")
    func testCompositeStrategyUsesNoOp() throws {
        let strategy = CompositeDeduplicationStrategy()

        let result = strategy.deduplicate(
            currentText: "hello world",
            previousEndWords: [],
            currentWords: nil
        )

        #expect(result.method == "noop")
        #expect(result.text == "hello world")
        #expect(result.wordsRemoved == 0)
    }

    // MARK: - Strategy Chaining Integration

    @Test("Sliding window default strategy deduplicates correctly")
    func testSlidingWindowDefaultStrategyDeduplication() throws {
        let config = SlidingWindowChunkingStrategy.SlidingWindowConfig()
        let strategy = config.deduplicationStrategy!

        let words = [
            WordTimestamp(word: "overlap", start: 20.0, end: 24.5, confidence: 0.9),
            WordTimestamp(word: "new", start: 25.5, end: 26.0, confidence: 0.9),
            WordTimestamp(word: "content", start: 26.5, end: 27.0, confidence: 0.9)
        ]

        let result = strategy.deduplicate(
            currentText: "overlap new content",
            previousEndWords: ["some", "overlap"],
            currentWords: words
        )

        // With default config (windowDuration=30, overlapDuration=5), hopDuration=25
        // Words ending before 25.0 should be filtered by timestamp strategy
        #expect(result.method == "timestamp")
        #expect(result.text == "new content")
    }
}
