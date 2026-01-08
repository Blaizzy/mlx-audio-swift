# Deduplication Strategies for Long Audio Transcription

## Overview

When processing long audio files with overlapping chunks, duplicate text can appear at chunk boundaries. MLX Audio Swift provides several strategies to handle this:

## Available Strategies

### 1. NoOpDeduplicationStrategy
Used when chunks don't overlap (VAD-based segmentation).

```swift
let strategy = NoOpDeduplicationStrategy()
```

### 2. LevenshteinDeduplicationStrategy
Uses edit distance to find matching sequences between chunks.
Best for: Real-time streaming, ~1ms computation time.

```swift
let strategy = LevenshteinDeduplicationStrategy(maxLookback: 10)
```

### 3. TimestampDeduplicationStrategy
Filters words based on their timestamps relative to overlap boundaries.
Best for: When word-level timestamps are available.

```swift
let strategy = TimestampDeduplicationStrategy(overlapEnd: 25.0)
```

### 4. CompositeDeduplicationStrategy (Default)
Combines all strategies with automatic fallback:
1. Uses timestamps when available
2. Falls back to Levenshtein when timestamps unavailable
3. Falls back to NoOp when no overlap

```swift
let strategy = CompositeDeduplicationStrategy(overlapEnd: 25.0)
```

## Usage

### Auto-select best strategy (default)

```swift
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .slidingWindow()
)
// SlidingWindowConfig automatically uses CompositeDeduplicationStrategy
```

### Use VAD (no deduplication needed)

```swift
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .vad()
)
// VAD creates non-overlapping chunks, deduplication not needed
```

### Custom configuration

```swift
let config = LongAudioProcessor.MergeConfig.withSmartDeduplication(overlapEnd: 25.0)
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    mergeConfig: config
)
```

## Strategy Selection Guide

| Strategy | When to Use | Performance |
|----------|-------------|-------------|
| **NoOp** | VAD chunks (non-overlapping) | O(1) |
| **Timestamp** | Word timestamps available | O(n) |
| **Levenshtein** | Streaming without timestamps | O(n*m) ~1ms |
| **Composite** | Automatic selection (recommended) | Varies |

## Implementation Details

### DeduplicationStrategy Protocol

All strategies implement the `DeduplicationStrategy` protocol:

```swift
public protocol DeduplicationStrategy: Sendable {
    func deduplicate(
        currentText: String,
        previousEndWords: [String],
        currentWords: [WordTimestamp]?
    ) -> DeduplicationResult

    var name: String { get }
}
```

### Integration Points

Deduplication is integrated at two levels:

1. **SlidingWindowChunkingStrategy**: Uses deduplication between overlapping chunks
2. **LongAudioProcessor**: Configures merge behavior via `MergeConfig`

### Configuration Options

```swift
// Default: auto-selects best strategy
let config = LongAudioProcessor.MergeConfig.default

// Disable deduplication (VAD mode)
let config = LongAudioProcessor.MergeConfig.concatenate

// Smart deduplication with custom overlap
let config = LongAudioProcessor.MergeConfig.withSmartDeduplication(overlapEnd: 25.0)
```

## Performance Considerations

- **NoOp**: Constant time, ideal for VAD-based segmentation
- **Timestamp**: Linear in number of words, requires word-level timing data
- **Levenshtein**: Quadratic with lookback window, ~1ms for typical chunks
- **Composite**: Adaptive performance based on available data

## Troubleshooting

### Too much deduplication (text being removed)

Increase the overlap threshold:

```swift
let config = LongAudioProcessor.MergeConfig.withSmartDeduplication(overlapEnd: 35.0)
```

### Not enough deduplication (duplicates remaining)

- Check if timestamps are available in your audio chunks
- Try explicit LevenshteinDeduplicationStrategy for better matching
- Reduce maxLookback if too many false positives

### Missing words at chunk boundaries

This is expected behavior:
- With overlapping chunks, some words appear in both chunks
- Deduplication removes the duplicate
- Use VAD-based segmentation to avoid overlaps entirely
