# DTW-Based Word Timestamps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract word-level timestamps from Whisper cross-attention weights using Dynamic Time Warping (DTW)

**Architecture:** Post-processing pipeline that takes decoded tokens + cross-attention weights and produces word timestamps via DTW alignment

**Tech Stack:** MLX, Swift, existing WhisperSession infrastructure

---

## Background

### How Word Timestamps Work in Whisper

Whisper's cross-attention mechanism naturally learns to align text tokens with audio frames. Specific attention heads (called "alignment heads") strongly correlate with the temporal position of words in audio. By extracting and processing these attention patterns, we can determine when each word was spoken.

The process:
1. During decoding, capture cross-attention weights from alignment heads
2. Average/median filter the weights to get a token-to-frame attention matrix
3. Use DTW to find optimal alignment between tokens and frames
4. Convert frame indices to timestamps
5. Merge subword tokens into words with their time spans

### Current Infrastructure (Already Implemented)

| Component | Location | Status |
|-----------|----------|--------|
| Cross-attention capture | `TextDecoder.swift:65-84` | ✅ Working |
| Alignment heads config | `AlignmentHeads.swift` | ✅ All models |
| `WordTimestamp` struct | `ChunkingTypes.swift:58-70` | ✅ Defined |
| Frame attention (streaming) | `StreamingDecoder.swift` | ✅ For AlignAtt |

### What's Missing

1. **DTW algorithm** - Core alignment algorithm
2. **QK processing** - Extracting/filtering attention from alignment heads
3. **Word merging** - Combining subword tokens into words
4. **Integration** - Connecting to WhisperSession output

---

## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      WhisperSession                              │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐   │
│  │ AudioEncoder │───▶│ TextDecoder │───▶│ TokenProcessor   │   │
│  └──────────────┘    └──────┬──────┘    └────────┬─────────┘   │
│                             │                      │             │
│                   cross_attn_weights           tokens           │
│                             │                      │             │
│                             ▼                      ▼             │
│                    ┌────────────────────────────────┐           │
│                    │   WordTimestampExtractor       │           │
│                    │  ┌─────────┐  ┌────────────┐  │           │
│                    │  │   DTW   │──│ WordMerger │  │           │
│                    │  └─────────┘  └────────────┘  │           │
│                    └────────────────────────────────┘           │
│                                  │                               │
│                                  ▼                               │
│                          [WordTimestamp]                         │
└─────────────────────────────────────────────────────────────────┘
```

### New Components

#### 1. `DTWAligner` - Core DTW Algorithm

```swift
// MLXAudioSTT/Models/Whisper/WordTimestamps/DTWAligner.swift

public struct DTWAligner: Sendable {
    public struct Config: Sendable {
        public var medianFilterWidth: Int = 7
        public var costMetric: CostMetric = .negativeLogSoftmax

        public enum CostMetric: Sendable {
            case negativeLogSoftmax  // -log(softmax(weights))
            case oneMinusAttention   // 1 - attention
        }
    }

    /// Align tokens to audio frames using DTW
    /// - Parameters:
    ///   - attention: Averaged attention matrix [tokens, frames]
    ///   - tokenCount: Number of tokens (excluding special tokens)
    /// - Returns: Array of frame indices, one per token
    public func align(attention: MLXArray, tokenCount: Int) -> [Int]
}
```

#### 2. `CrossAttentionProcessor` - QK Processing

```swift
// MLXAudioSTT/Models/Whisper/WordTimestamps/CrossAttentionProcessor.swift

public struct CrossAttentionProcessor: Sendable {
    /// Extract and average attention from alignment heads
    /// - Parameters:
    ///   - crossQK: Cross-attention weights from all layers
    ///   - alignmentHeads: (layer, head) tuples to use
    /// - Returns: Averaged attention matrix [tokens, frames]
    public func extractAlignmentAttention(
        crossQK: [MLXArray],
        alignmentHeads: [(layer: Int, head: Int)]
    ) -> MLXArray
}
```

#### 3. `WordMerger` - Token to Word Conversion

```swift
// MLXAudioSTT/Models/Whisper/WordTimestamps/WordMerger.swift

public struct WordMerger: Sendable {
    /// Merge subword tokens into words with timestamps
    /// - Parameters:
    ///   - tokens: Decoded token IDs
    ///   - frameIndices: Frame index per token from DTW
    ///   - tokenizer: Tokenizer for decoding
    ///   - frameToTime: Frame index to timestamp conversion factor
    /// - Returns: Array of word timestamps
    public func merge(
        tokens: [Int],
        frameIndices: [Int],
        tokenizer: Tokenizer,
        frameToTime: Double
    ) -> [WordTimestamp]
}
```

#### 4. `WordTimestampExtractor` - High-Level API

```swift
// MLXAudioSTT/Models/Whisper/WordTimestamps/WordTimestampExtractor.swift

public final class WordTimestampExtractor: Sendable {
    private let aligner: DTWAligner
    private let processor: CrossAttentionProcessor
    private let merger: WordMerger
    private let alignmentHeads: [(layer: Int, head: Int)]

    /// Extract word timestamps from decoding results
    /// - Parameters:
    ///   - tokens: Decoded token IDs (excluding SOT/EOT)
    ///   - crossQK: Cross-attention weights from all layers
    ///   - tokenizer: Tokenizer for decoding
    ///   - audioDuration: Total audio duration in seconds
    /// - Returns: Array of word timestamps
    public func extract(
        tokens: [Int],
        crossQK: [MLXArray],
        tokenizer: Tokenizer,
        audioDuration: TimeInterval
    ) -> [WordTimestamp]
}
```

### DTW Algorithm Details

The DTW algorithm finds the optimal monotonic alignment between tokens and frames by minimizing cumulative cost:

```
Input: Cost matrix C[tokens, frames] where C[i,j] = cost of aligning token i to frame j
Output: Path [(t0, f0), (t1, f1), ...] where ti is token index, fi is frame index

Algorithm:
1. Initialize DP matrix D[i,j] = ∞
2. D[0,0] = C[0,0]
3. For each (i,j):
   D[i,j] = C[i,j] + min(D[i-1,j-1], D[i-1,j], D[i,j-1])
4. Backtrack from D[n,m] to find optimal path
5. Extract frame index for each token from path
```

**Cost Function:**
```swift
// Negative log softmax gives good alignment
cost[i,j] = -log(softmax(attention[i,:])[j])
```

**Constraints:**
- Monotonic: frame indices must be non-decreasing
- Boundary: first token starts at frame 0, last token ends at last frame

### Frame to Time Conversion

Whisper uses 30-second segments with specific frame rates:
- Audio sample rate: 16000 Hz
- Mel frames: 3000 per 30s segment (100 frames/second)
- Time per frame: 0.01 seconds (10ms)

```swift
let frameToTime: Double = 0.01  // 10ms per frame
let timestamp = Double(frameIndex) * frameToTime
```

### Integration with WhisperSession

Modify `WhisperSession.transcribe()` to optionally return word timestamps:

```swift
// In TranscriptionOptions
public struct TranscriptionOptions: Sendable {
    // ... existing options ...
    public var extractWordTimestamps: Bool = false
}

// In StreamingResult
public struct StreamingResult: Sendable {
    public let text: String
    public let timestamp: ClosedRange<TimeInterval>
    public let isFinal: Bool
    public let words: [WordTimestamp]?  // NEW: Optional word timestamps
}
```

### Streaming Considerations

**Important:** Word timestamps require the full segment's cross-attention weights. They cannot be computed incrementally during streaming. The timestamps are available only when `isFinal == true`.

For streaming output:
1. Partial results: `words = nil`
2. Final result: `words = [WordTimestamp]` (if `extractWordTimestamps == true`)

---

## Implementation Tasks

### Task 1: DTW Algorithm Implementation

**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WordTimestamps/DTWAligner.swift`
- Test: `mlx_audio_swift/stt/Tests/DTWAlignerTests.swift`

**Step 1: Create test file with basic DTW test**

```swift
// Tests/DTWAlignerTests.swift
import Testing
import MLX
@testable import MLXAudioSTT

@Suite("DTW Aligner Tests")
struct DTWAlignerTests {
    @Test func simpleAlignment() {
        let aligner = DTWAligner()

        // 3 tokens, 5 frames with clear diagonal pattern
        let attention = MLXArray([
            [0.8, 0.1, 0.05, 0.03, 0.02],
            [0.1, 0.7, 0.15, 0.03, 0.02],
            [0.05, 0.1, 0.2, 0.4, 0.25]
        ])

        let frames = aligner.align(attention: attention, tokenCount: 3)

        #expect(frames.count == 3)
        #expect(frames[0] == 0)  // First token -> frame 0
        #expect(frames[1] == 1)  // Second token -> frame 1
        #expect(frames[2] >= 3)  // Third token -> frame 3 or 4
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter DTWAlignerTests`
Expected: FAIL - DTWAligner not found

**Step 3: Implement DTWAligner**

```swift
// MLXAudioSTT/Models/Whisper/WordTimestamps/DTWAligner.swift
import Foundation
import MLX

public struct DTWAligner: Sendable {
    public struct Config: Sendable {
        public var medianFilterWidth: Int

        public init(medianFilterWidth: Int = 7) {
            self.medianFilterWidth = medianFilterWidth
        }

        public static let `default` = Config()
    }

    private let config: Config

    public init(config: Config = .default) {
        self.config = config
    }

    public func align(attention: MLXArray, tokenCount: Int) -> [Int] {
        guard tokenCount > 0 else { return [] }

        // Apply median filter for smoothing
        let filtered = medianFilter(attention, width: config.medianFilterWidth)

        // Convert to cost matrix: -log(softmax(attention))
        let softmaxed = MLX.softmax(filtered, axis: 1)
        let cost = -MLX.log(softmaxed + 1e-10)  // Add epsilon for numerical stability

        // Run DTW
        return dtwBacktrack(cost: cost)
    }

    private func medianFilter(_ matrix: MLXArray, width: Int) -> MLXArray {
        guard width > 1 else { return matrix }

        let tokens = matrix.shape[0]
        let frames = matrix.shape[1]
        let halfWidth = width / 2

        var result = matrix

        // Apply 1D median filter along frames axis for each token
        for t in 0..<tokens {
            for f in 0..<frames {
                let start = max(0, f - halfWidth)
                let end = min(frames, f + halfWidth + 1)
                let window = matrix[t, start..<end]
                let sorted = MLX.sort(window)
                let medianIdx = (end - start) / 2
                result[t, f] = sorted[medianIdx]
            }
        }

        return result
    }

    private func dtwBacktrack(cost: MLXArray) -> [Int] {
        let tokens = cost.shape[0]
        let frames = cost.shape[1]

        // Build DP table
        var dp = [[Float]](repeating: [Float](repeating: Float.infinity, count: frames), count: tokens)
        var parent = [[(Int, Int)]](repeating: [(Int, Int)](repeating: (-1, -1), count: frames), count: tokens)

        let costData: [Float] = cost.asArray(Float.self)

        // Initialize first row
        dp[0][0] = costData[0]
        for f in 1..<frames {
            dp[0][f] = dp[0][f-1] + costData[f]
            parent[0][f] = (0, f-1)
        }

        // Fill DP table
        for t in 1..<tokens {
            for f in 0..<frames {
                let idx = t * frames + f
                let currentCost = costData[idx]

                var minCost = Float.infinity
                var minParent = (-1, -1)

                // From (t-1, f-1) - diagonal
                if f > 0 && dp[t-1][f-1] < minCost {
                    minCost = dp[t-1][f-1]
                    minParent = (t-1, f-1)
                }

                // From (t-1, f) - vertical (same frame, previous token)
                if dp[t-1][f] < minCost {
                    minCost = dp[t-1][f]
                    minParent = (t-1, f)
                }

                // From (t, f-1) - horizontal (same token, previous frame)
                if f > 0 && dp[t][f-1] < minCost {
                    minCost = dp[t][f-1]
                    minParent = (t, f-1)
                }

                dp[t][f] = minCost + currentCost
                parent[t][f] = minParent
            }
        }

        // Backtrack to find path
        var path: [(Int, Int)] = []
        var t = tokens - 1
        var f = frames - 1

        while t >= 0 && f >= 0 {
            path.append((t, f))
            let (pt, pf) = parent[t][f]
            if pt == -1 { break }
            t = pt
            f = pf
        }

        path.reverse()

        // Extract frame index for each token
        var tokenFrames = [Int](repeating: 0, count: tokens)
        for (token, frame) in path {
            tokenFrames[token] = frame
        }

        return tokenFrames
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter DTWAlignerTests`
Expected: PASS

**Step 5: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WordTimestamps/DTWAligner.swift mlx_audio_swift/stt/Tests/DTWAlignerTests.swift
git commit -m "feat(stt): implement DTW aligner for word timestamps"
```

---

### Task 2: Cross-Attention Processor

**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WordTimestamps/CrossAttentionProcessor.swift`
- Test: `mlx_audio_swift/stt/Tests/CrossAttentionProcessorTests.swift`

**Step 1: Create test file**

```swift
// Tests/CrossAttentionProcessorTests.swift
import Testing
import MLX
@testable import MLXAudioSTT

@Suite("Cross-Attention Processor Tests")
struct CrossAttentionProcessorTests {
    @Test func extractsAndAveragesAlignmentHeads() {
        let processor = CrossAttentionProcessor()

        // Simulate 2 layers, each with shape [1, 4, 3, 5] (batch, heads, tokens, frames)
        let layer0 = MLXArray.ones([1, 4, 3, 5]) * 0.1
        let layer1 = MLXArray.ones([1, 4, 3, 5]) * 0.2

        // Set specific values for alignment heads
        var layer0Mut = layer0
        var layer1Mut = layer1
        layer0Mut[0, 1, 0..., 0...] = MLXArray.ones([3, 5]) * 0.5  // (0, 1)
        layer1Mut[0, 2, 0..., 0...] = MLXArray.ones([3, 5]) * 0.7  // (1, 2)

        let crossQK = [layer0Mut, layer1Mut]
        let alignmentHeads = [(layer: 0, head: 1), (layer: 1, head: 2)]

        let result = processor.extractAlignmentAttention(
            crossQK: crossQK,
            alignmentHeads: alignmentHeads
        )

        // Result should be average of 0.5 and 0.7 = 0.6
        #expect(result.shape == [3, 5])
        let values: [Float] = result.asArray(Float.self)
        #expect(abs(values[0] - 0.6) < 0.01)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter CrossAttentionProcessorTests`
Expected: FAIL

**Step 3: Implement CrossAttentionProcessor**

```swift
// MLXAudioSTT/Models/Whisper/WordTimestamps/CrossAttentionProcessor.swift
import Foundation
import MLX

public struct CrossAttentionProcessor: Sendable {
    public init() {}

    public func extractAlignmentAttention(
        crossQK: [MLXArray],
        alignmentHeads: [(layer: Int, head: Int)]
    ) -> MLXArray {
        var weights: [MLXArray] = []

        for (layer, head) in alignmentHeads {
            guard layer < crossQK.count else { continue }
            let layerQK = crossQK[layer]

            // Shape: [batch, heads, tokens, frames] -> [tokens, frames]
            let attention = layerQK[0, head, 0..., 0...].asType(.float32)
            weights.append(attention)
        }

        guard !weights.isEmpty else {
            // Return empty array with appropriate shape
            return MLXArray([])
        }

        // Stack and average across heads
        let stacked = MLX.stacked(weights, axis: 0)  // [heads, tokens, frames]
        let averaged = stacked.mean(axis: 0)  // [tokens, frames]

        return averaged
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter CrossAttentionProcessorTests`
Expected: PASS

**Step 5: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WordTimestamps/CrossAttentionProcessor.swift mlx_audio_swift/stt/Tests/CrossAttentionProcessorTests.swift
git commit -m "feat(stt): add cross-attention processor for word timestamps"
```

---

### Task 3: Word Merger

**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WordTimestamps/WordMerger.swift`
- Test: `mlx_audio_swift/stt/Tests/WordMergerTests.swift`

**Step 1: Create test file**

```swift
// Tests/WordMergerTests.swift
import Testing
@testable import MLXAudioSTT

@Suite("Word Merger Tests")
struct WordMergerTests {
    @Test func mergesSubwordTokensIntoWords() {
        let merger = WordMerger()

        // Simulate tokenizer output for "Hello world"
        // Assuming tokens: ["Hello", " world"] with frames [0, 10]
        let tokenTexts = ["Hello", " world"]
        let frameIndices = [0, 10]
        let frameToTime: Double = 0.01  // 10ms per frame

        let words = merger.merge(
            tokenTexts: tokenTexts,
            frameIndices: frameIndices,
            frameToTime: frameToTime,
            totalFrames: 20
        )

        #expect(words.count == 2)
        #expect(words[0].word == "Hello")
        #expect(words[0].start == 0.0)
        #expect(words[0].end == 0.1)  // Next word starts at frame 10
        #expect(words[1].word == "world")
        #expect(words[1].start == 0.1)
    }

    @Test func combinesSubwordPrefixes() {
        let merger = WordMerger()

        // Simulate "playing" tokenized as ["play", "ing"]
        let tokenTexts = ["play", "ing", " ball"]
        let frameIndices = [0, 5, 15]
        let frameToTime: Double = 0.01

        let words = merger.merge(
            tokenTexts: tokenTexts,
            frameIndices: frameIndices,
            frameToTime: frameToTime,
            totalFrames: 30
        )

        #expect(words.count == 2)
        #expect(words[0].word == "playing")
        #expect(words[1].word == "ball")
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter WordMergerTests`
Expected: FAIL

**Step 3: Implement WordMerger**

```swift
// MLXAudioSTT/Models/Whisper/WordTimestamps/WordMerger.swift
import Foundation

public struct WordMerger: Sendable {
    public init() {}

    public func merge(
        tokenTexts: [String],
        frameIndices: [Int],
        frameToTime: Double,
        totalFrames: Int
    ) -> [WordTimestamp] {
        guard !tokenTexts.isEmpty, tokenTexts.count == frameIndices.count else {
            return []
        }

        var words: [WordTimestamp] = []
        var currentWord = ""
        var wordStartFrame = frameIndices[0]

        for (i, text) in tokenTexts.enumerated() {
            let isNewWord = text.hasPrefix(" ") || text.hasPrefix("Ġ") || currentWord.isEmpty

            if isNewWord && !currentWord.isEmpty {
                // Complete previous word
                let endFrame = frameIndices[i]
                let word = WordTimestamp(
                    word: currentWord.trimmingCharacters(in: .whitespaces),
                    start: Double(wordStartFrame) * frameToTime,
                    end: Double(endFrame) * frameToTime,
                    confidence: 1.0
                )
                if !word.word.isEmpty {
                    words.append(word)
                }

                // Start new word
                currentWord = text.trimmingCharacters(in: .whitespaces)
                wordStartFrame = frameIndices[i]
            } else {
                // Continue current word (subword token)
                if currentWord.isEmpty {
                    currentWord = text.trimmingCharacters(in: .whitespaces)
                    wordStartFrame = frameIndices[i]
                } else {
                    currentWord += text
                }
            }
        }

        // Add final word
        if !currentWord.isEmpty {
            let word = WordTimestamp(
                word: currentWord.trimmingCharacters(in: .whitespaces),
                start: Double(wordStartFrame) * frameToTime,
                end: Double(totalFrames) * frameToTime,
                confidence: 1.0
            )
            if !word.word.isEmpty {
                words.append(word)
            }
        }

        return words
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter WordMergerTests`
Expected: PASS

**Step 5: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WordTimestamps/WordMerger.swift mlx_audio_swift/stt/Tests/WordMergerTests.swift
git commit -m "feat(stt): add word merger for subword token aggregation"
```

---

### Task 4: Word Timestamp Extractor

**Files:**
- Create: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WordTimestamps/WordTimestampExtractor.swift`
- Test: `mlx_audio_swift/stt/Tests/WordTimestampExtractorTests.swift`

**Step 1: Create test file**

```swift
// Tests/WordTimestampExtractorTests.swift
import Testing
import MLX
@testable import MLXAudioSTT

@Suite("Word Timestamp Extractor Tests")
struct WordTimestampExtractorTests {
    @Test func extractsWordTimestampsFromCrossAttention() async throws {
        let alignmentHeads = [(layer: 0, head: 0)]
        let extractor = WordTimestampExtractor(alignmentHeads: alignmentHeads)

        // Create mock cross-attention: 1 layer, shape [1, 1, 3, 50]
        // 3 tokens, 50 frames (0.5 seconds at 100 frames/sec)
        var crossQK = MLXArray.zeros([1, 1, 3, 50])

        // Token 0 attends strongly to frames 0-15
        crossQK[0, 0, 0, 0..<15] = MLXArray.ones([15]) * 0.8
        // Token 1 attends strongly to frames 15-35
        crossQK[0, 0, 1, 15..<35] = MLXArray.ones([20]) * 0.8
        // Token 2 attends strongly to frames 35-50
        crossQK[0, 0, 2, 35..<50] = MLXArray.ones([15]) * 0.8

        let tokenTexts = ["Hello", " world", "!"]

        let words = extractor.extract(
            tokenTexts: tokenTexts,
            crossQK: [crossQK],
            audioDuration: 0.5
        )

        #expect(words.count == 3)
        #expect(words[0].word == "Hello")
        #expect(words[0].start < 0.2)
        #expect(words[1].word == "world")
        #expect(words[1].start >= 0.1 && words[1].start <= 0.4)
    }
}
```

**Step 2: Run test to verify it fails**

Run: `swift test --filter WordTimestampExtractorTests`
Expected: FAIL

**Step 3: Implement WordTimestampExtractor**

```swift
// MLXAudioSTT/Models/Whisper/WordTimestamps/WordTimestampExtractor.swift
import Foundation
import MLX

public final class WordTimestampExtractor: Sendable {
    private let aligner: DTWAligner
    private let processor: CrossAttentionProcessor
    private let merger: WordMerger
    private let alignmentHeads: [(layer: Int, head: Int)]

    public init(
        alignmentHeads: [(layer: Int, head: Int)],
        alignerConfig: DTWAligner.Config = .default
    ) {
        self.alignmentHeads = alignmentHeads
        self.aligner = DTWAligner(config: alignerConfig)
        self.processor = CrossAttentionProcessor()
        self.merger = WordMerger()
    }

    public func extract(
        tokenTexts: [String],
        crossQK: [MLXArray],
        audioDuration: TimeInterval
    ) -> [WordTimestamp] {
        guard !tokenTexts.isEmpty, !crossQK.isEmpty else {
            return []
        }

        // 1. Extract attention matrix from alignment heads
        let attention = processor.extractAlignmentAttention(
            crossQK: crossQK,
            alignmentHeads: alignmentHeads
        )

        guard attention.size > 0 else { return [] }

        // 2. Run DTW alignment
        let frameIndices = aligner.align(
            attention: attention,
            tokenCount: tokenTexts.count
        )

        // 3. Calculate frame parameters
        let totalFrames = attention.shape[1]
        let frameToTime = audioDuration / Double(totalFrames)

        // 4. Merge tokens into words
        return merger.merge(
            tokenTexts: tokenTexts,
            frameIndices: frameIndices,
            frameToTime: frameToTime,
            totalFrames: totalFrames
        )
    }
}
```

**Step 4: Run test to verify it passes**

Run: `swift test --filter WordTimestampExtractorTests`
Expected: PASS

**Step 5: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WordTimestamps/WordTimestampExtractor.swift mlx_audio_swift/stt/Tests/WordTimestampExtractorTests.swift
git commit -m "feat(stt): add word timestamp extractor with DTW pipeline"
```

---

### Task 5: Integration with WhisperSession

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/Models/Whisper/WhisperSession.swift`
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/TranscriptionTypes.swift`
- Test: `mlx_audio_swift/stt/Tests/WhisperSessionWordTimestampTests.swift`

**Step 1: Update TranscriptionOptions**

Add to `TranscriptionOptions`:

```swift
public var extractWordTimestamps: Bool = false
```

**Step 2: Update StreamingResult**

Add optional `words` field:

```swift
public struct StreamingResult: Sendable {
    public let text: String
    public let timestamp: ClosedRange<TimeInterval>
    public let isFinal: Bool
    public let words: [WordTimestamp]?  // New field

    // Update init
}
```

**Step 3: Modify WhisperSession.transcribe()**

At the point where final result is emitted, if `options.extractWordTimestamps`:

```swift
// In generateStreaming method, where final result is created:
var finalWords: [WordTimestamp]? = nil

if options.extractWordTimestamps {
    let extractor = WordTimestampExtractor(alignmentHeads: self.alignmentHeads)
    let tokenTexts = decodedTokens.map { tokenizer.decode([$0]) }
    finalWords = extractor.extract(
        tokenTexts: tokenTexts,
        crossQK: accumulatedCrossQK,
        audioDuration: segmentDuration
    )
}

let finalResult = StreamingResult(
    text: finalText,
    timestamp: 0...segmentDuration,
    isFinal: true,
    words: finalWords
)
```

**Step 4: Create integration test**

```swift
// Tests/WhisperSessionWordTimestampTests.swift
import Testing
import MLX
@testable import MLXAudioSTT

@Suite("WhisperSession Word Timestamp Integration")
struct WhisperSessionWordTimestampTests {
    @Test func wordTimestampsReturnedWhenEnabled() async throws {
        // This test requires a loaded model - may be integration test
        // For unit testing, mock the session or use test fixtures
    }
}
```

**Step 5: Run tests and commit**

```bash
swift test --filter WhisperSession
git add -A
git commit -m "feat(stt): integrate word timestamps into WhisperSession"
```

---

### Task 6: Update LongAudioProcessor Integration

**Files:**
- Modify: `mlx_audio_swift/stt/MLXAudioSTT/LongAudioProcessor.swift`

**Step 1: Update WhisperSessionTranscriber**

Modify `WhisperSessionTranscriber.transcribe()` to capture and return word timestamps:

```swift
func transcribe(
    audio: MLXArray,
    sampleRate: Int,
    previousTokens: [Int]?
) async throws -> ChunkResult {
    var options = TranscriptionOptions.default
    options.extractWordTimestamps = true  // Enable word extraction

    // ... existing code ...

    return ChunkResult(
        text: finalText,
        tokens: [],
        timeRange: finalTimestamp,
        confidence: 1.0,
        words: finalResult.words  // Pass through words
    )
}
```

**Step 2: Commit**

```bash
git add mlx_audio_swift/stt/MLXAudioSTT/LongAudioProcessor.swift
git commit -m "feat(stt): propagate word timestamps through LongAudioProcessor"
```

---

## Verification Checklist

- [ ] `swift test --filter DTWAligner` passes
- [ ] `swift test --filter CrossAttentionProcessor` passes
- [ ] `swift test --filter WordMerger` passes
- [ ] `swift test --filter WordTimestampExtractor` passes
- [ ] Integration test with real audio produces reasonable timestamps
- [ ] Word timestamps align with expected speech timing
- [ ] Subword tokens correctly merged into words

---

## Future Enhancements

1. **Confidence scores** - Calculate per-word confidence from attention entropy
2. **Punctuation handling** - Proper timing for punctuation marks
3. **Multi-language support** - Handle different tokenization patterns
4. **Streaming word timestamps** - Emit word timestamps as tokens are decoded (requires modified DTW)
5. **Whisper-timestamped parity** - Implement all features from whisper-timestamped

---

## References

- [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) - Python reference implementation
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) - MLX Python implementation
- [WhisperKit](https://github.com/argmaxinc/WhisperKit) - Swift reference (uses different approach)
- [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) - Enhanced word timestamps
