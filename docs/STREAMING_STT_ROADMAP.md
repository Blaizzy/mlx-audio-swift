# 🎯 Native Swift STT with AlignAtt Streaming

> **Design Document**: [2025-01-06-streaming-stt-design.md](plans/2025-01-06-streaming-stt-design.md)
>
> **Status**: Design approved, ready for implementation

## Overview

This document outlines the roadmap for implementing native Speech-to-Text (STT) with AlignAtt streaming support in MLX Swift, enabling low-latency transcription on Apple Silicon.

## Current Landscape

| Aspect | WhisperKit | Our Implementation (Python) |
|--------|------------|--------------------------|
| **Streaming Method** | VAD chunking | **AlignAtt** (cross-attention) |
| **Latency** | ~2-3s (chunk-based) | **~0.8-1.0s** (token-level) |
| **Platform** | CoreML | MLX |
| **Open Source** | Partial (Pro for full streaming) | ✅ Fully open |

### Key Advantage of AlignAtt

- WhisperKit waits for VAD chunk completion before decoding
- AlignAtt emits tokens **during decoding** by monitoring cross-attention weights
- Result: ~2-3x lower latency to first word

## Components to Port

### Python → Swift Mapping (~3,109 lines total)

```
Python Component              Swift Equivalent              Status
────────────────────────────────────────────────────────────────────
streaming.py (282 LoC)        StreamingDecoder.swift        🆕 New
├─ StreamingConfig            struct StreamingConfig
├─ StreamingResult            struct StreamingResult
├─ get_most_attended_frame()  func getMostAttendedFrame()
├─ should_emit()              func shouldEmit()
└─ StreamingDecoder           class StreamingDecoder

whisper.py (975 LoC)          WhisperModel.swift            🆕 New
├─ AudioEncoder               class AudioEncoder
├─ TextDecoder                class TextDecoder
├─ MultiHeadAttention         class MultiHeadAttention
└─ generate_streaming()       func generateStreaming()

audio.py (82 LoC)             AudioProcessor.swift          🆕 New
└─ log_mel_spectrogram()      func logMelSpectrogram()

decoding.py (767 LoC)         Decoding.swift                🆕 New
├─ DecodingOptions            struct DecodingOptions
├─ DecodingTask               class DecodingTask
└─ GreedyDecoder              class GreedyDecoder

tokenizer.py (398 LoC)        Use tiktoken-swift            ✅ Exists
```

## Effort Estimation

| Phase | Component | Complexity | Time |
|-------|-----------|-----------|-------|
| **1** | AudioEncoder + Mel Spectrogram | Medium | 3-4 days |
| **2** | TextDecoder + MultiHeadAttention | High | 5-7 days |
| **3** | Decoding (GreedyDecoder) | Medium | 3-4 days |
| **4** | **StreamingDecoder (AlignAtt)** | Medium | 2-3 days |
| **5** | Integration + Tests | Medium | 3-4 days |
| | **Total** | | **2-3 weeks** |

## Proposed Swift Architecture

```
MLXAudioSTT/
├── Models/
│   └── Whisper/
│       ├── WhisperModel.swift        // AudioEncoder + TextDecoder
│       ├── MultiHeadAttention.swift  // With cross-attention capture
│       ├── Streaming/
│       │   ├── StreamingConfig.swift
│       │   ├── StreamingResult.swift
│       │   └── StreamingDecoder.swift // AlignAtt logic
│       └── Decoding/
│           ├── DecodingOptions.swift
│           └── GreedyDecoder.swift
├── Audio/
│   ├── AudioProcessor.swift          // Mel spectrogram
│   └── MicrophoneCapture.swift       // Real-time input
└── Utils/
    └── WhisperTokenizer.swift        // tiktoken wrapper
```

## Streaming Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMING PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Microphone ──► Mel Spectrogram ──► AudioEncoder ──┐           │
│     (AVFoundation)   (vDSP/MLX)       (MLX)        │           │
│                                                     ▼           │
│                                              KV Cache           │
│                                                     │           │
│  ┌──────────────────────────────────────────────────┘           │
│  │                                                              │
│  ▼                                                              │
│  TextDecoder ──► Cross-Attention ──► AlignAtt ──► Emit Token   │
│    (MLX)         Weights capture      Check        if stable    │
│                                                                 │
│  Target Latency: < 500ms to first token                        │
└─────────────────────────────────────────────────────────────────┘
```

## Core AlignAtt Implementation (Swift)

```swift
// StreamingDecoder.swift - Core AlignAtt logic

import MLX
import MLXNN

public struct StreamingConfig {
    var frameThreshold: Int = 25      // Frames from audio end before emitting
    var minChunkDuration: Float = 0.5 // Minimum chunk duration in seconds
    var emitPartial: Bool = true      // Emit partial results
}

public struct StreamingResult {
    var text: String
    var tokens: [Int]
    var isFinal: Bool
    var startTime: Float
    var endTime: Float
    var progress: Float           // 0.0 to 1.0
    var audioPosition: Float      // Current position in seconds
    var audioDuration: Float      // Total duration in seconds
}

/// Extract the most attended audio frame from cross-attention weights
/// Uses alignment heads for word-level timestamp accuracy
public func getMostAttendedFrame(
    crossQK: [MLXArray],
    alignmentHeads: MLXArray
) -> Int {
    // Stack attention weights from alignment heads
    let weights = MLX.stack(
        alignmentHeads.map { pair in
            let layer = Int(pair[0].item(Int.self))
            let head = Int(pair[1].item(Int.self))
            return crossQK[layer][0, head, -1, ...]  // Last token's attention
        }
    )

    // Average across heads and find max
    let avgAttention = weights.mean(axis: 0)
    return Int(MLX.argmax(avgAttention).item(Int.self))
}

/// Determine if current token should be emitted based on attention stability
public func shouldEmit(
    mostAttendedFrame: Int,
    contentFrames: Int,
    threshold: Int
) -> Bool {
    return (contentFrames - mostAttendedFrame) >= threshold
}
```

## Benchmark Results (Python Reference)

Performance on 30s audio with `whisper-large-v3-turbo`:

| Mode | Total Time | Time to First | RTF |
|------|------------|---------------|-----|
| Batch | 1.13s | 1.13s | 0.038 |
| Streaming (0.5s chunks) | 34.9s | **1.00s** | 1.16 |
| Streaming (1.0s chunks) | 18.2s | **1.00s** | 0.61 |
| Streaming (2.0s chunks) | 9.9s | **1.13s** | 0.33 |
| Streaming (3.0s chunks) | 6.7s | **1.15s** | 0.22 |

**Key insight**: Streaming latency is independent of audio length (~1s to first result).

## Implementation Options

### Option A: Full MLX Swift Port (2-3 weeks)
- ✅ Full control over latency
- ✅ Native AVFoundation integration
- ✅ Can be added to mlx-audio-swift
- ❌ More work

### Option B: WhisperKit + AlignAtt Patch (1 week)
- ✅ Faster implementation
- ❌ CoreML dependency (less flexible)
- ❌ WhisperKit Pro required for full streaming

### Option C: Hybrid - Python backend + Swift UI (3-5 days)
- ✅ Use existing streaming implementation
- ✅ Swift UI for iOS/macOS
- ❌ Not fully native

## References

- [AlignAtt Paper](https://arxiv.org/abs/2211.00895) - SimulMT with AlignAtt
- [WhisperKit](https://github.com/argmaxinc/WhisperKit) - Swift Whisper implementation
- [MLX Swift](https://github.com/ml-explore/mlx-swift) - Apple's ML framework for Swift
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Python reference implementation

## Long Audio Support (>30 seconds)

### The 30-Second Limitation

Whisper has a **fixed 30-second context window**:
- Input: 3000 mel frames (30 sec × 100 fps)
- After Conv2 stride-2: 1500 frames (`nAudioCtx`)
- Positional embeddings: Fixed size array [1500, nAudioState]

Current behavior: `AudioUtils.padOrTrim()` truncates or pads audio to exactly 30 seconds.

### Solution: Sliding Window with Overlap

```
┌─────────────────────────────────────────────────────────────────┐
│                    LONG AUDIO CHUNKING                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Audio: |████████████████████████████████████████████████████| │
│         0s        30s       60s       90s      120s            │
│                                                                 │
│  Chunk 1: |██████████████████████████████|                     │
│           0s                            30s                     │
│                                                                 │
│  Chunk 2:          |██████████████████████████████|            │
│                   25s                            55s            │
│                    ↑ 5s overlap                                 │
│                                                                 │
│  Chunk 3:                   |██████████████████████████████|   │
│                            50s                            80s   │
│                             ↑ 5s overlap                        │
│                                                                 │
│  Merge: Align overlapping transcriptions using timestamps       │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

```swift
// LongAudioProcessor.swift

public struct ChunkingConfig {
    var chunkDuration: TimeInterval = 30.0  // Whisper max
    var overlapDuration: TimeInterval = 5.0 // Overlap for seamless merge
    var hopDuration: TimeInterval { chunkDuration - overlapDuration }  // 25s
}

public func transcribeLongAudio(
    _ audio: MLXArray,
    sampleRate: Int,
    config: ChunkingConfig = .default
) -> AsyncThrowingStream<StreamingResult, Error> {
    // 1. Split audio into overlapping chunks
    // 2. Process each chunk with WhisperSession
    // 3. Merge results using word-level timestamps
    // 4. Emit progressive results as chunks complete
}
```

### Merge Strategy

1. **Word-level timestamps** from AlignAtt cross-attention
2. **Overlap alignment**: Match words in overlap region
3. **Deduplication**: Remove repeated words at chunk boundaries
4. **Confidence scoring**: Prefer higher-confidence transcriptions

## Next Steps

### Phase 1: Core Implementation ✅
1. [x] Create `MLXAudioSTT` Swift package structure
2. [x] Port `AudioProcessor` (mel spectrogram)
3. [x] Port `MultiHeadAttention` with cross-attention capture
4. [x] Port `AudioEncoder` and `TextDecoder`
5. [x] Implement `StreamingDecoder` with AlignAtt
6. [x] Basic demo working with test audio

### Phase 2: Production Features
7. [ ] **Long audio chunking** (sliding window with overlap)
8. [ ] Add `MicrophoneCapture` for real-time input
9. [ ] Word-level timestamps extraction
10. [ ] Integration tests and benchmarks
11. [ ] iOS/macOS demo app

---

*Document created: January 2025*
*Branch: feat/streaming-stt*
