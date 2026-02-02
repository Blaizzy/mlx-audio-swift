# Silero VAD for MLX-Swift

Voice Activity Detection (VAD) module for Apple Silicon, powered by MLX. Detects speech segments in audio streams with real-time performance.

## Features

- **Real-time streaming**: <5ms latency per 32ms chunk on M1+
- **Batch processing**: `getSpeechTimestamps()` API compatible with Silero Python
- **Thread-safe**: Share `VADSession` across threads, create independent iterators
- **Configurable**: Presets for different use cases (sensitive, strict, conversation)

## Requirements

- macOS 13+ / iOS 16+
- Apple Silicon (M1/M2/M3)
- Audio: 16kHz, Float32, mono, normalized to [-1, 1]

## Quick Start

### Batch Processing

Process a complete audio file and get speech timestamps:

```swift
import MLXAudio
import MLX

// Load VAD session (loads model weights)
let vad = try await VADSession.make()

// Load your audio as MLXArray (16kHz, Float32, [-1, 1])
let audio: MLXArray = loadAudio("speech.wav")

// Get speech segments
let segments = try vad.getSpeechTimestamps(audio)

for segment in segments {
    print("Speech: \(segment.start)s - \(segment.end)s (duration: \(segment.duration)s)")
}
```

### Real-Time Streaming

Process audio chunks as they arrive:

```swift
let vad = try await VADSession.make()
let iterator = vad.makeIterator(config: .default)
let detector = SpeechDetector(config: .default)

// Process each 512-sample chunk (32ms at 16kHz)
for chunk in audioStream {
    let result = try iterator.process(chunk)

    if let event = detector.feed(result) {
        switch event {
        case .speechStarted(let time):
            print("🎤 Speech started at \(time)s")
        case .speechEnded(let time, let duration):
            print("🔇 Speech ended at \(time)s, duration: \(duration)s")
        case .speechDiscarded(let reason):
            print("⏭️ Speech discarded: \(reason)")
        }
    }
}

// Don't forget to finalize at end of stream
if let finalEvent = detector.finalize(at: iterator.currentTimestamp) {
    // Handle any ongoing speech at stream end
}
```

### Multiple Concurrent Streams

Each iterator maintains independent state:

```swift
let vad = try await VADSession.make()

// Stream 1
Task {
    let iterator1 = vad.makeIterator(config: .sensitive)
    for chunk in stream1 {
        let result = try iterator1.process(chunk)
        // Handle result...
    }
}

// Stream 2
Task {
    let iterator2 = vad.makeIterator(config: .strict)
    for chunk in stream2 {
        let result = try iterator2.process(chunk)
        // Handle result...
    }
}
```

## Configuration

### Presets

| Preset | Threshold | Use Case |
|--------|-----------|----------|
| `.default` | 0.5 | Balanced for general use |
| `.sensitive` | 0.35 | Quiet speakers, recording, transcription |
| `.strict` | 0.65 | Noisy environments, voice commands |
| `.conversation` | 0.5 | Quick response for voice assistants |

### Custom Configuration

```swift
let config = VADConfig(
    threshold: 0.6,           // Speech probability threshold [0-1]
    minSpeechDurationMs: 300, // Minimum speech duration to report
    minSilenceDurationMs: 150,// Silence duration to end speech
    speechPadMs: 50           // Padding added to segment boundaries
)

let iterator = vad.makeIterator(config: config)
```

### Parameters Explained

- **threshold**: Probability cutoff for classifying a frame as speech. Lower = more sensitive, higher = fewer false positives.
- **minSpeechDurationMs**: Speech segments shorter than this are discarded. Filters out clicks and noise bursts.
- **minSilenceDurationMs**: How long silence must last before ending a speech segment. Short pauses within this window are treated as continuous speech.
- **speechPadMs**: Padding added before speech start and after speech end. Helps capture word boundaries cleanly.

## Audio Format

The VAD expects audio in a specific format:

```swift
// Required format
VADAudioFormat.sampleRate    // 16000 Hz
VADAudioFormat.chunkSamples  // 512 samples per chunk
VADAudioFormat.chunkDuration // 0.032 seconds (32ms)
VADAudioFormat.valueRange    // -1.0...1.0

// Valid input shapes
[512]       // 1D array
[1, 512]    // 2D array with batch size 1
```

### Converting Audio

If your audio is in a different format:

```swift
// From AVAudioPCMBuffer (assuming 16kHz Float32)
let channelData = buffer.floatChannelData![0]
let samples = Array(UnsafeBufferPointer(start: channelData, count: Int(buffer.frameLength)))
let chunk = MLXArray(samples)

// Normalize Int16 audio to [-1, 1]
let int16Samples: [Int16] = ...
let normalized = int16Samples.map { Float($0) / 32768.0 }
let chunk = MLXArray(normalized)
```

## API Reference

### VADSession

Thread-safe entry point for VAD operations.

```swift
// Factory methods
static func make() async throws -> VADSession
static func make(weightsURL: URL) async throws -> VADSession

// Create iterator for streaming
func makeIterator(config: VADConfig = .default) -> VADIterator

// Batch processing
func getSpeechTimestamps(_ audio: MLXArray, config: VADConfig = .default) throws -> [SpeechSegment]
```

### VADIterator

Stateful chunk processor. **NOT thread-safe** - use one per stream.

```swift
// Process single chunk
func process(_ audio: MLXArray) throws -> VADResult

// Current position in stream
var currentTimestamp: TimeInterval { get }

// Reset state for new stream
func reset()

// Disable range validation for performance
var validateRange: Bool
```

### SpeechDetector

Event-based speech boundary detection. **NOT thread-safe**.

```swift
// Feed VAD result, get optional event
func feed(_ result: VADResult) -> SpeechEvent?

// Finalize at end of stream
func finalize(at timestamp: TimeInterval) -> SpeechEvent?

// Reset state
func reset()
```

### Types

```swift
struct VADResult {
    let probability: Float      // 0.0 - 1.0
    let isSpeech: Bool          // probability >= threshold
    let timestamp: TimeInterval
}

struct SpeechSegment {
    let start: TimeInterval
    let end: TimeInterval
    var duration: TimeInterval { get }
}

enum SpeechEvent {
    case speechStarted(at: TimeInterval)
    case speechEnded(at: TimeInterval, duration: TimeInterval)
    case speechDiscarded(reason: DiscardReason)
}

enum DiscardReason {
    case tooShort(duration: TimeInterval)
}
```

## Error Handling

```swift
do {
    let result = try iterator.process(chunk)
} catch VADError.invalidChunkSize(let expected, let got) {
    print("Wrong chunk size: expected \(expected), got \(got)")
} catch VADError.invalidDtype(let expected, let got) {
    print("Wrong dtype: expected \(expected), got \(got)")
} catch VADError.audioOutOfRange(let min, let max) {
    print("Audio values out of [-1, 1]: found [\(min), \(max)]")
} catch VADError.weightsNotFound(let path) {
    print("Model weights not found: \(path)")
}
```

## Performance Tips

1. **Disable range validation** for trusted audio pipelines:
   ```swift
   iterator.validateRange = false
   ```

2. **Reuse iterators** - creating new ones is cheap but unnecessary.

3. **Use appropriate config** - `.strict` reduces false positives in noisy environments.

4. **Batch process** when possible - `getSpeechTimestamps()` is optimized for complete audio.

## Thread Safety

| Type | Thread-Safe | Sendable |
|------|-------------|----------|
| `VADSession` | ✅ Yes | ✅ Yes |
| `VADIterator` | ❌ No | ❌ No |
| `SpeechDetector` | ❌ No | ❌ No |
| `VADConfig` | ✅ Yes | ✅ Yes |
| `VADResult` | ✅ Yes | ✅ Yes |
| `SpeechSegment` | ✅ Yes | ✅ Yes |

**Rule**: One `VADIterator` per Task/thread. Create multiple iterators for concurrent streams.

## Model Information

- **Architecture**: STFT → 4x Conv1d+ReLU → LSTM → Conv1d → Sigmoid
- **Parameters**: 309K
- **Weights**: `silero_vad_16k.safetensors` (1.2MB)
- **Input**: 512 samples @ 16kHz (32ms chunks)
- **Output**: Speech probability [0, 1]
