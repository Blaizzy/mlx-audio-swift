# MLXAudioSTT

Speech-to-text using Whisper models on Apple Silicon.

## Quick Start

```swift
import MLXAudioSTT

// Load model
let session = try await WhisperSession.fromPretrained(model: .largeTurbo)

// Transcribe
for try await result in session.transcribe(audio, sampleRate: 16000) {
    if result.isFinal {
        print(result.text)
    }
}
```

## Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `.tiny` | 39M | Fastest | Basic |
| `.base` | 74M | Fast | Good |
| `.small` | 244M | Medium | Better |
| `.largeTurbo` | 809M | Fast | Best |
| `.largeV3` | 1.5G | Slow | Best |

## Two APIs

### WhisperSession — Short audio (<30s)

```swift
let session = try await WhisperSession.fromPretrained(model: .largeTurbo)

// Streaming
for try await result in session.transcribe(audio, sampleRate: 16000) {
    print(result.isFinal ? "Final: \(result.text)" : "Partial: \(result.text)")
}

// Or simple async
let text = try await session.transcribe(audio, sampleRate: 16000)
```

### LongAudioProcessor — Long audio (>30s)

```swift
let processor = try await LongAudioProcessor.create(model: .largeTurbo)

for try await progress in processor.transcribe(audio, sampleRate: 16000) {
    print("[\(progress.chunkIndex)/\(progress.totalChunks)] \(progress.text)")
}
```

## Options

```swift
// Fast loading (int4 quantization)
let session = try await WhisperSession.fromPretrained(
    model: .largeTurbo,
    options: .fast
)

// Language hint
var options = TranscriptionOptions.default
options.language = "en"
for try await result in session.transcribe(audio, options: options) { ... }
```

## Chunking Strategies (Long Audio)

```swift
// Auto-select best strategy
let processor = try await LongAudioProcessor.create(
    model: .largeTurbo,
    strategy: .auto
)

// Or specify:
// .sequential()    — Fixed 30s chunks
// .vad()           — Voice Activity Detection
// .slidingWindow() — Overlapping windows
```

## Audio Format

- Sample rate: 16kHz (required)
- Channels: Mono
- Format: `MLXArray` of Float32 samples

```swift
// Load from file (see STTDemo for example)
let audio = try loadAudio(from: "speech.wav")  // Returns MLXArray
```
