# Qwen3-TTS

Alibaba's multilingual Qwen3-TTS family with Base, CustomVoice, and VoiceDesign checkpoints.

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
)

let audio = try await model.generate(
    text: "Hello from Qwen3-TTS.",
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: "English"
)

try AudioUtils.writeWavFile(
    samples: audio.asArray(Float.self),
    sampleRate: Double(model.sampleRate),
    fileURL: URL(fileURLWithPath: "/tmp/qwen3-tts.wav")
)
```

## Voice Cloning

Clone a voice with reference audio and its transcript:

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
)

let (_, refAudio) = try loadAudioArray(
    from: URL(fileURLWithPath: "sample_audio.wav"),
    sampleRate: model.sampleRate
)

let cloned = try await model.generate(
    text: "Hello from Qwen3-TTS.",
    voice: nil,
    refAudio: refAudio,
    refText: "This is what my voice sounds like.",
    language: "English"
)
```

For best results, keep `refText` closely aligned with the spoken content in the reference clip.

## CustomVoice (Emotion Control)

CustomVoice checkpoints support named speakers and style prompting. In the current Swift API, pass that conditioning through `voice`:

```swift
let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
)

let audio = try await model.generate(
    text: "I'm so excited to meet you!",
    voice: "Vivian, very happy and excited.",
    refAudio: nil,
    refText: nil,
    language: "English"
)
```

## VoiceDesign (Create Any Voice)

VoiceDesign checkpoints let you describe the target voice in natural language. In Swift, pass that description with `voice`:

```swift
let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
)

let audio = try await model.generate(
    text: "Big brother, you're back!",
    voice: "A cheerful young female voice with high pitch and energetic tone.",
    refAudio: nil,
    refText: nil,
    language: "English"
)
```

## Streaming

`generateStream(...)` yields tokens, timing info, and streamed audio chunks for lower-latency playback:

```swift
let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
)

var audioChunks = [[Float]]()

for try await event in model.generateStream(
    text: "Hello, how are you today?",
    voice: "A calm, friendly narrator",
    refAudio: nil,
    refText: nil,
    language: "English",
    generationParameters: GenerateParameters(
        maxTokens: 4096,
        temperature: 0.9,
        topP: 1.0,
        repetitionPenalty: 1.1
    ),
    streamingInterval: 0.32
) {
    switch event {
    case .token(let token):
        print("Generated token: \(token)")
    case .info(let info):
        print("Tokens/s: \(info.tokensPerSecond)")
    case .audio(let chunk):
        audioChunks.append(chunk.asArray(Float.self))
    }
}
```

`streamingInterval` controls how frequently chunks are emitted in seconds. Smaller values reduce latency but increase overhead.

## Batch Generation

The Python reference exposes batched multi-sequence generation. The current Swift port does not yet have a public batch-generation API, so issue multiple `generate(...)` or `generateStream(...)` calls at the application layer when you need concurrency.

## Available Models

| Model | Swift entry point | Description |
|-------|-------------------|-------------|
| `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` | `generate()` | Fast, predefined voices |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | `generate()` | Higher quality |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16` | `generate()` with `voice` prompt | Voices + emotion control |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | `generate()` with `voice` prompt | Better emotion control |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | `generate()` with `voice` description | Create any voice |

## Speakers (Base / CustomVoice)

Common preset speakers for Base and CustomVoice checkpoints include:

**Chinese:** `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`

**English:** `Ryan`, `Aiden`
