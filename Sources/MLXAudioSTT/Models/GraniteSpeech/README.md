# Granite Speech

MLX Swift implementation of IBM's Granite Speech, a speech-to-text model that combines a CTC Conformer encoder with a Granite LLM decoder via a BLIP-2 QFormer projector. Supports ASR (transcription) and AST (speech translation).

## Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| [mlx-community/granite-4.0-1b-speech-5bit](https://huggingface.co/mlx-community/granite-4.0-1b-speech-5bit) | ~1B | Speech recognition and translation (5-bit quantized) |

**Supported Languages:** English, French, German, Spanish, Portuguese, Japanese

## Swift Usage

### ASR (Transcription)

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL)

let model = try await GraniteSpeechModel.fromPretrained("mlx-community/granite-4.0-1b-speech-5bit")

// Basic transcription (default prompt)
let output = model.generate(audio: audio)
print(output.text)

// With custom prompt
let output = model.generate(audio: audio, prompt: "Translate the speech to text.")
print(output.text)
```

### AST (Speech Translation)

Use the `language` parameter to translate speech. Accepts full names or codes (`fr`, `de`, `es`, `pt`, `ja`):

```swift
// Translate speech to French (using language code)
let output = model.generate(audio: audio, language: "fr")
print(output.text)

// Translate speech to Spanish (using full name)
let output = model.generate(audio: audio, language: "Spanish")
print(output.text)

// Translate speech to Portuguese
let output = model.generate(audio: audio, language: "pt")
print(output.text)

// Or use a custom prompt directly
let output = model.generate(audio: audio, prompt: "Translate the speech to German.")
print(output.text)
```

> **Note:** If the model receives an unfamiliar prompt, it falls back to transcription as the default mode.

### Streaming

```swift
for try await event in model.generateStream(audio: audio) {
    switch event {
    case .token(let token):
        print(token, terminator: "")
    case .result(let result):
        print("\nFinal text: \(result.text)")
    case .info:
        break
    }
}
```

### Generation Parameters

```swift
let output = model.generate(
    audio: audio,
    maxTokens: 4096,
    temperature: 0.0,       // 0 = greedy decoding
    prompt: "Translate the speech to text.",
    verbose: true            // print timing info
)
```

## Architecture

- **Encoder**: CTC Conformer (16 layers, 1024 hidden dim, Shaw's relative positional embeddings, block-wise attention with context_size=200)
- **Projector**: BLIP-2 QFormer (2 layers, windowed cross-attention with window_size=15, downsample_rate=5)
- **Decoder**: Granite LLM (40 layers, 2048 hidden dim, GQA with 16/4 heads, RoPE, SwiGLU MLP)
- Audio input: 16kHz, 80-bin mel spectrogram with pair stacking (160-dim input)

## Output Format

```swift
STTOutput(
    text: "Full transcription text",
    promptTokens: 154,
    generationTokens: 42,
    totalTokens: 196,
    totalTime: 0.95,
    promptTps: 162.1,
    generationTps: 44.2,
    peakMemoryUsage: 0.85
)
```
