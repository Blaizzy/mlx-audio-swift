# Pocket TTS

A lightweight text-to-speech (TTS) model from Kyutai designed to run efficiently on CPUs.

[Blog Post](https://kyutai.org/blog/2026-01-13-pocket-tts)

## Supported Voices

- `alba`
- `marius`
- `javert`
- `jean`
- `fantine`
- `cosette`
- `eponine`
- `azelma`

## CLI Example

```bash
mlx-audio-swift-tts --model mlx-community/pocket-tts --text "Hello world."
```

## Swift Example

```swift
import MLXAudioTTS

let model = try await PocketTTSModel.fromPretrained("mlx-community/pocket-tts")
let audio = try await model.generate(
    text: "Hello world.",
    voice: "alba",
    generationParameters: GenerateParameters()
)
```
