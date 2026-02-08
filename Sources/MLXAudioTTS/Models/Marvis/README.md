# Marvis TTS

A fast conversational text-to-speech (TTS) model with built-in voices for English, French, and German.

[Hugging Face Model Repo](https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit)

## Supported Voices

- `conversational_a` (English)
- `conversational_b` (English)
- `conversational_fr` (French)
- `conversational_de` (German)

## CLI Example

```bash
mlx-audio-swift-tts --model Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit --voice conversational_a --text "Hello world."
```

## Swift Example

```swift
import MLXAudioTTS

let model = try await MarvisTTSModel.fromPretrained("Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit")
let audio = try await model.generate(
    text: "Hello world.",
    voice: "conversational_a",
    refAudio: nil,
    refText: nil,
    language: nil,
    generationParameters: GenerateParameters()
)
```
