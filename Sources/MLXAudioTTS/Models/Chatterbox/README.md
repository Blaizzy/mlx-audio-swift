# Chatterbox TTS

Two-stage speech synthesis with voice cloning. Converts text to speech tokens (T3), then speech tokens to 24kHz audio (S3Gen + HiFi-GAN). Two variants: Regular (LLaMA 520M, multilingual, emotion control) and Turbo (GPT-2 Medium, English, faster).

## Supported Models

- [mlx-community/Chatterbox-TTS-fp16](https://huggingface.co/mlx-community/Chatterbox-TTS-fp16) (Regular — 23 languages, ~500M params)
- [mlx-community/chatterbox-turbo-fp16](https://huggingface.co/mlx-community/chatterbox-turbo-fp16) (Turbo — English only, ~350M params)
- [mlx-community/chatterbox-turbo-8bit](https://huggingface.co/mlx-community/chatterbox-turbo-8bit) (Turbo — 8-bit quantized)
- [mlx-community/chatterbox-turbo-4bit](https://huggingface.co/mlx-community/chatterbox-turbo-4bit) (Turbo — 4-bit quantized)

## Swift Example

```swift
import MLXAudioTTS

// Load model (ships with a default voice)
let model = try await ChatterboxModel.fromPretrained("mlx-community/chatterbox-turbo-fp16")
let audio = try await model.generate(
    text: "Hello, this is a test of the Chatterbox model.",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: GenerateParameters(temperature: 0.8)
)
```

### Voice Cloning

Provide reference audio to clone a speaker's voice:

```swift
import MLXAudioCore

let (_, refAudio) = try loadAudioArray(from: referenceAudioURL)
let audio = try await model.generate(
    text: "This will sound like the reference speaker.",
    voice: nil, refAudio: refAudio, refText: nil, language: nil,
    generationParameters: GenerateParameters(temperature: 0.8)
)
```

## Streaming Example

```swift
for try await event in model.generateStream(
    text: "Streaming speech generation.",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: GenerateParameters(temperature: 0.8)
) {
    switch event {
    case .audio(let samples):
        // Process audio chunk
        break
    case .info(let info):
        print("Generated in \(info.generateTime)s")
    }
}
```
