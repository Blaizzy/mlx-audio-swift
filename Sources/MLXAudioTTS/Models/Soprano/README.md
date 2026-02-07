# Soprano

A compact autoregressive text-to-speech (TTS) model for efficient local synthesis.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/Soprano-80M-bf16)

## CLI Example

```bash
mlx-audio-swift-tts --model mlx-community/Soprano-80M-bf16 --text "Hello world."
```

## Swift Example

```swift
import MLXAudioTTS

let model = try await SopranoModel.fromPretrained("mlx-community/Soprano-80M-bf16")
let audio = try await model.generate(
    text: "Hello world.",
    parameters: GenerateParameters()
)
```

## Notes

- The `voice` argument is currently unused in the base `SopranoModel` implementation.
