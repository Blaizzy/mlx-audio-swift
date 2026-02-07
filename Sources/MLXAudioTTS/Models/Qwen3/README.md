# VyvoTTS

VyvoTTS is a text-to-speech model by Vyvo team using Qwen3 architecture.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/VyvoTTS-EN-Beta-4bit)

## CLI Example

```bash
mlx-audio-swift-tts --model mlx-community/VyvoTTS-EN-Beta-4bit --text "Hello world."
```

## Swift Example

```swift
import MLXAudioTTS

let model = try await Qwen3Model.fromPretrained("mlx-community/VyvoTTS-EN-Beta-4bit")
let audio = try await model.generate(
    text: "Hello world.",
    voice: "en-us-1",
    parameters: GenerateParameters()
)
```
