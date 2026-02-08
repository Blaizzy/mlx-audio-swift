# Llama TTS (Orpheus)

Orpheus TTS is a Llama-based Speech-LLM designed for high-quality, empathetic text-to-speech generation.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-bf16)

## Suggested Voices

- `tara`
- `leah`
- `jess`
- `leo`
- `dan`
- `mia`
- `zac`
- `zoe`

## CLI Example

```bash
mlx-audio-swift-tts --model mlx-community/orpheus-3b-0.1-ft-bf16 --voice tara --text "Hello world."
```

## Swift Example

```swift
import MLXAudioTTS

let model = try await LlamaTTSModel.fromPretrained("mlx-community/orpheus-3b-0.1-ft-bf16")
let audio = try await model.generate(
    text: "Hello world.",
    voice: "tara",
    parameters: GenerateParameters()
)
```
