# KugelAudio

KugelAudio generates 24 kHz mono speech in European languages using an autoregressive language model and acoustic diffusion. Guidance strength and diffusion steps are configurable. Generation decodes the complete utterance together; the stream API emits one final waveform.

Supported checkpoint: [kugelaudio/kugelaudio-0-open](https://huggingface.co/kugelaudio/kugelaudio-0-open).

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await KugelAudioModel.fromPretrained("kugelaudio/kugelaudio-0-open")
let audio = try await model.generate(
    text: "The next train leaves in ten minutes.",
    options: .init(guidanceScale: 3, diffusionSteps: 20),
    generationParameters: .init(maxTokens: 2048, seed: 42)
)
```

The shared `TTS.loadModel(modelRepo:)` API also loads this checkpoint and local checkpoint directories. The current generation API uses text conditioning; voice presets and reference-audio cloning are unavailable.

```bash
mlx-audio-swift-tts --model kugelaudio/kugelaudio-0-open \
  --text "The next train leaves in ten minutes." --output speech.wav
```

[KugelAudio code and model](https://github.com/Kugelaudio/kugelaudio-open) are MIT licensed. The implementation follows [mlx-audio](https://github.com/Blaizzy/mlx-audio) (MIT) and uses the [Qwen2.5 tokenizer](https://huggingface.co/Qwen/Qwen2.5-7B) (Apache 2.0) and VibeVoice acoustic architecture.
