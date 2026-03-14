# Echo TTS

Diffusion-based text-to-speech with voice cloning from a short reference clip.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/echo-tts-base)

## CLI Example

```bash
mlx-audio-swift-tts \
  --model mlx-community/echo-tts-base \
  --text "Hello from Echo TTS." \
  --ref-audio speaker.wav
```

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await EchoTTSModel.fromPretrained("mlx-community/echo-tts-base")
let (_, refAudio) = try loadAudioArray(
    from: URL(fileURLWithPath: "speaker.wav"),
    sampleRate: model.sampleRate
)

let audio = try await model.generate(
    text: "Hello from Echo TTS.",
    voice: nil,
    refAudio: refAudio,
    refText: nil,
    language: nil
)
```

## License

Echo-TTS and Fish S1 weights are released under `CC-BY-NC-SA-4.0`.
Use is non-commercial unless you have separate permission from the model authors.
