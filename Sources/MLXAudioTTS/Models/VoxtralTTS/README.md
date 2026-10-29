# Voxtral TTS

[Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) generates expressive 24 kHz speech with 20 preset voices in English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic and Hindi. It supports incremental audio output.

Supported models:

- [mlx-community/Voxtral-4B-TTS-2603-mlx-4bit](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-4bit)
- [mlx-community/Voxtral-4B-TTS-2603-mlx-bf16](https://huggingface.co/mlx-community/Voxtral-4B-TTS-2603-mlx-bf16)

```bash
mlx-audio-swift-tts \
  --model mlx-community/Voxtral-4B-TTS-2603-mlx-4bit \
  --text "Hello, how are you today?" \
  --voice casual_male \
  --output speech.wav
```

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await VoxtralTTSModel.fromPretrained(
    "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
)
let audio = try await model.generate(
    text: "Bonjour, comment allez-vous ?",
    voice: "fr_female",
    options: .init(denoisingSteps: 8, guidanceScale: 1.2),
    generationParameters: .init(maxTokens: 4096, seed: 42)
)
try AudioUtils.writeWavFile(
    samples: audio.asArray(Float.self),
    sampleRate: model.sampleRate,
    fileURL: URL(fileURLWithPath: "speech.wav")
)

for try await event in model.generateStream(
    text: "Hello, how are you today?", voice: "casual_male",
    refAudio: nil, refText: nil, language: nil,
    generationParameters: .init(maxTokens: 4096, seed: 42),
    streamingInterval: 0.64
) {
    if case .audio(let samples) = event {
        // Consume or concatenate each 24 kHz audio chunk.
    }
}
```

`TTS.loadModel(modelRepo:)` also accepts either repository ID or a local model directory. The default voice is `casual_male`; `availableVoices` lists the presets installed with the checkpoint. Set `sanitizeText: false` in `VoxtralTTSGenerationOptions` to preserve raw speech text. Semantic prediction is greedy; `seed` controls acoustic flow noise. `maxTokens` limits audio frames, with each frame representing 80 ms of speech.

English voices: `casual_female`, `casual_male`, `cheerful_female`, `neutral_female`, `neutral_male`.

Other voices: `fr_male`, `fr_female`, `es_male`, `es_female`, `de_male`, `de_female`, `it_male`, `it_female`, `pt_male`, `pt_female`, `nl_male`, `nl_female`, `ar_male`, `hi_male`, `hi_female`. Select the voice for the language of your text. This model uses preset voices and does not accept reference audio.

Voxtral weights use [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The MLX implementation derives from [mlx-audio](https://github.com/Blaizzy/mlx-audio), released under the [MIT license](https://github.com/Blaizzy/mlx-audio/blob/main/LICENSE).
