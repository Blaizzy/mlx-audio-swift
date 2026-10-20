# Pocket TTS and Phonon

A lightweight text-to-speech (TTS) model from Kyutai designed to run efficiently on CPUs.

[Blog Post](https://kyutai.org/blog/2026-01-13-pocket-tts)

Gradium Phonon extends the Pocket TTS architecture with speaker and duration
conditioning. Load the official checkpoints through `PocketTTSModel` or
`TTS.loadModel` for 24 kHz speech and incremental streaming.

| Model | Parameters | Languages |
|---|---|---|
| [gradium/phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en) | 31M | English |
| [gradium/phonon-40m-en](https://huggingface.co/gradium/phonon-40m-en) | 40M | English |
| [gradium/phonon-40m-fr](https://huggingface.co/gradium/phonon-40m-fr) | 40M | French |
| [gradium/phonon-40m-de](https://huggingface.co/gradium/phonon-40m-de) | 40M | German |
| [gradium/phonon-40m-es](https://huggingface.co/gradium/phonon-40m-es) | 40M | Spanish |
| [gradium/phonon-40m-pt](https://huggingface.co/gradium/phonon-40m-pt) | 40M | Portuguese |
| [gradium/phonon-40m-multilingual](https://huggingface.co/gradium/phonon-40m-multilingual) | 40M | English, French, German, Spanish, Portuguese |

For the 40M models, accept their Hugging Face access terms and provide `HF_TOKEN`.
The official checkpoints include `default-voice.safetensors`, selected when
`voice` is omitted or set to `"default"`. To use another embedding, pass its local
`.safetensors` path or an `hf://owner/repo/path.safetensors` URL (optionally with
`@revision`). Checkpoints that omit the speaker encoder require voice embeddings
instead of reference audio.

The earlier [nativ-community/phonon](https://huggingface.co/nativ-community/phonon)
checkpoint remains supported, including its `Freya` (default), `Harper`,
`Sterling`, and `Toby` voices. Voice names are case-insensitive.

```bash
mlx-audio-swift-tts --model gradium/phonon-31m-en --text "Hello world."
```

```swift
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "gradium/phonon-31m-en")
let audio = try await model.generate(
    text: "Hello world.",
    voice: nil,
    refAudio: nil, refText: nil, language: nil,
    generationParameters: model.defaultGenerationParameters
)

for try await event in model.generateStream(
    text: "Hello world.", voice: "default",
    refAudio: nil, refText: nil, language: nil,
    generationParameters: model.defaultGenerationParameters,
    streamingInterval: 0.32
) {
    if case .audio(let chunk) = event {
        // Play or collect successive chunks at model.sampleRate.
    }
}
```

## Pocket TTS Voices

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
