# Breeze TTS 2

Breeze TTS 2 is a bilingual English and Chinese text-to-speech model with natural-language voice design and zero-shot voice cloning.

## Load a model

The same Swift code works with the BF16, 8-bit, and 4-bit MLX checkpoints:

- `mlx-community/Breeze-TTS-2-mlx`
- `mlx-community/Breeze-TTS-2-mlx-8bit`
- `mlx-community/Breeze-TTS-2-mlx-4bit`

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Breeze-TTS-2-mlx-4bit"
)

let audio = try await model.generate(
    text: "Hello from Breeze TTS 2.",
    voice: "A clear and calm English narrator",
    refAudio: nil,
    refText: nil,
    language: "English"
)
```

`voice` is the voice-design instruction. The generic `language` field does not change Breeze prompt assembly; include language, accent, pace, tone, and style in the instruction when needed.

## Voice cloning

Pass one reference waveform and its exact transcript:

```swift
let cloned = try await model.generate(
    text: "This sentence uses the reference voice.",
    voice: nil,
    refAudio: referenceAudio,
    refText: "The exact words spoken in the reference clip.",
    language: "English"
)
```

You can also set `voice` while cloning to direct the cloned voice. The model uses classifier-free guidance for voice design and directed cloning.

## Output and streaming

Breeze returns mono audio at 24 kHz. `generateStream` reports token and timing events while it generates, then yields the decoded waveform as one audio event. The public API can add codec chunk streaming later without changing model loading or prompt controls.

## License

The Swift source in this repository uses the repository license. Breeze model weights, derived checkpoints, and self-hosted output use the Breeze model license, which limits use to research and non-commercial work. Review the [Breeze TTS 2 model card](https://huggingface.co/BreezeBlue/Breeze-TTS-2) before downloading or using a checkpoint.
