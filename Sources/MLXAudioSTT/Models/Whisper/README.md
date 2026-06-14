# Whisper

OpenAI's encoder-decoder ASR model. Supports every released size and both
the HuggingFace `transformers` and the OpenAI / mlx-whisper checkpoint
layouts.

## Supported Models

HuggingFace `transformers` layout — every `openai/whisper-*` size and `.en`
variant, including:

- [`openai/whisper-tiny`](https://huggingface.co/openai/whisper-tiny)
- [`openai/whisper-base`](https://huggingface.co/openai/whisper-base)
- [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
- [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium)
- [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3)
- [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)

OpenAI / mlx-whisper layout — every `mlx-community/whisper-*` mirror,
including:

- [`mlx-community/whisper-tiny-mlx`](https://huggingface.co/mlx-community/whisper-tiny-mlx)
- [`mlx-community/whisper-large-v3-mlx`](https://huggingface.co/mlx-community/whisper-large-v3-mlx)
- [`mlx-community/whisper-large-v3-turbo`](https://huggingface.co/mlx-community/whisper-large-v3-turbo)

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let model = try await WhisperModel.fromPretrained("openai/whisper-tiny")
let output = model.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(language: "en")
)
print(output.text)
```

## Streaming Example

```swift
for try await event in model.generateStream(audio: audio) {
    switch event {
    case .token(let token):
        print(token, terminator: "")
    case .result(let result):
        print("\nFinal text: \(result.text)")
    case .info:
        break
    }
}
```

## Notes

- Input audio should be mono 16 kHz.
- Whisper consumes a fixed 30 s window; longer audio is split into
  non-overlapping 30 s chunks and joined with a single space. Per-chunk
  offsets appear as `start` / `end` in `STTOutput.segments`.
- Pass `language` (ISO code or English name) to force the transcription
  language for multilingual variants. Omit it to let Whisper auto-detect.
  The `.en` repos ignore the parameter.
- `mlx-community/whisper-*` repos ship weights only; the loader fetches
  the matching tokenizer from the sibling `openai/whisper-*` repo on demand
  (no weight re-download).
