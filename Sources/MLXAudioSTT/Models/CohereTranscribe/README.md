# Cohere Transcribe 03-2026

Swift support for Cohere's encoder-decoder ASR model in `MLXAudioSTT`.

## Supported Model

- [`beshkenadze/cohere-transcribe-03-2026-mlx-fp16`](https://huggingface.co/beshkenadze/cohere-transcribe-03-2026-mlx-fp16)

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let model = try await CohereTranscribeModel.fromPretrained(
    "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"
)
let output = model.generate(audio: audio)
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

## Optional VAD Pre-processing

Cohere's encoder has a positional-encoding limit of ≈ 6.7 minutes. Long-form audio is therefore split into chunks before transcription. Two strategies are available:

| Strategy | When | Notes |
|---|---|---|
| Fixed-duration energy chunking (default) | Clean dense speech (audiobooks, narration) | Splits at low-energy points within each `chunkDuration` window |
| Silero VAD (opt-in) | Long-form audio with silences / non-speech (meetings, podcasts, interviews) | Trims silence, aligns chunks to natural pauses |

```swift
import MLXAudioSTT
import MLXAudioVAD

let model = try await CohereTranscribeModel.fromPretrained(
    "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"
)
let vad = try await SileroVAD.fromPretrained("mlx-community/silero-vad")

let output = model.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(language: "en"),
    vad: (model: vad, config: CohereVADConfig())
)
```

`CohereVADConfig` exposes `threshold`, `minSpeechMs`, `minSilenceMs`, `speechPadMs`, `mergeGapS`, `maxChunkS`. Defaults match Silero's recommendations and the encoder's 30 s safe chunk size.

### Measured trade-offs

10-min English meeting recording (silence + speech, M1 Max, Release):

| | Wall | 3-gram repeats (>=3) | Notable |
|---|---|---|---|
| Fixed chunking | 26 s | 6 (incl. `'a very strong' x3` hallucinated on initial silence) | Hallucinations on silent leading audio |
| `vad: ...` | 30 s (+15 %) | 2 (natural meeting filler only) | Clean start, natural sentence boundaries |

30-min concatenated LibriSpeech (clean audiobook reads, ground-truth WER):

| | Wall | WER | Insertions |
|---|---|---|---|
| Fixed chunking | 107 s | **1.66 %** | 6 |
| `vad: ...` | 117 s (+9 %) | 2.39 % | 27 |

Take-away: VAD pre-processing **is opt-in by design**. It improves long-form ASR on audio with silences or non-speech sections (meetings, podcasts), at a small wall-clock cost. On clean dense narration it produces no quality benefit and can add insertions at chunk boundaries — keep the default fixed chunking for that case.

## Notes

- Input audio should be mono 16 kHz.
- The current Swift port follows the model's default prompt format with punctuation enabled and timestamps disabled.
- The converted MLX checkpoint used for Swift integration was uploaded during this session.
