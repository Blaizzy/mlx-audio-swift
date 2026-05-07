# Silero VAD

Swift / MLX port of the Silero voice activity detector (`silero_vad`).

## Supported Models

| Model | HuggingFace Repo |
|-------|------------------|
| Silero VAD v5 | [`mlx-community/silero-vad`](https://huggingface.co/mlx-community/silero-vad) |
| Silero VAD v6 | [`mlx-community/silero-vad-v6`](https://huggingface.co/mlx-community/silero-vad-v6) |

Both ship the same `vad_16k.*` / `vad_8k.*` weight layout — the loader works for either.

## Usage

```swift
import MLXAudioVAD
import MLXAudioCore

let model = try await SileroVAD.fromPretrained("mlx-community/silero-vad")

let (sampleRate, audio) = try loadAudioArray(from: audioURL)
let timestamps = try model.getSpeechTimestamps(audio, sampleRate: sampleRate)
for ts in timestamps {
    let start = Float(ts.start) / Float(sampleRate)
    let end = Float(ts.end) / Float(sampleRate)
    print("[\(start)s - \(end)s]")
}
```

## Streaming

Feed 512 samples at 16 kHz (or 256 samples at 8 kHz) per call:

```swift
var state = try model.initialState(sampleRate: 16000)
let (probability, newState) = try model.feed(chunk: chunk, state: state, sampleRate: 16000)
state = newState
```

## Sample Rate

Silero supports 16 kHz and 8 kHz. Other rates are rejected with `SileroVADError.unsupportedSampleRate`. The Swift port does not auto-resample — convert your audio with `MLXAudioCore.resampleAudio` first if needed.
