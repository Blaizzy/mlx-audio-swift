# FireRed ASR 2

Swift support for FireRed ASR 2 in `MLXAudioSTT`.

## Supported Model

- [`mlx-community/FireRedASR2-AED-mlx`](https://huggingface.co/mlx-community/FireRedASR2-AED-mlx)

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let model = try await FireRedASR2Model.fromPretrained("mlx-community/FireRedASR2-AED-mlx")
let output = model.generate(audio: audio)
print(output.text)
```

## Custom Beam Search

```swift
let output = model.generate(
    audio: audio,
    beamSize: 5,
    softmaxSmoothing: 1.25,
    lengthPenalty: 0.6,
    eosPenalty: 1.0,
    maxLen: 0
)
```

## Notes

- Input audio should be mono 16 kHz. `loadAudioArray(from:sampleRate:)` can resample automatically.
