# Parakeet STT

Parakeet speech-to-text models support multilingual transcription, timestamps, and
chunked recordings in `MLXAudioSTT`. Parakeet Redux is Moondream's compact ternary
derivative of NVIDIA's Parakeet TDT 0.6B v3.

## Supported Models

- [mlx-community/parakeet-tdt-1.1b](https://huggingface.co/mlx-community/parakeet-tdt-1.1b)
- [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3)
- [mlx-community/parakeet-tdt-0.6b-v2](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2)
- [mlx-community/parakeet-tdt_ctc-1.1b](https://huggingface.co/mlx-community/parakeet-tdt-1.1b)
- [mlx-community/parakeet-ctc-0.6b](https://huggingface.co/mlx-community/parakeet-ctc-0.6b)
- [mlx-community/parakeet-ctc-1.1b](https://huggingface.co/mlx-community/parakeet-ctc-1.1b)
- [mlx-community/parakeet-rnnt-0.6b](https://huggingface.co/mlx-community/parakeet-rnnt-0.6b)
- [mlx-community/parakeet-rnnt-1.1b](https://huggingface.co/mlx-community/parakeet-rnnt-1.1b)
- [mlx-community/parakeet-tdt_ctc-110m](https://huggingface.co/mlx-community/parakeet-tdt_ctc-110m)

- Parakeet Redux: [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux),
  using an MLX 2-bit checkpoint prepared by Python `mlx-audio`.

## CLI Example

```bash
mlx-audio-swift-stt \
  --model mlx-community/parakeet-tdt-0.6b-v3 \
  --audio audio.wav \
  --output-path transcript \
  --format txt
```

For a local Parakeet Redux checkpoint:

```bash
mlx-audio-swift-stt \
  --model ./parakeet-redux-mlx \
  --audio audio.wav \
  --output-path transcript \
  --format txt
```

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL)

let model = try await ParakeetModel.fromPretrained("mlx-community/parakeet-tdt-0.6b-v3")
let output = model.generate(audio: audio)
print(output.text)
```

For a local Parakeet Redux checkpoint:

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try ParakeetModel.fromDirectory(URL(fileURLWithPath: "parakeet-redux-mlx"))
let (_, audio) = try loadAudioArray(from: URL(fileURLWithPath: "audio.wav"), sampleRate: 16000)
print(model.generate(audio: audio).text)
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

## Parakeet Redux License

Parakeet Redux weights are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution: [Moondream](https://huggingface.co/moondream/parakeet-redux) for Parakeet Redux
and [NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) for the original Parakeet model.
