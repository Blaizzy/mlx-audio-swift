# Parakeet STT

Parakeet speech-to-text model support for `MLXAudioSTT`.

## Supported Models

- [mlx-community/parakeet-tdt-1.1b](https://huggingface.co/mlx-community/parakeet-tdt-1.1b)
- [mlx-community/parakeet-tdt-0.6b-v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3)
- [mlx-community/parakeet-tdt_ctc-1.1b](https://huggingface.co/mlx-community/parakeet-tdt-1.1b)
- [mlx-community/parakeet-ctc-0.6b](https://huggingface.co/mlx-community/parakeet-ctc-0.6b)
- [mlx-community/parakeet-ctc-1.1b](https://huggingface.co/mlx-community/parakeet-ctc-1.1b)
- [mlx-community/parakeet-rnnt-0.6b](https://huggingface.co/mlx-community/parakeet-rnnt-0.6b)
- [mlx-community/parakeet-rnnt-1.1b](https://huggingface.co/mlx-community/parakeet-rnnt-1.1b)
- [mlx-community/parakeet-tdt_ctc-110m](https://huggingface.co/mlx-community/parakeet-tdt_ctc-110m)

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL)

let model = try await ParakeetModel.fromPretrained("mlx-community/parakeet-tdt-0.6b-v3")
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
