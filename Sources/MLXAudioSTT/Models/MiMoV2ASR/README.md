# MiMo-V2.5-ASR

MLX Swift implementation of XiaomiMiMo's MiMo-V2.5-ASR speech-to-text model.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/MiMo-V2.5-ASR-MLX)

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL)

let model = try await MiMoV2ASRModel.fromPretrained("mlx-community/MiMo-V2.5-ASR-MLX")

let output = model.generate(audio: audio)
print(output.text)
```

The published `mlx_manifest.json` resolves `mlx-community/MiMo-Audio-Tokenizer` automatically. For offline or fully local setups, you can still pass `tokenizerPath` explicitly.

The published tokenizer repo is the MLX encoder/RVQ subset used by MiMo ASR. It does not include the full decoder/vocoder stack from the original `XiaomiMiMo/MiMo-Audio-Tokenizer` release.

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
