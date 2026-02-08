# GLM-ASR

A speech-to-text (STT) model that combines a Whisper-style audio encoder with a GLM/LLaMA-style decoder.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/GLM-ASR-Nano-2512-4bit)

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (sampleRate, audio) = try loadAudioArray(from: audioURL)
_ = sampleRate

let model = try await GLMASRModel.fromPretrained("mlx-community/GLM-ASR-Nano-2512-4bit")
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
