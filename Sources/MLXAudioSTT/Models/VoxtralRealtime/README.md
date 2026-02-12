# Voxtral Mini 4B Realtime

A real-time speech-to-text model with a causal transformer encoder, 4x downsample adapter, and GQA decoder with AdaRMSNorm time conditioning.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit)

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (sampleRate, audio) = try loadAudioArray(from: audioURL)
_ = sampleRate

let model = try await VoxtralRealtimeModel.fromPretrained("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit")
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
