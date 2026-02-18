# Voxtral Realtime STT

Voxtral Realtime speech-to-text support for `MLXAudioSTT`.

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL)

let model = try await VoxtralRealtimeModel.fromPretrained("mlx-community/Voxtral-Mini-4B-Realtime-2602")
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
