# Voxtral Mini 4B Realtime

Voxtral Mini 4B Realtime 2602 is a multilingual, realtime speech-transcription model. It supports 13 languages and outperforms existing open-source baselines across a range of tasks, making it ideal for applications like voice assistants and live subtitling.

## Supported Models

[mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16)
[mlx-community/Voxtral-Mini-4B-Realtime-2602-6bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-6bit)
[mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit](https://huggingface.co/mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit)

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL)

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
