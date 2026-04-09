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

## Notes

- Input audio should be mono 16 kHz.
- The current Swift port follows the model's default prompt format with punctuation enabled and timestamps disabled.
- The converted MLX checkpoint used for Swift integration was uploaded during this session.
