# Qwen3-ASR & Qwen3-ForcedAligner

Speech-to-text and forced-alignment models based on Qwen3 ASR architecture.

## Supported Models

ASR (1.7B):

- [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)
- [mlx-community/Qwen3-ASR-1.7B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit)
- [mlx-community/Qwen3-ASR-1.7B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-6bit)
- [mlx-community/Qwen3-ASR-1.7B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-4bit)

ASR (0.6B):

- [mlx-community/Qwen3-ASR-0.6B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-bf16)
- [mlx-community/Qwen3-ASR-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-8bit)
- [mlx-community/Qwen3-ASR-0.6B-6bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-6bit)
- [mlx-community/Qwen3-ASR-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit)

Forced Alignment (0.6B):

- [mlx-community/Qwen3-ForcedAligner-0.6B-bf16](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-bf16)
- [mlx-community/Qwen3-ForcedAligner-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-8bit)
- [mlx-community/Qwen3-ForcedAligner-0.6B-6bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-6bit)
- [mlx-community/Qwen3-ForcedAligner-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-4bit)

## Swift Example (ASR)

```swift
import MLXAudioCore
import MLXAudioSTT

let (sampleRate, audio) = try loadAudioArray(from: audioURL)
_ = sampleRate

let model = try await Qwen3ASRModel.fromPretrained("mlx-community/Qwen3-ASR-0.6B-4bit")
let output = model.generate(audio: audio, language: "English")
print(output.text)
```

## Streaming Example (ASR)

```swift
for try await event in model.generateStream(audio: audio, language: "English") {
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

## Swift Example (Forced Alignment)

```swift
import MLXAudioCore
import MLXAudioSTT

let (sampleRate, audio) = try loadAudioArray(from: audioURL)
_ = sampleRate

let model = try await Qwen3ForcedAlignerModel.fromPretrained("mlx-community/Qwen3-ForcedAligner-0.6B-4bit")
let result = model.generate(
    audio: audio,
    text: "The transcript to align",
    language: "English"
)

for item in result.items {
    print("[\(item.startTime)s - \(item.endTime)s] \(item.text)")
}
```
