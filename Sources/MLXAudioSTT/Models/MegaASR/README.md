# Mega-ASR

Mega-ASR is a multilingual speech recognition model that automatically adapts to
clean, noisy, and far-field recordings. It supports Chinese, Cantonese, English,
German, Spanish, French, Italian, Portuguese, Russian, Korean, and Japanese.

## Supported Models

- [`mlx-community/Mega-ASR-bf16`](https://huggingface.co/mlx-community/Mega-ASR-bf16)

## CLI Example

```bash
mlx-audio-swift-stt \
  --model mlx-community/Mega-ASR-bf16 \
  --audio audio.wav \
  --output-path transcript \
  --format txt \
  --language English
```

The CLI uses automatic routing for recordings up to about 34 seconds. Split
longer recordings into shorter files. Add `--stream` to stream generated text.

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await MegaASRModel.fromPretrained("mlx-community/Mega-ASR-bf16")
let audioURL = URL(fileURLWithPath: "audio.wav")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let result = model.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(maxTokens: 512, language: "English")
)
print(result.text)
```

Routing is automatic by default. Use `transcribe` or `transcribeStream` to supply
context hints with `context:`, or choose a route with `routing: .base` or
`routing: .robust`.
