# Fun-ASR-Nano

Fun-ASR-Nano is a compact speech recognition model from FunAudioLLM for Chinese,
English, and Japanese. It supports language hints, hotword biasing, and inverse
text normalization for written-form transcripts.

## Supported Models

- [`mlx-community/Fun-ASR-Nano-2512`](https://huggingface.co/mlx-community/Fun-ASR-Nano-2512)

## CLI Example

```bash
mlx-audio-swift-stt \
  --model mlx-community/Fun-ASR-Nano-2512 \
  --audio audio.wav \
  --output-path transcript \
  --format txt \
  --language en
```

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await FunASRNanoModel.fromPretrained("mlx-community/Fun-ASR-Nano-2512")
let audioURL = URL(fileURLWithPath: "audio.wav")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let result = model.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(maxTokens: 512, language: "en"),
    hotwords: ["MLX", "Apple Silicon"],
    itn: true
)
print(result.text)
```

Use `language: "zh"` for Chinese or `language: "ja"` for Japanese. Inverse text
normalization is enabled by default; pass `itn: false` to `generate` or
`generateStream` to request spoken-form text. Hotwords and ITN options are
available through the Swift API.
