# MiMo-V2.5-ASR

MiMo-V2.5-ASR is Xiaomi's speech recognition model for English and Mandarin Chinese. It returns a transcript and a segment covering the supplied recording. Language can be detected automatically or guided with an English or Chinese hint.

Supply mono audio at 24 kHz. Text streaming returns the transcript as it is generated after the recording has been processed.

## Supported models

| Repository ID | Description |
| --- | --- |
| [`mlx-community/MiMo-V2.5-ASR-MLX`](https://huggingface.co/mlx-community/MiMo-V2.5-ASR-MLX) | 4-bit MiMo-V2.5-ASR |

The model also uses [`mlx-community/MiMo-Audio-Tokenizer`](https://huggingface.co/mlx-community/MiMo-Audio-Tokenizer). Loading a repository downloads and caches this dependency automatically. Local models can select it through their `mlx_manifest.json` or the Swift `audioTokenizerDirectory` argument.

## CLI

```bash
mlx-audio-swift-stt --model mlx-community/MiMo-V2.5-ASR-MLX --audio speech.wav --output-path transcript --format txt

mlx-audio-swift-stt --model mlx-community/MiMo-V2.5-ASR-MLX --audio speech.wav --language en --stream --output-path transcript --format txt
```

A local model directory can also be passed to `--model`. Use `--language zh` for a Mandarin hint. Longer transcripts may need a larger `--max-tokens` budget.

## Swift

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await MiMoASRModel.fromPretrained(
    "mlx-community/MiMo-V2.5-ASR-MLX"
)
let (_, audio) = try loadAudioArray(
    from: URL(fileURLWithPath: "speech.wav"), sampleRate: 24000
)

let output = model.generate(audio: audio)
print(output.text)
```

Guide the language and allow a longer transcript:

```swift
let output = model.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(maxTokens: 1024, language: "en")
)
print(output.text)
```

Omit `language` for automatic detection. Supported hints include `en`, `English`, `zh`, `Chinese`, and `Mandarin`.

### Streaming

```swift
for try await event in model.generateStream(audio: audio) {
    switch event {
    case .token(let text):
        print(text, terminator: "")
    case .result(let output):
        print("\n\(output.text)")
    case .info:
        break
    }
}
```

### Local loading

```swift
let model = try await MiMoASRModel.fromModelDirectory(
    URL(fileURLWithPath: "/models/MiMo-V2.5-ASR-MLX", isDirectory: true),
    audioTokenizerDirectory: URL(
        fileURLWithPath: "/models/MiMo-Audio-Tokenizer", isDirectory: true
    )
)
```

## License

MiMo-V2.5-ASR is released under Apache 2.0; see the [ASR model card](https://huggingface.co/mlx-community/MiMo-V2.5-ASR-MLX). The audio tokenizer weights use the MIT license; see the [tokenizer model card](https://huggingface.co/mlx-community/MiMo-Audio-Tokenizer).
