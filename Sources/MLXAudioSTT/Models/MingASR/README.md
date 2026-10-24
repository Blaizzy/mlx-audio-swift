# Ming-3.0 ASR Tiny

Ming-3.0 ASR Tiny transcribes Chinese and English speech, including regional Chinese dialects. It detects the language automatically and accepts language hints, hotwords, and topic or dialogue context through the Swift API.

Supply mono audio at 16 kHz. Output is plain transcription text and a language label. Audio is encoded in independent chunks within the model's 8,192-token context window. Token streaming emits text as it is decoded from a supplied recording.

## Supported models

| Repository ID | Precision |
| --- | --- |
| [`mlx-community/Ming-3.0-asr-tiny-4bit`](https://huggingface.co/mlx-community/Ming-3.0-asr-tiny-4bit) | 4-bit, lower memory use |
| [`mlx-community/Ming-3.0-asr-tiny-bf16`](https://huggingface.co/mlx-community/Ming-3.0-asr-tiny-bf16) | BF16 |

## CLI

```bash
mlx-audio-swift-stt --model mlx-community/Ming-3.0-asr-tiny-4bit --audio speech.wav --output-path transcript --format txt

mlx-audio-swift-stt --model mlx-community/Ming-3.0-asr-tiny-4bit --audio speech.wav --language zh --stream --output-path transcript --format json
```

A local model directory can also be passed to `--model`.

## Swift

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await STT.loadModel(
    modelRepo: "mlx-community/Ming-3.0-asr-tiny-4bit"
)
let (_, audio) = try loadAudioArray(
    from: URL(fileURLWithPath: "speech.wav"), sampleRate: 16000
)
let output = model.generate(audio: audio)
print(output.text)
print(output.language ?? "")

for try await event in model.generateStream(audio: audio) {
    if case .token(let text) = event { print(text, terminator: "") }
}
```

Use the model class for contextual transcription. Language hints accept `en` or `zh`; omit the hint for automatic detection. Custom prompts begin with `transcriptionInstruction` and keep context on the same line.

```swift
let ming = try await MingASRModel.fromPretrained(
    "mlx-community/Ming-3.0-asr-tiny-4bit"
)
let result = ming.generate(
    audio: audio,
    generationParameters: .init(maxTokens: 256, language: "en"),
    prompt: MingASRModel.transcriptionInstruction + " This recording discusses Apple development.",
    hotwords: ["MLX", "Swift", "Metal"]
)
print(result.text)
```

Ming model weights are released under the [MIT License](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/blob/main/LICENSE), copyright 2026 inclusionAI. The BailingMoE V3 source carries the [Apache 2.0 license](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny/blob/main/modeling_ming_v3_asr.py). The expert layers and delta recurrence are adapted from [mlx-lm](https://github.com/ml-explore/mlx-lm) under the [MIT License](https://github.com/ml-explore/mlx-lm/blob/main/LICENSE).
