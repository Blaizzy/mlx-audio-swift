# MOSS-Music

MOSS-Music understands music and transcribes Chinese and English lyrics. It can describe a recording, answer questions about instruments, mood, key or tempo, and produce timestamped lyrics when requested.

Supply mono audio at 16 kHz. The default prompt asks for a detailed musical description. Use the Swift API to provide a lyrics-transcription prompt or a music question. Timestamped responses populate `output.segments` for JSON, SRT and VTT output.

## Supported models

| Repository ID | Variant | Precision |
| --- | --- | --- |
| [`OpenMOSS-Team/MOSS-Music-8B-Instruct`](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) | Instruct | BF16 |
| [`rtikw/MOSS-Music-8B-Instruct-4bit`](https://huggingface.co/rtikw/MOSS-Music-8B-Instruct-4bit) | Instruct | 4-bit |
| [`OpenMOSS-Team/MOSS-Music-8B-Thinking`](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Thinking) | Thinking | BF16 |
| [`mlx-community/MOSS-Music-8B-Thinking-4bit`](https://huggingface.co/mlx-community/MOSS-Music-8B-Thinking-4bit) | Thinking | 4-bit |
| [`mlx-community/MOSS-Music-8B-Thinking-6bit`](https://huggingface.co/mlx-community/MOSS-Music-8B-Thinking-6bit) | Thinking | 6-bit |
| [`mlx-community/MOSS-Music-8B-Thinking-8bit`](https://huggingface.co/mlx-community/MOSS-Music-8B-Thinking-8bit) | Thinking | 8-bit |

## CLI

Generate a music description, optionally streaming the answer:

```bash
mlx-audio-swift-stt --model rtikw/MOSS-Music-8B-Instruct-4bit --audio song.wav --output-path description --format txt

mlx-audio-swift-stt --model rtikw/MOSS-Music-8B-Instruct-4bit --audio song.wav --stream --output-path description --format json
```

A local model directory can also be passed to `--model`.

## Swift

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await MossMusicModel.fromPretrained(
    "rtikw/MOSS-Music-8B-Instruct-4bit"
)
let (_, audio) = try loadAudioArray(
    from: URL(fileURLWithPath: "song.wav"), sampleRate: 16000
)
let output = model.generate(
    audio: audio,
    prompt: "Transcribe the lyrics of this song, with timestamps."
)
print(output.text)
for segment in output.segments ?? [] {
    print(segment)
}

for try await event in model.generateStream(
    audio: audio, prompt: "Describe the instruments and mood of this music."
) {
    if case .token(let text) = event { print(text, terminator: "") }
}
```

Thinking models may spend a substantial token budget reasoning before answering. Set `enableThinking: false` for a direct answer. Reasoning is hidden by default; `stripThinking: false` retains it. Time markers are enabled by default and can be disabled with `enableTimeMarkers: false`.

```swift
let thinking = try await MossMusicModel.fromPretrained(
    "mlx-community/MOSS-Music-8B-Thinking-4bit"
)
let answer = thinking.generate(
    audio: audio,
    generationParameters: .init(maxTokens: 512),
    prompt: "What instruments are playing?",
    enableThinking: false
)
print(answer.text)
```

MOSS-Music is released under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). See the [OpenMOSS project](https://github.com/OpenMOSS/MOSS-Music) for model credits.
