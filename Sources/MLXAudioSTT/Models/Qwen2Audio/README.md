# Qwen2-Audio

Qwen2-Audio is a conversational audio model from Qwen. It can transcribe speech, translate it, and answer questions about speech and other sounds. Supply mono audio at 16 kHz. Each clip is limited to 30 seconds; longer clips are truncated, so split longer recordings before passing them to the model.

Responses can include conversational text around a transcription. Multiple clips supplied together form one conversation.

## Supported models

| Repository ID | Description |
| --- | --- |
| [`mlx-community/Qwen2-Audio-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit) | Instruction-tuned model with 4-bit language-model weights |
| [`Qwen/Qwen2-Audio-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | Original instruction-tuned model |

## CLI

```bash
mlx-audio-swift-stt --model mlx-community/Qwen2-Audio-7B-Instruct-4bit --audio speech.wav --output-path transcript --format txt

mlx-audio-swift-stt --model mlx-community/Qwen2-Audio-7B-Instruct-4bit --audio speech.wav --stream --output-path transcript --format txt
```

You can also pass a local model directory to `--model`. Custom prompts and multiple clips are available through the Swift API.

## Swift

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await Qwen2AudioModel.fromPretrained(
    "mlx-community/Qwen2-Audio-7B-Instruct-4bit"
)
let (_, audio) = try loadAudioArray(
    from: URL(fileURLWithPath: "speech.wav"), sampleRate: 16000
)

let transcript = model.generate(audio: audio)
print(transcript.text)

let answer = model.generate(
    audio: audio, prompt: "What is the speaker talking about?"
)
print(answer.text)

let translation = model.generate(
    audio: audio, prompt: "Translate the speech into English."
)
print(translation.text)
```

Ask a question about several clips:

```swift
let (_, secondAudio) = try loadAudioArray(
    from: URL(fileURLWithPath: "second.wav"), sampleRate: 16000
)
let comparison = model.generate(
    audio: [audio, secondAudio], prompt: "Describe the differences between these clips."
)
print(comparison.text)
```

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

Use `STTGenerateParameters` to set generation limits and sampling options:

```swift
let output = model.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(maxTokens: 256, temperature: 0)
)
print(output.text)
```

## License

Qwen2-Audio is released under [Apache 2.0](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct/blob/main/LICENSE). See the model repositories for their model cards and license terms.
