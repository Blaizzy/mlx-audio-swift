# Voxtral Mini

Voxtral Mini is Mistral AI's offline audio model for speech transcription, audio questions, and summarization. It supports English, Spanish, French, Portuguese, Hindi, German, Dutch, and Italian, with automatic language detection or a language hint.

Supply mono audio at 16 kHz. Longer recordings are processed in 30-second blocks within one request. Token streaming returns text as it is generated after the supplied audio has been processed. For live microphone input, see [Voxtral Realtime](../VoxtralRealtime/README.md).

## Supported models

| Repository ID | Description |
| --- | --- |
| [`mlx-community/Voxtral-Mini-3B-2507-bf16`](https://huggingface.co/mlx-community/Voxtral-Mini-3B-2507-bf16) | BF16 model |
| [`mzbac/voxtral-mini-3b-4bit-mixed`](https://huggingface.co/mzbac/voxtral-mini-3b-4bit-mixed) | Mixed-precision model with lower memory use |

## CLI

```bash
mlx-audio-swift-stt --model mlx-community/Voxtral-Mini-3B-2507-bf16 --audio speech.wav --output-path transcript --format txt

mlx-audio-swift-stt --model mzbac/voxtral-mini-3b-4bit-mixed --audio speech.wav --language fr --stream --output-path transcript --format txt
```

A local model directory can also be passed to `--model`.

## Swift

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await VoxtralModel.fromPretrained(
    "mlx-community/Voxtral-Mini-3B-2507-bf16"
)
let (_, audio) = try loadAudioArray(
    from: URL(fileURLWithPath: "speech.wav"), sampleRate: 16000
)

let output = model.generate(audio: audio)
print(output.text)
```

Provide an ISO 639-1 language code to guide transcription:

```swift
let french = model.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(maxTokens: 256, language: "fr")
)
print(french.text)
```

Omit the language in `STTGenerateParameters` for automatic language detection:

```swift
let detected = model.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(maxTokens: 256)
)
print(detected.text)
```

Ask a question, summarize the recording, or request a translation with a custom prompt:

```swift
let answer = model.generate(
    audio: audio, prompt: "What is the speaker talking about?"
)
print(answer.text)

let summary = model.generate(audio: audio, prompt: "Summarize the recording.")
print(summary.text)

let translation = model.generate(
    audio: audio, prompt: "Translate the speech into English."
)
print(translation.text)
```

A custom prompt selects audio understanding mode. The transcription language hint applies when generating without a custom prompt.

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

## License

Voxtral Mini is released under Apache 2.0. See the [original model card](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) and each supported repository for license information.
