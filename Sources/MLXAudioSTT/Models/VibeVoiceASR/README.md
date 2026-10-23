# VibeVoice ASR

VibeVoice ASR transcribes speech with speaker labels. The long-form model returns structured text with speaker IDs and timestamps. The streaming models return speaker-labeled text as successive audio windows become available and retain context across windows.

Supply mono audio at 24 kHz. Streaming models support Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish. Language is detected automatically. Hotwords can guide names and specialized vocabulary through the Swift API.

## Supported models

| Repository ID | Output |
| --- | --- |
| [`microsoft/VibeVoice-ASR`](https://huggingface.co/microsoft/VibeVoice-ASR) | Long-form transcription with speaker IDs and timestamps |
| [`mlx-community/VibeVoice-ASR-4bit`](https://huggingface.co/mlx-community/VibeVoice-ASR-4bit) | Long-form model with lower memory use |
| [`microsoft/VibeVoice-ASR-Streaming-1.5B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B) | Streaming transcription with speaker labels |
| [`microsoft/VibeVoice-ASR-Streaming-7B`](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B) | Larger streaming model with speaker labels |

The long-form repositories use the `Qwen/Qwen2.5-7B` text tokenizer, downloaded automatically when the model repository does not include one. For a local setup, pass its directory as `tokenizerDirectory` to `fromModelDirectory`.

## CLI

```bash
mlx-audio-swift-stt --model mlx-community/VibeVoice-ASR-4bit --audio meeting.wav --output-path transcript --format json

mlx-audio-swift-stt --model microsoft/VibeVoice-ASR-Streaming-1.5B --audio meeting.wav --stream --output-path transcript --format txt
```

A local model directory can also be passed to `--model`. Streaming checkpoints automatically use their audio window settings. Use a long-form checkpoint when you need timestamps from the model.

## Swift

```swift
import Foundation
import MLX
import MLXAudioCore
import MLXAudioSTT

let model = try await VibeVoiceASRModel.fromPretrained(
    "mlx-community/VibeVoice-ASR-4bit"
)
let (_, audio) = try loadAudioArray(
    from: URL(fileURLWithPath: "meeting.wav"), sampleRate: 24000
)

let output = model.generate(audio: audio, hotwords: ["VibeVoice", "MLX"])
print(output.text)
for segment in output.segments ?? [] {
    print(segment)
}
```

The long-form model returns JSON text. `output.segments` provides its parsed `start`, `end`, `speaker_id`, and `text` fields. Recordings longer than 59 minutes are trimmed; split them before transcription to retain all audio.

Provide extra context or adjust the generation budget:

```swift
let output = model.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(maxTokens: 4096),
    context: "A discussion about audio machine learning.",
    hotwords: ["VibeVoice", "MLX"]
)
```

### File streaming

```swift
let model = try await VibeVoiceASRModel.fromPretrained(
    "microsoft/VibeVoice-ASR-Streaming-1.5B"
)
for try await event in model.generateStream(audio: audio, hotwords: ["MLX"]) {
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

Streaming models emit text for each audio window. Their segment times describe chunk intervals. `maxTokensPerChunk` controls the text budget per window and defaults to at most 256 tokens.

### Live audio

Create one state per recording and keep it while new audio arrives:

```swift
let state = model.makeStreamingState(hotwords: ["MLX"])

func transcribeWindow(_ window: MLXArray) -> String {
    let features = model.encodeSpeech(window)
    return model.streamingGenerateStep(audioFeatures: features, state: state)
}
```

Buffer `model.streamingWindowSamples` mono samples for each call, then advance by `model.streamingChunkSamples`, retaining the lookahead samples. At 24 kHz, the published streaming checkpoints consume approximately 2.93 seconds of new audio with 0.53 seconds of lookahead. Zero-pad the final partial window to `streamingWindowSamples` samples.

## License

VibeVoice ASR is released under the MIT license. See the [Microsoft model card](https://huggingface.co/microsoft/VibeVoice-ASR) and each supported repository for license information.
