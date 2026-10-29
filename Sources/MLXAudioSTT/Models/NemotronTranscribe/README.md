# Nemotron 3.5 Transcribe

NVIDIA Nemotron 3.5 Transcribe is a multilingual speech model for transcription,
translation, speaker-attributed transcription, rich transcription with event and
speaking-style tags, and questions about audio. Speaker IDs are local to each
recording. Text streaming starts after the recording has been encoded.

## Supported models

- [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
- [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit)

## CLI

```bash
mlx-audio-swift-stt \
  --model nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit \
  --audio recording.wav --output-path transcript --format txt
```

## Swift

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await STT.loadModel(
    modelRepo: "nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit"
)
let (_, audio) = try loadAudioArray(from: URL(fileURLWithPath: "recording.wav"), sampleRate: model.sampleRate)
let result = model.generate(audio: audio)
print(result.text)

for try await event in model.generateStream(audio: audio) {
    if case .token(let text) = event { print(text, terminator: "") }
}
```

Use the model-specific Swift API for speaker attribution, rich transcription,
translation, contextual vocabulary, and custom audio questions:

```swift
let nemotron = model as! NemotronTranscribeModel
let turns = nemotron.generate(audio: audio, options: .init(task: .speakerAttribution))
print(turns.text)
print(turns.segments ?? [])

let translated = nemotron.generate(
    audio: audio,
    generationParameters: STTGenerateParameters(language: "German", task: "translate"),
    options: .init()
)
print(translated.text)

let answer = nemotron.generate(
    audio: audio,
    options: .init(prompt: "Summarize the conversation.", hotwords: ["Nemotron"])
)
print(answer.text)
```

Supply mono audio at the model's 16 kHz sample rate. Speaker turns contain text
and speaker IDs. Word timestamps are unavailable; use text or JSON output.

## License

The NVIDIA checkpoint uses [OpenMDW-1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/blob/main/LICENSE).
The early-access checkpoint uses NVIDIA's
[Software and Model Evaluation License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/);
see its [model card](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit).
The implementation follows [mlx-audio](https://github.com/Blaizzy/mlx-audio) and
[NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) (Apache-2.0), with
Nemotron-H components adapted from [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT).
