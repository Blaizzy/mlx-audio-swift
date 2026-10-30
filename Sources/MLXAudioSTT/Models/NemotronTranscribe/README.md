# Nemotron 3.5 Transcribe

NVIDIA Nemotron 3.5 Transcribe is a multilingual speech model for transcription,
translation, speaker-attributed transcription, rich transcription with event and
speaking-style tags, and questions about audio. Speaker IDs are local to each
recording. Text streaming starts after the recording has been encoded.

## Supported models

- [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
- [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8)
- [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4)
- [nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit)

NVIDIA's BF16, FP8, and NVFP4 checkpoints include word timestamps. The
early-access 4-bit checkpoint supports transcription without timestamps.
FP8 needs roughly the same runtime memory as BF16; use NVFP4 to reduce memory use.

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
and speaker IDs.

## Word timestamps

Load an NVIDIA checkpoint and request timestamps in the Swift API:

```swift
let model = try await NemotronTranscribeModel.fromPretrained(
    "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
)
let result = model.generate(
    audio: audio,
    options: .init(task: .speakerAttribution, wordTimestamps: true)
)
for word in result.segments ?? [] {
    print(word["text"] ?? "", word["start"] ?? "", word["end"] ?? "", word["speaker"] ?? "")
}

for try await event in model.generateStream(audio: audio, options: .init(wordTimestamps: true)) {
    if case .result(let output) = event { print(output.segments ?? []) }
}
```

Each segment contains one word, `start` and `end` in seconds, and the transcript's
speaker ID. Timestamped segments arrive with the final streaming result.
Timestamps apply to transcription, speaker attribution, and rich transcription;
translation and audio questions return text. Chinese and Japanese use character
units; Thai word segmentation is unavailable. If timestamps are unavailable for
the checkpoint or transcript, generation returns the transcript and ordinary
speaker segments. Check `supportsWordTimestamps` before requesting them.

## License

The NVIDIA checkpoints use [OpenMDW-1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/blob/main/LICENSE).
The early-access checkpoint uses NVIDIA's
[Software and Model Evaluation License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-and-model-evaluation-license/);
see its [model card](https://huggingface.co/nativ-community/Nemotron-3.5-Transcribe-30B-A4B-EA-4bit).
The implementation follows [mlx-audio](https://github.com/Blaizzy/mlx-audio) and
[NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) (Apache-2.0), with
Nemotron-H components adapted from [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT).
Word alignment also follows [Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/nemotron3_5_transcribe) (Apache-2.0).
