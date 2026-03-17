# SenseVoice

SenseVoice is a speech foundation model with multiple speech understanding capabilities, including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and audio event detection (AED).

## Supported Model

- [`mlx-community/SenseVoiceSmall`](https://huggingface.co/mlx-community/SenseVoiceSmall)

## Swift Example

```swift
import MLXAudioCore
import MLXAudioSTT

let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let model = try await SenseVoiceModel.fromPretrained("mlx-community/SenseVoiceSmall")
let output = model.generate(audio: audio)
print(output.text)
print(output.language ?? "unknown")
```

## CLI Example

```bash
.build/debug/mlx-audio-swift-stt \
  --model mlx-community/SenseVoiceSmall \
  --audio Tests/media/conversational_a.wav \
  --output-path /tmp/sensevoice \
  --format txt
```

## Notes

- SenseVoice is a non-autoregressive CTC model.
- The output segment includes extra metadata for `language`, `emotion`, and `event`.
