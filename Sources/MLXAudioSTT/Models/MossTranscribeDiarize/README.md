# MOSS-Transcribe-Diarize

OpenMOSS MOSS-Transcribe-Diarize is an audio-conditioned Qwen3 decoder with a Whisper encoder for timestamped transcription and speaker labels.

## Model

- `OpenMOSS-Team/MOSS-Transcribe-Diarize`

## Usage

```swift
let model = try await STT.loadModel(
    modelRepo: "OpenMOSS-Team/MOSS-Transcribe-Diarize"
)
let output = model.generate(audio: audio)
print(output.text)
print(output.segments ?? [])
```

CLI:

```bash
swift run mlx-audio-swift-stt \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --audio sample.wav \
  --output-path moss-output \
  --format json
```
