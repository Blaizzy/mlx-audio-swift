# Mega-ASR

Mega-ASR is a multilingual speech recognition model that automatically adapts to
clean, noisy, and far-field recordings. It supports Chinese, Cantonese, English,
German, Spanish, French, Italian, Portuguese, Russian, Korean, and Japanese.

## Supported Models

- [`mlx-community/Mega-ASR-bf16`](https://huggingface.co/mlx-community/Mega-ASR-bf16)

## CLI Example

```bash
mlx-audio-swift-stt \
  --model mlx-community/Mega-ASR-bf16 \
  --audio audio.wav \
  --output-path transcript \
  --format txt \
  --language English \
  --context "Kaldi, Ethiopia"
```

Automatic routing accepts recordings up to about 34 seconds. For longer files,
split the recording or choose `--routing base` or `--routing robust`:

```bash
mlx-audio-swift-stt \
  --model mlx-community/Mega-ASR-bf16 \
  --audio long-recording.wav \
  --output-path transcript \
  --format txt \
  --routing robust \
  --stream
```

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await MegaASRModel.fromPretrained("mlx-community/Mega-ASR-bf16")
let audioURL = URL(fileURLWithPath: "audio.wav")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let result = try model.transcribe(
    audio: audio,
    generationParameters: STTGenerateParameters(maxTokens: 512, language: "English"),
    context: "Kaldi, Ethiopia"
)
print(result.text)
```
