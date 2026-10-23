# Higgs-Audio v3 STT

Higgs-Audio v3 STT is a speech-to-text model from Boson AI for transcribing
recorded speech. Its default output is lowercase text without punctuation.

## Supported Models

- [`bosonai/higgs-audio-v3-stt`](https://huggingface.co/bosonai/higgs-audio-v3-stt)

## CLI Example

```bash
mlx-audio-swift-stt \
  --model bosonai/higgs-audio-v3-stt \
  --audio audio.wav \
  --output-path transcript \
  --format txt
```

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await HiggsAudio3Model.fromPretrained("bosonai/higgs-audio-v3-stt")
let audioURL = URL(fileURLWithPath: "audio.wav")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let result = model.generate(audio: audio)
print(result.text)
```

In Swift, pass `prompt:` to customize the transcription request. Both `generate`
and `generateStream` also accept `prefillStepSize:` and `minP:`. Use `onToken:`
with `generate` or iterate over `generateStream` to receive text as it is generated.
