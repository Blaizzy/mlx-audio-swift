# Granite Speech 5.0 TurboCTC

Granite Speech 5.0 TurboCTC is a 470-million-parameter English speech recognition
model from IBM. Use it to transcribe recorded speech to text.

## Supported Models

- [`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)

## CLI Example

```bash
mlx-audio-swift-stt \
  --model ibm-granite/granite-speech-5.0-470m-turboctc \
  --audio audio.wav \
  --output-path transcript \
  --format txt
```

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await GraniteSpeech5CTCModel.fromPretrained(
    "ibm-granite/granite-speech-5.0-470m-turboctc"
)
let audioURL = URL(fileURLWithPath: "audio.wav")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let result = model.generate(audio: audio)
print(result.text)
```
