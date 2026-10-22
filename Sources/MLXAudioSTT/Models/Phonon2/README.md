# Phonon-2

Phonon-2 is a compact English speech recognition model from Fermion Research.
It supports timestamped transcription and chunked processing of recordings.

## Supported Models

- [`mlx-community/Phonon-2`](https://huggingface.co/mlx-community/Phonon-2)

## CLI Example

```bash
mlx-audio-swift-stt \
  --model mlx-community/Phonon-2 \
  --audio audio.wav \
  --output-path transcript \
  --format txt
```

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await Phonon2Model.fromPretrained("mlx-community/Phonon-2")
let audioURL = URL(fileURLWithPath: "audio.wav")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let result = model.generate(audio: audio)
print(result.text)
```

## License

Model weights derive from NVIDIA Parakeet TDT 0.6B v3 and are licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
Fermion Research code is licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0),
Copyright 2026 Fermion Research.
