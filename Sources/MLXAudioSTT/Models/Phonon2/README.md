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

## Python Example

Using the Python `mlx-audio` library:

```python
from mlx_audio.stt import load

model = load("mlx-community/Phonon-2")
result = model.generate("audio.wav", chunk_duration=30, overlap_duration=2)
print(result.text)
```
