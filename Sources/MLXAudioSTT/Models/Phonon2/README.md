# Phonon-2 and Phonon-2.5

Phonon is a compact speech recognition family from Fermion Research, based on
NVIDIA Parakeet TDT 0.6B v3. It supports timestamped transcription and chunked
processing of recordings. Phonon-2.5 recognizes 25 European languages; the
English variant and Phonon-2 focus on English.

## Supported Models

| Model | Languages |
| --- | --- |
| [`FermionResearch/Phonon-2.5`](https://huggingface.co/FermionResearch/Phonon-2.5) | English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian, Ukrainian, Romanian, Czech, Slovak, Bulgarian, Croatian, Swedish, Danish, Finnish, Estonian, Latvian, Lithuanian, Hungarian, Slovenian, Maltese, Greek |
| [`FermionResearch/Phonon-2.5-English`](https://huggingface.co/FermionResearch/Phonon-2.5-English) | English |
| [`mlx-community/Phonon-2`](https://huggingface.co/mlx-community/Phonon-2) | English |

## CLI Example

```bash
mlx-audio-swift-stt \
  --model FermionResearch/Phonon-2.5 \
  --audio audio.wav \
  --output-path transcript \
  --format txt
```

Use `FermionResearch/Phonon-2.5-English` for the English variant. The published
checkpoints load directly, and recordings should be provided at 16 kHz in the Swift API.

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioSTT

let model = try await Phonon2Model.fromPretrained("FermionResearch/Phonon-2.5")
let audioURL = URL(fileURLWithPath: "audio.wav")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)

let result = model.generate(audio: audio)
print(result.text)
```

The shared loader also supports these models:

```swift
let model = try await STT.loadModel(modelRepo: "FermionResearch/Phonon-2.5-English")
let result = model.generate(audio: audio)
print(result.text)
```

## License

Model weights derive from NVIDIA Parakeet TDT 0.6B v3 and are licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
Fermion Research code is licensed under [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0),
Copyright 2026 Fermion Research.
