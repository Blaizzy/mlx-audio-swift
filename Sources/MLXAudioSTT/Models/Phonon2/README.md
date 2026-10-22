# Phonon-2

Native MLX Swift support for the prepared
[`mlx-community/Phonon-2`](https://huggingface.co/mlx-community/Phonon-2)
checkpoint: English Parakeet TDT recognition with exact five-value packed encoder
projections. Conversion and repacking remain in Python `mlx-audio`.

```swift
import MLXAudioCore
import MLXAudioSTT

let model = try await STT.loadModel(modelRepo: "mlx-community/Phonon-2")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)
let result = model.generate(audio: audio)
print(result.text)
```

For a local prepared checkpoint, use
`try Phonon2Model.fromDirectory(directoryURL)`. Audio arrays must be mono 16 kHz
(the shared Parakeet path averages a trailing channel dimension). Very short
nonempty inputs are padded to 320 samples. The synchronous common STT protocol
returns an empty transcript for empty input; the streaming API throws an input
error. This differs from Python's synchronous empty-input rejection.

`generateStream` provides chunked transcription of a recording, not live-input
streaming. Translation, diarization and contextual hotword biasing are not
implemented for this family. Segments retain Parakeet TDT token timing.

Compute defaults to BF16; use `Phonon2Model.fromDirectory(..., computeDType:
.float32)` for FP32 comparison. Packed codes remain uint32 and learned row
palettes remain float16 for either compute dtype. Differences between palette
values, packed products and accumulation use FP32, matching Python. Generic
requantization is unsupported.

## Validation

Checkpoint revision: `1f228c19b37231d6ef8d839f8648bd4172bb097c` (350 MB weights).
Tests include deterministic Python fixtures for packed projections, short-input
frontend, a tiny complete encoder and strict model loading. Fixture generation:

```bash
PYTHONPATH=/path/to/mlx-audio /path/to/mlx-audio/.venv/bin/python Scripts/generate_phonon2_reference.py
```

Validated on macOS/Apple Silicon with Xcode 27.0 and Python MLX 0.32.4,
against sibling `mlx-audio` commit `d1e09119d5cce8c5506fcf8f5a4cc3780b641f31`:

| Comparison | Maximum absolute error |
| --- | ---: |
| Packed layer, FP32 / BF16 | 0 / 0 |
| Tiny-model mel / short-input mel | 2.81e-6 / 2.84e-6 |
| Tiny complete encoder | 2.39e-7 |
| Real checkpoint mel | 5.18e-4 |
| Real 24-layer encoder, FP32 | 2.19e-4 |

The real checkpoint comparison uses `Tests/media/conversational_a.wav`, resampled
to 16 kHz (212,139 samples, 13.259 seconds). The FP32 Python and Swift transcripts
matched exactly at both FP32 and default BF16, with 0% normalized word error rate.
It starts “Coffee's story
likely begins in Ethiopia” and ends “he tried them himself and felt invigorated.”
These measurements establish one reproducible regression case, not a corpus-wide
accuracy guarantee. Encoder comparison allows maximum absolute error `1e-2`;
real-transcript comparison allows up to 10% case/punctuation-normalized word error
against Python, avoiding an exact-text requirement at numerical near ties.

All 45 selected tests across `Phonon2Tests`, `ParakeetSTTTests`,
`NemotronASRTests`, and `ParakeetBatchParityTests` passed (the real-checkpoint case
is skipped by default). The opt-in Phonon-2 suite also passed with the real model.
The public Hub loader and Xcode-built STT CLI were also exercised on the same
audio file and produced the same transcript at default BF16 precision.

```bash
xcodebuild build-for-testing -scheme MLXAudio-Package -destination 'platform=macOS' \
  -derivedDataPath /tmp/phonon2-derived CODE_SIGNING_ALLOWED=NO
xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
  -derivedDataPath /tmp/phonon2-derived -parallel-testing-enabled NO \
  -only-testing:MLXAudioTests/Phonon2Tests \
  -only-testing:MLXAudioTests/ParakeetSTTTests \
  -only-testing:MLXAudioTests/NemotronASRTests \
  -only-testing:MLXAudioTests/ParakeetBatchParityTests CODE_SIGNING_ALLOWED=NO
```

For a real checkpoint, prepare Python numerical/transcript references with the
same script (it performs inference only):

```bash
PYTHONPATH=/path/to/mlx-audio /path/to/mlx-audio/.venv/bin/python \
  Scripts/generate_phonon2_reference.py --checkpoint /path/to/Phonon-2 \
  --output /tmp/phonon2-reference
```

The opt-in test requires `MLX_AUDIO_PHONON2_CHECKPOINT` (local checkpoint directory)
and `MLX_AUDIO_PHONON2_REFERENCE` (the reference output directory) in the **test
runner's** environment. Xcode does not necessarily forward shell variables; add
these values to the generated `.xctestrun` target's `EnvironmentVariables`, or
configure them in the Xcode test scheme. No test downloads weights automatically.

## Attribution

Packed layer integration is adapted from fermion-research 0.2.11, Copyright 2026
Fermion Research, Apache-2.0. Checkpoint weights derive from NVIDIA Parakeet TDT
0.6B v3 and are CC-BY-4.0; retain the checkpoint's license and notice files.
The adapted packed runtime's [Apache-2.0 license](LICENSE-CODE-Apache-2.0.txt)
is included in this directory.
