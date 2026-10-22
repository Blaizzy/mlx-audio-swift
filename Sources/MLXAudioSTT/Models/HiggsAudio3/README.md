# Higgs-Audio v3 STT

Native Swift transcription for [bosonai/higgs-audio-v3-stt](https://huggingface.co/bosonai/higgs-audio-v3-stt).
The model combines a Whisper-style audio encoder, pooled/temporal audio projector
and Qwen3 decoder. It has an independent text head; the Higgs audio codec is not
required. Compatible Python-prepared MLX checkpoints can also be loaded.

```swift
import MLXAudioCore
import MLXAudioSTT

let model = try await HiggsAudio3Model.fromPretrained("bosonai/higgs-audio-v3-stt")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)
let result = try model.transcribe(audio: audio)
print(result.text)
```

Use `STT.loadModel(modelRepo:)` or the `mlx-audio-swift-stt` CLI for the common
STT interface. `transcribe` is the throwing API and accepts a custom `prompt`,
`prefillStepSize`, `minP` and common `STTGenerateParameters`. The common
nonthrowing `generate` returns empty output on invalid input/load state; use the
throwing API or stream when errors must be surfaced. The default prompt
requests lowercase words without punctuation. `generateStream` emits token deltas
and the final result; it is output streaming from a complete recording, not a
live microphone protocol. Model instances require one inference caller at a time.

`fromPretrained` accepts a Hub ID or local directory. It tries native Silero VAD
when `vad_cut` is enabled; unavailable VAD falls back to fixed-duration chunks.
Pass `loadVAD: false` to avoid loading VAD. `fromModelDirectory` is an offline
loader and does not fetch VAD; a preloaded `SileroVAD` can be assigned to
`model.vadModel`. Chunk ranges follow Python's silence-preserving policy unless
`split_vads` is enabled. All chunks are padded to the longest chunk before encoding.
The common `chunkDuration` is capped at the checkpoint's `chunk_size_seconds`
(default 4 seconds); smaller requested chunks are honored.
Raw input must be nonempty mono 16 kHz; very short clips are padded to 640 samples
so the encoder's pair pooling remains defined. Segment end times describe input
audio duration rather than Python's inference wall time.

Model conversion remains in Python. Swift loads weights/config/tokenizer files
and handles runtime key/layout normalization. Quantized prepared weights use the
checkpoint quantization metadata; no new conversion dependencies are included.
The documented BF16 checkpoint is about 5.35 GB before buffers and KV caches.

## Validation

Validation commands, measured parity and real-checkpoint results are recorded
below once run. Numerical fixtures are generated from the sibling Python model
with `Tests/ReferenceGenerators/HiggsAudio3Reference.py`. Unit tests use a tiny
complete model and check frontend, encoder, projector, full decoder and cached
continuation with a maximum absolute error tolerance of `2e-4` in FP32.

Validated on macOS/Apple Silicon with Xcode 27.0, MLX Swift 0.32.3 and the
sibling Python `mlx-audio` revision `d1e09119d5cce8c5506fcf8f5a4cc3780b641f31`
(MLX Python 0.32.4). Real weights were loaded strictly from
`bosonai/higgs-audio-v3-stt` revision
`2ffd1aa39f5a1266931e405cba12e404a9f994b2` in BF16. Conversion is not part of
this port. Seven focused tests passed, including six tiny/unit tests and an
opt-in real-checkpoint test covering both recordings.

Tiny FP32 maximum absolute errors: frontend `2.146e-5`, encoder `1.819e-4`,
projector `1.025e-4`; decoder prefill and cached continuation `0`.

Real BF16 intermediate comparisons use the same waveform with VAD disabled.
Metrics are computed after casting to FP32 and flattening. The real-checkpoint
test requires cosine similarity above `0.995`, identical prompt IDs, and the
same first greedy token. Absolute errors are recorded rather than required to
be zero because BF16 accumulation/frontend rounding can propagate across layers.

| Stage | Maximum absolute error | RMS error | Cosine similarity |
| --- | ---: | ---: | ---: |
| Mel frontend | 0.0009765625 | 0.0000116472 | 1.0000001 |
| Audio encoder | 0.50390625 | 0.0124182 | 0.99872863 |
| Audio projector | 0.0048828125 | 0.000311431 | 0.9994928 |
| Prompt embeddings | 0.0048828125 | 0.000202364 | 0.99995387 |
| First-token logits | 0.265625 | 0.0329622 | 0.99992025 |

Both runtimes transcribed `Tests/media/intention.wav` as `intention` (45 prompt
and 2 generated tokens). On the 13.259-second `conversational_a.wav`, independently
resampled to 16 kHz by each runtime, both produced 226 prompt and 44 generated
tokens. Python spelled the proper name `kaldi`; Swift spelled it `caldi`; all
other words matched. This spelling is not an exact test invariant.

Swift transcript:

> coffee's story likely begins in ethiopia where legend tells of a goat herder named caldi who noticed his goats became energetic after eating red berries from a particular bush curious he tried them himself and felt invigorated

The normalized word comparison lowercases, removes punctuation except apostrophes
and splits on whitespace: **1 substitution / 37 reference words = 2.70% WER**.
This is measured against the Python
transcript, not a human ground-truth accuracy benchmark.

Reproduce the tiny fixture (fixture files are committed, so regeneration is
optional):

```bash
PYTHONPATH=/path/to/mlx-audio /path/to/mlx-audio/.venv/bin/python \
  Tests/ReferenceGenerators/HiggsAudio3Reference.py
```

Generate real references from an existing local checkpoint:

```bash
PYTHONPATH=/path/to/mlx-audio /path/to/mlx-audio/.venv/bin/python \
  Tests/ReferenceGenerators/HiggsAudio3CheckpointReference.py \
  --model-path /path/to/checkpoint --output /tmp/higgs-reference.safetensors
```

Run focused tests with Xcode (the real test runs only when its model-directory
variable is set; it never downloads the ASR weights automatically):

```bash
TEST_RUNNER_HIGGS_AUDIO3_MODEL_DIR=/path/to/checkpoint \
TEST_RUNNER_HIGGS_AUDIO3_REFERENCE=/tmp/higgs-reference.safetensors \
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
  -derivedDataPath /tmp/higgs-derived -parallel-testing-enabled NO -jobs 4 \
  -only-testing:MLXAudioTests/HiggsAudio3Tests \
  -only-testing:MLXAudioTests/HiggsAudio3NetworkTests CODE_SIGNING_ALLOWED=NO
```

Real-checkpoint validation here used the fixed-chunk fallback to isolate model
numerics; VAD cut-range policy has deterministic unit coverage. Quantized
prepared-artifact loading is implemented but was not included in this BF16
smoke. iOS memory/latency, live-input streaming, translation quality and broad
WER evaluation were not tested.
