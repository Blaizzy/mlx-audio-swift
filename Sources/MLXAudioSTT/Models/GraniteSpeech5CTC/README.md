# Granite Speech 5.0 TurboCTC

Native MLX Swift encoder-only English ASR for
[ibm-granite/granite-speech-5.0-470m-turboctc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc).
This is distinct from the autoregressive Granite Speech models. Conversion and
artifact preparation remain in Python `mlx-audio`.

```swift
import MLXAudioCore
import MLXAudioSTT

let model = try await GraniteSpeech5CTCModel.fromPretrained(
    "ibm-granite/granite-speech-5.0-470m-turboctc")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)
let result = try model.transcribe(audio: audio)
print(result.text)
```

The common `STT.loadModel` loader and `mlx-audio-swift-stt` CLI dispatch to this
model using `model_type=granite_speech5_ctc`. `fromPretrained` accepts a Hub ID or
local prepared checkpoint; `fromModelDirectory` is fully offline. Model arrays
are nonempty mono 16 kHz PCM. Very short recordings are zero-padded to keep time
nonzero through the internal subsampling blocks. The frontend API itself requires
at least 320 samples. `transcribe` is the throwing entry point; the nonthrowing
common `generate` API requires valid input and a loaded tokenizer.

Greedy CTC inference ignores language-model sampling/token-budget parameters.
There is no translation, hotword biasing, diarization or trained live streaming.
`generateStream` emits the full recording's decoded text and final result after
encoding completes. It does not emit partial audio-chunk hypotheses. Word times
are not produced.

Swift includes the precise HTK log-mel/delta/pair-stacking frontend, block-local
Shaw attention, internal strided Conformer blocks, self-conditioning and tied
CTC head, inference BatchNorm and decode-only BPE/byte-fallback tokenizer. It
loads prepared MLX or original supported convolution layouts and quantization
metadata without a converter or Python inference dependency. Only the main
Apache-2.0 checkpoint is targeted by the real validation; the `-nc` variant's
checkpoint license and tokenizer must be checked separately.

## Validation

Numerical fixtures and checkpoint comparisons use the sibling Python
`mlx-audio` implementation. Real checkpoint revision:
`1dd23f46f6d4e6eae2531982ba2e33220086dd97`, BF16, about 946 MB.
Large-weight tests are opt-in and never download in CI. Tests were run on macOS
Apple Silicon with Xcode 27.0, MLX Swift 0.32.3 and MLX Python 0.32.4, using Python
`mlx-audio` revision `d1e09119d5cce8c5506fcf8f5a4cc3780b641f31`.
All eight focused Xcode tests passed, including the opt-in real-checkpoint test.

The committed tiny FP32 fixture covers a complete four-layer Conformer with two
internal subsampling blocks, self-conditioning, odd lengths and explicit masks.
Maximum absolute errors: frontend `2.16e-5`; final logits `0` both with and without
an attention mask. Additional tests cover CTC blank/repeat behavior, replicated
delta edges, configuration validation, strict checkpoint loading, byte-fallback
decoding, short input, quantized input dtype handling, common loader dispatch and
full-buffer streaming/error propagation.

Real checkpoint comparisons use exactly the Python waveform to isolate model
numerics. Logits are cast to FP32 before calculating scale-aware metrics:

| Recording | Frontend max abs | Logits max abs | Logits relative RMS | Logits cosine |
| --- | ---: | ---: | ---: | ---: |
| `intention.wav` (1.52 s) | 4.61e-5 | 0.25 | 0.3385% | 0.9999943 |
| `conversational_a.wav` (13.259 s) | 1.03e-4 | 0.625 | 0.3661% | 0.9999933 |

With the same waveform, both transcripts match Python exactly. Independent
Swift loading/resampling of the original WAV files preserves all normalized
words: the single-word recording yields `Intention.` instead of `intention.`,
and the longer recording matches exactly:

> Coffee's story likely begins in Ethiopia, where legend tells of a goat herder named Kaldi, who notices goats became energetic after eating red berries from a particular bush. Curious, he tried them himself and felt invigorated.

The real tests require logit relative RMS below 1%, cosine above 0.9999 and
normalized word disagreement at most 10%; measured disagreement is 0% on both
clips. Word normalization lowercases and splits at non-alphanumeric characters.
This compares implementations, not recognition accuracy against human labels.

Regenerate the committed tiny fixture using the Python library:

```bash
PYTHONPATH=/path/to/mlx-audio /path/to/mlx-audio/.venv/bin/python \
  Tests/ReferenceGenerators/GraniteSpeech5CTCReference.py
```

Generate real reference directories from already prepared checkpoint weights:

```bash
PYTHONPATH=/path/to/mlx-audio /path/to/mlx-audio/.venv/bin/python \
  Tests/ReferenceGenerators/GraniteSpeech5CTCReference.py \
  --model-path /path/to/checkpoint --audio Tests/media/intention.wav \
  --output /tmp/granite-reference
PYTHONPATH=/path/to/mlx-audio /path/to/mlx-audio/.venv/bin/python \
  Tests/ReferenceGenerators/GraniteSpeech5CTCReference.py \
  --model-path /path/to/checkpoint --audio Tests/media/conversational_a.wav \
  --output /tmp/granite-reference-conversational_a
```

Run the Xcode suites (omit both `TEST_RUNNER_` variables for tiny/unit tests only):

```bash
TEST_RUNNER_GRANITE_TURBOCTC_MODEL_DIR=/path/to/checkpoint \
TEST_RUNNER_GRANITE_TURBOCTC_REFERENCES=/tmp/granite-reference:/tmp/granite-reference-conversational_a \
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
  -derivedDataPath /tmp/granite-derived -parallel-testing-enabled NO -jobs 4 \
  -only-testing:MLXAudioTests/GraniteSpeech5CTCTests \
  -only-testing:MLXAudioTests/GraniteSpeech5CTCCheckpointTests CODE_SIGNING_ALLOWED=NO
```

The real model and both reference directories must already exist. The validation
run used `build-for-testing`, followed by `test-without-building` with the two
environment variables injected into the generated `.xctestrun` test target. This
ensures the opt-in checkpoint test actually runs rather than being skipped.

BF16 is the real tested precision. Prepared quantized checkpoints have loader
support and a focused quantized-projection regression test, but no full quantized
checkpoint smoke was run. iOS memory/latency, the `-nc` checkpoint, long recordings
and broad accuracy benchmarks were not tested. Model instances should have one
inference caller at a time.
