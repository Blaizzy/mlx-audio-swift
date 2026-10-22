# Mega-ASR

Native routed speech recognition using
[`mlx-community/Mega-ASR-bf16`](https://huggingface.co/mlx-community/Mega-ASR-bf16).
An audio-quality classifier selects either dense Qwen3-ASR or its noise-robust
LoRA path. Conversion, quantization and repacking remain entirely in Python.

```swift
import MLXAudioCore
import MLXAudioSTT

let model = try await MegaASRModel.fromPretrained("mlx-community/Mega-ASR-bf16")
let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)
let output = try model.transcribe(audio: audio,
    generationParameters: .init(maxTokens: 256, language: "English"),
    context: "Kaldi, Ethiopia")
print(output.text)
```

`fromPretrained` accepts a Hub ID or local prepared checkpoint directory;
`fromModelDirectory` is an offline loader. The generic `STT.loadModel` dispatcher
recognizes `mega_asr`. The checkpoint must contain the base safetensors, config,
text tokenizer files, and both configured router/LoRA sidecars (normally
`extras/router.safetensors` and `extras/lora.safetensors`). Loading is strict;
missing sidecars or incomplete factors fail rather than leaving random heads.

Raw arrays are mono or time-major stereo at 16 kHz. Automatic routing needs at
least 201 samples and the published router's 850 subsampled positional frames
limit it to roughly 34 seconds per request. This is also a limitation of the
Python router. Split longer recordings before automatic routing, or explicitly
choose `routing: .base` / `.robust`; a forced route uses the ordinary Qwen3-ASR
chunking path for the whole recording. Forced routes do not claim an audio-quality
probability. `model.router.route(audio:)` exposes the automatic decision and
its degraded probability (threshold `0.5`).

`transcribe` is the throwing API for input errors and accepts the common
`STTGenerateParameters`, a context string, and `.automatic`, `.base` or `.robust`
routing. Temperature/top-p/top-k and repetition controls are applied by the native
decoder. Quantized KV caches and quantized base weights are currently rejected.
The common nonthrowing `generate` API requires valid input; use `transcribe` or
`transcribeStream` when input errors must be handled.

`transcribeStream` accepts the same controls and emits text-token deltas followed
by the final result. It streams output from a complete recording, not live PCM.
Requests are serialized per model, including the whole streaming worker lifetime.
Cancellation checks occur at chunk, prefill and token boundaries before a later
request can change the active adapter.

LoRA activation uses cached original/adapted dense tensors. Disabling restores
original tensors exactly instead of subtracting a rounded delta repeatedly.
The published sidecar contains distinct `layers.*` and `model.layers.*` factors;
Python aliases both paths to the same text layers. Swift composes both
contributions in deterministic order, retaining per-update dtype rounding. The
prepared factor tensors already contain their scaling. This is runtime loading,
not a checkpoint converter. Expect additional memory for the robust dense weights.

## CLI

```bash
mlx-audio-swift-stt --model mlx-community/Mega-ASR-bf16 \
  --audio audio.wav --output-path transcript --context 'Kaldi, Ethiopia'

mlx-audio-swift-stt --model /path/to/prepared-checkpoint \
  --audio long-recording.wav --output-path transcript --routing robust --stream
```

`--gen-kwargs '{"routing":"base","context":"domain terms"}'` is also supported.
Route overrides are specific to Mega-ASR.

## Validation

The committed tiny reference fixture is generated from Python's native router
and compares its frontend, complete classifier, probability and BF16 LoRA math.
Additional tests cover strict sidecar validation, text-layer alias composition,
20 repeated adapter toggles with exact original restoration, real clean/noisy
transcription, cancellation followed by a different route, and complete output
streaming. Large-checkpoint tests are opt-in and never download weights in CI.

Generate references from the sibling Python project:

```bash
PYTHONPATH=/path/to/mlx-audio /path/to/mlx-audio/.venv/bin/python \
  Tests/ReferenceGenerators/MegaASRReference.py

PYTHONPATH=/path/to/mlx-audio /path/to/mlx-audio/.venv/bin/python \
  Tests/ReferenceGenerators/MegaASRCheckpointReference.py \
  --model-path /path/to/prepared-checkpoint --output /tmp/mega-references
```

Build with Xcode to produce the bundled Metal library:

```bash
xcodebuild build-for-testing -scheme MLXAudio-Package \
  -destination 'platform=macOS' -derivedDataPath /tmp/mega-derived \
  -parallel-testing-enabled NO -jobs 4 CODE_SIGNING_ALLOWED=NO
```

For the opt-in suite, add `MEGA_ASR_MODEL_DIR=/path/to/prepared-checkpoint` and
`MEGA_ASR_REFERENCE_ROOT=/tmp/mega-references` to the test target's environment in
the generated `.xctestrun`, then run:

```bash
xcodebuild test-without-building -xctestrun /path/to/configured.xctestrun \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -only-testing:MLXAudioTests/MegaASRTests \
  -only-testing:MLXAudioTests/MegaASRCheckpointTests CODE_SIGNING_ALLOWED=NO
```

### Recorded results

Validated with Xcode 27.0 on macOS/Apple Silicon, MLX Swift 0.32.3 and Python MLX
0.32.4. Python source revision was `d1e09119d5cce8c5506fcf8f5a4cc3780b641f31`;
the dense BF16 checkpoint revision was
`58b2aa4ed62343871f6dba78a4f71db015406152`. Xcode `build-for-testing` succeeded,
and all **6 focused/checkpoint tests** passed. Both CLI paths below also passed.

- Tiny frontend maximum/mean absolute errors: `7.03e-5` / `9.55e-7`.
- Tiny complete router logits: exact match in the tested FP32 fixture.
- BF16 LoRA update: exact match; all 20 toggle cycles restored the original
  weights exactly. The distinct text-layer alias composition test also matched.

| Real clip | Frontend max abs error | Router logit max abs error | Swift degraded probability | Route | Normalized word disagreement |
| --- | ---: | ---: | ---: | --- | ---: |
| `conversational_a.wav` (13.259 s) | 7.94e-5 | 2.62e-6 | 0.04078388 | Base | 0% |
| `noisy_audio.wav` (10 s) | 7.93e-5 | 8.11e-6 | 0.95510346 | Robust LoRA | 0% |

These comparisons use the same Python-resampled waveform, keeping codec/resampler
variation separate from model numerics. The real test tolerances are `5e-4` for
router features and `5e-3` for router logits; observed errors were substantially
smaller. Word disagreement lowercases and splits on non-alphanumeric characters;
it compares Swift with Python, not with a human ground-truth benchmark. Noisy
punctuation/casing differs, while the words agree.

The clean → robust → clean sequence reproduced the original clean transcript
exactly. Cancellation followed by a forced-base request and complete streamed
transcription also reproduced it. The implementation uses one gated worker for
streaming; no detached backbone worker survives cancellation and adapter changes.

The clean transcript begins “Coffee's story likely begins in Ethiopia” and ends
“Curious, he tried them himself and felt invigorated.” The noisy transcript is:

> He wrote many books about these adventures. Too one of the books I read of his quite a few years ago now was the maldives mystery.

CLI checks used the Xcode-built executable, loaded the local checkpoint and
independently decoded/resampled the original WAVs: automatic routing with
`--context 'Kaldi, Ethiopia'` on the clean clip, and `--routing robust --stream`
on the noisy clip. Both produced plausible complete transcripts.

The new router and LoRA layers have direct numerical references; the Qwen3-ASR
encoder/decoder reuse the existing native implementation and were validated here
through real transcription. Broad WER evaluation, iOS memory/latency and
quantized checkpoints were not tested. Dense weights plus cached robust weights
can require substantially more memory than the roughly 4.08 GB base checkpoint.
