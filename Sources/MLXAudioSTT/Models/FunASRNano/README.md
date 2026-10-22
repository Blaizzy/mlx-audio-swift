# Fun-ASR-Nano

Native Swift inference for `mlx-community/Fun-ASR-Nano-2512`: a SenseVoice-style
SANM encoder, Transformer audio adaptor and tied Qwen3-0.6B text decoder.
Model conversion remains in Python `mlx-audio`.

```swift
import MLXAudioCore
import MLXAudioSTT

let model = try await FunASRNanoModel.fromPretrained("mlx-community/Fun-ASR-Nano-2512")
// audio contains mono 16 kHz floating-point PCM.
let result = try model.transcribe(
    audio: audio,
    generationParameters: STTGenerateParameters(maxTokens: 512, language: "en"),
    hotwords: ["MLX", "Apple Silicon"],
    itn: true
)
print(result.text)
```

`fromModelDirectory(_:)` accepts a local Python-prepared checkpoint, including its
`Qwen3-0.6B/` tokenizer directory. Language hints accept `zh`, `en`, `ja` and
Chinese dialect aliases such as `yue`, `wuu` and `nan`; `nil`/`auto` omit the hint.
Use either `hotwords` or `context`. `itn: false` requests spoken-form text.

`transcribeStream` emits token deltas from a complete recording and supports the
same model-specific options. It runs in a cancellable task. This checkpoint does
not support live-input ASR, diarization or word/segment timestamp heads. Returned
segment times describe recording chunks, not learned speech alignments. The
common STT generation API uses default hotword/ITN settings; use the throwing
`transcribe` API to validate inputs, or `generateStream` for throwing event errors.
Arrays must already match `sampleRate`; file decoding/resampling uses MLXAudioCore.
Generation calls on one model instance should be serialized.

```bash
mlx-audio-swift-stt --model mlx-community/Fun-ASR-Nano-2512 \
  --audio speech.wav --output-path transcript --language en \
  --context 'MLX, Apple Silicon' --gen-kwargs '{"itn":true}'
```

Only the text backbone is eligible for prepared checkpoint quantization; the
encoder/adaptor retain their published precision. KV-cache quantization is not
currently supported. No PyTorch, FunASR or Python inference runtime is needed.

## Validation

Focused tests include deterministic Python reference values for the exact Kaldi
fbank/LFR frontend, SANM encoder, audio adaptor, Qwen3 forward pass and cached
continuation. The fixture uses fixed weights and a synthetic waveform.

Validated on macOS/Apple Silicon with Xcode 27, MLX Swift 0.32.3 and Python
`mlx-audio` revision `d1e09119d5cce8c5506fcf8f5a4cc3780b641f31` (MLX 0.32.4).
Checkpoint: `mlx-community/Fun-ASR-Nano-2512` revision
`a7bc96fceaafce39ed6748e0c0fa9a9508b67f86`, published BF16 weights.

| Stage | Measured maximum absolute error | Relative RMS error |
| --- | ---: | ---: |
| Tiny FP32 fbank/LFR | 0.0000296 | — |
| Tiny FP32 encoder | 0.0002853 | — |
| Tiny FP32 adaptor | 0.000000276 | — |
| Tiny FP32 decoder | 0 | — |
| Tiny FP32 cached continuation | 0.0001612 | — |
| Checkpoint features | 0.001147 | 0.00018% |
| Checkpoint encoder | 0.005693 | 0.0407% |
| Checkpoint adaptor | 0.43912 | 0.0747% |
| Checkpoint merged embeddings | 0.5 | 0.0965% |

The larger absolute adaptor errors are small relative to its tensor scale
(reference RMS about 28); encoder/adaptor/embedding cosine similarity exceeded
0.9999995. Fused attention and BF16 rounding differ between the two runtimes.
First-step logits had max absolute error 0.3125 and centered relative RMS error
3.34%, but probability KL divergence was only `3.36e-6` and total variation
`5.04e-5`. The first token and full greedy transcript matched Python. Tests allow
1% intermediate relative RMS, cosine above 0.9999, 5% centered-logit relative
RMS, KL below 0.001 and total variation below 0.01. Frontend comparisons have a
separate tight absolute/relative tolerance.

Both runtimes produced this transcript for the checkpoint's 7.176-second
`example/en.mp3`; token streaming also returned the same final text:

> The tribal chieftain called for the boy and presented him with fifty pieces of gold.

The CLI also passed local-directory loading, 48 kHz MP3 → 16 kHz resampling,
contextual biasing, ITN options, chunked prefill and streamed output on this clip.

This is a numerical check and one real English smoke clip, not a multilingual
accuracy benchmark. No timestamp heads are present or initialized.

Build and run the small tests without downloading weights:

```bash
xcodebuild build-for-testing -scheme MLXAudio-Package -destination 'platform=macOS' \
  -derivedDataPath /tmp/FunASRNanoDerivedData CODE_SIGNING_ALLOWED=NO
xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
  -derivedDataPath /tmp/FunASRNanoDerivedData -parallel-testing-enabled NO \
  -only-testing:MLXAudioTests/FunASRNanoTests CODE_SIGNING_ALLOWED=NO
```

Real-checkpoint tests are opt-in, load local weights, and never download in CI.
Generate reference tensors using the Python checkout and already-downloaded
checkpoint (this performs inference, not conversion):

```bash
PYTHONPATH=/path/to/mlx-audio /path/to/python-venv/bin/python \
  Tests/media/fun_asr_nano_checkpoint_reference.py \
  /path/to/Fun-ASR-Nano-2512 /tmp/fun-asr-parity
TEST_RUNNER_FUN_ASR_MODEL_DIR=/path/to/Fun-ASR-Nano-2512 \
TEST_RUNNER_FUN_ASR_PARITY_DIR=/tmp/fun-asr-parity \
xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
  -derivedDataPath /tmp/FunASRNanoDerivedData -parallel-testing-enabled NO \
  -only-testing:MLXAudioTests/FunASRNanoCheckpointTests CODE_SIGNING_ALLOWED=NO
```

`Tests/media/fun_asr_nano_reference.py` regenerates the tiny JSON fixture. Use
Xcode/xcodebuild for Swift validation so the MLX Metal library is bundled.
