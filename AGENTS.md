# AGENTS.md

## Background

We are porting **Chatterbox Turbo** from the Python `mlx-audio` reference
implementation into the **mlx-audio-swift** refactor SDK (PR #1). The goal is to
enable **local TTS** for Clawdbot’s macOS app without any Python subprocesses,
using Apple Silicon + MLX only. This port is required because the current Swift
SDK does **not** expose Chatterbox Turbo yet, and we want a Swift-first
implementation that uses the standard Hugging Face cache layout.

## Why this port

- Chatterbox Turbo provides higher-quality, expressive local TTS than system
  voices.
- We need a Swift-native pipeline for macOS Talk Mode (no Python bridge).
- The refactor SDK will be the long-term API surface; we want Chatterbox Turbo
  to live inside it.

## Development environment setup

- **MLX**: Install via Homebrew: `brew install mlx` (currently v0.30.1)
- **mlx-audio**: For reference implementation, create a virtual environment and install:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install mlx-audio mlx-lm
  ```
  This installs the Python reference implementation (currently mlx-audio v0.2.9) which includes Chatterbox Turbo for comparison during the port.
- **Activate environment**: `source venv/bin/activate` to use the Python reference implementation.

## Debugging / parity tools

This repo includes a Swift-native parity/debug executable for Chatterbox Turbo:

- **`ChatterboxTurboCompare`** (Swift executable target)
  - Generates deterministic speech tokens + mel + WAV given a seed (no Python subprocess).
  - Writes artifacts to disk for diffing against the Python reference.
  - Common usage:
    ```bash
    # Generates Swift WAV + token/mel dumps (seeded)
    swift run -c debug ChatterboxTurboCompare \
      --repo mlx-community/Chatterbox-Turbo-TTS-4bit \
      --text "Quick quality check. Does this sound natural?" \
      --seed 0 --maxTokens 200 \
      --out Artifacts/swift.wav \
      --tokensOut Artifacts/swift_tokens.txt \
      --melOut Artifacts/swift_mel.bin
    ```
  - You can also dump a model parameter by flattened key:
    ```bash
    swift run -c debug ChatterboxTurboCompare \
      --dumpParamKey "s3gen.decoder.estimator.down_blocks.0.downsample.conv.conv.weight" \
      --dumpParamOut Artifacts/param.bin
    ```

For Python reference output, use the `mlx-audio` implementation (we typically have it cloned under `~/projects/xaden/src/mlx-audio`):

```bash
cd ~/projects/xaden/src/mlx-audio
source .venv311/bin/activate  # or your venv
python - <<'PY'
import os
import numpy as np
import mlx.core as mx
import soundfile as sf
from mlx_audio.tts.utils import load_model

text = "Quick quality check. Does this sound natural?"
model = load_model("mlx-community/Chatterbox-Turbo-TTS-4bit")
mx.random.seed(0)  # IMPORTANT: seed right before generation for parity

segments = [np.array(r.audio, dtype=np.float32) for r in model.generate(text, split_pattern=None, max_tokens=200)]
audio = np.concatenate(segments)
out_path = os.path.expanduser("~/projects/xaden/src/mlx-audio-swift/Artifacts/python.wav")
sf.write(out_path, audio, model.sample_rate, subtype="FLOAT")
print("wrote python.wav", model.sample_rate, audio.shape[0])
PY
```

Quick numeric compare (Python):

```bash
python - <<'PY'
import numpy as np
import soundfile as sf

swift_audio, sr1 = sf.read("Artifacts/swift.wav", dtype="float32")
py_audio, sr2 = sf.read("Artifacts/python.wav", dtype="float32")
assert sr1 == sr2
n = min(len(swift_audio), len(py_audio))
diff = swift_audio[:n] - py_audio[:n]
print("sr", sr1, "samples", n)
print("max_abs", float(np.max(np.abs(diff))))
print("mean_abs", float(np.mean(np.abs(diff))))
print("rms", float(np.sqrt(np.mean(diff**2))))
PY
```

## Repository expectations

- Work on branch `pc/refactor-core`.
- Keep the public API stable and additive.
- Align cache handling with HF CLI default (`~/.cache/huggingface/hub`).

## Testing requirements

You **must add tests as you port**. Each major stage of the port should include
verification tests so we can validate correctness and catch regressions early.

Minimum expected tests:

1. **Config decoding tests**
   - Validate that Chatterbox Turbo config files decode correctly.

2. **Weight loading tests**
   - Ensure weight prefix splits (`t3.*`, `s3gen.*`, `ve.*`) map correctly.
   - Validate missing-file failures are clear and actionable.

3. **Generation smoke test**
   - Generate audio from a short prompt and assert:
     - non-empty output
     - expected sample rate (24kHz)
     - reasonable RMS / peak range

4. **Reference-audio path tests** (once implemented)
   - Validate ref-audio conditioning pipeline executes without error.

5. Produce an audio sample into a wav file and then determine if the audio is
   correct. You may use something like `whisper` to transcribe the audio to
   text. If it doens't work then it probably means the implementation of the
   port of Chatterbox Turbo from Python to swift is incorrect somewhere. The
   final test of knowing our port implementation is correct is when we have an
   audio sample that is being produced that is validated to be correct.

If a test requires large model weights, use a gated/integration test approach
and keep unit tests lightweight.

## Notes

- Chatterbox Turbo depends on a separate S3Tokenizer repo. Ensure cache lookup
  covers both the main model repo and tokenizer repo.
- Aim for parity with Python outputs (shape, length, sample rate). Exact bitwise
  matching is not required, but quality must be comparable.
- `sox` and `whisper` are also installed to help with any debugging.
