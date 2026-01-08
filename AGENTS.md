# AGENTS.md

## Background

We are porting **Chatterbox Turbo** from the Python `mlx-audio` reference implementation
into the **mlx-audio-swift** refactor SDK (PR #1). The goal is to enable **local TTS**
for Clawdbot’s macOS app without any Python subprocesses, using Apple Silicon + MLX
only. This port is required because the current Swift SDK does **not** expose
Chatterbox Turbo yet, and we want a Swift-first implementation that uses the
standard Hugging Face cache layout.

## Why this port

- Chatterbox Turbo provides higher-quality, expressive local TTS than system voices.
- We need a Swift-native pipeline for macOS Talk Mode (no Python bridge).
- The refactor SDK will be the long-term API surface; we want Chatterbox Turbo to
  live inside it.

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

If a test requires large model weights, use a gated/integration test approach
and keep unit tests lightweight.

## Notes

- Chatterbox Turbo depends on a separate S3Tokenizer repo. Ensure cache lookup
  covers both the main model repo and tokenizer repo.
- Aim for parity with Python outputs (shape, length, sample rate). Exact bitwise
  matching is not required, but quality must be comparable.
