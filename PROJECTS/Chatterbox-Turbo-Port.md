# Chatterbox Turbo Swift Port (MLXAudioTTS)

## Goal

Port **Chatterbox Turbo** from the Python `mlx-audio` reference implementation into the
**mlx-audio-swift** refactor SDK (PR #1), so it can run fully in Swift/MLX on Apple Silicon
without Python. This doc covers the staged plan and success criteria.

## Constraints

- Target branch: `pc/refactor-core` (refactor SDK)
- Focus: **Chatterbox Turbo only** (standard Chatterbox may follow later)
- No Python subprocess in the macOS app
- Use HF cache layout compatible with `hf` CLI

## Reference Sources

- Python implementation: `mlx-audio/mlx_audio/tts/models/chatterbox_turbo/*`
- Shared components: `mlx-audio/mlx_audio/tts/models/chatterbox/*` (tokenizers, S3Tokenizer, VoiceEncoder)

## Deliverables

- New Swift model module under `Sources/MLXAudioTTS/Models/ChatterboxTurbo`
- Public API: `ChatterboxTurboModel.fromPretrained(repoId:)` and `generate(...)`
- End-to-end text → audio waveform generation at 24kHz
- Weight loading + quantization parity with Python
- Tests: config load + model load + generate smoke test

## Phases

### Phase 1 — Architecture + API shape

- Map Python modules to Swift targets and file layout
- Define Swift API and public types
- Decide on minimal generation parameters for v1 (text, temperature, top_p, max_tokens)

### Phase 2 — Config + Weight Loading

- Implement config parsing (Turbo-specific config)
- Implement weight splitting and prefix mapping (`t3.*`, `s3gen.*`, `ve.*`)
- Implement S3Tokenizer loading from separate repo (e.g. `mlx-community/S3TokenizerV2`)

### Phase 3 — Core Model Ports

- Port T3 (token generator)
- Port S3Gen (decoder/vocoder)
- Port VoiceEncoder
- Port S3TokenizerV2

### Phase 4 — Generation Pipeline

- Wire T3 → S3Gen flow
- Implement reference-audio optional path (if present in Turbo)
- Validate output waveform shape + sample rate

### Phase 5 — Tests + Parity Checks

- Unit tests for config decoding
- Weight load smoke test
- Generation smoke test (short phrase)
- Optional: compare output statistics vs Python (duration, RMS, peak)

## Risks

- Weight key mapping / shape mismatches
- Quantization handling differences
- Memory spikes during generation

## Success Criteria

- Generate intelligible audio locally from Swift without Python
- Compatible with HF cache layout
- Stable API under refactor SDK modules

## Follow-ups (not in scope)

- Standard Chatterbox (non-turbo)
- Streaming chunked playback
- Advanced voice cloning controls
- GPU/memory optimizations beyond parity
