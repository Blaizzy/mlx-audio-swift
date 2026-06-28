# OmniVoice

Swift/MLX port of [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), a multilingual zero-shot TTS model: a bidirectional diffusion LM over a Qwen3 backbone (28 layers, 1024 hidden) generating 9 RVQ codebooks at 24 kHz, decoded by a HiggsAudioV2 acoustic codec.

## Weights

Two MLX conversions are available, both with the complete audio codec
(acoustic **and** semantic encode path), so both support voice cloning:

- `mlx-community/OmniVoice` — fp32.
- `mlx-community/OmniVoice-bf16` — bfloat16: ~half the size and MLX's native
  compute dtype on Apple Silicon.

## Usage

```swift
let model = try await TTS.loadModel(modelRepo: "mlx-community/OmniVoice")

// Voice design (instruction)
let audio = try await model.generate(
    text: "Hello from OmniVoice on Apple Silicon.",
    voice: "female, warm, clear voice",
    refAudio: nil, refText: nil, language: nil,
    generationParameters: .init()
)
```

CLI:

```bash
swift run mlx-audio-swift-tts --model mlx-community/OmniVoice \
    --text "Hello!" --voice "male, british accent" --output out.wav
```

Generation knobs live in `OmniVoiceGenerateParameters` (`numStep`, `guidanceScale`, `speed`, `positionTemperature`, `tShift`, ...). Defaults match the Python reference (`num_steps=32`, `guidance_scale=2.0`).

## Modes

- **Auto voice** (`voice: nil`) — works
- **Voice design** (`voice: "female, low pitch, ..."`) — works
- **Voice cloning** (`refAudio` + `refText`) — works. Reference audio is
  encoded through the full HiggsAudioV2 semantic path (HuBERT + SemanticEncoder
  + fusion projection + residual RVQ). Requires the full `mlx-community/OmniVoice`
  checkpoint; the `-bf16` variant strips `semantic_model.*` and cannot encode
  reference audio.

## Implementation notes

- The backbone runs **bidirectional attention** (NAR diffusion); no causal
  mask is ever applied, matching the Python reference.
- The `<|denoise|>` style token is emitted only when reference audio is
  present (`denoise=has_ref` in the reference implementation).
- Timestep schedule and per-step unmask counts mirror
  `mlx_audio/tts/models/omnivoice/generation.py`; the final step reveals all
  remaining masked positions.
- Checkpoints with fused `audio_embeddings.weight` / `audio_heads.weight`
  (`[C*V, H]`) are split into per-codebook tensors during sanitize;
  `codebook_layer_offsets` is dropped (not needed — embeddings are indexed
  per codebook).
