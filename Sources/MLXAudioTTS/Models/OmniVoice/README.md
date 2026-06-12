# OmniVoice

Swift/MLX port of [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice), a multilingual zero-shot TTS model: a bidirectional diffusion LM over a Qwen3 backbone (28 layers, 1024 hidden) generating 8 RVQ codebooks at 24 kHz, decoded by a HiggsAudioV2 acoustic codec.

## Weights

Use the full conversion `mlx-community/OmniVoice` (fp32, includes the complete
audio tokenizer). `mlx-community/OmniVoice-bf16` ships a stripped tokenizer
(no `semantic_model.*` weights) and a different fused embedding layout; both
layouts are supported by the loader, but the stripped tokenizer cannot encode
reference audio.

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
- **Voice cloning** (`refAudio` + `refText`) — incomplete: the HiggsAudioV2
  semantic encode path (HuBERT + SemanticEncoder + fusion projection) is not
  ported yet, so reference audio is tokenized through an acoustic-only
  approximation and the cloned voice will not match the reference. Semantic
  weights are dropped at load time until the port lands.

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
