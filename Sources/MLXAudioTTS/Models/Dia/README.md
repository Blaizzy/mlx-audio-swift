# Dia

Swift/MLX port of [nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B), a 1.6B dialogue TTS model: an encoder–decoder transformer that byte-tokenizes the input text and autoregressively decodes 9 delayed DAC codebooks with classifier-free guidance, decoded to a 44.1 kHz waveform by the Descript audio codec.

## Weights

- `mlx-community/Dia-1.6B` — fp32.

The Descript codec (`mlx-community/descript-audio-codec-44khz`) is downloaded automatically on first use.

## Usage

```swift
let model = try await TTS.loadModel(modelRepo: "mlx-community/Dia-1.6B")

let audio = try await model.generate(
    text: "[S1] Hello from Dia on Apple Silicon. [S2] It runs entirely on MLX.",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: .init(maxTokens: 3072, temperature: 1.3, topP: 0.95)
)
```

CLI:

```bash
swift run mlx-audio-swift-tts --model mlx-community/Dia-1.6B \
    --text "[S1] Hello! [S2] How are you?" --output out.wav
```

## Notes

- Speaker turns are written as `[S1]`/`[S2]` tags inside the text; they are mapped to the control bytes the model was trained on.
- Generation is stochastic (top-k CFG filtering, `cfg_scale=3.0`); pass `temperature: 0` for greedy, reproducible output.
- The encoder pass, first decode-step logits, and the full greedy waveform match the Python `mlx-audio` reference to a relative error of ~1e-4 (float32).
