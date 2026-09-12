# Nemotron ASR

Swift support for NVIDIA Nemotron ASR streaming checkpoints converted to MLX.

```swift
import MLXAudioSTT

let model = try await NemotronASRModel.fromPretrained(
    "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"
)
let output = model.generate(audio: audio, generationParameters: .init(language: "auto"))
print(output.text)
```

Supported repositories:

| Model | Description |
| --- | --- |
| `animaslabs/nemotron-speech-streaming-en-0.6b-mlx-8bit` | English-only 8-bit quantized checkpoint |
| `mlx-community/nemotron-3.5-asr-streaming-0.6b` | bf16 checkpoint |
| `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | 8-bit quantized checkpoint |

The implementation supports the NeMo prompt-conditioned and English-only RNN-T layouts: causal FastConformer encoder, chunked-limited relative attention, optional language prompt conditioning, and greedy RNN-T decoding.
