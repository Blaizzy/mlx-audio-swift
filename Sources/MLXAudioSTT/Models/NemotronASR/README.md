# Nemotron ASR

Swift support for NVIDIA Nemotron 3.5 ASR streaming checkpoints converted to MLX.

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
| `mlx-community/nemotron-3.5-asr-streaming-0.6b` | bf16 checkpoint |
| `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | 8-bit quantized checkpoint |

The implementation follows the NeMo `EncDecRNNTBPEModelWithPrompt` layout: causal FastConformer encoder, chunked-limited relative attention, language prompt conditioning, and greedy RNN-T decoding.
