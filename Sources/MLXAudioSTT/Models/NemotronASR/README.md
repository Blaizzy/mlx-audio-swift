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

## Nemotron Unified

The same model type loads [`nvidia/nemotron-labs-unified-asr-0.6b`](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b) converted with mlx-audio's `nemotron_asr.convert`. Its encoder is non-causal, so it streams buffered: each step re-encodes left context + chunk + right context and keeps the chunk's frames, as NeMo's buffered streaming does.

```swift
let model = try NemotronASRModel.fromDirectory(url)
let offline = model.generate(audio: audio, generationParameters: .init(language: "fr"))

// chunkMs is the latency (chunk + right context): 160, 240, 320 (default), 560, 1120 or 2080 ms.
let session = model.makeStreamSession(language: "fr", chunkMs: 320)
let delta = session.step(samples)
let tail = session.finish()
```

Offline, `generate` runs one full-context pass up to 60 s and 10/10/5 s windows beyond. Pass the spoken language: at low latency `auto` can emit a few first tokens in the wrong language.
