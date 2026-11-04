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
| `mlx-community/nemotron-labs-unified-asr-0.6b-bf16` | Nemotron Unified, bf16 checkpoint (~1.25 GB) |
| `mlx-community/nemotron-labs-unified-asr-0.6b-8bit` | Nemotron Unified, 8-bit quantized checkpoint |
| `mlx-community/nemotron-labs-unified-asr-0.6b-4bit` | Nemotron Unified, 4-bit quantized checkpoint (~472 MB) |

The implementation supports the NeMo prompt-conditioned and English-only RNN-T layouts: causal FastConformer encoder, chunked-limited relative attention, optional language prompt conditioning, and greedy RNN-T decoding.

## Nemotron Unified

The same model type loads [`nvidia/nemotron-labs-unified-asr-0.6b`](https://huggingface.co/nvidia/nemotron-labs-unified-asr-0.6b) converted with mlx-audio's `nemotron_asr.convert` ([bf16](https://huggingface.co/mlx-community/nemotron-labs-unified-asr-0.6b-bf16), [8-bit](https://huggingface.co/mlx-community/nemotron-labs-unified-asr-0.6b-8bit), [4-bit](https://huggingface.co/mlx-community/nemotron-labs-unified-asr-0.6b-4bit)). Its encoder is non-causal, so it streams buffered: each step re-encodes left context + chunk + right context and keeps the chunk's frames, as NeMo's buffered streaming does.

```swift
let model = try await NemotronASRModel.fromPretrained("mlx-community/nemotron-labs-unified-asr-0.6b-8bit")
let offline = model.generate(audio: audio, generationParameters: .init(language: "fr"))

// chunkMs is the latency (chunk + right context), 160-2080 ms; 320 by default.
let session = model.makeStreamSession(language: "fr", chunkMs: 320)
let delta = session.step(samples)
let tail = session.finish()
```

`chunkMs` is rounded to whole 80 ms frames and clamped to 160-2080 ms. NVIDIA's recommended contexts give the chunk (160/240/320 ms: 80 ms; 560 ms: 160 ms; 1120 ms: 560 ms; 2080 ms: 1040 ms); a latency between two rows takes the lower row's chunk and the rest as right context.

Offline, `generate` runs one full-context pass up to 60 s and 10/10/5 s windows beyond. Pass the spoken language: at low latency `auto` can emit a few first tokens in the wrong language. A bare code matches a unique locale key, as in NeMo; an unknown or ambiguous language falls back to the default with a warning on stderr.

`makeStreamSession(language:bufferedContext:)` takes an explicit layout in 80 ms frames (`NemotronASRBufferedContext(left:chunk:right:)`). Both throw `NemotronASRStreamingError` for an invalid layout or a cache-aware checkpoint.

`NemotronUnifiedNeMoParityTests` checks NeMo's token ids for `media/false-turn.wav` at five latencies. Point it at an fp32 conversion (`nemotron_asr.convert --dtype float32` in mlx-audio) to rerun it:

```bash
TEST_RUNNER_MLXAUDIO_ENABLE_MLX_RUNTIME_TESTS=1 \
TEST_RUNNER_NEMOTRON_UNIFIED_MLX_PATH=/path/to/nemotron-labs-unified-asr-0.6b-fp32 \
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
  -only-testing:MLXAudioTests/NemotronUnifiedNeMoParityTests
```
