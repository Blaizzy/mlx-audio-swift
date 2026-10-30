# Higgs Audio v2

Higgs Audio v2 from Boson AI synthesizes expressive speech at 24 kHz mono. It supports a reference voice with its transcript, conversational audio context, and incremental audio streaming.

Supported models:

- [mlx-community/higgs-audio-v2-3B-mlx-q8](https://huggingface.co/mlx-community/higgs-audio-v2-3B-mlx-q8)
- [mlx-community/higgs-audio-v2-3B-mlx-q6](https://huggingface.co/mlx-community/higgs-audio-v2-3B-mlx-q6)

The [Higgs Audio v2 tokenizer](https://huggingface.co/mlx-community/higgs-audio-v2-tokenizer) downloads automatically. Reference audio should be mono at 24 kHz and include an accurate transcript.

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/higgs-audio-v2-3B-mlx-q8"
)
let (_, reference) = try loadAudioArray(
    from: URL(fileURLWithPath: "reference.wav"), sampleRate: model.sampleRate
)
let audio = try await model.generate(
    text: "Hello from Higgs Audio.",
    voice: nil,
    refAudio: reference,
    refText: "The words spoken in the reference recording.",
    language: nil
)
```

For model-specific controls, use `HiggsV2Model` with `HiggsV2GenerationOptions`. The options expose system instructions, conversation turns with optional audio codebooks, repetition avoidance, sampling warmup, fades, and streaming overlap. `GenerateParameters` controls temperature, top-p, top-k, seed, and maximum audio frames.

```swift
let higgs = try await HiggsV2Model.fromPretrained(
    "mlx-community/higgs-audio-v2-3B-mlx-q8"
)
for try await event in higgs.generateStream(
    text: "Welcome to the conversation.",
    parameters: GenerateParameters(maxTokens: 500, temperature: 0.7, topP: 0.95),
    streamingInterval: 0.64
) {
    if case .audio(let chunk) = event {
        // Append or play each chunk at 24 kHz.
    }
}
```

```bash
mlx-audio-swift-tts \
  --model mlx-community/higgs-audio-v2-3B-mlx-q8 \
  --text "Hello from Higgs Audio." \
  --ref_audio reference.wav \
  --ref_text "The words spoken in the reference recording." \
  --output speech.wav
```

Model and codec by [Boson AI](https://github.com/boson-ai/higgs-audio), with MLX checkpoints from [mlx-community](https://huggingface.co/mlx-community). Review the upstream [generation model license](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base/blob/main/LICENSE), [audio tokenizer license](https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/main/LICENSE), and [Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/LICENSE).
