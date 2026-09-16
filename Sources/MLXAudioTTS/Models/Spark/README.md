# Spark-TTS

MLX Swift port of [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) (SparkAudio),
a text-to-speech model where a Qwen2 language model emits [BiCodec](https://arxiv.org/abs/2503.01710)
semantic + global tokens that the BiCodec decodes to a 16 kHz waveform.

## Status

- **Controllable TTS** (gender / pitch / speed) — implemented.
- **Voice cloning** (reference audio) — implemented via the full BiCodec encode
  path: mel-spectrogram, Wav2Vec2-large-xlsr-53 features, feature encoder + FVQ
  (semantic tokens), and ECAPA-TDNN + perceiver resampler + FSQ (global speaker
  tokens).

## Usage

Controllable TTS:

```swift
import MLXAudioTTS

let model = try await SparkModel.fromPretrained("mlx-community/Spark-TTS-0.5B-bf16")
let audio = try await model.generate(
    text: "Hello world, this is a test.",
    voice: "female",              // "female" | "male"
    refAudio: nil, refText: nil, language: nil,
    generationParameters: model.defaultGenerationParameters
)
```

Voice cloning — pass a 16 kHz mono reference clip as `refAudio` (and optionally its
transcript as `refText` to also seed the reference semantic tokens):

```swift
let audio = try await model.generate(
    text: "This sentence is spoken in the reference voice.",
    voice: nil,
    refAudio: referenceWaveform16k, refText: nil, language: nil,
    generationParameters: model.defaultGenerationParameters
)
```

`TTS.loadModel(modelRepo: "mlx-community/Spark-TTS-0.5B-bf16")` also resolves to
this model.

## Validation

Every stage of the reference-audio encode path is numerically checked against the
reference Python implementation (`mlx-audio`, in eval mode): wav2vec2 features and
the encoder latents match within ~1e-3, the ECAPA latent matches exactly, and the
global speaker tokens and clone prompt match exactly. The BiCodec synthesis path
matches the decoded waveform within ~0.09% relative error given identical tokens.
