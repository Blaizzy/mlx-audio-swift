# Spark-TTS

MLX Swift port of [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) (SparkAudio),
a text-to-speech model where a Qwen2 language model emits [BiCodec](https://arxiv.org/abs/2503.01710)
semantic + global tokens that the BiCodec decodes to a 16 kHz waveform.

## Status

- **Controllable TTS** (gender / pitch / speed) is implemented.
- **Voice cloning** (reference-audio encode via mel + Wav2Vec2 + ECAPA + perceiver)
  is not yet ported; the BiCodec encode path is omitted and its checkpoint weights
  are dropped on load.

## Usage

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

`TTS.loadModel(modelRepo: "mlx-community/Spark-TTS-0.5B-bf16")` also resolves to
this model.

## Validation

The BiCodec synthesis path is numerically validated against the reference Python
implementation (`mlx-audio`): with identical global/semantic tokens the decoded
waveform matches within ~0.09% relative error.
