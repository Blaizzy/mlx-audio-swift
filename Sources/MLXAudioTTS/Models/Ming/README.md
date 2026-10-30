# Ming dense speech synthesis

Ming dense uses a Qwen2.5 backbone to generate 44.1 kHz mono speech. Ming-3.0 supports Chinese, English, German, Spanish, French, Korean and Japanese, built-in voices, delivery instructions and reference voice conditioning.

Supported checkpoints:

- [mlx-community/Ming-3.0-tts-0.5B-bf16](https://huggingface.co/mlx-community/Ming-3.0-tts-0.5B-bf16)
- [inclusionAI/Ming-omni-tts-0.5B](https://huggingface.co/inclusionAI/Ming-omni-tts-0.5B)

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(
    modelRepo: "mlx-community/Ming-3.0-tts-0.5B-bf16")
let audio = try await model.generate(
    text: "Good evening, and welcome to the news.",
    voice: "UK_Charlotte", refAudio: nil, refText: nil, language: "en",
    generationParameters: GenerateParameters(temperature: 0, seed: 42))
```

Reference audio must be mono at the model's 44.1 kHz sample rate. Supplying its transcript also conditions synthesis on its acoustic history. Model instances require serialized use; create a fresh reference tensor for every generation call.

```swift
let model = try await MingDenseModel.fromPretrained(
    "mlx-community/Ming-3.0-tts-0.5B-bf16")
let (_, reference) = try loadAudioArray(
    from: URL(fileURLWithPath: "reference.wav"), sampleRate: model.sampleRate)
let audio = try await model.generate(
    text: "The evening wind blew in from the mountain pass.",
    voice: nil, refAudio: reference,
    refText: "The exact words spoken in the reference clip.", language: "en",
    generationParameters: GenerateParameters(temperature: 0, seed: 42))
```

Swift callers can set `model.options.instruction`, `cfgScale`, `sigma`, `flowSteps`, `rawText`, `fragmentLength`, `trimSilence`, `zeroSpeakerEmbedding` or a precomputed 192-value `speakerEmbedding`. Use `MingText.instruction(emotion: "高兴")` for structured Ming-3.0 controls, or `MingText.instruction(style: "成熟沉稳的男性旁白")` with `zeroSpeakerEmbedding = true` for voice design. Set `language` to `de`, `es`, `fr`, `ja` or `ko` for the additional target languages.

`generateStream` yields each completed Ming-3.0 text fragment. Legacy Omni returns its completed utterance. `maxTokens` limits generated latent patches; Ming-3.0 also has a 512-position budget per fragment. Shorten long reference clips or instructions if they exhaust that budget. Numbers, dates and units should be written out as spoken words.

```bash
mlx-audio-swift-tts \
  --model mlx-community/Ming-3.0-tts-0.5B-bf16 \
  --text "Good evening, and welcome to the news." \
  --voice UK_Charlotte --output ming.wav
```

For legacy Omni, omit `refText` to condition on the reference speaker. Acoustic continuation with a reference transcript can produce near-silence; use Ming-3.0 for audio-and-transcript conditioning.

When loading a local legacy Omni folder, pass a native CAMPPlus safetensors file as `speakerWeightsURL` to enable automatic speaker extraction. Repository loading obtains the compatible pinned speaker component automatically.

Ming-3.0 weights use the [MIT license](https://huggingface.co/mlx-community/Ming-3.0-tts-0.5B-bf16/blob/main/LICENSE). Legacy Omni weights use [Apache 2.0](https://huggingface.co/inclusionAI/Ming-omni-tts-0.5B). See the [Ming-3.0 project](https://github.com/inclusionAI/Ming-3.0-tts), [Ming project](https://github.com/inclusionAI/Ming) and [CAMPPlus project](https://github.com/modelscope/3D-Speaker) for attribution and voice guidance. The legacy speaker component uses Apache 2.0 and is supplied by [StepAudio2](https://huggingface.co/mlx-community/Step-Audio-2-token2wav).

The 16.8B Bailing MoE model uses the separate [Ming Omni model](../MingOmni/README.md).
