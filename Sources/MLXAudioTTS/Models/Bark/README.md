# Bark

Bark generates multilingual speech, vocal expressions and sound effects as mono 24 kHz audio. It accepts speaker presets and token histories for continuation. Text is limited to 256 multilingual BERT tokens per utterance; longer text is truncated.

Supported checkpoint: [mlx-community/bark-small](https://huggingface.co/mlx-community/bark-small). The codec is [mlx-community/encodec-24khz-float32](https://huggingface.co/mlx-community/encodec-24khz-float32).

```swift
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "mlx-community/bark-small")
let audio = try await model.generate(
    text: "Hello, this is Bark speaking.",
    voice: "v2/en_speaker_6",
    refAudio: nil,
    refText: nil,
    language: nil,
    generationParameters: GenerateParameters(maxTokens: 768, temperature: 0.7, seed: 42)
)
```

The text determines the language. Omit `voice` for unconditioned speech. Presets such as `en_speaker_0`, `announcer`, and nested `v2/en_speaker_6` are resolved through the checkpoint's speaker manifest. Some preset files download on first use.

```sh
mlx-audio-swift-tts --model mlx-community/bark-small \
  --text "Hello, this is Bark speaking." --voice v2/en_speaker_6 \
  --output bark.wav
```

For explicit continuation, load `BarkModel`, construct a `BarkHistory` from semantic tokens and coarse/fine codebooks, and call `generate(text:history:parameters:)`. `BarkModel.options` controls the coarse history/window and fine-stage temperature. Set `options.useFineHistory = true` to also condition the fine stage on the supplied history. `generateStream` emits the complete utterance once synthesis finishes.

Bark is by [Suno](https://github.com/suno-ai/bark) under the [MIT license](https://github.com/suno-ai/bark/blob/main/LICENSE). This implementation follows [MLX Audio](https://github.com/Blaizzy/mlx-audio) ([MIT](https://github.com/Blaizzy/mlx-audio/blob/main/LICENSE)). Encodec code is [MIT licensed](https://github.com/facebookresearch/encodec/blob/main/LICENSE); its pretrained weights use [CC BY-NC 4.0](https://huggingface.co/facebook/encodec_24khz). These codec terms apply separately from Bark's license.

The [multilingual BERT tokenizer](https://huggingface.co/google-bert/bert-base-multilingual-cased) is distributed under Apache 2.0.
