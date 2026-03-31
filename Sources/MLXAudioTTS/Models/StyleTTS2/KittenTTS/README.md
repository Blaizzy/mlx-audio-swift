# KittenTTS

Compact non-autoregressive English TTS built on the StyleTTS2 stack.
Uses an ALBERT encoder, duration-based prosody prediction, and an iSTFT-based vocoder.
Official KittenTTS docs describe it as a developer-preview English TTS family with 24kHz output.

## Supported Models

Official KittenTTS repositories:

- [KittenML/kitten-tts-mini-0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8) - 80M params
- [KittenML/kitten-tts-micro-0.8](https://huggingface.co/KittenML/kitten-tts-micro-0.8) - 40M params
- [KittenML/kitten-tts-nano-0.8-fp32](https://huggingface.co/KittenML/kitten-tts-nano-0.8-fp32) - 15M params
- [KittenML/kitten-tts-nano-0.8-int8](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) - 15M params, int8

Verified MLX ports currently include at least `mlx-community/kitten-tts-mini-0.8`
and `mlx-community/kitten-tts-micro-0.8`.

## Swift Example

```swift
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "mlx-community/kitten-tts-mini-0.8")
let audio = try await model.generate(
    text: "Hello from Kitten TTS.",
    voice: "Bella"
)
```

## CLI Example

```bash
mlx-audio-swift-tts \
  --model mlx-community/kitten-tts-mini-0.8 \
  --voice Bella \
  --text "Hello from Kitten TTS."
```

## Voices

Official v0.8 voice list:

- `Bella`
- `Jasper`
- `Luna`
- `Bruno`
- `Rosie`
- `Hugo`
- `Kiki`
- `Leo`

Legacy v0.2 checkpoints used the older voice IDs:

- `expr-voice-2-f`, `expr-voice-2-m`
- `expr-voice-3-f`, `expr-voice-3-m`
- `expr-voice-4-f`, `expr-voice-4-m`
- `expr-voice-5-f`, `expr-voice-5-m`

Models can also define voice aliases in `config.json`.
In the current Swift port, aliases from the checkpoint config are resolved automatically.

## English G2P

By default, KittenTTS uses the built-in `MisakiTextProcessor` for English phonemization.
That path combines:

- rule-based English preprocessing
- lexicon lookup from downloaded Kitten G2P resources on Hugging Face
- a BART fallback model for unknown words

G2P resources are downloaded automatically during model loading.

If your input is already phonemized IPA, instantiate `KittenTTSModel` directly
with `textProcessor: nil`.

`TTS.loadModel()` does not disable G2P when `textProcessor` is `nil`; it
installs the default `MisakiTextProcessor()` automatically.

```swift
let model = try await KittenTTSModel.fromPretrained(
    "mlx-community/kitten-tts-mini-0.8",
    textProcessor: nil
)
```

## Streaming

```swift
for try await event in model.generateStream(
    text: "Streaming synthesis with Kitten.",
    voice: "Leo"
) {
    switch event {
    case .audio(let samples):
        break
    case .info(let info):
        print("Generated in \(info.generateTime)s")
    }
}
```

## Notes

- English only in the current official release
- built-in text preprocessing covers numbers, currencies, and units
- `nano-int8` is the smallest official variant, but upstream notes that some users have reported issues with it
- quantized checkpoints are supported through the normal `quantization` config path
- upstream project docs: [KittenML/KittenTTS](https://github.com/KittenML/KittenTTS)
