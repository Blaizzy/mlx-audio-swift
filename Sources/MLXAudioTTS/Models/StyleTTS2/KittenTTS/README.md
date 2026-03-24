# KittenTTS

Compact non-autoregressive English TTS built on the StyleTTS2 stack.
Uses an ALBERT encoder, duration-based prosody prediction, and an iSTFT-based vocoder.
Output is 24kHz mono audio.

## Supported Models

KittenTTS model loading is selected by model type `kitten_tts`.
Tested repository names include:

- `mlx-community/kitten-tts-mini-0.8`
- `mlx-community/kitten-tts-nano-0.8-8bit`

## Swift Example

```swift
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "mlx-community/kitten-tts-mini-0.8")
let audio = try await model.generate(
    text: "Hello from Kitten TTS.",
    voice: "expr-voice-5-m"
)
```

## CLI Example

```bash
mlx-audio-swift-tts \
  --model mlx-community/kitten-tts-mini-0.8 \
  --voice expr-voice-5-m \
  --text "Hello from Kitten TTS."
```

## Voices

KittenTTS reads voices from the model repository's `voices.safetensors` file.
The default voice is `expr-voice-5-m`.

Known voices in the current Kitten checkpoints:

- `expr-voice-2-f`
- `expr-voice-2-m`
- `expr-voice-3-f`
- `expr-voice-3-m`
- `expr-voice-4-f`
- `expr-voice-4-m`
- `expr-voice-5-f`
- `expr-voice-5-m`

Models can also define voice aliases in `config.json`.
For example, `Bella` may map to `expr-voice-2-f`.

## English G2P

By default, KittenTTS uses the built-in `MisakiTextProcessor` for English phonemization.
That path combines:

- rule-based English preprocessing
- lexicon lookup from the Kitten G2P resource bundle on Hugging Face
- a BART fallback model for unknown words

G2P resources are downloaded automatically during model loading.

If your input is already phonemized IPA, you can disable text processing:

```swift
let model = try await TTS.loadModel(
    modelRepo: "mlx-community/kitten-tts-mini-0.8",
    textProcessor: nil
)
```

## Streaming

```swift
for try await event in model.generateStream(
    text: "Streaming synthesis with Kitten.",
    voice: "expr-voice-5-m"
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

- language support is English-centric
- voice aliases and speed priors come from the model config
- quantized checkpoints are supported through the normal `quantization` config path
