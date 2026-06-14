# Irodori TTS

Japanese flow-matching text-to-speech (Echo-TTS family): a Rectified-Flow DiT over
continuous `Semantic-DACVAE-Japanese-32dim` latents at 48 kHz, with v3 automatic
duration prediction and **VoiceDesign** — the voice is described by a Japanese
caption rather than a reference clip.

[Hugging Face Model Repo](https://huggingface.co/mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit)

## CLI Example

```bash
mlx-audio-swift-tts \
  --model mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit \
  --text "こんにちは。今日はいい天気ですね。" \
  --voice "落ち着いた自然な女性の声で、はっきりと読み上げてください。"
```

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await IrodoriTTSModel.fromPretrained(
    "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit"
)

// `voice` is a Japanese VoiceDesign caption describing the speaker.
let audio = try await model.generate(
    text: "こんにちは。今日はいい天気ですね。",
    voice: "落ち着いた自然な女性の声で、はっきりと読み上げてください。",
    refAudio: nil,
    refText: nil,
    language: nil
)
```

If `voice` is `nil`, a neutral default caption is used. The model is
Japanese-only — pass Japanese text directly (no `language` argument needed).

## Notes

- The tokenizer is `llm-jp/llm-jp-3-150m` (SentencePiece Unigram), downloaded
  separately on first load. It is **not** bundled with the model weights.
- The DACVAE codec ships in the model repo under `dacvae/` and is loaded
  automatically.
- On memory-constrained devices, lower `sampler.sequence_length` and/or use
  `cfg_guidance_mode: "alternating"` to reduce peak memory.

## License

See the [Irodori-TTS model repository](https://huggingface.co/mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit)
and the upstream [Aratako/Irodori-TTS](https://huggingface.co/Aratako) project for
model weight licensing.
