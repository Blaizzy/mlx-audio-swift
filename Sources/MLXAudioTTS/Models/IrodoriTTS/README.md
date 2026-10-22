# Irodori TTS

Japanese flow-matching text-to-speech (Echo-TTS family): a DiT over continuous
`Semantic-DACVAE-Japanese-32dim` latents at 48 kHz, with automatic duration
prediction and **VoiceDesign**: the voice is described by a Japanese caption
rather than a reference clip.

## Supported checkpoints

| Version | Model repo | Text encoder | Sampler |
|---|---|---|---|
| v3 VoiceDesign | [mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit](https://huggingface.co/mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit) | scratch-trained, `llm-jp/llm-jp-3-150m` tokenizer | 40-step Euler + CFG |
| v4.1 | [mlx-community/Irodori-TTS-v4.1-Small-8bit](https://huggingface.co/mlx-community/Irodori-TTS-v4.1-Small-8bit) | pretrained ModernBERT-ja-310m (bundled) | 40-step Euler + CFG |
| v4.1-MF | [mlx-community/Irodori-TTS-v4.1-Small-MF-8bit](https://huggingface.co/mlx-community/Irodori-TTS-v4.1-Small-MF-8bit) | pretrained ModernBERT-ja-310m (bundled) | 4-step MeanFlow, no CFG |

All versions use `model_type: irodori_tts`; the loader picks the architecture from
`config.json` (`text_encoder_type`, `flow_parameterization`). v4-Large (T5Gemma2
text encoder) is not supported.

v4.1-MF is a MeanFlow distillation of v4.1. Classifier-free guidance is baked in,
so it runs one forward pass per step for 4 steps instead of up to 3 batched passes
for 40 steps. On an M-series Mac its sampler takes ~0.15–0.2 s per sentence versus
~3.5–5 s for v3, which leaves the DACVAE decode as the largest cost.

## CLI Example

```bash
mlx-audio-swift-tts \
  --model mlx-community/Irodori-TTS-v4.1-Small-MF-8bit \
  --text "こんにちは。今日はいい天気ですね。" \
  --voice "落ち着いた自然な女性の声"
```

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "mlx-community/Irodori-TTS-v4.1-Small-MF-8bit")

// `voice` is a Japanese VoiceDesign caption describing the speaker; no reference audio needed.
let audio = try await model.generate(
    text: "こんにちは。今日はいい天気ですね。",
    voice: "落ち着いた自然な女性の声",
    refAudio: nil,
    refText: nil,
    language: nil
)
```

The same call works for every version above (`IrodoriTTSModel.fromPretrained(_:)`
loads them too). If `voice` is `nil`, a neutral default caption is used; an empty
caption is masked out. The model is Japanese-only, so no `language` argument is
needed. `refAudio` (mono, 48 kHz) adds speaker conditioning; v4 caps it at the
checkpoint's `ref_max_seconds` (120 s).

## Notes

- v3: the tokenizer is `llm-jp/llm-jp-3-150m` (SentencePiece Unigram), downloaded
  separately on first load.
- v4: the ModernBERT-ja tokenizer ships in the model repo under `tokenizer/`. A
  sanitized copy is written to the user Caches directory so swift-transformers can
  load it; the downloaded files are left untouched. Runs of several consecutive
  ASCII spaces can tokenize differently from Python (rare in Japanese text).
- v4 drops the token padding before the text encoder. Every consumer masks padding
  out, so the ModernBERT encoder and the DiT's context attention only see real tokens.
- v4 uses the upstream text normaliser (keeps 。/、, NFKC). v3 keeps the original
  Swift normaliser, so its output is unchanged.
- Without reference audio, v4 over-predicts duration for very short texts (a few
  tokens) and may repeat the sentence; this is upstream model behaviour.
- The DACVAE codec ships in the model repo under `dacvae/` and is loaded
  automatically.
- On memory-constrained devices running a CFG checkpoint (v3, v4.1), lower
  `sampler.sequence_length` and/or use `cfg_guidance_mode: "alternating"` to reduce
  peak memory.

## License

See the model repositories above and the upstream
[Aratako/Irodori-TTS](https://huggingface.co/Aratako) project for model weight
licensing.
