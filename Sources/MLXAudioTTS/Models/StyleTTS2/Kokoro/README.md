# Kokoro TTS

Lightweight non-autoregressive TTS (82M params, 24kHz). Uses ALBERT encoder + prosody prediction + iSTFT-based vocoder. Supports 9 languages with 40+ voices.

## Supported Models

- [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16)

## Languages & Voices

| Language | Prefix | Voices |
|----------|--------|--------|
| 🇺🇸 American English | `af_*` / `am_*` | af_heart, af_bella, am_adam, am_michael, ... |
| 🇬🇧 British English | `bf_*` / `bm_*` | bf_emma, bf_isabella, bm_george, bm_lewis |
| 🇪🇸 Spanish | `ef_*` / `em_*` | ef_dora, em_alex, em_santa |
| 🇫🇷 French | `ff_*` | ff_siwis |
| 🇮🇳 Hindi | `hf_*` / `hm_*` | hf_alpha, hf_beta, hm_omega, hm_psi |
| 🇮🇹 Italian | `if_*` / `im_*` | if_sara, im_nicola |
| 🇯🇵 Japanese | `jf_*` / `jm_*` | jf_alpha, jf_gongitsune, jm_kumo |
| 🇧🇷 Portuguese | `pf_*` / `pm_*` | pf_dora, pm_alex, pm_santa |
| 🇨🇳 Chinese | `zf_*` / `zm_*` | zf_xiaobei, zf_xiaoni, zm_yunjian |

Language is auto-detected from voice prefix.

## Swift Example

```swift
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "mlx-community/Kokoro-82M-bf16")
let audio = try await model.generate(
    text: "Hello from Kokoro!",
    voice: "af_heart"
)
```

### Multilingual

```swift
let audio = try await model.generate(
    text: "Hola, esto es una prueba.",
    voice: "ef_dora"  // Spanish auto-detected from "e" prefix
)
```

### Explicit Language

```swift
let audio = try await model.generate(
    text: "Bonjour le monde",
    voice: "ff_siwis",
    language: "fr"
)
```

## Warm-Up

First call for each language downloads and loads resources on demand:

| Language | Resource | Cold Start |
|----------|----------|------------|
| English | CMUdict + G2P rules | ~0.4s |
| EU languages (es, fr, it, pt, de, ru, ...) | IPA lexicon TSV | ~0.3–3.8s (depends on size) |
| JA, HI, ZH | ByT5 neural G2P model (20MB) | ~1.2s |

**Large lexicons** (es: 595K entries, ru: 533K) take longer on first load. Subsequent calls use the in-memory cache and are <1ms.

To avoid latency on the first `generate()` call, pre-warm the processor:

```swift
let model = try await TTS.loadModel(modelRepo: "mlx-community/Kokoro-82M-bf16")

// Pre-warm: downloads lexicon/model and loads into memory
if let kokoro = model as? KokoroModel,
   let processor = kokoro.textProcessor as? KokoroMultilingualProcessor
{
    try await processor.prepare(for: "es")
}

// First generate() is now fast
let audio = try await model.generate(text: "Hola", voice: "ef_dora")
```

## Streaming

```swift
for try await event in model.generateStream(
    text: "Streaming speech synthesis.",
    voice: "af_heart"
) {
    switch event {
    case .audio(let samples):
        // Process audio chunk (24kHz Float32)
        break
    case .info(let info):
        print("Generated in \(info.generateTime)s")
    }
}
```

## G2P Pipeline

| Language | Method | Source |
|----------|--------|--------|
| English | CMUdict + rule-based (Misaki) | MIT |
| ES, FR, IT, PT, DE, RU, AR, CS, FA, NL, SV, SW | IPA lexicon lookup | gruut (MIT) |
| JA, HI, ZH | ByT5 neural G2P | MIT |

Lexicons: [beshkenadze/kokoro-ipa-lexicons](https://huggingface.co/beshkenadze/kokoro-ipa-lexicons)
Neural G2P: [beshkenadze/g2p-multilingual-byT5-tiny-mlx](https://huggingface.co/beshkenadze/g2p-multilingual-byT5-tiny-mlx)
