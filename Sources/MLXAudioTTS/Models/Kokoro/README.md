# Kokoro TTS

Kokoro is an 82M parameter text-to-speech model that generates natural-sounding speech using BERT-based phoneme encoding, duration/prosody prediction, and an iSTFT HiFi-GAN decoder. Output audio is 24kHz mono.

## Supported Models

- [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16)

## Quick Start (Pre-Phonemized Text)

By default, Kokoro expects pre-phonemized IPA text as input:

```swift
import MLXAudioTTS

let model = try await KokoroModel.fromPretrained("mlx-community/Kokoro-82M-bf16")
let audio = try await model.generate(
    text: "hɛloʊ, ðɪs ɪz ə tɛst.",  // IPA phonemes
    voice: "af_heart"
)
```

## Plain Text with a TextProcessor

To use plain natural language text, provide a `TextProcessor` that handles
grapheme-to-phoneme (G2P) conversion. [MisakiSwift](https://github.com/mlalma/MisakiSwift)
(Apache 2.0) is the recommended G2P library for English:

```swift
import MLXAudioTTS

// Create your G2P adapter (implement TextProcessor protocol)
struct MisakiTextProcessor: TextProcessor {
    let g2p: EnglishG2P
    func process(text: String, language: String?) throws -> String {
        let (phonemes, _) = g2p.phonemize(text: text)
        return phonemes
    }
}

// Load model with text processor
let processor = MisakiTextProcessor(british: false)
let model = try await KokoroModel.fromPretrained(
    "mlx-community/Kokoro-82M-bf16",
    textProcessor: processor
)

// Now you can use plain text
let audio = try await model.generate(text: "Hello, this is a test.", voice: "af_heart")
```

You can also set or change the text processor after loading:

```swift
model.setTextProcessor(processor)  // enable G2P
model.setTextProcessor(nil)        // back to pre-phonemized mode
```

## Voices

Kokoro ships with multiple voice embeddings. List available voices:

```swift
let voices = model.availableVoices()  // ["af_heart", "af_bella", ...]
```

## Streaming

```swift
for try await event in model.generateStream(text: ipaText, voice: "af_heart") {
    switch event {
    case .audio(let samples):
        // Process audio chunk
        break
    }
}
```

## Why No Built-In G2P?

This library intentionally does not bundle a G2P engine. Here is why:

1. **Quality matters more than convenience.** A dictionary-only G2P (e.g., CMU Pronouncing Dictionary) produces poor results for heteronyms ("read" vs "read"), proper nouns, and out-of-vocabulary words. A bad default would lead users to blame the model rather than the G2P, creating a misleading experience.

2. **Resource footprint.** A quality G2P requires 6-18MB of pronunciation dictionaries and neural fallback models. Bundling these in the library would bloat the package for all consumers, including those who never use Kokoro.

3. **The `TextProcessor` protocol guides you toward quality solutions.** Libraries like [MisakiSwift](https://github.com/mlalma/MisakiSwift) (Apache 2.0) provide high-quality English phonemization with dictionary lookup, POS-aware disambiguation, and neural fallback for unknown words. The protocol makes integration straightforward while keeping the choice of G2P implementation yours.
