# MLXAudioG2P

`MLXAudioG2P` is a text-only grapheme-to-phoneme module for MLX Audio Swift.

## Scope

- Deterministic normalization
- Deterministic tokenization
- Lexicon-first phonemization
- Rule-based fallback phonemization
- Optional token/phoneme alignment

## Clean-room and provenance

This module is a self-authored MIT implementation in this repository. It is not a direct source port of MisakiSwift or hexgrad/misaki.

## Quick usage

```swift
import MLXAudioG2P

let pipeline = G2PPipeline.preview()
let result = try pipeline.convert("Hello world")
print(result.phonemes.render())
```

## Notes

- Neural fallback is intentionally out of scope for v1.
- Future TTS integration should happen through adapters from consuming modules.
