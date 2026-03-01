# MMS-LID Language Identification

Swift port of Meta's MMS-LID (Massively Multilingual Speech — Language Identification) model. Identifies the spoken language from raw audio waveforms using a Wav2Vec2 backbone with a classification head.

[Hugging Face Model Repo](https://huggingface.co/facebook/mms-lid-256)

## Architecture

1. **Feature Extractor** — 7 temporal convolution layers (stride-based downsampling) converting raw waveform to latent representations
2. **Feature Projection** — LayerNorm + Linear projection to hidden dimension
3. **Wav2Vec2 Encoder** — Positional convolutional embedding + 48 transformer encoder layers with pre-layer-norm
4. **Classifier** — Mean pooling → Linear projector → Linear classifier over 256 languages

## Quick Start

```swift
import MLXAudioCore
import MLXAudioLID

let model = try await Wav2Vec2ForSequenceClassification.fromPretrained("facebook/mms-lid-256")

let (_, audio) = try loadAudioArray(from: audioURL)
let output = model.predict(waveform: audio, topK: 5)

print("Language: \(output.language) (\(output.confidence * 100)%)")
for pred in output.topLanguages {
    print("  \(pred.language): \(pred.confidence * 100)%")
}
```

## API

### `model.predict()`

Run language identification on a 16 kHz mono audio waveform.

```swift
let output = model.predict(
    waveform: audioData,   // MLXArray — 1-D audio samples (16 kHz)
    topK: 5                // number of top language predictions to return
)
```

**Returns** a `LIDOutput` with:

| Field | Type | Description |
|-------|------|-------------|
| `language` | `String` | ISO 639-3 code of the top predicted language |
| `confidence` | `Float` | Softmax probability of the top prediction (0–1) |
| `topLanguages` | `[LanguagePrediction]` | Top-K predictions sorted by confidence |

Each `LanguagePrediction` contains:

| Field | Type | Description |
|-------|------|-------------|
| `language` | `String` | ISO 639-3 language code (e.g. `"eng"`, `"fra"`, `"deu"`) |
| `confidence` | `Float` | Softmax probability (0–1) |

### `model.callAsFunction()`

Low-level forward pass returning raw logits.

```swift
let logits = model(waveform)  // MLXArray (1, numLabels)
```

### `Wav2Vec2ForSequenceClassification.fromPretrained()`

Download and load a model from Hugging Face.

```swift
let model = try await Wav2Vec2ForSequenceClassification.fromPretrained(
    "facebook/mms-lid-256",   // HuggingFace repo ID
    hfToken: nil              // optional HF token for private models
)
```

## Supported Models

| Model | Languages | Size | Description |
|-------|-----------|------|-------------|
| `facebook/mms-lid-256` | 256 | 3.86 GB | Primary MMS-LID model |

## Notes

- Input audio is automatically normalized (zero-mean, unit-variance) inside `predict()`
- Audio should be 16 kHz mono; use `loadAudioArray(from:)` from `MLXAudioCore` for automatic resampling
- The model uses 48 transformer encoder layers — first inference includes weight loading (~3-4s), subsequent calls are fast (~250ms for 10s audio on M1)
- Language codes follow ISO 639-3 (e.g. `"eng"` for English, `"fra"` for French, `"rus"` for Russian)
- Ported from [facebook/mms-lid-256](https://huggingface.co/facebook/mms-lid-256) via the [MLX Audio Python LID implementation](https://github.com/Blaizzy/mlx-audio)
