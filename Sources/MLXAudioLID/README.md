# MLXAudioLID — Language Identification

Swift MLX implementations of spoken language identification models. Identifies the spoken language from raw audio waveforms.

## Supported Models

| Model | Class | Languages | Size | Latency (M1, 10s) | Description |
|-------|-------|-----------|------|--------------------|-------------|
| `facebook/mms-lid-256` | `Wav2Vec2ForSequenceClassification` | 256 | 3.86 GB | ~250ms | Wav2Vec2-based LID |
| `beshkenadze/lang-id-voxlingua107-ecapa-mlx` | `EcapaTdnn` | 107 | 81 MB | ~15ms | ECAPA-TDNN speaker/language embeddings |

## Quick Start

### MMS-LID-256 (Wav2Vec2)

```swift
import MLXAudioCore
import MLXAudioLID

let model = try await Wav2Vec2ForSequenceClassification.fromPretrained("facebook/mms-lid-256")

let (_, audio) = try loadAudioArray(from: audioURL)
let output = model.predict(waveform: audio, topK: 5)

print("Language: \(output.language) (\(output.confidence * 100)%)")
```

### ECAPA-TDNN (VoxLingua107)

```swift
import MLXAudioCore
import MLXAudioLID

let model = try await EcapaTdnn.fromPretrained("beshkenadze/lang-id-voxlingua107-ecapa-mlx")

let (_, audio) = try loadAudioArray(from: audioURL)
let output = model.predict(waveform: audio, topK: 5)

print("Language: \(output.language) (\(output.confidence * 100)%)")
```

## API

Both models share the same `LIDOutput` type and `predict()` interface.

### `model.predict(waveform:topK:)`

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
| `language` | `String` | ISO code of the top predicted language |
| `confidence` | `Float` | Probability of the top prediction (0–1) |
| `topLanguages` | `[LanguagePrediction]` | Top-K predictions sorted by confidence |

### `model.callAsFunction(_:)`

Low-level forward pass returning raw logits/log-probabilities.

```swift
// Wav2Vec2: raw waveform → logits
let logits = wav2vec2Model(waveform)

// ECAPA-TDNN: mel spectrogram → log-probabilities
let mel = EcapaMelSpectrogram.compute(audio: waveform)
let logProbs = ecapaModel(mel)
```

### `fromPretrained(_:)`

Download and load a model from Hugging Face. Uses `HF_TOKEN` environment variable or Info.plist key.

```swift
let wav2vec2 = try await Wav2Vec2ForSequenceClassification.fromPretrained("facebook/mms-lid-256")
let ecapa = try await EcapaTdnn.fromPretrained("beshkenadze/lang-id-voxlingua107-ecapa-mlx")
```

## Architecture

### MMS-LID-256

1. **Feature Extractor** — 7 temporal convolution layers converting raw waveform to latent representations
2. **Feature Projection** — LayerNorm + Linear projection to hidden dimension
3. **Wav2Vec2 Encoder** — Positional convolutional embedding + 48 transformer encoder layers
4. **Classifier** — Mean pooling → Linear projector → Linear classifier over 256 languages

### ECAPA-TDNN

1. **Mel Spectrogram** — SpeechBrain-compatible 60-bin log-mel (Hamming window, HTK scale, top_db=80)
2. **TDNN + SE-Res2Net Blocks** — 1 TDNN entry block + 3 SE-Res2Net blocks with multi-scale processing
3. **Multi-layer Feature Aggregation** — Concatenation of block outputs → TDNN
4. **Attentive Statistics Pooling** — Attention-weighted mean + std pooling
5. **Classifier** — BatchNorm → DNN → Linear over 107 languages

## Notes

- Audio should be 16 kHz mono; use `loadAudioArray(from:)` from `MLXAudioCore` for automatic resampling
- MMS-LID-256 normalizes audio internally (zero-mean, unit-variance)
- ECAPA-TDNN computes mel spectrogram internally from raw waveform
- MMS-LID language codes follow ISO 639-3 (e.g. `"eng"`, `"fra"`, `"rus"`)
- ECAPA-TDNN language codes follow ISO 639-1/3 (e.g. `"en"`, `"fr"`, `"ceb"`)
- ECAPA-TDNN is ~16x faster and ~47x smaller than MMS-LID-256
- Set `HF_TOKEN` environment variable or Info.plist key for private model access
