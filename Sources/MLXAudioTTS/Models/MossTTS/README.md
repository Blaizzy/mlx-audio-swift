# MOSS-TTS

MOSS-TTS covers the non-quantized full OpenMOSS text-to-speech checkpoints: the standard delay-pattern model, the dialogue model, and the local-transformer variant. It uses a Qwen3 backbone, multi-codebook audio generation, and the full MOSS Audio Tokenizer.

## Supported Models

- `OpenMOSS-Team/MOSS-TTS`
- `OpenMOSS-Team/MOSS-TTSD-v1.0`
- `OpenMOSS-Team/MOSS-TTS-Local-Transformer`

The full audio tokenizer is loaded automatically from:

- `OpenMOSS-Team/MOSS-Audio-Tokenizer`

## CLI Examples

Standard MOSS-TTS:

```bash
swift run mlx-audio-swift-tts \
  --model OpenMOSS-Team/MOSS-TTS \
  --text "Hello, this is MOSS-TTS running from Swift." \
  --output moss-tts.wav
```

Dialogue model:

```bash
swift run mlx-audio-swift-tts \
  --model OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --text "[S1] Hello. [S2] Hi, this is MOSS-TTSD running from Swift." \
  --output moss-ttsd.wav
```

Local-transformer model:

```bash
swift run mlx-audio-swift-tts \
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer \
  --text "Hello, this is the MOSS local-transformer model running from Swift." \
  --output moss-local-transformer.wav
```

Voice cloning with a single reference clip:

```bash
swift run mlx-audio-swift-tts \
  --model OpenMOSS-Team/MOSS-TTS \
  --text "This is a short cloned voice sample." \
  --ref_audio speaker.wav \
  --ref_text "Reference transcript for the speaker." \
  --output moss-clone.wav
```

## Swift Example

```swift
import Foundation
import MLXAudioCore
import MLXAudioTTS

let model = try await TTS.loadModel(modelRepo: "OpenMOSS-Team/MOSS-TTS")
let audio = try await model.generate(
    text: "Hello, this is MOSS-TTS running from Swift.",
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: "English",
    generationParameters: GenerateParameters(
        maxTokens: 120,
        temperature: 1.1,
        topP: 0.9
    )
)
```
