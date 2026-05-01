# VoxCPM2 TTS

Autoregressive MiniCPM backbone + CFM diffusion + AudioVAE decoder. Produces 48kHz speech with voice cloning from a reference audio clip. 2B parameters, supports 30+ languages.

## Supported Models

- [mlx-community/VoxCPM2-4bit](https://huggingface.co/mlx-community/VoxCPM2-4bit) (4-bit quantized, ~2.3GB)

## Swift Example

```swift
import MLXAudioTTS
import MLXAudioCore

// Load model
let model = try await VoxCPM2Model.fromPretrained("mlx-community/VoxCPM2-4bit")

// Voice cloning requires reference audio
let (_, refAudio) = try loadAudioArray(from: referenceAudioURL)
let audio = try await model.generate(
    text: "Hello, this is a test of VoxCPM2.",
    voice: nil, refAudio: refAudio, refText: nil, language: nil,
    generationParameters: GenerateParameters(maxTokens: 100, temperature: 1.0)
)
// audio is an MLXArray of Float32 samples at 48kHz
```

### Zero-Shot (No Reference)

```swift
let audio = try await model.generate(
    text: "Zero-shot generation without a reference voice.",
    voice: nil, refAudio: nil, refText: nil, language: nil,
    generationParameters: GenerateParameters(maxTokens: 100, temperature: 1.0)
)
```

## Streaming Example

VoxCPM2 streaming wraps the full generation and yields the result as a single audio event:

```swift
for try await event in model.generateStream(
    text: "Streaming test.", voice: nil, refAudio: refAudio,
    refText: nil, language: nil,
    generationParameters: GenerateParameters(maxTokens: 100, temperature: 1.0)
) {
    switch event {
    case .audio(let samples):
        // Full audio output (48kHz Float32)
        break
    case .info(let info):
        print("Generated in \(info.generateTime)s")
    case .token(_):
        break
    }
}
```

## Output Format

- **Sample rate**: 48kHz
- **Format**: Mono Float32 PCM
- **VAE encoder rate**: 16kHz (internal; resampling is handled automatically)

## Known Limitations

- The 4-bit quantized stop predictor can be unreliable — callers should set `maxTokens` conservatively to avoid runaway generation.
- Short prompts (especially in Chinese) may have text-following issues, consistent with the upstream Python implementation.
- Streaming currently wraps the full `generate()` call rather than yielding incremental audio chunks.
