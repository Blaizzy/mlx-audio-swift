# OmniVoice Support for MLX Audio Swift

This implementation adds comprehensive support for the **OmniVoice** text-to-speech model to the MLX Audio Swift project, mirroring the functionality of the Python `omnivoice-infer` CLI.

## Overview

OmniVoice is a massively multilingual zero-shot TTS model supporting over 600 languages. Built on a novel diffusion language model architecture with a Qwen3 LLM backbone, it provides:

- **Voice Cloning**: Clone any voice from a reference audio + transcript
- **Voice Design**: Create custom voices via text instructions (e.g., "male, British accent")
- **Auto Voice**: Default voice when no voice specification is provided

## Architecture

### Files Created

```
Sources/MLXAudioTTS/Models/OmniVoice/
├── OmniVoice.swift                    # Main model implementation
├── OmniVoiceConfig.swift              # Configuration structs
└── OmniVoiceGenerateParameters.swift  # Generation parameters

Sources/MLXAudioTTS/
└── TTSModel.swift                     # Modified to register OmniVoice

Sources/Tools/mlx-audio-swift-tts/
└── App.swift                          # Extended CLI with OmniVoice options
```

### Model Architecture

Based on the `mlx-community/OmniVoice-bf16` HuggingFace model:

- **Backbone**: Qwen3 LLM (28 layers, 1024 hidden dims, 16 attention heads, 8 KV heads)
- **Audio Codebooks**: 8 codebooks with hierarchical weighting [8, 8, 6, 6, 4, 4, 2, 2]
- **Audio Vocab Size**: 1025 (1024 tokens + 1 mask)
- **Sample Rate**: 24 kHz
- **Max Context**: 40,960 tokens
- **Vocabulary**: 151,676 tokens

### Components

1. **OmniVoiceModel**: Main TTS model conforming to `SpeechGenerationModel` protocol
2. **OmniVoiceAudioTokenizer**: Handles audio encoding/decoding with:
   - DAC-based acoustic model for fine-grained audio tokens
   - Hubert-based semantic model for high-level audio features
3. **OmniVoiceGenerateParameters**: Diffusion-specific parameters

## Usage

### CLI Usage

The existing `mlx-audio-swift-tts` CLI now supports OmniVoice:

```bash
# Voice cloning
swift run mlx-audio-swift-tts --model mlx-community/OmniVoice-bf16 \
    --text "Hello, this is a text for text-to-speech." \
    --ref_audio ref.wav --ref_text "Reference transcript." --output out.wav

# Voice design
swift run mlx-audio-swift-tts --model mlx-community/OmniVoice-bf16 \
    --text "Hello, this is a text for text-to-speech." \
    --instruct "male, British accent" --output out.wav

# Auto voice
swift run mlx-audio-swift-tts --model mlx-community/OmniVoice-bf16 \
    --text "Hello, this is a text for text-to-speech." --output out.wav

# Custom generation parameters
swift run mlx-audio-swift-tts --model mlx-community/OmniVoice-bf16 \
    --text "Hello, this is a test." \
    --num_step 64 --guidance_scale 2.5 --speed 0.9 --output out.wav
```

### CLI Options

#### General Options
- `--text, -t`: Text to synthesize (required)
- `--voice, -v`: Voice ID
- `--model`: HuggingFace repo ID (default: Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit)
- `--output, -o`: Output WAV path (default: ./output.wav)
- `--ref_audio`: Path to reference audio for voice cloning
- `--ref_text`: Caption for reference audio
- `--language, -l`: Language code

#### OmniVoice-Specific Options
- `--instruct <string>`: Voice design instruction (e.g., "male, British accent")
- `--num_step <int>`: Number of diffusion steps (default: 32, range: 8-64)
- `--guidance_scale <float>`: Classifier-free guidance scale (default: 2.0, range: 1.0-5.0)
- `--speed <float>`: Speech speed factor (default: 1.0, range: 0.5-2.0)
- `--duration <float>`: Fixed output duration in seconds (optional)
- `--t_shift <float>`: Time shift for diffusion (default: 0.1)
- `--denoise <bool>`: Denoise output audio (default: true)
- `--postprocess_output <bool>`: Postprocess output audio (default: true)
- `--layer_penalty_factor <float>`: Layer penalty factor (default: 5.0)
- `--position_temperature <float>`: Position temperature (default: 5.0)
- `--class_temperature <float>`: Class temperature (default: 0.0)

### Programmatic Usage

```swift
import MLXAudioTTS
import MLXAudioCore

// Load the model
let model = try await TTS.loadModel(modelRepo: "mlx-community/OmniVoice-bf16")

// Configure OmniVoice-specific parameters (if needed)
if let omnivoice = model as? OmniVoiceModel {
    omnivoice.setGenerationConfig(
        numStep: 32,
        guidanceScale: 2.0,
        speed: 1.0,
        tShift: 0.1,
        denoise: true,
        postprocessOutput: true,
        layerPenaltyFactor: 5.0,
        positionTemperature: 5.0,
        classTemperature: 0.0
    )
}

// Voice cloning
let clonedAudio = try await model.generate(
    text: "Hello, this is a test.",
    voice: nil,
    refAudio: refAudioArray,
    refText: "Reference transcript",
    language: "English",
    generationParameters: GenerateParameters(maxTokens: 4096, temperature: 1.0, topP: 0.95)
)

// Voice design
let designedAudio = try await model.generate(
    text: "Hello, this is a test.",
    voice: "male, British accent",  // instruct serves as voice description
    refAudio: nil,
    refText: nil,
    language: "English",
    generationParameters: GenerateParameters(maxTokens: 4096, temperature: 1.0, topP: 0.95)
)

// Auto voice
let autoAudio = try await model.generate(
    text: "Hello, this is a test.",
    voice: nil,
    refAudio: nil,
    refText: nil,
    language: "English",
    generationParameters: GenerateParameters(maxTokens: 4096, temperature: 1.0, topP: 0.95)
)

// Save to WAV
try AudioUtils.writeWavFile(
    samples: clonedAudio.asArray(Float.self),
    sampleRate: model.sampleRate,
    fileURL: outputURL
)
```

## Generation Parameters

### OmniVoiceGenerateParameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `numStep` | 32 | 8-64 | Diffusion steps (higher = better quality, slower) |
| `guidanceScale` | 2.0 | 1.0-5.0 | How strongly to follow the prompt |
| `speed` | 1.0 | 0.5-2.0 | Speech speed multiplier |
| `duration` | nil | - | Fixed output duration (seconds) |
| `tShift` | 0.1 | 0.0-1.0 | Diffusion time shift |
| `denoise` | true | - | Apply denoising to output |
| `postprocessOutput` | true | - | Apply postprocessing to output |
| `layerPenaltyFactor` | 5.0 | 1.0-10.0 | Layer influence penalty |
| `positionTemperature` | 5.0 | 0.1-10.0 | Positional sampling randomness |
| `classTemperature` | 0.0 | 0.0-5.0 | Class sampling randomness |

### Presets

```swift
// Fast generation (lower quality, quicker)
let fastParams = OmniVoiceGenerateParameters.fast
// numStep: 16, guidanceScale: 1.5

// High quality generation (slower, better quality)
let hqParams = OmniVoiceGenerateParameters.highQuality
// numStep: 64, guidanceScale: 2.5
```

## Implementation Details

### Three Operating Modes

1. **Voice Cloning Mode**
   - Triggered when `refAudio` AND `refText` are provided
   - Encodes reference audio to extract voice characteristics
   - Uses the voice timbre from reference for synthesis

2. **Voice Design Mode**
   - Triggered when `voice` (instruct) parameter is provided without refAudio
   - Creates a voice based on the description (e.g., "female, deep voice, cheerful")
   - Language parameter influences voice characteristics

3. **Auto Voice Mode**
   - Triggered when neither voice nor refAudio is provided
   - Uses the model's default voice characteristics

### Diffusion Process

OmniVoice uses a diffusion-based generation approach:

1. **Input Preparation**: Text is tokenized and embedded using the Qwen3 tokenizer
2. **Noise Initialization**: Audio tokens start as random noise
3. **Iterative Denoising**: Over `numStep` iterations, the model predicts and removes noise
4. **Guidance**: Classifier-free guidance scale controls prompt adherence
5. **Token Sampling**: Audio codebooks are sampled with temperature control
6. **Audio Decoding**: Final audio tokens are decoded to waveform via the acoustic model

### Model Loading

The model is automatically loaded from HuggingFace:

```swift
let model = try await OmniVoiceModel.fromPretrained("mlx-community/OmniVoice-bf16")
```

This downloads and caches:
- `config.json` - Model architecture configuration
- `model.safetensors` - Model weights (1.23 GB, bfloat16)
- `tokenizer.json` + `tokenizer_config.json` - Text tokenizer
- `audio_tokenizer/` - Audio tokenizer components

## Comparison with Python CLI

This implementation mirrors the Python `omnivoice-infer` CLI functionality:

| Feature | Python CLI | Swift CLI |
|---------|-----------|-----------|
| Voice cloning | ✅ `--ref_audio` + `--ref_text` | ✅ `--ref_audio` + `--ref_text` |
| Voice design | ✅ `--instruct` | ✅ `--instruct` |
| Auto voice | ✅ text only | ✅ text only |
| Diffusion steps | ✅ `--num_step` | ✅ `--num_step` |
| Guidance scale | ✅ `--guidance_scale` | ✅ `--guidance_scale` |
| Speed control | ✅ `--speed` | ✅ `--speed` |
| Duration control | ✅ `--duration` | ✅ `--duration` |
| Time shift | ✅ `--t_shift` | ✅ `--t_shift` |
| Denoising | ✅ `--denoise` | ✅ `--denoise` |
| Postprocessing | ✅ `--postprocess_output` | ✅ `--postprocess_output` |
| Layer penalty | ✅ `--layer_penalty_factor` | ✅ `--layer_penalty_factor` |
| Position temp | ✅ `--position_temperature` | ✅ `--position_temperature` |
| Class temp | ✅ `--class_temperature` | ✅ `--class_temperature` |
| Language | ✅ `--language` | ✅ `--language` |
| Benchmark | ❌ | ✅ `--benchmark` |
| Streaming | ❌ | ✅ Built-in |

## Notes

- **Sample Rate**: OmniVoice outputs 24 kHz audio (higher than many TTS models)
- **Memory Usage**: The model requires significant memory (~2-3 GB for bf16 weights)
- **Generation Speed**: RTF ~0.025 (40x real-time) on Apple Silicon with optimized settings
- **Language Support**: 600+ languages supported via the multilingual training

## Future Enhancements

- [ ] Optimize diffusion loop for faster generation
- [ ] Add streaming audio output during generation
- [ ] Implement voice mixing (combine multiple reference voices)
- [ ] Add phoneme-level control for precise pronunciation
- [ ] Support for non-verbal expressions (laughter, sighs, etc.)

## References

- Original OmniVoice: https://huggingface.co/k2-fsa/OmniVoice
- MLX Version: https://huggingface.co/mlx-community/OmniVoice-bf16
- Python CLI: https://github.com/k2-fsa/OmniVoice
