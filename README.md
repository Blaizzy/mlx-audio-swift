# MLX Audio Swift

A modular Swift SDK for audio processing with MLX on Apple Silicon

![Platform](https://img.shields.io/badge/platform-macOS%2014%2B%20%7C%20iOS%2017%2B-lightgrey)
![Swift](https://img.shields.io/badge/Swift-5.9%2B-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

## Architecture

MLXAudio follows a modular design allowing you to import only what you need:

- **MLXAudioCore**: Base types, protocols, and utilities
- **MLXAudioCodecs**: Audio codec implementations (SNAC, Vocos, Mimi)
- **MLXAudioTTS**: Text-to-Speech models (Soprano, VyvoTTS, Orpheus, Marvis TTS, Pocket TTS)
- **MLXAudioSTT**: Speech-to-Text models (GLMASR)
- **MLXAudioVAD**: Voice Activity Detection & Speaker Diarization (Sortformer)
- **MLXAudioSTS**: Speech-to-Speech (future)
- **MLXAudioUI**: SwiftUI components for audio interfaces

## Installation

Add MLXAudio to your project using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/Blaizzy/mlx-audio-swift.git", branch: "main")
]

// Import only what you need
.product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
.product(name: "MLXAudioCore", package: "mlx-audio-swift")
```

## Quick Start

### Text-to-Speech

```swift
import MLXAudioTTS
import MLXAudioCore

// Load a TTS model from HuggingFace
let model = try await SopranoModel.fromPretrained("mlx-community/Soprano-80M-bf16")

// Generate audio
let audio = try await model.generate(
    text: "Hello from MLX Audio Swift!",
    parameters: GenerateParameters(
        maxTokens: 200,
        temperature: 0.7,
        topP: 0.95
    )
)

// Save to file
try saveAudioArray(audio, sampleRate: Double(model.sampleRate), to: outputURL)
```

### Speech-to-Text

```swift
import MLXAudioSTT
import MLXAudioCore

// Load audio file
let (sampleRate, audioData) = try loadAudioArray(from: audioURL)

// Load STT model
let model = try await GLMASRModel.fromPretrained("mlx-community/GLM-ASR-Nano-2512-4bit")

// Transcribe
let output = model.generate(audio: audioData)
print(output.text)
```

### Speaker Diarization

```swift
import MLXAudioVAD
import MLXAudioCore

// Load audio file
let (sampleRate, audioData) = try loadAudioArray(from: audioURL)

// Load diarization model
let model = try await SortformerModel.fromPretrained(
    "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
)

// Detect who is speaking when
let output = try await model.generate(audio: audioData, threshold: 0.5)
for segment in output.segments {
    print("Speaker \(segment.speaker): \(segment.start)s - \(segment.end)s")
}
```

### Streaming Generation

```swift
for try await event in model.generateStream(text: text, parameters: parameters) {
    switch event {
    case .token(let token):
        print("Generated token: \(token)")
    case .audio(let audio):
        print("Final audio shape: \(audio.shape)")
    case .info(let info):
        print(info.summary)
    }
}
```

## Supported Models

### TTS Models

| Model | Model README | HuggingFace Repo |
|-------|--------------|------------------|
| Soprano | [Soprano README](Sources/MLXAudioTTS/Models/Soprano/README.md) | [mlx-community/Soprano-80M-bf16](https://huggingface.co/mlx-community/Soprano-80M-bf16) |
| VyvoTTS | [VyvoTTS README](Sources/MLXAudioTTS/Models/Qwen3/README.md) | [mlx-community/VyvoTTS-EN-Beta-4bit](https://huggingface.co/mlx-community/VyvoTTS-EN-Beta-4bit) |
| Orpheus | [Orpheus README](Sources/MLXAudioTTS/Models/Llama/README.md) | [mlx-community/orpheus-3b-0.1-ft-bf16](https://huggingface.co/mlx-community/orpheus-3b-0.1-ft-bf16) |
| Marvis TTS | [Marvis TTS README](Sources/MLXAudioTTS/Models/Marvis/README.md) | [Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit](https://huggingface.co/Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit) |
| Pocket TTS | [Pocket TTS README](Sources/MLXAudioTTS/Models/PocketTTS/README.md) | [mlx-community/pocket-tts](https://huggingface.co/mlx-community/pocket-tts) |

### STT Models

| Model | Model README | HuggingFace Repo |
|-------|--------------|------------------|
| GLMASR | [GLMASR README](Sources/MLXAudioSTT/Models/GLMASR/README.md) | [mlx-community/GLM-ASR-Nano-2512-4bit](https://huggingface.co/mlx-community/GLM-ASR-Nano-2512-4bit) |

### VAD / Speaker Diarization Models

| Model | Model README | HuggingFace Repo |
|-------|--------------|------------------|
| Sortformer | [Sortformer README](Sources/MLXAudioVAD/Models/Sortformer/README.md) | [mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16](https://huggingface.co/mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16) |

## Features

- **Modular architecture** for minimal app size - import only what you need
- **Automatic model downloading** from HuggingFace Hub
- **Native async/await support** for seamless Swift integration
- **Streaming audio generation** for real-time TTS
- **Type-safe Swift API** with comprehensive error handling
- **Optimized for Apple Silicon** with MLX framework

## Advanced Usage

### Custom Generation Parameters

```swift
let parameters = GenerateParameters(
    maxTokens: 1200,
    temperature: 0.7,
    topP: 0.95,
    repetitionPenalty: 1.5,
    repetitionContextSize: 30
)

let audio = try await model.generate(text: "Your text here", parameters: parameters)
```

### Audio Codec Usage

```swift
import MLXAudioCodecs

// Load SNAC codec
let snac = try await SNAC.fromPretrained("mlx-community/snac_24khz")

// Encode audio to tokens
let tokens = try snac.encode(audio)

// Decode tokens back to audio
let reconstructed = try snac.decode(tokens)
```

### Voice Selection for Multi-Voice Models

```swift
// For models supporting multiple voices (like LlamaTTS/Orpheus)
let audio = try await model.generate(
    text: "Hello!",
    voice: "tara",  // Options: tara, leah, jess, leo, dan, mia, zac, zoe
    parameters: parameters
)
```

## Requirements

- **macOS 14+** or **iOS 17+**
- **Apple Silicon** (M1 or later) recommended for optimal performance
- **Xcode 15+**
- **Swift 5.9+**

## Examples

Check out the [Examples/VoicesApp](Examples/VoicesApp) directory for a complete SwiftUI application demonstrating:
- Loading and running TTS models
- Playing generated audio
- UI components for model interaction

Additional usage examples can be found in the test files.

## Credits

- Built on [MLX Swift](https://github.com/ml-explore/mlx-swift)
- Uses [swift-transformers](https://github.com/huggingface/swift-transformers)
- Inspired by [MLX Audio (Python)](https://github.com/Blaizzy/mlx-audio)

## License

MIT License - see [LICENSE](LICENSE) file for details.
