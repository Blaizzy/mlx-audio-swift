# LFM2.5-Audio for MLX Swift

MLX Swift implementation of [LiquidAI's LFM2.5-Audio](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B), a multimodal foundation model for audio understanding and generation.

## Features

- **Text-to-Speech (TTS)**: Generate natural speech from text
- **Speech-to-Text (STT)**: Transcribe audio to text
- **Speech-to-Speech (STS)**: Voice conversations with audio input and output
- **Text-to-Text (T2T)**: Standard text chat
- **Interleaved Generation**: Mixed text and audio responses in a single turn
- **Streaming**: Real-time token-by-token generation via `AsyncThrowingStream`

## Quick Start

### Text-to-Speech (TTS)

```swift
import MLXAudioSTS
import MLX

// Load model (downloads from HuggingFace Hub)
let model = try await LFM2AudioModel.fromPretrained("mlx-community/LFM2.5-Audio-1.5B-6bit")
let processor = model.processor!

// Build chat prompt
let chat = ChatState(processor: processor)
chat.newTurn(role: "system")
chat.addText("Perform TTS. Use a UK male voice.")
chat.endTurn()
chat.newTurn(role: "user")
chat.addText("Hello, welcome to MLX Audio Swift!")
chat.endTurn()
chat.newTurn(role: "assistant")
chat.addAudioStartToken()

// Generate audio sequentially
let genConfig = LFMGenerationConfig(
    maxNewTokens: 2048,
    audioTemperature: 0.8,
    audioTopK: 4
)

var audioCodes: [MLXArray] = []
for try await (token, modality) in model.generateSequential(
    textTokens: chat.getTextTokens(),
    audioFeatures: chat.getAudioFeatures(),
    modalities: chat.getModalities(),
    config: genConfig
) {
    eval(token)
    if modality == .audioOut {
        if token[0].item(Int.self) == lfmAudioEOSToken { break }
        audioCodes.append(token)
    }
}

// Decode audio with the detokenizer
let stacked = MLX.stacked(audioCodes, axis: 0)
let codesInput = stacked.transposed(1, 0).expandedDimensions(axis: 0)
let detokenizer = try LFM2AudioDetokenizer.fromPretrained(modelPath: model.modelDirectory!)
let waveform = detokenizer(codesInput) // 24kHz sample rate
```

### Speech-to-Text (STT)

```swift
import MLXAudioSTS
import MLX

let model = try await LFM2AudioModel.fromPretrained("mlx-community/LFM2.5-Audio-1.5B-6bit")
let processor = model.processor!

// Load audio as MLXArray (16kHz for input)
let audioData: MLXArray = ...

let chat = ChatState(processor: processor)
chat.newTurn(role: "system")
chat.addText("You are a helpful assistant that transcribes audio.")
chat.endTurn()
chat.newTurn(role: "user")
chat.addAudio(audioData, sampleRate: 16000)
chat.addText("Transcribe the audio.")
chat.endTurn()
chat.newTurn(role: "assistant")

var textTokens: [Int] = []
for try await (token, modality) in model.generateInterleaved(
    textTokens: chat.getTextTokens(),
    audioFeatures: chat.getAudioFeatures(),
    modalities: chat.getModalities()
) {
    eval(token)
    if modality == .text {
        textTokens.append(token.item(Int.self))
    }
}

let transcription = processor.decodeText(textTokens)
```

### Speech-to-Speech (STS)

```swift
import MLXAudioSTS
import MLX

let model = try await LFM2AudioModel.fromPretrained("mlx-community/LFM2.5-Audio-1.5B-6bit")
let processor = model.processor!

let audioData: MLXArray = ... // 16kHz input audio

let chat = ChatState(processor: processor)
chat.newTurn(role: "system")
chat.addText("Respond with interleaved text and speech audio. Use a UK male voice.")
chat.endTurn()
chat.newTurn(role: "user")
chat.addAudio(audioData, sampleRate: 16000)
chat.endTurn()
chat.newTurn(role: "assistant")

let genConfig = LFMGenerationConfig(maxNewTokens: 2048)

var textTokens: [Int] = []
var audioCodes: [MLXArray] = []

for try await (token, modality) in model.generateInterleaved(
    textTokens: chat.getTextTokens(),
    audioFeatures: chat.getAudioFeatures(),
    modalities: chat.getModalities(),
    config: genConfig
) {
    eval(token)
    if modality == .text {
        textTokens.append(token.item(Int.self))
    } else if modality == .audioOut {
        if token[0].item(Int.self) != lfmAudioEOSToken {
            audioCodes.append(token)
        }
    }
}

let text = processor.decodeText(textTokens)

// Decode audio
let stacked = MLX.stacked(audioCodes, axis: 0)
let codesInput = stacked.transposed(1, 0).expandedDimensions(axis: 0)
let detokenizer = try LFM2AudioDetokenizer.fromPretrained(modelPath: model.modelDirectory!)
let waveform = detokenizer(codesInput)
```

## Generation Modes

### Interleaved (`generateInterleaved`)

Produces alternating text and audio tokens. The model alternates between generating `interleavedNText` (default 6) text tokens and `interleavedNAudio` (default 12) audio tokens per cycle. Best for STT, STS, and T2T.

Each audio token is a frame of shape `(8,)` containing all 8 codebook values.

### Sequential (`generateSequential`)

Generates all text first, then all audio. Best for TTS where the model transitions to audio mode after emitting the audio start token (128).

Requires calling `chat.addAudioStartToken()` before generation to trigger audio output.

## Generation Configuration

```swift
let config = LFMGenerationConfig(
    maxNewTokens: 2048,      // Maximum tokens to generate
    temperature: 0.7,         // Text sampling temperature
    topK: 50,                 // Text top-K sampling
    topP: 1.0,                // Text nucleus sampling
    audioTemperature: 0.8,    // Audio sampling temperature
    audioTopK: 4              // Audio top-K sampling
)
```

## Audio Decoding

LFM2.5-Audio generates 8-codebook audio tokens that must be decoded to waveforms using the neural detokenizer (ISTFT-based vocoder):

```swift
// Stack audio frames: list of (8,) -> (T, 8) -> (8, T) -> (1, 8, T)
let stacked = MLX.stacked(audioCodes, axis: 0)
let codesInput = stacked.transposed(1, 0).expandedDimensions(axis: 0)

// Decode using detokenizer
let detokenizer = try LFM2AudioDetokenizer.fromPretrained(modelPath: model.modelDirectory!)
let waveform = detokenizer(codesInput)  // (1, T_audio) at 24kHz
```

## CLI Tool

The `mlx-audio-swift-sts` CLI tool provides a command-line interface:

```bash
# Text-to-Speech
swift run mlx-audio-swift-sts \
    --model mlx-community/LFM2.5-Audio-1.5B-6bit \
    --mode tts \
    --text "Hello, welcome to MLX Audio!" \
    -o output.wav

# Speech-to-Text
swift run mlx-audio-swift-sts \
    --model mlx-community/LFM2.5-Audio-1.5B-6bit \
    --mode stt \
    --audio input.wav \
    --stream

# Speech-to-Speech
swift run mlx-audio-swift-sts \
    --model mlx-community/LFM2.5-Audio-1.5B-6bit \
    --mode sts \
    --audio input.wav \
    --stream \
    -o response.wav

# Text-to-Text
swift run mlx-audio-swift-sts \
    --model mlx-community/LFM2.5-Audio-1.5B-6bit \
    --mode t2t \
    --text "What is machine learning?" \
    --stream
```

### CLI Options

| Option | Description | Default |
|---|---|---|
| `--model <repo>` | HuggingFace model repo | - |
| `--mode <t2t\|tts\|stt\|sts>` | Generation mode | `sts` |
| `-t, --text <string>` | Input text | - |
| `-i, --audio <path>` | Input audio file | - |
| `--system <string>` | System prompt | Per-mode default |
| `--max-new-tokens <int>` | Max tokens | `512` |
| `--temperature <float>` | Text temperature | `0.7` |
| `--top-k <int>` | Text top-K | `50` |
| `--audio-temperature <float>` | Audio temperature | `0.8` |
| `--audio-top-k <int>` | Audio top-K | `4` |
| `--stream` | Stream text output | `false` |
| `-o, --output-target <path>` | Audio output path | `lfm_output.wav` |
| `--output-text <path>` | Text output path | - |

## Model Architecture

LFM2.5-Audio consists of:

- **Audio Encoder**: Conformer-based encoder (17 layers, 512d) for processing 16kHz input audio
- **Audio Adapter**: MLP projecting encoder output to backbone dimensions
- **LFM Backbone**: 1.5B parameter Liquid Foundation Model (16 layers, mix of conv and attention) for multimodal reasoning
- **Audio Head**: Depthformer (6 layers) for generating 8-codebook audio tokens
- **Detokenizer**: ISTFT-based neural vocoder for reconstructing 24kHz waveforms from audio codes

## API Reference

### LFM2AudioModel

```swift
public class LFM2AudioModel: Module, STSModel {
    /// Load pretrained model from HuggingFace Hub
    public static func fromPretrained(_ modelNameOrPath: String) async throws -> LFM2AudioModel

    /// Generate interleaved text and audio tokens
    /// Yields (token, modality) tuples:
    ///   - TEXT: token is scalar Int
    ///   - AUDIO_OUT: token is (8,) array with codebook values
    public func generateInterleaved(
        textTokens: MLXArray?, audioFeatures: MLXArray?,
        audioCodes: MLXArray?, modalities: MLXArray?,
        config: LFMGenerationConfig
    ) -> AsyncThrowingStream<(MLXArray, LFMModality), Error>

    /// Generate text then audio sequentially (best for TTS)
    public func generateSequential(
        textTokens: MLXArray?, audioFeatures: MLXArray?,
        audioCodes: MLXArray?, modalities: MLXArray?,
        config: LFMGenerationConfig
    ) -> AsyncThrowingStream<(MLXArray, LFMModality), Error>

    /// Output sample rate (24kHz)
    public var sampleRate: Int { get }
}
```

### ChatState

```swift
public class ChatState {
    public init(processor: LFM2AudioProcessor, addBos: Bool = true)

    public func newTurn(role: String)        // Start turn: "system", "user", or "assistant"
    public func endTurn()                     // End current turn
    public func addText(_ text: String)       // Add text to current turn
    public func addAudio(_ audio: MLXArray, sampleRate: Int = 16000) // Add audio input
    public func addAudioStartToken()          // Required before TTS sequential generation

    public func getTextTokens() -> MLXArray   // (1, T) Int32
    public func getAudioFeatures() -> MLXArray? // (1, frames, features)
    public func getModalities() -> MLXArray   // (1, T) Int32
}
```

### LFM2AudioProcessor

```swift
public class LFM2AudioProcessor {
    public func tokenize(_ text: String) -> [Int]
    public func decodeText(_ tokens: [Int]) -> String
    public func preprocessAudio(_ audio: MLXArray, sampleRate: Int) -> MLXArray
}
```

### LFMModality

```swift
public enum LFMModality: Int {
    case text = 1
    case audioIn = 2
    case audioOut = 3
}
```

## Supported Models

| Model | Size | Quantization |
|---|---|---|
| `mlx-community/LFM2.5-Audio-1.5B-bf16` | ~3 GB |  bfloat16 |
| `mlx-community/LFM2.5-Audio-1.5B-8bit` | ~1.7 GB | 8-bit |
| `mlx-community/LFM2.5-Audio-1.5B-6bit` | ~1.3 GB | 6-bit |
| `mlx-community/LFM2.5-Audio-1.5B-5bit` | ~1.1 GB | 5-bit |
| `mlx-community/LFM2.5-Audio-1.5B-4bit` | ~0.9 GB | 4-bit |


## License

This implementation follows the license terms of the original LFM2.5-Audio model.
See [LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) for details.

## Acknowledgements

- [LiquidAI](https://liquid.ai/) for the LFM2.5-Audio model
- [MLX](https://github.com/ml-explore/mlx) team for the framework
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) for the Python reference implementation
