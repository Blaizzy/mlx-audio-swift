# Adding a New Model

## Module layout

Place the model under the appropriate Swift module:

```
Sources/
├── MLXAudioTTS/Models/<ModelName>/   # Text-to-speech
├── MLXAudioSTT/Models/<ModelName>/   # Speech-to-text
├── MLXAudioSTS/Models/<ModelName>/   # Speech-to-speech / enhancement
└── MLXAudioCodecs/Models/<ModelName>/ # Standalone audio codecs
```

Typical file structure for a model:

```
Sources/MLXAudioTTS/Models/MyModel/
├── MyModel.swift          # Main class — protocol conformance + generate()
├── MyModelConfig.swift    # Configuration struct (Codable + Sendable)
├── MyModelDecoder.swift   # Decoder component (if needed)
└── MyModelUtils.swift     # Utilities (if needed)
```

**Naming conventions:**
- Main class: `MyModelModel` (e.g., `SopranoModel`, `Qwen3ASRModel`)
- Config struct: `MyModelConfiguration` or `MyModelConfig`
- Supporting types follow the same `MyModel` prefix

## Protocols

Implement the protocol that matches your model's task:

**TTS — `SpeechGenerationModel`:**
```swift
public protocol SpeechGenerationModel {
    var sampleRate: Int { get }
    var defaultGenerationParameters: GenerationParameters { get }

    func generate(text: String, parameters: GenerationParameters) async throws -> MLXArray
    func generateStream(text: String, parameters: GenerationParameters)
        -> AsyncThrowingStream<GenerationToken, Error>
}
```

**STT — `STTGenerationModel`:**
```swift
public protocol STTGenerationModel {
    func generate(audio: MLXArray, language: String?) async throws -> STTOutput
    func generateStream(audio: MLXArray, language: String?)
        -> AsyncThrowingStream<STTGeneration, Error>
}
```

**Codec — `AudioCodecModel`:**
```swift
public protocol AudioCodecModel: AudioDecoderModel {
    func encodeAudio(audio: MLXArray) async throws -> EncodedAudio
}

public protocol AudioDecoderModel {
    var codecSampleRate: Int? { get }
    func decodeAudio(input: DecoderInput) async throws -> MLXArray
}
```

## Configuration struct

```swift
public struct MyModelConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numLayers: Int
    public let sampleRate: Int

    // Map Swift camelCase ↔ JSON snake_case
    enum CodingKeys: String, CodingKey {
        case hiddenSize  = "hidden_size"
        case numLayers   = "num_layers"
        case sampleRate  = "sample_rate"
    }
}
```

All config structs must be `Codable` (for JSON loading) and `Sendable` (for
concurrency safety). Use `CodingKeys` to map JSON snake_case to Swift camelCase.

## Model class

```swift
import MLX
import MLXAudioCore

public class MyModelModel: SpeechGenerationModel {
    public let sampleRate: Int
    public var defaultGenerationParameters = GenerationParameters()

    private let config: MyModelConfig
    // ... layers

    public init(config: MyModelConfig) {
        self.config = config
        self.sampleRate = config.sampleRate
        // initialize layers …
    }

    public func generate(
        text: String,
        parameters: GenerationParameters = .init()
    ) async throws -> MLXArray {
        // inference …
        return audio  // MLXArray of shape [samples]
    }

    public func generateStream(
        text: String,
        parameters: GenerationParameters = .init()
    ) -> AsyncThrowingStream<GenerationToken, Error> {
        AsyncThrowingStream { continuation in
            Task {
                // yield tokens progressively …
                continuation.finish()
            }
        }
    }

    /// Load weights from a HuggingFace repository.
    public static func fromPretrained(
        _ repository: String,
        token: String? = nil
    ) async throws -> MyModelModel {
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repository, token: token
        )
        let config = try JSONDecoder().decode(
            MyModelConfig.self,
            from: Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        )
        let model = MyModelModel(config: config)
        let weights = try ModelUtils.loadWeights(from: modelDir)
        let sanitized = model.sanitize(weights)
        try model.update(parameters: sanitized)
        return model
    }

    /// Rename / reshape PyTorch keys to match Swift attribute paths.
    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            var k = key
            if k.hasPrefix("model.") { k = String(k.dropFirst(6)) }
            // Conv2d: PyTorch (out,in,kH,kW) → MLX (out,kH,kW,in)
            // if isConvWeight { result[k] = value.transposed(0,2,3,1); continue }
            result[k] = value
        }
        return result
    }
}
```

## Factory registration

**This step is mandatory** — unlike the Python library, Swift does not
auto-discover models. Add your model type to the factory switch in
`Sources/MLXAudioTTS/TTSModel.swift` (or the STT/codec equivalent):

```swift
// In TTS.loadModel(from:withToken:modelType:)
switch resolvedType {
case .myModel:
    return try await MyModelModel.fromPretrained(repository, token: token)
// … existing cases
}
```

Also add the case to the `TTSModelType` enum and the type-resolution logic that
maps a HuggingFace `model_type` string to your enum case.

## Publishing weights to mlx-community

Model weights must be published to
[mlx-community](https://huggingface.co/mlx-community) on HuggingFace, not
bundled in this repository.

### Naming convention

```
mlx-community/<ModelName>[-<Variant>]-<ParameterCount>-<Dtype>
```

| part | description | examples |
|---|---|---|
| `ModelName` | base model name, preserve original casing | `Soprano`, `Qwen3-TTS`, `Kokoro` |
| `Variant` | optional variant tag | `Base`, `VoiceDesign`, `Realtime` |
| `ParameterCount` | size indicator | `80M`, `0.6B`, `1.7B` |
| `Dtype` | precision / quantization level | `bf16`, `fp16`, `8bit`, `4bit` |

Real examples from mlx-community:

```
mlx-community/Soprano-80M-bf16
mlx-community/Kokoro-82M-bf16
mlx-community/Kokoro-82M-4bit
mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
mlx-community/whisper-large-v3-turbo-asr-fp16
mlx-community/parakeet-tdt-0.6b-v3
```

**Notes:**
- Prefer `bf16` as the primary upload; add quantized variants for large models.
- Include the parameter count when the model family has multiple sizes.
- Link the mlx-community repo in your PR description.

## Tests

Add tests to the relevant test file (`Tests/MLXAudioTTSTests.swift`, etc.):

```swift
func testMyModelGeneration() async throws {
    let model = try await MyModelModel.fromPretrained(
        "mlx-community/MyModel-1B-bf16"
    )
    let audio = try await model.generate(text: "Hello world")
    XCTAssertGreaterThan(audio.shape[0], 0)
}

func testMyModelStream() async throws {
    let model = try await MyModelModel.fromPretrained(
        "mlx-community/MyModel-1B-bf16"
    )
    var count = 0
    for try await _ in model.generateStream(text: "Hello") { count += 1 }
    XCTAssertGreaterThan(count, 0)
}
```

## PR checklist

- [ ] Model class conforms to the correct protocol (`SpeechGenerationModel`, `STTGenerationModel`, or `AudioCodecModel`)
- [ ] Config struct is `Codable` + `Sendable` with `CodingKeys` for snake_case mapping
- [ ] `fromPretrained()` loads `config.json` and all `.safetensors` from HF
- [ ] `sanitize()` covers all key renames and Conv weight transposes
- [ ] Model type added to factory switch statement
- [ ] Weights published to `mlx-community` following naming convention
- [ ] Tests cover `generate()` and `generateStream()`
- [ ] Model listed in `README.md` model table
