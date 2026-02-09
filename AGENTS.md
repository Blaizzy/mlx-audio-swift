# AI Agent Instructions for mlx-audio-swift

## Project Overview

mlx-audio-swift is a modular Swift SDK for audio processing on Apple Silicon using the MLX framework. It provides text-to-speech (TTS), speech-to-text (STT), and audio codec implementations, all optimized for M-series chips. The upstream repo is `Blaizzy/mlx-audio-swift`; this fork is `intrusive-memory/mlx-audio-swift`.

## Build System

**CRITICAL**: Never use `swift build` or `swift test`. Always use `xcodebuild`.

```bash
# Build the package
xcodebuild build -scheme MLXAudio-Package -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO

# Build for testing (separate from running tests)
xcodebuild build-for-testing -scheme MLXAudio-Package -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO

# Run a specific test suite
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
  -only-testing:MLXAudioTests/VocosTests CODE_SIGNING_ALLOWED=NO

# Run tests without rebuilding
xcodebuild test-without-building -scheme MLXAudio-Package -destination 'platform=macOS' \
  -only-testing:MLXAudioTests/VocosTests CODE_SIGNING_ALLOWED=NO
```

- **Swift version**: 6.2+
- **Platforms**: macOS 14+, iOS 17+
- **CI runner**: `macos-26`

## Repository Structure

```
mlx-audio-swift/
├── Package.swift                    # SPM manifest (swift-tools-version: 6.2)
├── Sources/
│   ├── MLXAudioCore/                # Base types, protocols, utilities
│   │   ├── AudioUtils.swift         # WAV I/O, loadAudioArray, saveAudioArray
│   │   ├── ModelUtils.swift         # HuggingFace model download/cache
│   │   ├── DSP.swift                # Signal processing utilities
│   │   └── Generation/              # GenerateParameters, protocols, errors
│   ├── MLXAudioCodecs/              # Audio codec implementations
│   │   ├── Vocos/                   # Vocos vocoder
│   │   ├── Encodec/                 # Meta Encodec codec
│   │   ├── SNAC/                    # SNAC neural audio codec
│   │   ├── Mimi/                    # Mimi codec
│   │   └── DACVAE/                  # DAC VAE codec
│   ├── MLXAudioTTS/                 # Text-to-speech models
│   │   ├── TTSModelUtils.swift      # Model type resolution + loading
│   │   ├── Generation.swift         # SpeechGenerationModel protocol
│   │   └── Models/
│   │       ├── Soprano/             # Soprano TTS (80M params)
│   │       ├── Qwen3/               # VyvoTTS / Qwen3 TTS
│   │       ├── Qwen3TTS/            # Qwen3-TTS conditional generation (VoiceDesign)
│   │       ├── Llama/               # Orpheus / LlamaTTS (3B params)
│   │       ├── PocketTTS/           # Pocket TTS (small, multi-voice)
│   │       └── Marvis/              # Marvis TTS (250M params)
│   ├── MLXAudioSTT/                 # Speech-to-text models
│   │   └── Models/
│   │       ├── GLMASR/              # GLM-ASR model
│   │       └── Qwen3ASR/            # Qwen3 ASR + ForcedAligner
│   ├── MLXAudioSTS/                 # Speech-to-speech (placeholder)
│   ├── MLXAudioUI/                  # SwiftUI components (placeholder)
│   └── mlx-audio-swift-tts/         # CLI executable
├── Tests/
│   ├── MLXAudioCodecsTests.swift    # Codec unit + integration tests
│   ├── MLXAudioTTSTests.swift       # TTS integration tests
│   ├── MLXAudioSTTTests.swift       # STT unit + integration tests
│   └── media/                       # Test audio fixtures (WAV files)
├── Examples/
│   ├── VoicesApp/                   # SwiftUI TTS demo app
│   └── SimpleChat/                  # Chat-based TTS/STT example
└── .github/workflows/tests.yaml     # CI workflow
```

## Dependencies

| Package | Version | Products Used |
|---------|---------|---------------|
| mlx-swift | v0.30.3+ | MLX, MLXNN |
| mlx-swift-lm | v2.30.3+ | MLXLMCommon, MLXLLM |
| swift-transformers | v1.1.6+ | Transformers |
| swift-huggingface | v0.6.0+ | HuggingFace |

## Architecture Patterns

### Protocol-oriented model design

All TTS models conform to `SpeechGenerationModel` (defined in `Sources/MLXAudioTTS/Generation.swift`). New models must implement:
- `sampleRate: Int`
- `generate(text:voice:parameters:) async throws -> MLXArray`
- `generateStream(...) -> AsyncThrowingStream<AudioGeneration, Error>`

### Model loading

Models are loaded from HuggingFace via `fromPretrained()`. Internally this calls `ModelUtils.resolveOrDownloadModel()` which caches to `~/Library/Caches/mlx-audio/<namespace>_<repo>/`. The cache checks for `.safetensors`, `.json`, `.txt`, `.model` files before downloading.

### Concurrency

- All model loading and generation uses `async/await`
- Streaming uses `AsyncThrowingStream` with `AudioGeneration` enum cases: `.token`, `.audio`, `.info`
- Types are annotated `Sendable`; models use `@unchecked Sendable` where needed
- Use `@preconcurrency import MLX` to suppress concurrency warnings from MLX

### Neural network layers

MLX neural network modules use `@ModuleInfo` property wrappers for layer registration. All layer classes inherit from `Module` (MLXNN). Weight sanitization methods (e.g., `sanitize(weights:)`) handle PyTorch-to-MLX weight format conversion.

### Error handling

Custom errors use `AudioGenerationError` enum with `LocalizedError` conformance and associated values for context.

## Coding Conventions

- **Imports**: `@preconcurrency import MLX` and `import MLXNN` for neural network code
- **Organization**: `// MARK: -` comments to separate sections within files
- **Classes vs structs**: Neural network modules are `class` (inheriting `Module`); configs and data types are `struct`
- **Enum namespaces**: Utility classes use `enum` with static methods (e.g., `ModelUtils`, `TTSModelUtils`)
- **Documentation**: Doc comments on public API methods with `/// - Parameters:` and `/// - Returns:` sections
- **File headers**: Standard Xcode file header with name, target, author, date
- **No linter**: No SwiftLint or swift-format configuration exists
- **Print statements**: Used for user-facing progress feedback; CI flags them as warnings

## Test Organization

Tests use **Swift Testing** framework (`@Test`, `#expect`), not XCTest.

### Test suites that run without model downloads (safe for CI)

| Suite | File | What it tests |
|-------|------|---------------|
| `VocosTests` | MLXAudioCodecsTests.swift | Vocos vocoder components, layer shapes |
| `EncodecTests` | MLXAudioCodecsTests.swift | Encodec codec components, config, RVQ |
| `DACVAETests` | MLXAudioCodecsTests.swift | DACVAE codec components, encoder/decoder |
| `GLMASRModuleSetupTests` | MLXAudioSTTTests.swift | GLM-ASR config, layers, shapes |
| `Qwen3ASRModuleSetupTests` | MLXAudioSTTTests.swift | Qwen3 ASR config, layers, weight sanitization |
| `ForceAlignProcessorTests` | MLXAudioSTTTests.swift | Text tokenization, timestamp encoding |
| `ForcedAlignResultTests` | MLXAudioSTTTests.swift | Alignment result data structures |
| `Qwen3ASRHelperTests` | MLXAudioSTTTests.swift | Feature extraction length calculations |
| `SplitAudioIntoChunksTests` | MLXAudioSTTTests.swift | Audio chunking logic |

### Test suites that require model downloads (local only)

| Suite | File | Model required |
|-------|------|----------------|
| `SNACTests` | MLXAudioCodecsTests.swift | `mlx-community/snac_24khz` |
| `MimiTests` | MLXAudioCodecsTests.swift | `kyutai/moshiko-pytorch-bf16` |
| `Qwen3TTSTests` | MLXAudioTTSTests.swift | `mlx-community/VyvoTTS-EN-Beta-4bit` |
| `LlamaTTSTests` | MLXAudioTTSTests.swift | `mlx-community/orpheus-3b-0.1-ft-bf16` |
| `SopranoTTSTests` | MLXAudioTTSTests.swift | `mlx-community/Soprano-80M-bf16` |
| `PocketTTSTests` | MLXAudioTTSTests.swift | `mlx-community/pocket-tts` |
| `Qwen3TTSVoiceDesignTests` | MLXAudioTTSTests.swift | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` |
| `Qwen3ASRTests` | MLXAudioSTTTests.swift | `mlx-community/Qwen3-ASR-0.6B-4bit` |
| `GLMASRTests` | MLXAudioSTTTests.swift | `mlx-community/GLM-ASR-Nano-2512-4bit` |

### Test conventions

- **Struct naming**: `<Component>Tests` (e.g., `VocosTests`, `SopranoTTSTests`)
- **Function naming**: `@Test func testFeatureName()` or `@Test func featureNameBehavior()`
- **Test resources**: Access via `Bundle.module.url(forResource:withExtension:subdirectory:"media")`
- **Async tests**: Use `async throws` for anything involving model loading or generation
- **Streaming tests**: Iterate `AsyncThrowingStream`, count tokens, verify final audio shape
- **Output files**: Write to `FileManager.default.temporaryDirectory`

## Git Workflow

- **Branches**: `main` (protected), `development` (working branch)
- **PRs**: Always `development` → `main`
- **Branch protection on main**: Required status checks (`Code Quality`, `macOS Tests`), enforce admins, no force push, no deletions
- **Commit style**: Imperative mood, concise subject line (e.g., "Add Qwen3 ASR", "Fix weight loading in Vocos")
- **Remotes**: `origin` (intrusive-memory fork), `upstream` (Blaizzy/mlx-audio-swift)

## CI/CD

**Workflow**: `.github/workflows/tests.yaml`

| Job | Runs on | Purpose |
|-----|---------|---------|
| Code Quality | macos-26 | Flags TODOs, large files, print() in Sources/ |
| macOS Tests | macos-26 | Builds package, runs unit tests (no downloads) |
| Download Models | macos-26 | Caches HuggingFace models for integration tests |
| Model Tests | macos-26 | Runs model-dependent tests when cache is warm |

**Triggers**: `workflow_dispatch` + PRs to `main` (opened, synchronize, reopened)

**Model cache**: `~/Library/Caches/mlx-audio` cached with key `mlx-models-v1`. Prime via `workflow_dispatch`. Model tests skip when cache is cold.

## Adding a New TTS Model

1. Create a new directory under `Sources/MLXAudioTTS/Models/<ModelName>/`
2. Implement the model class conforming to `SpeechGenerationModel`
3. Add a `<ModelName>Config.swift` with `Codable` config struct
4. Implement `fromPretrained()` using `ModelUtils.resolveOrDownloadModel()`
5. Implement `sanitize(weights:)` for PyTorch-to-MLX weight conversion
6. Add the model type to `TTSModelUtils.swift` for routing
7. Add tests to `Tests/MLXAudioTTSTests.swift`
8. Add a `README.md` in the model directory

## Adding a New Audio Codec

1. Create a new directory under `Sources/MLXAudioCodecs/<CodecName>/`
2. Implement `encode()` and `decode()` methods
3. Add `fromPretrained()` for HuggingFace loading
4. Add tests to `Tests/MLXAudioCodecsTests.swift`

## Common Pitfalls

- **Never use `swift build`/`swift test`** — always `xcodebuild`
- **PyTorch weight conversion**: Conv2d weights need transposition from `(O,I,H,W)` to `(O,H,W,I)` in `sanitize()`
- **Model cache path**: `~/Library/Caches/mlx-audio/<namespace>_<repo>` — replace `/` with `_` in repo ID
- **Bundle resources in tests**: Use `.copy("media")` in Package.swift, access via `Bundle.module`
- **Concurrency warnings**: Use `@preconcurrency import MLX` and `@unchecked Sendable` on Module subclasses
- **CI test selection**: Only add tests to CI that work without model downloads. Model-dependent tests go in the `model-tests` job
