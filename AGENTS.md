# AI Agent Instructions for mlx-audio-swift

## Project Overview

mlx-audio-swift is a modular Swift SDK for audio processing on Apple Silicon using the MLX framework. It provides text-to-speech (TTS), speech-to-text (STT), and audio codec implementations, all optimized for M-series chips. The upstream repo is `Blaizzy/mlx-audio-swift`; this fork is `intrusive-memory/mlx-audio-swift`.

## Build System

**CRITICAL**: Never use `swift build` or `swift test`. Always use `xcodebuild`.

```bash
# Build the package
xcodebuild build -scheme MLXAudio-Package -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO

# Run a specific test suite
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
  -only-testing:MLXAudioTests/VocosTests CODE_SIGNING_ALLOWED=NO

# Run all CI-safe tests (no model downloads)
xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
  -only-testing:MLXAudioTests/VocosTests \
  -only-testing:MLXAudioTests/EncodecTests \
  -only-testing:MLXAudioTests/DACVAETests \
  -only-testing:MLXAudioTests/GLMASRModuleSetupTests \
  -only-testing:MLXAudioTests/Qwen3ASRModuleSetupTests \
  -only-testing:MLXAudioTests/ForceAlignProcessorTests \
  -only-testing:MLXAudioTests/ForcedAlignResultTests \
  -only-testing:MLXAudioTests/Qwen3ASRHelperTests \
  -only-testing:MLXAudioTests/SplitAudioIntoChunksTests \
  -only-testing:MLXAudioTests/Qwen3TTSSpeechTokenizerTests \
  -only-testing:MLXAudioTests/Qwen3TTSSpeechTokenizerEncodeTests \
  -only-testing:MLXAudioTests/Qwen3TTSSpeechTokenizerWeightTests \
  -only-testing:MLXAudioTests/Qwen3TTSLanguageTests \
  -only-testing:MLXAudioTests/Qwen3TTSConfigTests \
  -only-testing:MLXAudioTests/Qwen3TTSRoutingTests \
  -only-testing:MLXAudioTests/Qwen3TTSPrepareBaseInputsTests \
  -only-testing:MLXAudioTests/Qwen3TTSGenerateCustomVoiceTests \
  -only-testing:MLXAudioTests/Qwen3TTSSpeakerEncoderTests \
  -only-testing:MLXAudioTests/Qwen3TTSSpeakerEncoderWeightTests \
  -only-testing:MLXAudioTests/Qwen3TTSSpeakerEmbeddingTests \
  -only-testing:MLXAudioTests/Qwen3TTSPrepareICLInputsTests \
  -only-testing:MLXAudioTests/Qwen3TTSGenerateICLTests \
  -only-testing:MLXAudioTests/Qwen3TTSSpeakerEncoderSmokeTests \
  CODE_SIGNING_ALLOWED=NO
```

- **Swift version**: 6.2+
- **Platforms**: macOS 26+, iOS 26+
- **CI runner**: `macos-26`

## Repository Structure

```
mlx-audio-swift/
├── Package.swift                    # SPM manifest (swift-tools-version: 6.2)
├── AGENTS.md                        # This file — agent instructions
├── CLAUDE.md                        # Claude Code specific instructions
├── GEMINI.md                        # Gemini specific instructions
├── Sources/
│   ├── MLXAudioCore/                # Base types, protocols, utilities
│   │   ├── AudioUtils.swift         # WAV I/O, loadAudioArray, saveAudioArray
│   │   ├── AudioPlayerManager.swift # Audio playback management
│   │   ├── AudioSessionManager.swift# Audio session configuration
│   │   ├── ConvWeighted.swift       # Weighted convolution helper
│   │   ├── DSP.swift                # Signal processing (mel spectrogram, STFT, etc.)
│   │   ├── MLX+Extensions.swift     # MLX array convenience extensions
│   │   ├── ModelUtils.swift         # HuggingFace model download/cache
│   │   └── Generation/
│   │       └── GenerationTypes.swift # AudioGeneration, AudioGenerationError, GenerateParameters
│   ├── MLXAudioCodecs/              # Audio codec implementations
│   │   ├── Vocos/                   # Vocos vocoder (2 files)
│   │   ├── Encodec/                 # Meta Encodec codec (4 files)
│   │   ├── SNAC/                    # SNAC neural audio codec (5 files)
│   │   ├── Mimi/                    # Mimi codec (5 files) — components reused by Qwen3TTS encoder
│   │   └── DACVAE/                  # DAC VAE codec (6 files)
│   ├── MLXAudioTTS/                 # Text-to-speech models
│   │   ├── TTSModelUtils.swift      # Model type resolution + loading dispatch
│   │   ├── Generation.swift         # SpeechGenerationModel protocol
│   │   └── Models/
│   │       ├── Soprano/             # Soprano TTS (80M params, 4 files)
│   │       ├── Qwen3/               # VyvoTTS / Qwen3 TTS (2 files)
│   │       ├── Qwen3TTS/            # Qwen3-TTS conditional generation (8 files, 4747 lines)
│   │       ├── Llama/               # Orpheus / LlamaTTS (3B params, 2 files)
│   │       ├── PocketTTS/           # Pocket TTS (small, multi-voice, 9 files)
│   │       └── Marvis/              # Marvis TTS (250M params, 3 files)
│   ├── MLXAudioSTT/                 # Speech-to-text models
│   │   └── Models/
│   │       ├── GLMASR/              # GLM-ASR model (4 files)
│   │       └── Qwen3ASR/            # Qwen3 ASR + ForcedAligner (3 files)
│   ├── MLXAudioSTS/                 # Speech-to-speech (placeholder)
│   ├── MLXAudioUI/                  # SwiftUI components (placeholder)
│   └── mlx-audio-swift-tts/         # CLI executable
├── Tests/
│   ├── MLXAudioCodecsTests.swift    # Codec unit + integration tests
│   ├── MLXAudioTTSTests.swift       # TTS unit + integration tests (~2700 lines)
│   ├── MLXAudioSTTTests.swift       # STT unit + integration tests
│   └── media/                       # Test audio fixtures (WAV files)
├── Examples/
│   ├── VoicesApp/                   # SwiftUI TTS demo app
│   └── SimpleChat/                  # Chat-based TTS/STT example
├── docs/
│   └── EXECUTION_PLAN.md            # Completed 22-task execution plan for Qwen3-TTS
└── .github/workflows/tests.yaml     # CI workflow
```

## Dependencies

| Package | Version | Products Used |
|---------|---------|---------------|
| mlx-swift | v0.30.3+ | MLX, MLXNN |
| mlx-swift-lm | v2.30.3+ | MLXLMCommon, MLXLLM |
| swift-transformers | v1.1.6+ | Transformers |
| swift-huggingface | v0.6.0+ | HuggingFace |
| SwiftAcervo | v0.1.0+ | SwiftAcervo |

## Architecture Patterns

### SpeechGenerationModel protocol

All TTS models conform to `SpeechGenerationModel` (defined in `Sources/MLXAudioTTS/Generation.swift`):

```swift
public protocol SpeechGenerationModel: AnyObject {
    var sampleRate: Int { get }
    func generate(text:voice:refAudio:refText:language:generationParameters:) async throws -> MLXArray
    func generateStream(text:voice:refAudio:refText:language:generationParameters:) -> AsyncThrowingStream<AudioGeneration, Error>
}
```

### AudioGeneration types (`Sources/MLXAudioCore/Generation/GenerationTypes.swift`)

```swift
public enum AudioGeneration: Sendable {
    case token(Int)                    // Generated token ID
    case info(AudioGenerationInfo)     // Generation statistics
    case audio(MLXArray)               // Final generated audio
}

public enum AudioGenerationError: Error, LocalizedError {
    case modelNotInitialized(String)
    case generationFailed(String)
    case invalidInput(String)
    case audioDecodingFailed(String)
    case audioEncodingFailed(String)
}

public struct AudioGenerateParameters {
    var maxTokens: Int = 1200
    var temperature: Float = 0.6
    var topP: Float = 0.8
    var repetitionPenalty: Float = 1.3
    var repetitionContextSize: Int = 20
}
```

### Model cache — SwiftAcervo shared directory

All `intrusive-memory` projects share models via **SwiftAcervo** at `~/Library/SharedModels/`. Models downloaded by any app are available to all others.

| Component | Path |
|-----------|------|
| **Shared models** | `~/Library/SharedModels/<namespace>_<repo>/` |
| **Legacy path** (auto-migrated) | `~/Library/Caches/intrusive-memory/Models/Audio/<namespace>_<repo>/` |
| **Marvis prompt cache** | `~/Library/SharedModels/Marvis-AI_marvis-tts-250m-v0.2-MLX-8bit/prompt_cache/` |

The `<namespace>_<repo>` directory name is the HuggingFace repo ID with `/` replaced by `_` (e.g., `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` → `mlx-community_Qwen3-TTS-12Hz-1.7B-Base-bf16`).

Model resolution uses `ModelResolver.resolve(modelId:)` which delegates to SwiftAcervo. Legacy cache paths are auto-migrated on first use via `Acervo.migrateFromLegacyPaths()`.

### Audio Component Registry (ComponentDescriptor Pattern)

Audio codec modules (SNAC, Mimi) register with the **Acervo Component Registry** at module initialization using `ComponentDescriptor`. This declarative approach enables intelligent model management—Acervo knows exactly what to download, verify, and cache before model code runs.

#### Why ComponentDescriptor?

The pattern solves a critical coordination problem:
- **Model code** (in `SNAC.swift`) loads weights from disk
- **Acervo** needs to know what files to download before model code runs
- **Without descriptors**, these concerns are tightly coupled; Acervo doesn't know what files a model needs until model code actually calls it

**Solution**: Register component metadata at module initialization time, independent of model loading.

#### Pattern structure

Each codec with model variants defines (e.g., in `SNACModelManager.swift`):

1. **Enum for model repos** (e.g., `SNACModelRepo`)
   ```swift
   enum SNACModelRepo: String {
     case snac24kHz = "mlx-community/snac_24khz"
     
     var componentId: String {
       switch self {
       case .snac24kHz: return "snac-24khz"
       }
     }
   }
   ```

2. **File lists** (e.g., `snac24kHzRequiredFiles`)
   ```swift
   private let snac24kHzRequiredFiles: [ComponentFile] = [
     ComponentFile(relativePath: "config.json"),
     ComponentFile(relativePath: "model.safetensors"),
   ]
   ```
   Used by `Acervo.ensureComponentReady()` to verify all required files exist.

3. **Component descriptors** (e.g., `snacComponentDescriptors`)
   ```swift
   private let snacComponentDescriptors: [ComponentDescriptor] = [
     ComponentDescriptor(
       id: "snac-24khz",
       type: .decoder,
       displayName: "SNAC 24 kHz Audio Codec",
       repoId: "mlx-community/snac_24khz",
       files: snac24kHzRequiredFiles,
       estimatedSizeBytes: 158_809_902,
       minimumMemoryBytes: 200_000_000,
       metadata: ["sampleRate": "24000", "bitrate": "0.98 kbps", "rvqLevels": "3"]
     ),
   ]
   ```
   Each descriptor contains:
   - `id`: Unique Acervo component identifier
   - `type`: Component category (`.decoder`, `.encoder`, `.codec`)
   - `displayName`: Human-readable name for logging
   - `repoId`: HuggingFace repo path
   - `files`: Required files for verification
   - `estimatedSizeBytes`: Download size for progress UI
   - `minimumMemoryBytes`: Inference memory requirement
   - `metadata`: Codec-specific info (stored as `[String: String]`)

4. **Module-level registration trigger** (e.g., `_registerSNACComponents`)
   ```swift
   private let _registerSNACComponents = {
     do {
       try Acervo.register(snacComponentDescriptors)
     } catch {
       print("SNAC ComponentDescriptor registration failed (non-fatal): \(error)")
     }
   }()
   ```
   - Evaluated once (lazily) on first access
   - Gracefully handles registration failures (models still load via fallback)

#### Usage in model loading

```swift
// In SNACModelManager.swift
extension SNAC {
  public static func ensureComponentsRegistered() {
    _ = _registerSNACComponents  // Trigger lazy initialization
  }
}

// In SNAC.swift fromPretrained()
public static func fromPretrained(_ modelRepo: String) async throws -> SNAC {
  Self.ensureComponentsRegistered()  // Register before any Acervo call
  
  let resolved = try await Acervo.ensureComponentReady(
    componentId: SNACModelRepo(rawValue: modelRepo)?.componentId ?? modelRepo
  )
  // Model code can now safely assume all required files exist at resolved.path
  let weights = try loadWeights(from: resolved.path)
  // ... initialize and return model
}
```

#### Benefits

- **Declarative**: Component metadata lives in one place (e.g., `SNACModelManager.swift`)
- **Discoverable**: Acervo knows download size, memory requirements, and file structure without running model code
- **Testable**: `ComponentDescriptor` structs can be unit tested independently
- **Resilient**: Registration failures are non-fatal; models load via fallback if Acervo isn't available
- **Efficient**: Metadata exists at app startup, enabling smart UI (download estimates, verification) before model code runs

#### P1 Models Using ComponentDescriptors

| Model | Component ID | HuggingFace Repo | Manager File | Status |
|-------|--------------|------------------|--------------|--------|
| SNAC 24 kHz | `snac-24khz` | `mlx-community/snac_24khz` | `Sources/MLXAudioCodecs/SNAC/SNACModelManager.swift` | P1 |
| Mimi PyTorch | `mimi-pytorch-bf16` | `kyutai/moshiko-pytorch-bf16` | `Sources/MLXAudioCodecs/Mimi/MimiModelManager.swift` | P1 |

#### Shared Model Cache

All audio models (P1 codecs and TTS) download to:
```
~/Library/SharedModels/<namespace>_<repo>/
```

The shared cache allows:
- Any intrusive-memory app to reuse models downloaded by another
- Offline operation (models persist across app restarts)
- Faster repeated model loading (no redundant downloads)

Legacy paths (`~/Library/Caches/intrusive-memory/Models/`) are auto-migrated on first use via `Acervo.migrateFromLegacyPaths()`.

### Model loading

Models are loaded from HuggingFace via `fromPretrained()`. Internally this calls `ModelResolver.resolve()` (SwiftAcervo) which caches to `~/Library/SharedModels/`.

### TTSModelUtils — model type dispatch

`TTSModelUtils.loadModel(modelRepo:)` reads `config.json` from the HuggingFace repo and dispatches based on `model_type`:

| `model_type` | Swift class | Example repo |
|-------------|-------------|--------------|
| `qwen3_tts` | `Qwen3TTSModel` | `mlx-community/Qwen3-TTS-12Hz-1.7B-*` |
| `qwen3` / `qwen` | `Qwen3Model` | `mlx-community/VyvoTTS-EN-Beta-4bit` |
| `llama_tts` / `orpheus` | `LlamaTTSModel` | `mlx-community/orpheus-3b-0.1-ft-bf16` |
| `csm` / `sesame` | `MarvisTTSModel` | `Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit` |
| `soprano_tts` | `SopranoModel` | `mlx-community/Soprano-80M-bf16` |
| `pocket_tts` | `PocketTTSModel` | `mlx-community/pocket-tts` |

### Concurrency

- All model loading and generation uses `async/await`
- Streaming uses `AsyncThrowingStream` with `AudioGeneration` enum cases
- Types are annotated `Sendable`; models use `@unchecked Sendable` where needed
- Use `@preconcurrency import MLX` to suppress concurrency warnings from MLX

### Neural network layers

MLX neural network modules use `@ModuleInfo` property wrappers for layer registration. All layer classes inherit from `Module` (MLXNN). Weight sanitization methods (`sanitize(weights:)`) handle PyTorch-to-MLX weight format conversion (e.g., Conv1d weight transposition from `[O,I,K]` to `[O,K,I]`).

### Error handling

Custom errors use `AudioGenerationError` enum with `LocalizedError` conformance and associated values for context.

## Qwen3-TTS Model Architecture

The Qwen3-TTS family (`Sources/MLXAudioTTS/Models/Qwen3TTS/`) supports four generation paths.

### File map

| File | Lines | Description |
|------|-------|-------------|
| `Qwen3TTS.swift` | 1666 | Main model: routing, generation methods, shared loop, embedding extraction |
| `Qwen3TTSSpeechDecoder.swift` | 1047 | Speech tokenizer: VQ decoder, encoder, streaming decode, weight sanitization |
| `Qwen3TTSConfig.swift` | 598 | All config structs (talker, code predictor, tokenizer, speaker encoder) |
| `Qwen3TTSSpeakerEncoder.swift` | 388 | ECAPA-TDNN speaker encoder: TDNN, Res2Net, SE blocks, attentive pooling |
| `Qwen3TTSTalker.swift` | 363 | Talker transformer with 3D RoPE, embedding layers |
| `Qwen3TTSVoiceClonePrompt.swift` | 306 | VoiceClonePrompt struct, serialization, createVoiceClonePrompt, generateWithClonePrompt |
| `Qwen3TTSCodePredictor.swift` | 241 | Predicts codec tokens 2-16 from first codebook token + hidden states |
| `Qwen3TTSSpeechEncoder.swift` | 138 | SeanetEncoder + transformer encoder (reuses Mimi codec components) |

### Model variants

Three model variants are distinguished by `tts_model_type` in `config.json`:

| Model | `tts_model_type` | HuggingFace repo | Voice source | Speaker encoder | `spk_id` |
|-------|-----------------|------------------|--------------|----------------|----------|
| Base | `"base"` | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | x-vector from ref audio or none | Yes (`enc_dim: 2048`) | Empty |
| VoiceDesign | `"voice_design"` | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | Text description (instruct) | No | Empty |
| CustomVoice | `"custom_voice"` | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | Predefined speakers by name | No | 9 named speakers |

### Generation routing

```
generate(text:voice:refAudio:refText:language:generationParameters:)
├── tts_model_type == "voice_design"  → generateVoiceDesign()
├── tts_model_type == "custom_voice"  → generateCustomVoice(speaker: voice)
└── tts_model_type == "base"
    ├── refAudio + refText + hasEncoder → generateICL() (voice cloning)
    └── else                            → generateBase(speaker: voice)
```

All generation paths follow the same pattern:
1. Prepare input embeddings (text + codec prefix + optional speaker/instruct)
2. Run shared autoregressive loop (`generateFromEmbeddings()`)
3. Decode codec tokens to audio via speech tokenizer
4. Trim to valid audio length

### Public API — Qwen3TTSModel

```swift
// Core protocol methods
public var sampleRate: Int { get }
public func generate(text:voice:refAudio:refText:language:generationParameters:) async throws -> MLXArray
public func generateStream(text:voice:refAudio:refText:language:generationParameters:) -> AsyncThrowingStream<AudioGeneration, Error>

// Model loading
public static func fromPretrained(_ modelRepo: String) async throws -> Qwen3TTSModel

// Language resolution
public static func resolveLanguage(_ code: String, config: Qwen3TTSTalkerConfig? = nil) -> String?

// Voice cloning (from VoiceClonePrompt extension)
public func createVoiceClonePrompt(refAudio:refText:language:) throws -> VoiceClonePrompt
public func generateWithClonePrompt(text:clonePrompt:language:temperature:topP:repetitionPenalty:maxTokens:) throws -> MLXArray
```

### VoiceClonePrompt

`VoiceClonePrompt` caches encoded reference audio and speaker embedding for reuse:

```swift
public struct VoiceClonePrompt: Sendable {
    public let refCodes: MLXArray        // [1, 16, ref_time]
    public let speakerEmbedding: MLXArray? // [1, enc_dim] (nil for VoiceDesign)
    public let refText: String
    public let language: String

    public func serialize() throws -> Data
    public static func deserialize(from data: Data) throws -> VoiceClonePrompt
}
```

### Internal methods (not public, but important for understanding)

| Method | File | Purpose |
|--------|------|---------|
| `resolveGenerationPath()` | Qwen3TTS.swift | Determines which generation path to use |
| `prepareBaseInputs()` | Qwen3TTS.swift | Builds input embeddings for Base/CustomVoice |
| `prepareICLInputs()` | Qwen3TTS.swift | Builds combined text+codec embeddings for ICL (two overloads) |
| `generateFromEmbeddings()` | Qwen3TTS.swift | Shared autoregressive generation loop |
| `generateBase()` | Qwen3TTS.swift | Base model generation |
| `generateCustomVoice()` | Qwen3TTS.swift | CustomVoice with speaker validation |
| `generateICL()` | Qwen3TTS.swift | ICL voice cloning from raw reference audio |
| `generateVoiceDesign()` | Qwen3TTS.swift | VoiceDesign generation (text-described voice) |
| `extractSpeakerEmbedding()` | Qwen3TTS.swift | Extract x-vector embedding via ECAPA-TDNN |
| `sampleToken()` | Qwen3TTS.swift | Token sampling with temperature, topP, repetition penalty |

### Speaker encoder (ECAPA-TDNN)

`Qwen3TTSSpeakerEncoder.swift` implements the full ECAPA-TDNN architecture:
- Input: mel spectrogram `[batch, time, 128]`
- Components: TimeDelayNet → 3x SqueezeExcitation-Res2Net blocks → AttentiveStatisticsPooling → Conv1d projection
- Output: x-vector embedding `[batch, enc_dim]` (typically 2048)
- `sanitize(weights:)` strips `speaker_encoder.` prefix and transposes Conv1d weights

### Speech tokenizer encoder

`Qwen3TTSSpeechEncoder.swift` encodes audio waveforms into codec tokens:
- Reuses Mimi codec components (`SeanetEncoder`, `ProjectedTransformer`, `ConvDownsample1d`, `SplitResidualVectorQuantizer`)
- Input: audio `[batch, 1, samples]`
- Output: codec codes `[1, 16, time]`
- Only present on Base models (checked via `hasEncoder` property)

## Coding Conventions

- **Imports**: `@preconcurrency import MLX` and `import MLXNN` for neural network code
- **Organization**: `// MARK: -` comments to separate sections within files
- **Classes vs structs**: Neural network modules are `class` (inheriting `Module`); configs and data types are `struct`
- **Enum namespaces**: Utility classes use `enum` with static methods (e.g., `ModelUtils`, `TTSModelUtils`)
- **Documentation**: Doc comments on public API methods with `/// - Parameters:` and `/// - Returns:` sections
- **No linter**: No SwiftLint or swift-format configuration exists
- **Print statements**: Used for user-facing progress feedback in model loading methods; matches convention across all model implementations

## Test Organization

Tests use **Swift Testing** framework (`@Test`, `#expect`, `@Suite`), not XCTest.

### Test suites that run without model downloads (safe for CI)

**Codecs:**

| Suite | File | Tests | What it tests |
|-------|------|-------|---------------|
| `VocosTests` | MLXAudioCodecsTests.swift | — | Vocos vocoder components, layer shapes |
| `EncodecTests` | MLXAudioCodecsTests.swift | — | Encodec codec components, config, RVQ |
| `DACVAETests` | MLXAudioCodecsTests.swift | — | DACVAE codec components, encoder/decoder |

**STT:**

| Suite | File | Tests | What it tests |
|-------|------|-------|---------------|
| `GLMASRModuleSetupTests` | MLXAudioSTTTests.swift | — | GLM-ASR config, layers, shapes |
| `Qwen3ASRModuleSetupTests` | MLXAudioSTTTests.swift | — | Qwen3 ASR config, layers, weight sanitization |
| `ForceAlignProcessorTests` | MLXAudioSTTTests.swift | — | Text tokenization, timestamp encoding |
| `ForcedAlignResultTests` | MLXAudioSTTTests.swift | — | Alignment result data structures |
| `Qwen3ASRHelperTests` | MLXAudioSTTTests.swift | — | Feature extraction length calculations |
| `SplitAudioIntoChunksTests` | MLXAudioSTTTests.swift | — | Audio chunking logic |

**Qwen3-TTS (no model downloads):**

| Suite | File | Tests | What it tests |
|-------|------|-------|---------------|
| `Qwen3TTSSpeechTokenizerTests` | MLXAudioTTSTests.swift | 2 | hasEncoder default and setter |
| `Qwen3TTSSpeechTokenizerEncodeTests` | MLXAudioTTSTests.swift | 1 | encode() throws without encoder |
| `Qwen3TTSSpeechTokenizerWeightTests` | MLXAudioTTSTests.swift | — | Weight loading, encoder weight mapping, Q/K/V combining |
| `Qwen3TTSLanguageTests` | MLXAudioTTSTests.swift | 18 | Language code resolution (ISO, full name, auto, config) |
| `Qwen3TTSConfigTests` | MLXAudioTTSTests.swift | 10 | Config parsing for Base, VoiceDesign, CustomVoice |
| `Qwen3TTSRoutingTests` | MLXAudioTTSTests.swift | 10 | Generation path routing logic |
| `Qwen3TTSPrepareBaseInputsTests` | MLXAudioTTSTests.swift | 14 | Input preparation, speaker lookup, dialect override |
| `Qwen3TTSGenerateCustomVoiceTests` | MLXAudioTTSTests.swift | — | CustomVoice validation (nil speaker, invalid speaker, speaker lookup) |
| `Qwen3TTSSpeakerEncoderTests` | MLXAudioTTSTests.swift | 13 | ECAPA-TDNN shapes, reflect padding, blocks |
| `Qwen3TTSSpeakerEncoderWeightTests` | MLXAudioTTSTests.swift | 13 | Weight sanitization, prefix stripping, transposition |
| `Qwen3TTSSpeakerEmbeddingTests` | MLXAudioTTSTests.swift | 8 | extractSpeakerEmbedding output shapes, errors, determinism |
| `Qwen3TTSPrepareICLInputsTests` | MLXAudioTTSTests.swift | — | ICL input preparation, ref code handling, embedding composition |
| `Qwen3TTSGenerateICLTests` | MLXAudioTTSTests.swift | — | ICL generation validation (missing encoder, tokenizer, ref audio/text) |
| `Qwen3TTSSpeakerEncoderSmokeTests` | MLXAudioTTSTests.swift | — | Smoke tests for speaker encoder integration with Base model |

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
| `Qwen3TTSBaseModelTests` | MLXAudioTTSTests.swift | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` |
| `Qwen3TTSCustomVoiceTests` | MLXAudioTTSTests.swift | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` |
| `Qwen3TTSCloningTests` | MLXAudioTTSTests.swift | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` |
| `Qwen3TTSSpeakerEncoderIntegrationTests` | MLXAudioTTSTests.swift | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` |
| `Qwen3ASRTests` | MLXAudioSTTTests.swift | `mlx-community/Qwen3-ASR-0.6B-4bit` |
| `GLMASRTests` | MLXAudioSTTTests.swift | `mlx-community/GLM-ASR-Nano-2512-4bit` |

### Test conventions

- **Struct naming**: `<Component>Tests` (e.g., `VocosTests`, `Qwen3TTSRoutingTests`)
- **Function naming**: `@Test func testFeatureName()` or `@Test func featureNameBehavior()`
- **Test resources**: Access via `Bundle.module.url(forResource:withExtension:subdirectory:"media")`
- **Async tests**: Use `async throws` for anything involving model loading or generation
- **Streaming tests**: Iterate `AsyncThrowingStream`, count tokens, verify final audio shape
- **Output files**: Write to `FileManager.default.temporaryDirectory`

## HuggingFace Model Repos

| Repo | Type | Used by |
|------|------|---------|
| `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` | TTS (Base) | Qwen3TTSModel |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` | TTS (VoiceDesign) | Qwen3TTSModel |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` | TTS (CustomVoice) | Qwen3TTSModel |
| `mlx-community/VyvoTTS-EN-Beta-4bit` | TTS | Qwen3Model |
| `mlx-community/orpheus-3b-0.1-ft-bf16` | TTS | LlamaTTSModel |
| `mlx-community/Soprano-80M-bf16` | TTS | SopranoModel |
| `mlx-community/pocket-tts` | TTS | PocketTTSModel |
| `Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit` | TTS | MarvisTTSModel |
| `mlx-community/snac_24khz` | Codec | SNAC |
| `kyutai/moshiko-pytorch-bf16` | Codec | Mimi |
| `mlx-community/encodec-24khz-float32` | Codec | Vocos/Encodec |
| `mlx-community/Qwen3-ASR-0.6B-4bit` | STT | Qwen3ASR |
| `mlx-community/GLM-ASR-Nano-2512-4bit` | STT | GLMASR |
| `mlx-community/Qwen3-ForcedAligner-0.6B-4bit` | STT | Qwen3ForcedAligner |

## Git Workflow

- **Branches**: `main` (protected), `development` (working branch)
- **PRs**: Always `development` -> `main` within the fork
- **Branch protection on main**: Required status checks (`Code Quality`, `macOS Tests`), enforce admins, no force push, no deletions
- **Commit style**: Imperative mood, concise subject line (e.g., "Add Qwen3 ASR", "Fix weight loading in Vocos")
- **Remotes**: `origin` (intrusive-memory fork), `upstream` (Blaizzy/mlx-audio-swift), `inqtr` (INQTR fork)
- **Upstream contributions**: **Do NOT create upstream pull requests.** This fork is independent and does not contribute back to `Blaizzy/mlx-audio-swift`. All work stays within the `intrusive-memory` fork.

## CI/CD

**Workflow**: `.github/workflows/tests.yaml`

| Job | Runs on | Purpose |
|-----|---------|---------|
| Code Quality | macos-26 | Flags TODOs, large files, print() in Sources/ |
| macOS Tests | macos-26 | Builds package, runs unit tests (no downloads) |
| Download Models | macos-26 | Caches HuggingFace models for integration tests |
| Model Tests | macos-26 | Runs model-dependent tests when cache is warm |

**Triggers**: `workflow_dispatch` + PRs to `main` (opened, synchronize, reopened)

**Model cache**: `~/Library/SharedModels` cached with key `mlx-models-v2`. Prime via `workflow_dispatch`. Model tests skip when cache is cold.

## Adding a New TTS Model

1. Create a new directory under `Sources/MLXAudioTTS/Models/<ModelName>/`
2. Implement the model class conforming to `SpeechGenerationModel`
3. Add a `<ModelName>Config.swift` with `Codable` config struct
4. Implement `fromPretrained()` using `ModelResolver.resolve(modelId:)`
5. Implement `sanitize(weights:)` for PyTorch-to-MLX weight conversion
6. Add the model type to `TTSModelUtils.swift` for routing
7. Add tests to `Tests/MLXAudioTTSTests.swift`

## Adding a New Audio Codec

1. Create a new directory under `Sources/MLXAudioCodecs/<CodecName>/`
2. Implement `encode()` and `decode()` methods
3. Add `fromPretrained()` for HuggingFace loading
4. Add tests to `Tests/MLXAudioCodecsTests.swift`

## Common Pitfalls

- **Never use `swift build`/`swift test`** — always `xcodebuild`
- **PyTorch weight conversion**: Conv1d weights need transposition from `(O,I,K)` to `(O,K,I)`, Conv2d from `(O,I,H,W)` to `(O,H,W,I)` in `sanitize()`
- **Model cache path**: `~/Library/SharedModels/<namespace>_<repo>` — replace `/` with `_` in repo ID. All `intrusive-memory` projects share models via SwiftAcervo
- **Bundle resources in tests**: Use `.copy("media")` in Package.swift, access via `Bundle.module`
- **Concurrency warnings**: Use `@preconcurrency import MLX` and `@unchecked Sendable` on Module subclasses
- **CI test selection**: Only add tests to CI that work without model downloads. Model-dependent tests go in the `model-tests` job
- **@ModuleInfo mutation**: Never mutate `@ModuleInfo` properties directly after init; use `update(modules:)` or `update(parameters:)` instead
- **ICL repetition penalty**: Minimum 1.5 for ICL generation to prevent code degeneration with long reference prefills
