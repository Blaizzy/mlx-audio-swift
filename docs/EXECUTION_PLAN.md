# mlx-audio-swift Fork — Execution Plan

## Overview

This plan covers all work in the `intrusive-memory/mlx-audio-swift` fork to add full
Qwen3-TTS support: Base model generation, CustomVoice generation, voice cloning (ICL),
speaker encoder (ECAPA-TDNN), speech tokenizer encoder, instruct parameter support,
and ultimately an upstream PR back to `Blaizzy/mlx-audio-swift`.

The fork already exists at `github.com/intrusive-memory/mlx-audio-swift` with PR #23
(VoiceDesign support by INQTR) merged into main.

### What This Fork Provides

1. **VoiceDesign model support** (already implemented from PR #23)
2. **Base model generation** — text + language → audio (no voice description needed)
3. **CustomVoice generation** — predefined speakers via `spk_id` config
4. **Speech tokenizer encoder** — encode reference audio to codec tokens
5. **Speaker encoder (ECAPA-TDNN)** — extract x-vector speaker embeddings from audio
6. **Voice cloning (ICL)** — reference audio + text → cloned voice generation
7. **Clone prompt persistence** — pre-compute speaker embedding + ref codes, serialize for reuse
8. **Instruct parameter** — voice delivery hints for Base and CustomVoice modes
9. **Language parameter** — multi-language generation support

### Python Reference

- `github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/qwen3_tts/qwen3_tts.py`
- `github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/qwen3_tts/speech_tokenizer.py`
- `github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/qwen3_tts/speaker_encoder.py`
- `github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/tts/models/qwen3_tts/config.py`

### Model Type Architecture

The Qwen3-TTS family has three model variants, distinguished by `tts_model_type` in `config.json`:

| Model | `tts_model_type` | Voice Source | `speaker_encoder_config` | `spk_id` |
|-------|-----------------|-------------|-------------------------|----------|
| Base | `"base"` | x-vector from ref audio OR none | Present (`enc_dim: 2048`) | Empty `{}` |
| VoiceDesign | `"voice_design"` | Text description via instruct | Absent | Empty `{}` |
| CustomVoice | `"custom_voice"` | Predefined speakers by name | Absent | 9 named speakers |

All three share the same Talker, CodePredictor, and SpeechDecoder architectures. The differences are:
- How voice identity is specified (text description vs speaker ID vs reference audio)
- Whether a speaker encoder exists (Base only)
- Whether `spk_id` has entries (CustomVoice only)

### Generation Routing (Python reference)

```
generate()
├── tts_model_type == "voice_design" → generate_voice_design()        [IMPLEMENTED]
├── tts_model_type == "custom_voice" → generate_custom_voice()        [NOT IMPLEMENTED]
└── tts_model_type == "base"
    ├── ref_audio + ref_text + has_encoder → _generate_icl()          [NOT IMPLEMENTED]
    └── else → _generate_with_instruct() (standard base generation)   [NOT IMPLEMENTED]
```

### Current Swift Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Talker model | Complete | `Qwen3TTSTalker.swift` |
| Code Predictor | Complete | `Qwen3TTSCodePredictor.swift` |
| Speech tokenizer decoder | Complete | `Qwen3TTSSpeechDecoder.swift` |
| Speech tokenizer encoder | **Not implemented** (weights skipped at line 759) | — |
| Speaker encoder (ECAPA-TDNN) | **Not implemented** | — |
| VoiceDesign generation | Complete | `Qwen3TTS.swift:generateVoiceDesign()` |
| Base generation | **Not implemented** | — |
| CustomVoice generation | **Not implemented** | — |
| ICL generation | **Not implemented** | — |
| Config parsing (ttsModelType) | Parsed but not branched on | `Qwen3TTSConfig.swift:327` |
| Config parsing (spkId) | Parsed but not used | `Qwen3TTSConfig.swift:100` |
| Config parsing (speakerEncoderConfig) | **Not parsed** | — |
| Mimi encoder components (reusable) | Available in `MLXAudioCodecs/Mimi/` | `Seanet.swift`, `Transformer.swift`, `Conv.swift`, `Quantization.swift` |
| Mel spectrogram DSP | Available | `MLXAudioCore/DSP.swift` |

---

## Prioritized Execution Order

Tasks are ordered to reach ICL voice cloning (VoxAlta critical path) as fast as possible.
Unit tests ship inline with their implementation tasks. Standalone Base and CustomVoice
generation are deferred until after the ICL pipeline is complete.

| # | Task | Area | Tier | Deps | Parallel with |
|---|------|------|------|------|---------------|
| 1 | Model type routing + routing tests | A1 | 0 | — | 2, 3, 4, 5 |
| 2 | Parse speaker_encoder_config + config tests | A2 | 0 | — | 1, 3, 4, 5 |
| 3 | Add hasEncoder property | A3 | 0 | — | 1, 2, 4, 5 |
| 4 | Language code mapping + lang tests | H2 | 0 | — | 1, 2, 3, 5 |
| 5 | Extract shared generation loop | I1 | 0 | — | 1, 2, 3, 4 |
| 6 | Implement prepareBaseInputs() | B1 | 1 | 1 | 7, 8 |
| 7 | Implement speech tokenizer encoder | D1 | 1 | 3 | 6, 8 |
| 8 | Port ECAPA-TDNN speaker encoder + shape tests | E1 | 1 | 2 | 6, 7 |
| 9 | Update sanitize() for encoder weights | D2 | 2 | 7 | 10 |
| 10 | Speaker encoder weight loading | E2 | 2 | 8, 2 | 9 |
| 11 | Speech tokenizer encode() method | D3 | 3 | 7, 9 | 12 |
| 12 | Implement extractSpeakerEmbedding() | E3 | 3 | 10 | 11 |
| 13 | Implement prepareICLInputs() | F1 | 4 | 11, 12, 6 | — |
| 14 | Implement generateICL() | F2 | 4 | 13, 1 | — |
| 15 | Implement VoiceClonePrompt struct | G1 | 5 | 11, 12 | — |
| 16 | Implement generateWithClonePrompt() | G2 | 5 | 15, 14 | — |
| 17 | Implement generateBase() | B2 | 6 | 1, 6, 5 | 18, 19 |
| 18 | Implement generateCustomVoice() | C1 | 6 | 1, 6, 5 | 17, 19 |
| 19 | Wire instruct parameter | H1 | 6 | 6 | 17, 18 |
| 20 | Integration tests (model downloads) | J5 | 7 | all impl | — |
| 21 | Clean up fork changes | K1 | 7 | all impl | — |
| 22 | Create upstream PR | K2 | 7 | 21 | — |

### Tier Summary

| Tier | Name | Tasks | Goal |
|------|------|-------|------|
| 0 | Foundation | 1-5 | Config, routing, shared loop — all parallel, no deps |
| 1 | Building blocks | 6-8 | Input prep + both encoder implementations — all parallel |
| 2 | Weight loading | 9-10 | Load encoder weights into both encoder models — parallel |
| 3 | Encode + embed | 11-12 | Functional encode() and extractSpeakerEmbedding() — parallel |
| 4 | ICL pipeline | 13-14 | Voice cloning generation — sequential |
| 5 | Persistence | 15-16 | Clone prompt caching for VoxAlta — sequential |
| 6 | Standalone gen | 17-19 | Base + CustomVoice generation — all parallel |
| 7 | Ship | 20-22 | Integration tests, cleanup, upstream PR |

### Critical Path (longest chain to VoxAlta unblock)

```
A2 → E1 → E2 → E3 ──┐
A3 → D1 → D2 → D3 ──┼→ F1 → F2 → G1 → G2
A1 → B1 ─────────────┘
```

Three parallel chains converge at F1 (task 13). The encoder chains (8 tasks each) are
the bottleneck. Total critical path length: **10 tasks** (A2→E1→E2→E3→F1→F2→G1→G2
or A3→D1→D2→D3→F1→F2→G1→G2, with A1→B1 completing sooner).

---

## Task Details

Each task is atomic (single deliverable), testable (has explicit verification criteria),
and fully scoped (all inputs, outputs, and dependencies documented). Unit tests ship
inline with implementation.

---

### Task 1: Add model type routing in generate() [A1]

**Tier**: 0 — Foundation
**Parallel with**: Tasks 2, 3, 4, 5

**What**: Branch on `config.ttsModelType` in `Qwen3TTS.swift:generate()` to route
to the correct generation method based on model variant.

**Current state**: `generate()` always calls `generateVoiceDesign()` regardless of
`ttsModelType`. The `refAudio` and `refText` parameters are accepted but ignored.

**Changes**:
- In `Qwen3TTS.swift:generate()` (line 33-66), add a `switch config.ttsModelType`:
  - `"voice_design"` → `generateVoiceDesign()` (existing)
  - `"custom_voice"` → `generateCustomVoice()` (new, stub initially)
  - `"base"` → check if `refAudio != nil && refText != nil && speechTokenizer.hasEncoder`:
    - If yes → `generateICL()` (new, stub initially)
    - If no → `generateBase()` (new, stub initially)

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification** (includes routing unit tests — formerly J2):
- Unit test: Create Qwen3TTSModelConfig with `ttsModelType: "voice_design"`, verify it
  routes to VoiceDesign path (existing behavior unchanged)
- Unit test: Create config with `ttsModelType: "custom_voice"`, verify it routes to
  CustomVoice path (can be a stub that throws "not implemented" initially)
- Unit test: Create config with `ttsModelType: "base"`, verify it routes to Base path
- Unit test: Base config with refAudio + refText + hasEncoder → routes to ICL path
- Unit test: Base config with refAudio but no refText → routes to Base path (not ICL)
- Build passes
- Existing CI tests still pass (no regression)

**Dependencies**: None

---

### Task 2: Parse speaker_encoder_config from config.json [A2]

**Tier**: 0 — Foundation
**Parallel with**: Tasks 1, 3, 4, 5

**What**: Add `speakerEncoderConfig` field to `Qwen3TTSModelConfig` to parse the
`speaker_encoder_config` section from Base model configs.

**Current state**: `Qwen3TTSModelConfig` does not have this field. Base model configs
contain `"speaker_encoder_config": {"enc_dim": 2048, "sample_rate": 24000}`.

**Changes**:
- Add `Qwen3TTSSpeakerEncoderConfig` struct (Codable) with fields:
  - `melDim: Int` (default 128)
  - `encDim: Int` (default 1024)
  - `encChannels: [Int]` (default [512, 512, 512, 512, 1536])
  - `encKernelSizes: [Int]` (default [5, 3, 3, 3, 1])
  - `encDilations: [Int]` (default [1, 2, 3, 4, 1])
  - `encAttentionChannels: Int` (default 128)
  - `encRes2netScale: Int` (default 8)
  - `encSeChannels: Int` (default 128)
  - `sampleRate: Int` (default 24000)
- Add `speakerEncoderConfig: Qwen3TTSSpeakerEncoderConfig?` to `Qwen3TTSModelConfig`
- Parse with `decodeIfPresent`

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSConfig.swift`

**Verification** (includes config parsing tests — formerly J1):
- Unit test: Parse Base model config JSON, verify `speakerEncoderConfig` is non-nil
  with `encDim == 2048`, `sampleRate == 24000`
- Unit test: Parse VoiceDesign config JSON, verify `speakerEncoderConfig` is nil
- Unit test: Parse CustomVoice config JSON, verify `speakerEncoderConfig` is nil
- Unit test: Parse Base config → `ttsModelType == "base"`, `spkId` empty
- Unit test: Parse VoiceDesign config → `ttsModelType == "voice_design"`, `spkId` empty
- Unit test: Parse CustomVoice config → `ttsModelType == "custom_voice"`,
  `spkId` has entries (e.g., `"serena": [3066]`)
- Unit test: Parse `codecLanguageId` → 10 entries for Base/VoiceDesign, 12 for CustomVoice
- Build passes

**Dependencies**: None

---

### Task 3: Add hasEncoder property to speech tokenizer [A3]

**Tier**: 0 — Foundation
**Parallel with**: Tasks 1, 2, 4, 5

**What**: Add a `hasEncoder` boolean property to `Qwen3TTSSpeechTokenizer` that indicates
whether the encoder is available (needed for ICL routing decision).

**Current state**: No way to check if encoder is loaded.

**Changes**:
- Add `var encoderModel: Qwen3TTSSpeechTokenizerEncoder?` property to `Qwen3TTSSpeechTokenizer`
  (initially nil)
- Add `var hasEncoder: Bool { encoderModel != nil }` computed property

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSSpeechDecoder.swift`

**Verification**:
- Unit test: `hasEncoder` returns false when encoder is not loaded
- Build passes

**Dependencies**: None

---

### Task 4: Implement language code mapping [H2]

**Tier**: 0 — Foundation
**Parallel with**: Tasks 1, 2, 3, 5

**What**: Map ISO 639-1 language codes (en, zh, ja, ko, etc.) to Qwen3-TTS internal
language strings (english, chinese, japanese, korean, etc.) and validate against the
model's supported languages.

**Current state**: `codecLanguageId` config field is parsed and used in
`prepareGenerationInputs()` but there's no user-friendly mapping from ISO codes.

**Python behavior**: Accepts language strings like "english", "chinese", "auto".
The Swift API should accept both ISO codes and full strings.

**Changes**:
- Add `static func resolveLanguage(_ code: String, config: Qwen3TTSTalkerConfig) -> String?`
  utility method
- Map: "en"→"english", "zh"→"chinese", "ja"→"japanese", "ko"→"korean",
  "de"→"german", "fr"→"french", "ru"→"russian", "pt"→"portuguese",
  "es"→"spanish", "it"→"italian"
- Accept "auto" as pass-through
- Accept full strings ("english", "chinese") as pass-through

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification** (includes language mapping tests — formerly J3):
- Unit test: `resolveLanguage("en")` → "english"
- Unit test: `resolveLanguage("zh")` → "chinese"
- Unit test: `resolveLanguage("english")` → "english" (pass-through)
- Unit test: `resolveLanguage("auto")` → "auto"
- Unit test: `resolveLanguage("xx")` → nil (unsupported)
- Build passes

**Dependencies**: None

---

### Task 5: Extract shared autoregressive generation loop [I1]

**Tier**: 0 — Foundation
**Parallel with**: Tasks 1, 2, 3, 4

**What**: The autoregressive loop (Talker forward → sample first code → CodePredictor
loop for codes 2-16 → prepare next input embedding) is identical across VoiceDesign,
Base, CustomVoice, and ICL. Extract it into a shared method to avoid code duplication.

**Current state**: The loop exists only in `generateVoiceDesign()` (lines 160-230).
Tasks 14, 17, and 18 would each need to duplicate this loop.

**Changes**:
- Extract `generateFromEmbeddings(inputEmbeds:trailingTextHidden:ttsPadEmbed:temperature:topP:repetitionPenalty:maxTokens:) -> [MLXArray]`
  that returns the generated code sequence
- `generateVoiceDesign()`, `generateBase()`, `generateCustomVoice()`, `generateICL()`
  all call this after their respective input preparation
- Each method handles its own post-processing (e.g., ICL prepends ref_codes and trims)

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification**:
- Existing VoiceDesign tests still pass (no regression after refactor)
- Build passes

**Dependencies**: None

---

### Task 6: Implement prepareBaseInputs() [B1]

**Tier**: 1 — Building blocks
**Parallel with**: Tasks 7, 8

**What**: Port `_prepare_generation_inputs()` from Python for Base/CustomVoice models.
This is the input embedding preparation that handles speaker ID lookup, language ID,
dialect override, instruct embedding, and codec prefix construction.

**Current state**: Only `prepareGenerationInputs()` exists for VoiceDesign. The Base
path is different: it looks up speaker in `spk_id`, supports dialect override, and
embeds a speaker ID token instead of a voice description.

**Python reference**: `_prepare_generation_inputs()` in `qwen3_tts.py:249-404`

**Key differences from VoiceDesign**:
- Speaker embedding: looks up `voice` in `config.talkerConfig.spkId`, gets token ID,
  embeds it via `talker.getInputEmbeddings()` → `[1, 1, hidden]`
- OR if `refAudio` is provided and speaker encoder exists, extracts x-vector embedding
- Dialect override: if speaker has `spkIsDialect[speaker] != false`, uses dialect
  language ID instead of the specified language
- Instruct embedding: optional, same format as VoiceDesign (`<|im_start|>user\n{instruct}<|im_end|>\n`)
- Codec prefix construction: `[think/nothink, thinkBos, langId?, thinkEos, speaker?, pad, bos]`

**Input**: `(text: String, language: String, speaker: String?, refAudio: MLXArray?, instruct: String?)`
**Output**: `(inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray)`

**Changes**:
- Add `prepareBaseInputs()` method to `Qwen3TTSModel`
- Implement speaker ID lookup from `config.talkerConfig.spkId`
- Implement dialect override from `config.talkerConfig.spkIsDialect`
- Implement optional speaker embedding injection into codec prefix
- Implement optional instruct embedding prepending

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification**:
- Unit test: Given a config with `spkId: {"alice": [3066]}`, calling `prepareBaseInputs(speaker: "alice")`
  produces input embeddings with the speaker token embedded in the codec prefix
- Unit test: Given `spkIsDialect: {"eric": "sichuan_dialect"}` and `language: "chinese"`,
  verify the codec prefix uses the Sichuan dialect language ID
- Unit test: Verify instruct embedding is prepended when provided, absent when nil
- Build passes

**Dependencies**: Task 1 (routing exists to reach this code path)

---

### Task 7: Implement speech tokenizer encoder [D1]

**Tier**: 1 — Building blocks
**Parallel with**: Tasks 6, 8

**What**: Port the speech tokenizer encoder from Python. This encodes raw audio waveforms
into discrete codec tokens `[batch, num_quantizers, time]` — required for ICL voice cloning.

**Current state**: Only the decoder exists. Encoder weights are explicitly skipped in
`sanitize()` at `Qwen3TTSSpeechDecoder.swift:759`.

**Key finding**: The encoder reuses **Mimi codec components** that already exist in Swift:
- `SeanetEncoder` → `Sources/MLXAudioCodecs/Mimi/Seanet.swift`
- `ProjectedTransformer` → `Sources/MLXAudioCodecs/Mimi/Transformer.swift`
- `ConvDownsample1d` → `Sources/MLXAudioCodecs/Mimi/Conv.swift`
- `SplitResidualVectorQuantizer.encode()` → `Sources/MLXAudioCodecs/Mimi/Quantization.swift`

**Python reference**: `Qwen3TTSSpeechTokenizerEncoder` in `speech_tokenizer.py:889-990`

**Architecture**:
```
audio [batch, 1, samples]
  → SeanetEncoder (conv downsampling chain)
  → ProjectedTransformer (causal attention with RoPE, cache)
  → ConvDownsample1d (stride = encoder_frame_rate / frame_rate)
  → SplitResidualVectorQuantizer.encode() (nearest-neighbor codebook lookup)
  → codes [batch, 16, time]
```

**Encoder config** (from `speech_tokenizer/config.json`):
- 32 quantizers in encoder, but only first 16 used (`valid_num_quantizers = 16`)
- 8 transformer layers, 8 attention heads
- Codebook size: 2048 (semantic: 4096)
- Sample rate: 24000, downsample rate: 1920

**Changes**:
- Create `Qwen3TTSSpeechTokenizerEncoder` class in `Qwen3TTSSpeechDecoder.swift` (or new file)
- Import and compose existing Mimi components: `SeanetEncoder`, `ProjectedTransformer`,
  `ConvDownsample1d`, `SplitResidualVectorQuantizer`
- Add encoder config parsing to `Qwen3TTSTokenizerConfig` (encoder-specific fields)
- Implement `encode(audio:) -> MLXArray` method:
  1. Reset encoder state and cache
  2. Run through `SeanetEncoder`
  3. Create causal attention mask
  4. Run through `ProjectedTransformer` with cache
  5. Run through `ConvDownsample1d`
  6. Run through `quantizer.encode()`
  7. Slice to `[:, :validNumQuantizers, :]`
- Needs investigation: verify Mimi components in MLXAudioCodecs are API-compatible
  with the encoder's usage (may need minor adaptations or public API exposure)

**Note on cross-module dependency**: The encoder needs types from `MLXAudioCodecs`
(Mimi module). Verify that `MLXAudioTTS` can import from `MLXAudioCodecs` in `Package.swift`.

**Files modified**:
- `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSSpeechDecoder.swift` (or new `Qwen3TTSSpeechEncoder.swift`)
- `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSConfig.swift` (encoder config)
- Possibly `Package.swift` (dependency)

**Verification**:
- Unit test: Encoder initializes from config without errors
- Unit test: Verify encoder layer structure matches expected architecture
- Integration test (requires Base model download): Encode a test WAV file,
  verify output tensor shape is `[1, 16, N]` where `N = samples / 1920`
- Build passes

**Dependencies**: Task 3 (hasEncoder property)

---

### Task 8: Port ECAPA-TDNN speaker encoder [E1]

**Tier**: 1 — Building blocks
**Parallel with**: Tasks 6, 7

**What**: Port the `Qwen3TTSSpeakerEncoder` from Python — an ECAPA-TDNN architecture
that extracts x-vector speaker embeddings from mel spectrograms.

**Current state**: No speaker encoder exists in the Swift codebase.

**Python reference**: `speaker_encoder.py` — 6 classes, ~250 lines total.

**Architecture**:
```
mel spectrogram [batch, time, 128]
  → transpose to [batch, 128, time]
  → TimeDelayNetBlock (mel_dim=128 → 512, kernel=5, dilation=1)
  → 3x SqueezeExcitationRes2NetBlock (512→512, kernels=[3,3,3], dilations=[2,3,4])
  → concatenate hidden states from SE-Res2Net blocks
  → TimeDelayNetBlock (MFA, 1536→1536, kernel=1, dilation=1)
  → AttentiveStatisticsPooling (1536 → 3072 via mean+std)
  → Conv1d (3072 → enc_dim=2048, kernel=1)
  → squeeze → [batch, enc_dim]
```

**Classes to port**:
1. `TimeDelayNetBlock` — Conv1d with reflect padding, ReLU
2. `Res2NetBlock` — Multi-scale feature extraction (scale=8, splits input channels)
3. `SqueezeExcitationBlock` — Channel attention (mean → Conv1d → ReLU → Conv1d → sigmoid → scale)
4. `SqueezeExcitationRes2NetBlock` — TDNN → Res2Net → TDNN → SE, with residual
5. `AttentiveStatisticsPooling` — Attention-weighted mean+std pooling
6. `Qwen3TTSSpeakerEncoder` — Orchestrates all components

**Helper needed**: `reflectPad1d(x:pad:)` — reflect padding in time dimension

**Files to create**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSSpeakerEncoder.swift`

**Verification** (includes encoder shape tests — formerly J4):
- Unit test: Initialize encoder from config, verify layer structure (correct number of
  SE-Res2Net blocks, correct channel dimensions)
- Unit test: Feed random mel spectrogram `[1, 100, 128]`, verify output shape is `[1, 2048]`
- Unit test: Feed batch of 2 mels `[2, 100, 128]`, verify output shape `[2, 2048]`
- Build passes

**Dependencies**: Task 2 (speaker encoder config)

---

### Task 9: Update sanitize() to load encoder weights [D2]

**Tier**: 2 — Weight loading
**Parallel with**: Task 10

**What**: Remove the encoder weight skip in `Qwen3TTSSpeechTokenizer.sanitize()` and
add proper weight remapping for encoder-prefixed keys.

**Current state**: Line 759 of `Qwen3TTSSpeechDecoder.swift`:
```swift
if k.hasPrefix("encoder.") { continue }
```

**Changes**:
- Remove the `continue` for encoder-prefixed keys
- Add weight remapping logic for encoder keys (same pattern as decoder: strip prefix,
  handle Conv1d weight transpositions)
- Conditionally load encoder weights only when encoder model exists (to avoid errors
  on VoiceDesign models that don't have encoder weights)

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSSpeechDecoder.swift`

**Verification**:
- Integration test (requires Base model download): Load Base model, verify encoder
  weights are loaded (encoder layer count > 0, no nil weights)
- VoiceDesign model still loads correctly (no regression — VoiceDesign safetensors
  don't contain encoder keys, so nothing to skip)
- Build passes

**Dependencies**: Task 7

---

### Task 10: Add speaker encoder weight loading [E2]

**Tier**: 2 — Weight loading
**Parallel with**: Task 9

**What**: Load speaker encoder weights from the Base model safetensors. Add sanitization
logic for `speaker_encoder.*` prefixed keys.

**Python reference**: `Qwen3TTSSpeakerEncoder.sanitize()` — strips `speaker_encoder.` prefix,
transposes 3D Conv1d weights from PyTorch `(O,I,W)` to MLX `(O,W,I)` format.

**Changes**:
- Add `sanitize(weights:)` static method to `Qwen3TTSSpeakerEncoder`
- In `Qwen3TTSModel.fromPretrained()`, filter and load speaker encoder weights
- Only instantiate speaker encoder when `config.ttsModelType == "base"` and
  `config.speakerEncoderConfig != nil`

**Files modified**:
- `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSSpeakerEncoder.swift`
- `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift` (init and fromPretrained)

**Verification**:
- Integration test (requires Base model download): Load Base model, verify
  `model.speakerEncoder` is non-nil with correct `encDim`
- Integration test: VoiceDesign model loads with `speakerEncoder == nil` (no regression)
- Build passes

**Dependencies**: Tasks 8, 2

---

### Task 11: Implement speech tokenizer encode() method [D3]

**Tier**: 3 — Encode + embed
**Parallel with**: Task 12

**What**: Add `encode(audio:) -> MLXArray` method to `Qwen3TTSSpeechTokenizer` wrapper
that delegates to the encoder model.

**Current state**: Only `decode()` and `streamingDecode()` methods exist.

**Changes**:
- Add `encode(audio:) -> MLXArray` method to `Qwen3TTSSpeechTokenizer`
- Guard on `encoderModel != nil`, throw descriptive error if not available
- Input: audio waveform `[batch, 1, samples]`
- Output: codec tokens `[batch, 16, time]`

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTSSpeechDecoder.swift`

**Verification**:
- Unit test: Calling `encode()` when encoder is nil throws appropriate error
- Integration test (requires Base model download): Encode test WAV, decode back,
  verify round-trip produces valid audio (lossy but recognizable)
- Build passes

**Dependencies**: Tasks 7, 9

---

### Task 12: Implement extractSpeakerEmbedding() [E3]

**Tier**: 3 — Encode + embed
**Parallel with**: Task 11

**What**: Implement the method that takes raw audio, computes mel spectrogram, and
runs it through the speaker encoder to get an x-vector embedding.

**Python reference**: `extract_speaker_embedding()` — computes mel spectrogram
(n_fft=1024, num_mels=128, sr=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000),
then calls `self.speaker_encoder(mels)`.

**Changes**:
- Add `extractSpeakerEmbedding(audio:) -> MLXArray` method to `Qwen3TTSModel`
- Use existing `melSpectrogram()` from `MLXAudioCore/DSP.swift` (verify parameter compatibility)
- Input: audio waveform `[samples]` or `[batch, samples]`
- Output: speaker embedding `[1, enc_dim]` (enc_dim=2048 for 1.7B models)

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification**:
- Integration test (requires Base model download): Extract speaker embedding from test
  WAV, verify output shape is `[1, 2048]`
- Integration test: Verify two different speakers produce different embeddings
- Build passes

**Dependencies**: Task 10

---

### Task 13: Implement prepareICLInputs() [F1]

**Tier**: 4 — ICL pipeline
**Sequential**: Must complete before Task 14

**What**: Port `_prepare_icl_generation_inputs()` from Python. This builds the combined
embedding sequence for ICL: reference text embeddings + target text embeddings overlaid
with reference codec embeddings.

**Python reference**: `_prepare_icl_generation_inputs()` in `qwen3_tts.py:406-590`

**Algorithm**:
1. Encode reference audio → `ref_codes [1, 16, ref_time]` via `speechTokenizer.encode()`
2. Tokenize ref_text: `<|im_start|>assistant\n{ref_text}<|im_end|>\n` → extract pure text tokens (skip role prefix/suffix)
3. Tokenize target_text: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n` → extract pure text tokens
4. Build `text_embed = textProjection(textEmbedding(ref_text_tokens + target_text_tokens)) + eos`
5. Build `codec_embed = codecBos + sum_of_all_16_codebook_embeddings(ref_codes)`
6. Non-streaming overlay mode: `text_with_codec_pad` then `codec_with_text_pad`
7. Extract x-vector speaker embedding (if speaker encoder available)
8. Build codec prefix: `[think/nothink, thinkBos, langId?, thinkEos, speaker?, pad, bos]`
9. Assemble: `role_embed + codec_prefix + icl_input_embed`

**Input**: `(text: String, refAudio: MLXArray, refText: String, language: String)`
**Output**: `(inputEmbeds: MLXArray, trailingTextHidden: MLXArray, ttsPadEmbed: MLXArray, refCodes: MLXArray)`

**Changes**:
- Add `prepareICLInputs()` method to `Qwen3TTSModel`

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification**:
- Integration test (requires Base model download): Prepare ICL inputs with a test WAV
  and text, verify output embedding tensor shapes are consistent
- Verify `refCodes` shape is `[1, 16, N]`
- Build passes

**Dependencies**: Tasks 11 (encode), 12 (speaker embedding), 6 (shared codec prefix logic)

---

### Task 14: Implement generateICL() [F2]

**Tier**: 4 — ICL pipeline
**Sequential**: Must complete after Task 13

**What**: Port `_generate_icl()` from Python. This is the ICL voice cloning generation
method — same autoregressive loop but with ICL-specific input preparation and post-processing.

**Python reference**: `_generate_icl()` in `qwen3_tts.py:1250-1548`

**Key ICL-specific behaviors**:
1. Minimum repetition penalty of 1.5 (stronger than standard 1.05) to prevent code
   degeneration with long reference audio prefills
2. After generation, **prepend** reference codes to generated codes before decoding:
   `full_codes = [ref_codes_transposed, gen_codes]` (so decoder sees full audio context)
3. **Proportional trimming**: After decoding, remove the reference audio portion:
   `cut = ref_len / total_len * audio_samples; audio = audio[cut:]`
4. Max tokens capped at `min(maxTokens, max(75, targetTokenCount * 6))`

**Changes**:
- Add `generateICL(text:refAudio:refText:language:temperature:topP:repetitionPenalty:maxTokens:)` method
- Calls `prepareICLInputs()` for input preparation
- Reuses shared autoregressive loop (same as generateBase/generateVoiceDesign)
- After loop: prepend ref_codes, decode full sequence, proportionally trim

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification**:
- Integration test (requires Base model download + test WAV): Clone a voice from
  reference audio, verify output is non-empty valid audio
- Integration test: Verify output audio length is reasonable (not including reference
  audio — proportional trimming works)
- Build passes

**Dependencies**: Tasks 13, 1 (ICL routing in generate())

---

### Task 15: Implement VoiceClonePrompt struct [G1]

**Tier**: 5 — Persistence
**Sequential**: Must complete before Task 16

**What**: Create a serializable struct that captures the pre-computed ICL prompt data
(encoded reference codes + speaker embedding) for reuse across multiple generation calls.

**Note**: This feature does NOT exist in the Python reference. It's a Swift-specific
convenience for VoxAlta to avoid re-encoding reference audio on every generation call.

**Changes**:
- Create `VoiceClonePrompt` struct (Codable):
  - `refCodes: MLXArray` — encoded reference audio codes `[1, 16, ref_time]`
  - `speakerEmbedding: MLXArray?` — x-vector embedding `[1, enc_dim]` (if available)
  - `refText: String` — the reference transcript
  - `language: String` — the language code used
- Add `createVoiceClonePrompt(refAudio:refText:language:) -> VoiceClonePrompt` method
- Add serialization to `Data` and deserialization from `Data`

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift` (or new file)

**Verification**:
- Integration test: Create clone prompt from ref audio, serialize to Data, deserialize,
  verify `refCodes` shapes match and values are equal
- Build passes

**Dependencies**: Tasks 11 (encode), 12 (speaker embedding)

---

### Task 16: Implement generateWithClonePrompt() [G2]

**Tier**: 5 — Persistence
**Sequential**: Must complete after Task 15

**What**: Generate audio using a pre-computed `VoiceClonePrompt` instead of raw
reference audio. Skips re-encoding and re-extracting speaker embedding.

**Changes**:
- Add `generate(text:language:clonePrompt:instruct:temperature:topP:repetitionPenalty:maxTokens:)` method
- Build ICL inputs from cached refCodes and speakerEmbedding instead of raw audio
- Rest of generation is same as generateICL()

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification**:
- Integration test: Create clone prompt, generate two different texts with same prompt,
  verify both produce valid audio
- Integration test: Compare audio from direct ICL vs clone prompt — should be bitwise
  identical given same random seed
- Build passes

**Dependencies**: Tasks 15, 14

---

### Task 17: Implement generateBase() [B2]

**Tier**: 6 — Standalone generation
**Parallel with**: Tasks 18, 19

**What**: Implement the standard Base model generation method — same autoregressive
loop as VoiceDesign but using `prepareBaseInputs()` instead of VoiceDesign inputs.

**Current state**: Only `generateVoiceDesign()` exists.

**Changes**:
- Add `generateBase(text:voice:language:instruct:refAudio:temperature:topP:repetitionPenalty:maxTokens:)` method
- Calls `prepareBaseInputs()` for input preparation
- Uses the shared `generateFromEmbeddings()` extracted in Task 5

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification**:
- Integration test (requires Base model download): Load `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`,
  generate audio from "Hello world", verify non-empty valid audio output
- Build passes
- Existing VoiceDesign tests still pass (no regression)

**Dependencies**: Tasks 1, 6, 5

---

### Task 18: Implement generateCustomVoice() [C1]

**Tier**: 6 — Standalone generation
**Parallel with**: Tasks 17, 19

**What**: Implement generation using predefined speakers from the CustomVoice model.
CustomVoice uses `spk_id` (9 named speakers with codec token IDs) and supports
optional instruct parameter.

**Current state**: Not implemented.

**Python reference**: `generate_custom_voice()` in `qwen3_tts.py` — validates speaker
name against `supported_speakers`, then delegates to `_generate_with_instruct()` which
calls `_prepare_generation_inputs(speaker=voice)`.

**Key insight**: CustomVoice generation is functionally identical to Base generation
with a speaker name. The only difference is validation (speaker must exist in spk_id)
and the model type check. Both use `_prepare_generation_inputs()`.

**Changes**:
- Add `generateCustomVoice(text:speaker:language:instruct:temperature:topP:repetitionPenalty:maxTokens:)` method
- Validate speaker name exists in `config.talkerConfig.spkId`
- Populate `supportedSpeakers` list from config on init
- Delegate to `prepareBaseInputs()` + shared `generateFromEmbeddings()`

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification**:
- Unit test: Calling with invalid speaker name throws error listing available speakers
- Integration test (requires CustomVoice model download): Load `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16`,
  generate audio with `speaker: "serena"`, verify non-empty valid audio
- Build passes

**Dependencies**: Tasks 1, 6, 5

---

### Task 19: Wire instruct parameter through Base/CustomVoice paths [H1]

**Tier**: 6 — Standalone generation
**Parallel with**: Tasks 17, 18

**What**: Ensure the `instruct` parameter (voice delivery hints like "speak in a whisper")
flows through Base and CustomVoice generation paths.

**Current state**: Instruct works for VoiceDesign (implemented in PR #23). The Python
`_prepare_generation_inputs()` already handles instruct for Base/CustomVoice too — it's
the same embedding format: `<|im_start|>user\n{instruct}<|im_end|>\n`.

**Key insight**: If `prepareBaseInputs()` (Task 6) is implemented correctly following
the Python reference, instruct is already handled there. This task is about verification
and wiring the parameter from the protocol through to the method.

**Changes**:
- Verify `voice` parameter in `SpeechGenerationModel.generate()` passes through as
  `instruct` for VoiceDesign but as `speaker` for CustomVoice
- Add `instruct` as a separate parameter or use `generationParameters` extension
- OR: for CustomVoice, `voice` = speaker name; for VoiceDesign, `voice` = instruct text;
  for Base with ICL, both voice and refAudio are used

**Files modified**: `Sources/MLXAudioTTS/Models/Qwen3TTS/Qwen3TTS.swift`

**Verification**:
- Integration test (requires model download): Generate two clips with same text,
  different instruct descriptions, verify audibly different output
- Build passes

**Dependencies**: Task 6

---

### Task 20: Integration tests [J5]

**Tier**: 7 — Ship
**Parallel with**: Task 21

**What**: Full end-to-end tests that load actual models and generate audio. These cannot
run in CI and should be grouped in a separate test suite.

**Test suites to create**:
- `Qwen3TTSBaseModelTests`:
  - Load Base model, generate audio, verify valid output
  - Load Base model, generate with different quantizations (bf16, 8bit, 4bit)
- `Qwen3TTSCustomVoiceTests`:
  - Load CustomVoice model, generate with named speaker, verify valid output
  - Verify invalid speaker name throws error
- `Qwen3TTSCloningTests`:
  - Load Base model with encoder, encode test WAV, verify codes shape
  - Clone voice from reference audio, verify valid output
  - Create clone prompt, serialize, deserialize, generate — verify valid output
- `Qwen3TTSSpeakerEncoderTests`:
  - Load Base model, extract speaker embedding, verify shape
  - Two different audio clips produce different embeddings

**Files**: `Tests/MLXAudioTTSTests.swift`

**Dependencies**: All implementation tasks (1-19)

---

### Task 21: Clean up fork changes [K1]

**Tier**: 7 — Ship
**Parallel with**: Task 20

**What**: Ensure all changes follow mlx-audio-swift coding conventions, remove any
VoxAlta-specific code, add inline documentation for public API methods.

**Checklist**:
- All new public methods have `///` doc comments with `- Parameters:` and `- Returns:`
- No VoxAlta references in source code
- File headers follow existing convention
- `// MARK: -` sections for organization
- No unnecessary `print()` statements in production code
- Build with no warnings: `xcodebuild build -scheme MLXAudio-Package -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO`

**Dependencies**: All implementation tasks (1-19)

---

### Task 22: Create upstream PR [K2]

**Tier**: 7 — Ship

**What**: Create PR against `Blaizzy/mlx-audio-swift` from intrusive-memory fork.

**PR content**:
- Title: "Add Qwen3-TTS Base, CustomVoice, and ICL voice cloning support"
- Description: What's added (Base model, CustomVoice, ICL cloning, speaker encoder,
  instruct, language), architecture overview, test plan
- Reference: PR #23 (VoiceDesign by INQTR), PR #17 (by smdesai), issue #9
- Include test instructions for reviewers

**Dependencies**: Task 21

---

## Build Rules

- **NEVER** use `swift build` — always `xcodebuild`
- macOS 26+ / Swift 6.2+
- Build: `xcodebuild build -scheme MLXAudio-Package -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO`
- Test: `xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO`
- CI runner: `macos-26`

## External Consumer

The VoxAlta project (`/Users/stovak/Projects/SwiftVoxAlta`) depends on this fork via
Swift Package Manager. VoxAlta's voice cloning feature blocks on Tasks 14 (ICL generation)
and 16 (clone prompt persistence) completing here.
