# mlx-audio-swift Acervo Integration Requirements

**Date**: 2026-04-17  
**Status**: ⚠️ PARTIAL — Legacy HF discovery pattern in use  
**Pattern**: Pattern C (Dynamic File Discovery) → migrate to Pattern B (ComponentDescriptor)

---

## Current Implementation

mlx-audio-swift discovers model files at runtime via HuggingFace Hub API, then downloads them using SwiftAcervo.

**Location**: `Sources/MLXAudioCore/ModelResolver.swift`

```swift
// 1. Check local cache
if Acervo.isModelAvailable(modelId) {
    return try Acervo.modelDirectory(for: modelId)
}

// 2. Discover files via HuggingFace Hub API
let files = try await huggingFaceClient.listFiles(in: repoID)

// 3. Build dynamic file list
var filesToDownload = Set(requiredFiles)
for entry in files where extensions.contains(entry.extension) {
    filesToDownload.insert(entry.path)
}

// 4. Download via Acervo
try await Acervo.download(
    modelId,
    files: Array(filesToDownload).sorted(),
    progress: { progress in ... }
)
```

**Model Coverage**:
- Mimi (codec)
- Snac (codec)
- ASR (speech-to-text) encoders
- Other audio models

### Gaps vs. Best Pattern

| Issue | Impact | Severity |
|-------|--------|----------|
| **Dynamic file discovery at runtime** | Adds latency on first use; cannot pre-validate manifest | ⚠️ Medium |
| **No ComponentDescriptor registration** | Files not pre-declared; harder to estimate download size | ⚠️ Medium |
| **No memory requirements declared** | App cannot plan GPU allocation upfront | ⚠️ Medium |
| **HF API calls on every startup** | Unnecessary network requests even if model cached | ⚠️ Medium |
| **Harder to test** | Mock discovery instead of using declared descriptors | ⚠️ Low |

---

## Required Changes

### R1: Determine Model File Stability (Prerequisite)

**AUDIT COMPLETE — See detailed findings below**

#### Audit Results (April 17, 2026)

**Stable Models (Suitable for ComponentDescriptor)**:

1. **SNAC Codec** ✅
   - Repository: `mlx-community/snac_24khz`
   - Files: `config.json`, `model.safetensors`
   - Used by: Qwen3TTS, LlamaTTS (hard-coded)
   - Stability: Hard-coded repo path, no variants
   - Reference: Tests/MLXAudioCodecsTests.swift:45

2. **Mimi Codec** ✅
   - Repository: `kyutai/moshiko-pytorch-bf16`
   - Files: Single file `tokenizer-e351c8d8-checkpoint125.safetensors`
   - Used by: MarvisTTSModel (hard-coded)
   - Stability: Hard-coded repo path, exact file name
   - Reference: Tests/MLXAudioCodecsTests.swift:96

**Dynamic Models (Keep HF Discovery)**:

- **Vocos, Encodec, DACVAE**: User-specified repo IDs at runtime
- **TTS Models** (Qwen3TTS, Qwen3, LlamaTTS, Soprano, PocketTTS, MarvisTTS): User-specified repos
- **ASR Models** (GLMASRModel, Qwen3ASR): User-specified repos

**Summary**:
```
Stable (ComponentDescriptor): 2 models (SNAC, Mimi)
Dynamic (HF discovery):       5+ model families
```

**Recommendation**: Migrate SNAC and Mimi to ComponentDescriptor; keep HF discovery for all others

---

### R2: Migrate Stable Models to ComponentDescriptor (High Priority)

For models with stable file lists, register ComponentDescriptor at module init:

**Location**: `Sources/MLXAudioCore/ModelRegistry.swift` (new or existing init)

```swift
import SwiftAcervo

private let _registerAudioComponents = {
    let descriptors = [
        ComponentDescriptor(
            id: "mimi-codec-48khz",
            type: .audioModel,
            displayName: "Mimi Codec (48 kHz)",
            repoId: "luctoma/mimi",
            files: [
                "config.json",
                "model.safetensors",
                "tokenizer.json"
            ],
            estimatedSizeBytes: 500_000_000, // ~500 MB
            minimumMemoryBytes: 2_000_000_000 // 2 GB
        ),
        ComponentDescriptor(
            id: "snac-codec-16khz",
            type: .audioModel,
            displayName: "Snac Codec (16 kHz)",
            repoId: "hubblestack/snac",
            files: [
                "config.json",
                "model.safetensors",
                "tokenizer.json"
            ],
            estimatedSizeBytes: 300_000_000, // ~300 MB
            minimumMemoryBytes: 1_500_000_000 // 1.5 GB
        ),
        // ... other stable models
    ]
    Acervo.register(descriptors)
}()
```

**Acceptance Criteria**:
- [ ] Descriptors registered for all stable models
- [ ] File lists match current HuggingFace repository
- [ ] estimatedSizeBytes is accurate (within 10%)
- [ ] minimumMemoryBytes covers typical inference workload

---

### R3: Update ModelResolver to Use ensureComponentReady() (High Priority)

Replace `Acervo.download()` calls with `ensureComponentReady()`:

**Before**:
```swift
try await Acervo.download(modelId, files: discoveredFiles, progress: callback)
```

**After**:
```swift
try await Acervo.ensureComponentReady(componentId, progress: callback)
```

**Changes**:
- [ ] ModelResolver.resolveModelPath() uses ensureComponentReady()
- [ ] Skip HuggingFace API discovery for registered models
- [ ] Fall back to HF discovery only if descriptor not found
- [ ] Update error handling to catch AcervoError

**Code Pattern**:
```swift
// Check if model is registered as a component
if let descriptor = Acervo.registeredComponent(modelId) {
    // Use ComponentDescriptor path
    try await Acervo.ensureComponentReady(modelId, progress: callback)
} else {
    // Fall back to HF discovery for new/unregistered models
    let files = try await discoverFilesFromHuggingFace(modelId)
    try await Acervo.download(modelId, files: files, progress: callback)
}
```

**Acceptance Criteria**:
- [ ] ensureComponentReady() used for all registered models
- [ ] HF discovery skipped for registered models (avoids network calls)
- [ ] Download still works for new/unregistered models via HF fallback
- [ ] Progress callbacks use structured AcervoProgress

---

### R4: Update Memory Pre-Validation (Optional, Recommended)

Add memory validation before loading models:

```swift
try await MemoryManager.validateCanLoad(
    estimatedBytes: descriptor.minimumMemoryBytes
)

// If validation passes, proceed with load
try await Acervo.ensureComponentReady(modelId)
let model = try await loadModel(from: modelPath)
```

**Benefit**: Early fail if device memory is insufficient (rather than OOM crash)

**Acceptance Criteria**:
- [ ] Memory check happens before download
- [ ] Clear error message if memory is insufficient
- [ ] User can retry after freeing memory

---

### R5: Update Tests (Medium Priority)

Update test fixtures and CI to use descriptors:

- [ ] Unit tests mock ComponentDescriptor instead of HF API
- [ ] Integration tests use real Acervo cache (no live HF calls)
- [ ] CI tests marked as `requires-model-download` can still use fallback
- [ ] Add test that verifies registered models skip HF discovery

---

### R6: Update Documentation (Low Priority)

- [ ] AGENTS.md documents supported models and their variants
- [ ] README.md explains download behavior (first-run auto-download)
- [ ] CLAUDE.md updated if build/test instructions changed
- [ ] Mention that HuggingFace fallback still works for new models

---

## Compliance Checklist

Track implementation progress:

- [x] **R1**: Audio model file lists audited and documented ✅ COMPLETE (Apr 17, 2026)
  - [x] Stable models identified: SNAC, Mimi
  - [x] Dynamic models documented: Vocos, Encodec, DACVAE, TTS, ASR
  - [x] Audit report: See REQUIREMENTS.md § R1
- [ ] **R2**: ComponentDescriptors registered at module init (NEXT SORTIE: 2.1)
  - [ ] SNAC descriptor created
  - [ ] Mimi descriptor created
  - [ ] File lists are accurate
  - [ ] Memory requirements specified
- [ ] **R3**: ModelResolver updated (NEXT SORTIE: 2.2)
  - [ ] Uses ensureComponentReady() for SNAC and Mimi
  - [ ] HF discovery fallback for all other models
  - [ ] Error handling updated
- [ ] **R4**: Memory validation (optional, future consideration)
  - [ ] Pre-validates before download/load
  - [ ] Clear error messages
- [ ] **R5**: Tests updated (NEXT SORTIE: 3.1)
  - [ ] Unit tests for descriptor registration
  - [ ] CI uses fallback for dynamic models
- [ ] **R6**: Documentation updated (NEXT SORTIE: 3.2)
  - [ ] AGENTS.md documents stable models
  - [ ] README.md explains ComponentDescriptor pattern
  - [ ] CLAUDE.md updated if needed

---

## Migration Path

**Effort**: 2–3 hours total

### Phase 1: Audit (30 min)
- Identify stable vs. dynamic models
- Document findings in this file (R1 section)

### Phase 2: Register Descriptors (1 hour)
- Create descriptors for all stable models
- Register at module init
- Test descriptor registration

### Phase 3: Update Resolver (1 hour)
- Update ModelResolver to use ensureComponentReady()
- Add fallback for unregistered models
- Test download path

### Phase 4: Update Tests & Docs (30 min–1 hour)
- Update unit tests
- Update AGENTS.md and README.md
- Verify CI still passes

---

## Reference

See `/Users/stovak/Projects/ACERVO_INTEGRATION_REQUIREMENTS.md` (master reference) for:
- Pattern C (Dynamic File Discovery) details
- Pattern B (ComponentDescriptor) ideal implementation
- Shared error handling policies
- SwiftAcervo/AGENTS.md for complete API reference
