---
title: "mlx-audio-swift — SwiftAcervo v2 Integration Requirements"
date: 2026-04-18
source: "ACERVO_CONSUMER_AUDIT.md (lines 96–114)"
priority: "🔴 HIGH"
version: "2.0"
status: "READY FOR EXECUTION"
---

# mlx-audio-swift — SwiftAcervo v2 Integration Requirements

**Mission Supervisor Briefing**  
**Audit Date**: April 18, 2026  
**Classification**: Hybrid Pattern (Partial Registration, Zero v2 Access)

---

## Audit Findings Summary

| Finding | Status | Severity |
|---------|--------|----------|
| **Component Registration** | ⚠️ PARTIAL | Descriptors defined for P1 (SNAC, Mimi) but incomplete |
| **Registry Integration** | ⚠️ PARTIAL | `Acervo.register()` called but HF fallback still in use |
| **v1 Path Access** | ✅ ACTIVE | `ModelResolver.resolve()` returns URLs for all models |
| **v2 Access (`withComponentAccess`)** | ❌ MISSING | Never invoked; file access happens outside managed closure |
| **Integrity Verification** | ✅ PARTIAL | Checksums declared for P1 but verification skipped |

**Overall Status**: 🔴 **HIGH PRIORITY** — Hybrid pattern, no v2 access

---

## Current Architecture

**Pattern**: Pattern C (Dynamic File Discovery with Partial Registration)

**Location**: `Sources/MLXAudioCore/ModelResolver.swift`, `Sources/MLXAudioCore/AudioModelManager.swift`

**Issue Map**:
- Line 102: Defines `ComponentDescriptor` for SNAC/Mimi but doesn't leverage them
- Line 103: `Acervo.register()` called, but HuggingFace fallback still active
- Line 104: `ModelResolver.resolve()` returns URLs—bypasses Acervo's managed access layer
- Line 105: `withComponentAccess()` never used; `Acervo.modelDirectory()` used directly
- Line 106: SHA-256 checksums declared but not verified on access

---

## Work Items (Sorties)

### Sortie 4.1: Remove HuggingFace Fallback for P1 Models

**Objective**: Eliminate `huggingFaceClient.listFiles()` calls for SNAC/Mimi

**Entry Criteria**:
- Audit complete (ACERVO_CONSUMER_AUDIT.md, lines 96–114)
- ComponentDescriptor exists in codebase

**Changes Needed**:
1. Remove HF API discovery for `snac_24khz` and `mimi-codec-48khz`
2. Hard-code file lists in ComponentDescriptor (already stable)
3. Eliminate `Acervo.download()` calls for P1 models
4. Verify no test fixtures call HF API for P1 models

**Exit Criteria**:
- No HF API calls for SNAC/Mimi in production code
- Unit tests pass (no model downloads required)
- P1 models load via `Acervo.ensureComponentReady()` only

**Implementation Hint**:
```swift
// Before: Discovers files at runtime via HF
let files = try await huggingFaceClient.listFiles(in: "mlx-community/snac_24khz")

// After: Uses pre-declared descriptor
try await Acervo.ensureComponentReady("snac-24khz")
```

---

### Sortie 4.2: Replace ModelResolver.resolve() with loadAudioModel()

**Objective**: Create `loadAudioModel()` that uses `withComponentAccess()` closure pattern

**Entry Criteria**:
- Sortie 4.1 complete (HF fallback removed for P1)
- Understand current `ModelResolver.resolve()` return signature

**Changes Needed**:
1. Create new `AudioModelManager.loadAudioModel()` method
2. Takes `componentId` (String) and returns model via closure
3. Wraps all file access in `withComponentAccess()` closure
4. Validates SHA-256 checksums before returning
5. Remove or deprecate `ModelResolver.resolve()` for P1 models

**Exit Criteria**:
- `loadAudioModel()` method exists and is called from inference paths
- All P1 file access happens inside `withComponentAccess()` closure
- Checksums validated on every access
- No direct `Acervo.modelDirectory()` calls for P1 models

**Implementation Pattern**:
```swift
func loadAudioModel<T>(_ componentId: String, load: (_ modelPath: URL) throws -> T) async throws -> T {
    return try await Acervo.withComponentAccess(componentId) { (metadata, modelPath) in
        // Verify checksums
        try validateChecksums(at: modelPath, expected: metadata.checksums)
        // Load model
        return try load(modelPath)
    }
}
```

---

### Sortie 4.3: Move File Discovery into AudioModelManager

**Objective**: Consolidate model file access; remove scattered ModelResolver calls

**Entry Criteria**:
- Sortie 4.2 complete (`loadAudioModel()` exists)
- Inventory of all sites that call `ModelResolver.resolve()`

**Changes Needed**:
1. Identify all callsites of `ModelResolver.resolve()` for P1 models
2. Redirect them to `AudioModelManager.loadAudioModel()`
3. For P2 models (user-specified repos), keep HF discovery but delegate to new method
4. Remove unused `ModelResolver` methods (or deprecate)

**Exit Criteria**:
- No direct file access outside AudioModelManager
- ModelResolver used only for P2 (dynamic) models
- All P1 access routed through `loadAudioModel()` + `withComponentAccess()`

---

### Sortie 4.4: Register P2 Models or Remove from Pipeline

**Objective**: Decide P2 model fate: register as ComponentDescriptor or keep as HF-fallback-only

**Entry Criteria**:
- Sortie 4.3 complete (P1 consolidated)
- Audit of P2 usage patterns (Vocos, Encodec, DACVAE, TTS, ASR)

**Changes Needed** (Choose One):
- **Option A**: Register P2 models with ComponentDescriptor (stable repos only)
- **Option B**: Deprecate P2 support; keep P1 only
- **Option C**: Keep P2 as HF fallback; clearly document as "advanced/unsupported"

**Exit Criteria**:
- P2 model handling documented in REQUIREMENTS.md § P2 Strategy
- No ambiguity about supported vs. experimental models
- Tests reflect decision

**Recommendation**: Option C (HF fallback for P2) aligns with existing usage patterns

---

### Sortie 4.5: Update Tests and CI

**Objective**: Verify v2 integration works; remove reliance on HF API in CI

**Entry Criteria**:
- Sortie 4.4 complete (P2 strategy decided)
- Understand current CI test matrix

**Changes Needed**:
1. Mock `withComponentAccess()` in unit tests (no real downloads)
2. Add integration test: verify checksums are validated
3. Update CI to NOT call HF API for P1 models
4. Mark P2 tests as `requires-network` or `requires-model-download`
5. Verify tests pass without model downloads (P1 only)

**Exit Criteria**:
- Unit tests pass with mocked Acervo (no network)
- Integration tests validate checksums
- CI test matrix updated to match new paths

---

## Reference & Links

**Master Requirements**: `/Users/stovak/Projects/REQUIREMENTS.md`

**Audit Source**: `/Users/stovak/Projects/ACERVO_CONSUMER_AUDIT.md` (lines 96–114)

**SwiftAcervo API Reference**: `/Users/stovak/Projects/SwiftAcervo/AGENTS.md`

**Related Projects**:
- SwiftProyecto (Phi-3 LLM) — ✅ Completed Wave 2
- SwiftBruja (Qwen3 LLM) — ✅ Completed Wave 2
- SwiftVoxAlta (Qwen3 TTS) — 🟡 Medium priority (registration excellent, access missing)

---

## Acceptance Checklist

Complete integration = All criteria met

- [ ] **P1 Registration**: SNAC & Mimi fully registered with stable descriptors
- [ ] **P1 Access**: All SNAC/Mimi access via `withComponentAccess()` closure
- [ ] **P1 Verification**: Checksums validated on every access
- [ ] **HF Fallback Removed**: No HF API calls for P1 models
- [ ] **P2 Strategy**: Decided and documented (recommended: HF fallback)
- [ ] **Tests Updated**: Unit & integration tests reflect v2 pattern
- [ ] **CI Green**: All tests pass; no model downloads in fast test path
- [ ] **Documentation**: AGENTS.md/README.md updated with new patterns

---

## Mission Timeline

**Estimated Effort**: 3–4 hours total

| Sortie | Task | Est. Time | Blocker |
|--------|------|-----------|---------|
| 4.1 | Remove HF fallback (P1) | 45 min | None |
| 4.2 | Create `loadAudioModel()` | 1 hour | 4.1 complete |
| 4.3 | Move file discovery | 1 hour | 4.2 complete |
| 4.4 | P2 model decision | 30 min | 4.3 complete |
| 4.5 | Update tests & CI | 1 hour | 4.4 complete |

**Recommended Sequence**: 4.1 → 4.2 → 4.3 → 4.4 → 4.5

---

**Status**: READY FOR EXECUTION  
**Next Agent Action**: Claim Sortie 4.1 via mission-supervisor skill
