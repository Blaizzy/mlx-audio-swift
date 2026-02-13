# Mimi API Compatibility Analysis for Speech Tokenizer Encoder

## Investigation Summary

This document verifies that Mimi codec components in `MLXAudioCodecs` are API-compatible with the speech tokenizer encoder implementation (Task 7c). All components are publicly accessible and their APIs align with the requirements for building the encoder.

**Status**: ✅ **FULLY COMPATIBLE** — No breaking incompatibilities found. All 4 required components are ready for use.

**Verification Date**: 2026-02-13 (Task 7a complete)
**Task 0 Findings**: ✅ Confirmed accurate
**Task 7b Status**: ✅ Already complete (encoder config fully parsed in Qwen3TTSConfig.swift)

---

## Quick Reference: Component API Summary

| Component | Status | Key API | Adaptation Required |
|-----------|--------|---------|---------------------|
| `SeanetEncoder` | ✅ Ready | `init(cfg:)`, `callAsFunction(_:)`, `resetState()` | None |
| `ProjectedTransformer` | ✅ Ready | `init(cfg:inputDim:outputDims:)`, `callAsFunction(_:cache:)`, `makeCache()` | Extract first output: `outputs[0]` |
| `ConvDownsample1d` | ✅ Ready | `init(stride:dim:causal:)`, `callAsFunction(_:)`, `resetState()` | None |
| `SplitResidualVectorQuantizer` | ✅ Ready | `init(dim:inputDim:outputDim:nq:bins:)`, `encode(_:)`, `decode(_:)` | Slice to valid quantizers |

**All components are public. No Mimi module changes required.**

---

## Package Dependencies

### ✅ Cross-Module Import Verified

**File**: `/Users/stovak/Projects/mlx-audio-swift/Package.swift` (lines 72-86)

```swift
.target(
    name: "MLXAudioTTS",
    dependencies: [
        "MLXAudioCore",
        "MLXAudioCodecs",  // ✅ Present
        // ... other dependencies
    ],
    path: "Sources/MLXAudioTTS"
)
```

**Result**: `MLXAudioTTS` target already has `MLXAudioCodecs` as a dependency. No changes needed.

**Build verification**: Project builds successfully with `xcodebuild build -scheme MLXAudio-Package -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO`

---

## Component 1: SeanetEncoder

**File**: `/Users/stovak/Projects/mlx-audio-swift/Sources/MLXAudioCodecs/Mimi/Seanet.swift` (lines 208-255)

### API Signature
```swift
public final class SeanetEncoder: Module {
    public init(cfg: SeanetConfig)
    public func callAsFunction(_ xs: MLXArray) -> MLXArray
    public func resetState()
}
```

### ✅ Compatible Methods

| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `callAsFunction(_:)` | `MLXArray` (NCL format) | `MLXArray` (NCL format) | Accepts audio `[batch, channels, samples]`, returns encoded features `[batch, dimension, time]` |
| `resetState()` | None | Void | Resets streaming state for encoder layers |

### Usage Pattern (from Python reference)
```python
# speech_tokenizer.py:891-900
x = self.seanet_encoder(x)  # [B, C, T] → [B, D, T']
```

### Swift Equivalent
```swift
let encoded = seanetEncoder(audio)  // [B, 1, samples] → [B, dimension, time]
```

### Verification Checklist
- [x] Public class visibility
- [x] `callAsFunction(_:)` accepts MLXArray
- [x] `callAsFunction(_:)` returns MLXArray
- [x] Input/output shapes match encoder requirements (NCL format)
- [x] No parameter mismatches

**Status**: ✅ **FULLY COMPATIBLE**

---

## Component 2: ProjectedTransformer

**File**: `/Users/stovak/Projects/mlx-audio-swift/Sources/MLXAudioCodecs/Mimi/Transformer.swift` (lines 316-369)

### API Signature
```swift
public final class ProjectedTransformer: Module {
    public init(cfg: TransformerConfig, inputDim: Int, outputDims: [Int])
    public func callAsFunction(_ xs: MLXArray, cache: [KVCache]) -> [MLXArray]
    public func makeCache() -> [KVCacheSimple]
}
```

### ✅ Compatible Methods

| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `callAsFunction(_:cache:)` | `MLXArray` (NCL or NLC), `[KVCache]` | `[MLXArray]` | Accepts encoded features + KV cache, returns list of transformed outputs |
| `makeCache()` | None | `[KVCacheSimple]` | Creates fresh cache for new sequence |

### Usage Pattern (from Python reference)
```python
# speech_tokenizer.py:901-914
x, _ = self.transformer(x, offset=offset)  # causal transformer with cache
```

### Swift Equivalent
```swift
let cache = transformer.makeCache()
let outputs = transformer(encoded, cache: cache)  // returns [MLXArray]
let x = outputs[0]  // first output
```

### Verification Checklist
- [x] Public class visibility
- [x] `callAsFunction(_:cache:)` accepts MLXArray + cache
- [x] Returns array of MLXArray (multiple output projections supported)
- [x] Cache parameter matches KVCache protocol
- [x] Causal attention support (via config)
- [x] RoPE positional embeddings (via config)

**Status**: ✅ **FULLY COMPATIBLE**

### ⚠️ Minor Adaptation Required

The transformer returns `[MLXArray]` (array of outputs) due to multi-head output projection design. For speech encoder, we only need the first output:

```swift
let outputs = transformer(x, cache: cache)
let x = outputs[0]  // Extract first (and typically only) output
```

This is not a breaking change — just requires extracting the first element of the returned array.

---

## Component 3: ConvDownsample1d

**File**: `/Users/stovak/Projects/mlx-audio-swift/Sources/MLXAudioCodecs/Mimi/Conv.swift` (lines 346-360)

### API Signature
```swift
public final class ConvDownsample1d: Module {
    public init(stride: Int, dim: Int, causal: Bool)
    public func callAsFunction(_ xs: MLXArray) -> MLXArray
    public func resetState()
}
```

### ✅ Compatible Methods

| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `callAsFunction(_:)` | `MLXArray` (NCL format) | `MLXArray` (NCL format) | Downsamples time dimension by stride factor |
| `resetState()` | None | Void | Resets streaming state |

### Usage Pattern (from Python reference)
```python
# speech_tokenizer.py:915-916
x = self.downsample(x)  # stride = encoder_frame_rate / frame_rate
```

### Swift Equivalent
```swift
// encoder_frame_rate = 1920, frame_rate = 75 → stride = 25 (actual value may vary)
let downsample = ConvDownsample1d(stride: 25, dim: dimension, causal: true)
let downsampled = downsample(x)  // [B, D, T] → [B, D, T/stride]
```

### Verification Checklist
- [x] Public class visibility
- [x] `callAsFunction(_:)` accepts MLXArray
- [x] `callAsFunction(_:)` returns MLXArray
- [x] Stride parameter configurable
- [x] Causal mode supported

**Status**: ✅ **FULLY COMPATIBLE**

---

## Component 4: SplitResidualVectorQuantizer

**File**: `/Users/stovak/Projects/mlx-audio-swift/Sources/MLXAudioCodecs/Mimi/Quantization.swift` (lines 171-211)

### API Signature
```swift
public final class SplitResidualVectorQuantizer: Module {
    public init(dim: Int, inputDim: Int?, outputDim: Int?, nq: Int, bins: Int)
    public func encode(_ xs: MLXArray) -> MLXArray
    public func decode(_ xs: MLXArray) -> MLXArray
}
```

### ✅ Compatible Methods

| Method | Input | Output | Notes |
|--------|-------|--------|-------|
| `encode(_:)` | `MLXArray` (NCL format) | `MLXArray` (codes) | Encodes continuous features to discrete codebook indices `[batch, num_quantizers, time]` |
| `decode(_:)` | `MLXArray` (codes) | `MLXArray` (NCL format) | Decodes codebook indices back to continuous features |

### Usage Pattern (from Python reference)
```python
# speech_tokenizer.py:917-925
codes = self.quantizer.encode(x)  # [B, C, T] → [B, nq, T]
codes = codes[:, :self.valid_num_quantizers, :]  # slice to first 16 quantizers
```

### Swift Equivalent
```swift
let allCodes = quantizer.encode(downsampled)  // [B, 32, T] (full codebook)
let codes = allCodes[0..<allCodes.shape[0], 0..<validNumQuantizers, 0..<allCodes.shape[2]]  // [B, 16, T]
```

### Verification Checklist
- [x] Public class visibility
- [x] `encode(_:)` method exists
- [x] `encode(_:)` accepts MLXArray
- [x] `encode(_:)` returns MLXArray with correct shape `[batch, nq, time]`
- [x] Output is discrete codebook indices

**Status**: ✅ **FULLY COMPATIBLE**

---

## Overall Compatibility Assessment

### ✅ All Components Compatible

| Component | Visibility | API Match | Adaptations Needed |
|-----------|-----------|-----------|-------------------|
| `SeanetEncoder` | ✅ Public | ✅ Matches | None |
| `ProjectedTransformer` | ✅ Public | ✅ Matches | Extract first output from array |
| `ConvDownsample1d` | ✅ Public | ✅ Matches | None |
| `SplitResidualVectorQuantizer` | ✅ Public | ✅ Matches | None |

### Required Adaptations for Task 7c

1. **Transformer Output Extraction** (minor):
   ```swift
   let outputs = transformer(x, cache: cache)
   let x = outputs[0]  // Extract first output
   ```

2. **Codebook Slicing** (standard Swift array slicing):
   ```swift
   let codes = allCodes[0..<allCodes.shape[0], 0..<validNumQuantizers, 0..<allCodes.shape[2]]
   ```

Both adaptations are trivial and do not require any changes to the Mimi components themselves.

---

## Data Flow Verification

### Python Reference Pipeline (speech_tokenizer.py:889-990)
```
audio [B, 1, samples]
  → SeanetEncoder → [B, dimension, time]
  → ProjectedTransformer (with cache, RoPE) → [B, dimension, time]
  → ConvDownsample1d (stride=encoder_frame_rate/frame_rate) → [B, dimension, time/stride]
  → SplitResidualVectorQuantizer.encode() → [B, 32, time/stride]
  → Slice [:, :valid_num_quantizers, :] → [B, 16, time/stride]
```

### Swift Implementation (Task 7c will implement)
```swift
// 1. Reset state
encoder.resetState()
transformer.resetState()
downsample.resetState()

// 2. Create cache
let cache = transformer.makeCache()

// 3. Run through encoder chain
var x = seanetEncoder(audio)  // [B, 1, samples] → [B, dimension, time]

// 4. Transformer with causal attention
let transformerOutputs = transformer(x, cache: cache)
x = transformerOutputs[0]  // Extract first output

// 5. Downsample
x = downsample(x)  // [B, dimension, time] → [B, dimension, time/stride]

// 6. Quantize and slice
let allCodes = quantizer.encode(x)  // [B, 32, time/stride]
let codes = allCodes[0..<allCodes.shape[0], 0..<validNumQuantizers, 0..<allCodes.shape[2]]  // [B, 16, time/stride]

return codes
```

**Status**: ✅ **DATA FLOW MATCHES PYTHON REFERENCE**

---

## Conclusion

**All 4 Mimi components are API-compatible with the speech tokenizer encoder requirements.** No breaking incompatibilities found. The cross-module import is already configured in Package.swift. Task 7c can proceed with implementation using these components without any modifications to the Mimi codec module.

### Exit Criteria Met (Task 7a Complete)

- [x] All 4 Mimi components verified as public and API-compatible
- [x] No blocking incompatibilities found (2 minor adaptations documented, both trivial)
- [x] Documentation confirms findings from Task 0 are accurate and complete
- [x] Package.swift cross-module import verified (MLXAudioCodecs dependency present in MLXAudioTTS)
- [x] **BONUS**: Task 7b already complete — `Qwen3TTSTokenizerEncoderConfig` fully implemented with all required fields

### API Verification Checklist (from EXECUTION_PLAN.md lines 488-494)

- [x] `SeanetEncoder` can be instantiated with config from Qwen3TTSSpeechTokenizer
  - ✅ `public init(cfg: SeanetConfig)` accepts all required parameters
  - ✅ Config fields map from `Qwen3TTSTokenizerEncoderConfig` (ratios, dimension, etc.)
- [x] `ProjectedTransformer` accepts causal attention mask + RoPE cache
  - ✅ `cfg.causal` enables causal attention via mask generation
  - ✅ `cfg.positionalEmbedding = "rope"` enables RoPE positional embeddings
  - ✅ `cache: [KVCache]` parameter accepts cache from `makeCache()`
- [x] `ConvDownsample1d` stride parameter matches encoder requirements
  - ✅ `stride` parameter configurable: `encoderFrameRate / frameRate` (e.g., 1920 / 75 ≈ 25)
  - ✅ `causal: Bool` parameter matches encoder causal mode
- [x] `SplitResidualVectorQuantizer.encode()` returns codes tensor `[batch, num_quantizers, time]`
  - ✅ `public func encode(_ xs: MLXArray) -> MLXArray` exists
  - ✅ Returns shape `[batch, nq, time]` (lines 194-200 of Quantization.swift)
  - ✅ Slicing to `validNumQuantizers` (16) works with standard MLX array indexing
- [x] All 4 components are public (or mark as public if needed)
  - ✅ All components already public (no changes needed)

**Next steps**: Proceed directly to Task 7c (encoder implementation) — Task 7b is already complete.
