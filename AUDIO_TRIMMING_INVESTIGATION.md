# Audio Trimming Investigation

**Symptom**: Generated audio cuts off prematurely. Example: the line "The First Principal. All spellcraft begins with the practitioner's volition. The mirror does not act without the one who stands before it." is cut off after the word "All".

**Model path**: Qwen3TTS ICL (voice cloning via `generateWithClonePrompt` or `generateICL`)

---

## Suspect List

### SUSPECT 1: Token cap based on text length (effectiveMaxTokens)
- **File**: `Qwen3TTS.swift:1183`, `Qwen3TTSVoiceClonePrompt.swift:259`
- **Code**: `let effectiveMaxTokens = min(maxTokens, max(200, targetTokenCount * 12))`
- **How it could cause the bug**: The tokenizer encodes the text into tokens. If the text is short (say ~20 tokens), the cap becomes `max(200, 240) = 240` tokens. If the model needs more than 240 codec steps to speak the full sentence, the generation loop exits early at the `for step in 0 ..< maxTokens` boundary. The audio simply stops mid-word.
- **Why it's suspicious**: The 12x multiplier is a heuristic. Some voices speak slowly, some text has long pauses. A slow-speaking voice clone could easily exceed 12 codec tokens per text token.
- **Diagnostic**: Check stderr telemetry for `[ICL] targetTokens=X, effectiveMaxTokens=Y`. If generation produced exactly Y codes (no EOS hit), this is the culprit.
- **Status**: NOT YET ELIMINATED

---

### SUSPECT 2: Valid length trimming (audioLengths from decoder)
- **File**: `Qwen3TTS.swift:1213-1217`, `Qwen3TTSVoiceClonePrompt.swift:300-306`
- **Code**:
  ```swift
  let validLen = Int(audioLengths[0].item(Int32.self))
  if validLen > 0 && validLen < audioOut.dim(0) {
      audioOut = audioOut[..<validLen]
  }
  ```
- **How it could cause the bug**: `audioLengths` is computed by counting non-zero values in the first codebook: `(audioCodes[0..., 0..., 0] .> 0).sum(axis: 1) * decodeUpsampleRate`. If some valid codec tokens happen to have a zero in the first codebook (codebook index 0), they get miscounted as "invalid", producing a `validLen` that is shorter than the actual audio content.
- **Why it's suspicious**: The assumption that "codebook 0 == 0 means padding" may not hold for all generated sequences. A single zero in codebook 0 mid-sequence would cause the count to underestimate. The count sums ALL non-zero values but doesn't account for non-contiguous zeros — a zero at position 50 in a 200-token sequence means only 199 are counted, not 200.
- **Diagnostic**: Check `[ICL] decodedAudio=X samples, validLen=Y samples`. If Y < X and the trim is significant, this may be cutting real audio.
- **Status**: NOT YET ELIMINATED

---

### SUSPECT 3: Proportional reference audio trimming
- **File**: `Qwen3TTS.swift:1219-1225`, `Qwen3TTSVoiceClonePrompt.swift:308-320`
- **Code**:
  ```swift
  let refLen = refCodes.dim(2)  // refTime
  let totalLen = fullCodes.dim(1)  // refTime + genLen
  let cut = Int(Float(refLen) / Float(max(totalLen, 1)) * Float(audioOut.dim(0)))
  audioOut = audioOut[cut...]
  ```
- **How it could cause the bug**: This computes a proportional cut to remove the reference audio from the decoded waveform. The ratio `refLen / totalLen` is applied to the audio sample count. However, the decoder doesn't produce samples at a perfectly linear rate — the upsampling/transformer layers may produce non-uniform temporal distribution. If the reference portion decodes to fewer samples than the ratio predicts, the cut removes some of the generated audio.
- **Why it's suspicious**: This is an approximation. The actual boundary between reference and generated audio in the decoded waveform is not precisely at `ratio * totalSamples`. Codec decoders have receptive fields and cross-timestep dependencies that blur boundaries.
- **Diagnostic**: Check `[ICL] TRIM: refLen=X, totalLen=Y, ratio=Z, cut=W samples, remaining=R samples (~Ss)`. Compare the remaining seconds to expected speech duration.
- **Status**: NOT YET ELIMINATED

---

### SUSPECT 4: EOS token fired prematurely
- **File**: `Qwen3TTS.swift:809-812`
- **Code**: `if tokenId == eosTokenId { break }`
- **How it could cause the bug**: The model might emit the EOS token (`codec_eos_token_id = 2150`) before finishing the sentence. This is a model behavior issue — repetition penalty, temperature, or the input embeddings could cause the model to decide it's "done" early.
- **Why it's suspicious**: The recently removed forced repetition penalty (`max(repPenalty, 1.5)` → pass-through) means the effective penalty might now be lower (default 1.3 from `GenerateParameters`). Lower rep penalty = higher chance of degenerate patterns that trigger early EOS.
- **Diagnostic**: Count generated codes vs effectiveMaxTokens. If generated codes < effectiveMaxTokens, the model hit EOS early. Also check if the generation token count is suspiciously low for the text length.
- **Status**: NOT YET ELIMINATED

---

### SUSPECT 5: CLI default maxTokens = 1200
- **File**: `App.swift:192`
- **Code**: `var maxTokens: Int = 1200`
- **How it could cause the bug**: The CLI defaults to 1200 max tokens. For Qwen3TTS ICL, this becomes `min(1200, max(200, targetTokenCount * 12))`. For a sentence with ~20 text tokens, the cap is `min(1200, 240) = 240`. The 1200 only helps if the text is long enough that `targetTokenCount * 12 > 1200`, which requires ~100+ text tokens.
- **Why it's suspicious**: This interacts with Suspect 1. Even if the user passes `--max_tokens 4096`, the `effectiveMaxTokens` formula still caps it to 12x text length.
- **Diagnostic**: Check what `--max_tokens` value was passed. If default 1200, the formula is `min(1200, max(200, tokens*12))`.
- **Status**: NOT YET ELIMINATED

---

### SUSPECT 6: Memory cache limit set too low
- **File**: `App.swift:67`
- **Code**: `Memory.cacheLimit = 100 * 1024 * 1024` (100 MB)
- **How it could cause the bug**: The cache limit is 100 MB. The `flushGPUState()` function clears cache when it exceeds 256 MB. But with a 100 MB cache limit, MLX may be aggressively evicting intermediate tensors needed during long generation sequences, potentially causing numerical issues that lead to early EOS.
- **Why it's suspicious**: Qwen3-TTS generation allocates significant KV cache memory. A 100 MB cache limit is quite restrictive for a model that typically uses 500MB+ during generation.
- **Diagnostic**: Check `Memory.snapshot()` output. If cache is consistently at 100 MB, tensors are being evicted.
- **Status**: NOT YET ELIMINATED (lower probability)

---

### SUSPECT 7: Repetition penalty causing early EOS
- **File**: `Qwen3TTS.swift:1525-1538` (sampleToken), `GenerationTypes.swift:115`
- **Code**: Default `repetitionPenalty: Float = 1.3` with context window of 20 tokens
- **How it could cause the bug**: Repetition penalty modifies logits for previously seen tokens. In codec space, repeated tokens are common (sustained sounds, silence). If the penalty is too aggressive, it can suppress natural codec patterns and push the sampling distribution toward EOS.
- **Why it's suspicious**: The recent commit `314edad` removed the forced minimum of 1.5 for ICL, using the caller's value (1.3 default) instead. But 1.3 may still be too high for codec token generation where repetition is natural, or it could be that 1.5 was actually helping prevent issues and 1.3 is now too low (allowing degenerate loops that EOS-out).
- **Diagnostic**: Try generating with `--repetition_penalty 1.0` (disabled) to see if the cut-off disappears.
- **Status**: NOT YET ELIMINATED

---

### SUSPECT 8: GPU memory cleanup corrupting state
- **File**: `Qwen3TTS.swift:869-882`
- **Code**:
  ```swift
  func flushGPUState() {
      Stream.defaultStream(.gpu).synchronize()
      let cachedBytes = Memory.cacheMemory
      if cachedBytes > 256 * 1024 * 1024 {
          Memory.clearCache()
      }
  }
  ```
- **How it could cause the bug**: Called after each generation completes (`eval(audioOut); flushGPUState()`). If the previous generation's cleanup somehow affects the next generation's initial state — e.g., by evicting cached weights or pre-computed embeddings — the next generation could start in a bad state.
- **Why it's suspicious**: The commit `e5834f8` specifically added this to fix "Metal memory pool fragmentation corrupting subsequent generations." If the threshold is wrong or the timing is off, it could actually cause the problem it was trying to fix.
- **Diagnostic**: Only relevant if this is a **second or later** generation in a batch. First generation would be unaffected.
- **Status**: NOT YET ELIMINATED (only applies to sequential generations)

---

### SUSPECT 9: Periodic Memory.clearCache() during generation loop
- **File**: `Qwen3TTS.swift:861-862`
- **Code**: `if step > 0 && step % 50 == 0 { Memory.clearCache() }`
- **How it could cause the bug**: Every 50 steps, the GPU cache is cleared during autoregressive generation. This is meant to free memory, but if it evicts KV cache entries or intermediate computation results that are still needed, the model's hidden state becomes corrupt, leading to nonsensical outputs or premature EOS.
- **Why it's suspicious**: Clearing cache every 50 steps is aggressive. MLX's cache includes computed tensors that may be reused. If the KV cache relies on cached GPU memory, clearing it mid-generation could cause issues.
- **Diagnostic**: Check if the cutoff consistently happens around 50-token boundaries (50, 100, 150, 200 steps).
- **Status**: NOT YET ELIMINATED (moderate probability)

---

## Priority Order for Investigation

1. **SUSPECT 1 (token cap)** — Most likely. Simple math could produce a cap that's too low.
2. **SUSPECT 2 (valid length trim)** — High probability. The non-zero counting heuristic is fragile.
3. **SUSPECT 3 (proportional trim)** — High probability. Approximation that doesn't account for decoder non-linearity.
4. **SUSPECT 4 (premature EOS)** — Medium-high. Model behavior, but interacts with rep penalty.
5. **SUSPECT 7 (rep penalty)** — Medium. Changed recently, could have side effects.
6. **SUSPECT 5 (CLI maxTokens)** — Medium. Interacts with Suspect 1.
7. **SUSPECT 9 (periodic cache clear)** — Medium. Could corrupt mid-generation state.
8. **SUSPECT 8 (flushGPUState)** — Low-medium. Only for sequential generations.
9. **SUSPECT 6 (cache limit)** — Low. Unlikely to cause this specific symptom.

---

## Investigation Plan

For each suspect, we need to either:
- **ELIMINATE**: Prove it cannot cause the observed symptom
- **CONFIRM**: Prove it is the cause (or a contributing cause)

### Step 1: Capture telemetry
Run the failing generation with stderr visible and capture all `[ICL]` log lines. This gives us:
- `targetTokens`, `effectiveMaxTokens` → eliminates/confirms Suspects 1, 5
- `generatedCodes` count vs `effectiveMaxTokens` → eliminates/confirms Suspect 4
- `decodedAudio`, `validLen` → eliminates/confirms Suspect 2
- `TRIM: refLen, totalLen, ratio, cut, remaining` → eliminates/confirms Suspect 3

### Step 2: Compare generation count to cap
If `generatedCodes == effectiveMaxTokens`, the cap was hit (Suspect 1/5).
If `generatedCodes < effectiveMaxTokens`, EOS was hit early (Suspect 4/7).

### Step 3: Test with modified parameters
- Remove token cap entirely → isolates Suspect 1
- Set `repetitionPenalty = 1.0` → isolates Suspect 7
- Disable valid length trimming → isolates Suspect 2
- Disable proportional trimming → isolates Suspect 3
- Disable periodic cache clear → isolates Suspect 9

---

---

## Reproduction Run (2026-02-27)

**Command**: `produciesta generate ../podcast-confessions --episode episode_24.highland --provider voxalta --verbose --regenerate`

### Telemetry for the problematic line

```
Text (138 chars): "The First Principal. All spellcraft begins with the practitioner's volition. The mirror does not act..."
targetTokens=28, effectiveMaxTokens=336, effectiveRepPenalty=1.3
refCodes shape=[1, 16, 63], speakerEmbedding=true
generatedCodes=132 codes, refCodes=63 codes, fullCodes=195 codes
decodedAudio=374400 samples, validLen=374400 samples
TRIM: refLen=63, totalLen=195, ratio=0.323, cut=120960 samples, remaining=253440 samples (~10.56s)
finalAudio=253440 samples (~10.56s)
```

### Full episode statistics

- **60 dialogue lines** generated
- **Zero hit the token cap** — every generation completed with natural EOS
- No valid-length trimming occurred on any line (validLen == decodedAudio for all)
- Proportional reference trimming was the only trim applied (expected behavior for ICL)

### Suspects eliminated by this run

| Suspect | Status | Evidence |
|---------|--------|----------|
| #1 Token cap | **ELIMINATED** | 132 generated vs 336 cap. No line hit cap. |
| #2 Valid length trim | **ELIMINATED** | validLen=374400 == decodedAudio=374400. No trim. |
| #3 Proportional ref trim | **WORKING AS EXPECTED** | 32.3% cut, leaving 10.56s. Sufficient for text. |
| #4 Premature EOS | **ELIMINATED (this run)** | 132 codes → 10.56s is plenty. |
| #5 CLI maxTokens | **ELIMINATED** | maxTokens=16384 from Produciesta. |
| #7 Rep penalty | **ELIMINATED** | 1.3 penalty, model generated enough codes. |
| #9 Periodic cache clear | **ELIMINATED** | Only 132 steps, full audio produced. |
| #6 Cache limit | **N/A** | Produciesta manages its own memory. |
| #8 flushGPUState | **NOT TESTED** | Would only affect sequential gens. |

### Conclusion

**The mlx-audio-swift generation pipeline produced 10.56s of audio for the problematic line.** This is more than sufficient to speak the full sentence.

The cutoff after "ALL" observed in the original generation must be caused by one of:

1. **Stochastic model behavior** — A previous run may have hit premature EOS at a different point. TTS generation is non-deterministic; running again may produce different results.
2. **Downstream audio processing in Produciesta** — The audio stitching, format conversion (WAV→M4A), or timeline assembly in `HeadlessExporter.swift` may be truncating audio segments during concatenation.

### NEW SUSPECT: Produciesta HeadlessExporter

**File**: `Produciesta/ProduciestaCore/HeadlessExporter.swift`
- Reads individual WAV segments into `AVAudioPCMBuffer`
- Converts format using `AVAudioConverter`
- Concatenates buffers and writes to M4A with AAC codec

Potential issues:
- `AVAudioConverter.convert(to:error:)` may not convert all frames
- AAC encoder frame alignment could truncate trailing samples
- Buffer capacity calculation during sample rate conversion could be too small
- `AVAudioFile.read(into:)` may not read all samples if buffer is undersized

### Notes

- The text "The First Principal. All spellcraft begins with the practitioner's volition. The mirror does not act without the one who stands before it." is ~25 words. Tokenizer encoded it as 28 tokens. With 12x multiplier → cap of 336. Model generated 132 codes (natural EOS), producing 10.56s of final audio.
- The cutoff after "All" in the ORIGINAL run (~2-3s of audio) suggests either premature EOS in that specific run, or downstream truncation in Produciesta's audio assembly pipeline.
