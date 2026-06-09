# autoresearch results — Streaming CoreML (Nemotron, Option A)

Metric: cosine(CoreML stream, torch stream) + transcript token-match. Direction: higher_is_better. Target: ≥0.999.
Env: alex-mac (M-series, ANE), uv venv NeMo 2.x + coremltools + torch 2.10. Clip: conversational_a.wav (13.3s).

| iter | change | metric | check2 (transcript) | status |
|---|---|---|---|---|
| 0a | naive right-pad NeMo variable chunks → F=121 (CHECK2 vs offline) | 0.645 | — | discard (padding corrupts; length baked) |
| 0b | uniform-121 feeding, CHECK2 vs **offline chunked-limited** | 0.012 | — | discard (wrong reference — frame-cosine mirage) |
| 0c | wrong att_context [56,13] (model needs [70,13]) | 0.012 | — | fix arg |
| 0d | **transcript test**: uniform-121 vs NeMo native stream (torch) | — | identical tokens | keep (recipe validated) |
| 0e | convert blocked: coremltools `aten::Int` on (1,)-const (torch 2.10) | crash | — | patch op (squeeze→scalar) |
| **1** | **uniform-121 + CoreML, CHECK1 + transcript** | **0.999922** | **True (token-identical)** | **KEEP — target met** |

## What works (validated, Python/CoreML)
- **Recipe (Swift-replicable):** F = pre_encode_cache_size[1] + chunk_size[1] = 9 + 112 = 121. Feed every
  chunk as `[9 prev-mel ++ 112 new-mel]`, stride 112 (zeros for first prepend & last new-tail). The model
  (`cache_aware_stream_step`) applies `drop_extra_pre_encoded` internally → keep its output frames as-is;
  last chunk keeps only `ceil(real_new/8)` valid frames. att_context **[70,13]** (NOT [56,13]).
- **Caches threaded as explicit I/O** (functional, no MLState): cache_last_channel [L,1,70,d] fp16,
  cache_last_time [L,1,d,Cconv] fp16, cache_last_channel_len [1] int32. Feed in → read `new_*` → feed back.
- **Convert:** `aten::Int` coremltools patch (override squeezes (1,)-const to scalar) — avoids torch≤2.7 pin.
- **Result:** CHECK1 cosine 0.9999, transcript token-identical to torch native streaming, on a 13s clip.

## Key lessons (the 3 wrong turns)
1. **Frame-cosine-vs-offline is a MIRAGE for streaming ASR.** Streaming has less right-context than any
   offline forward → frames differ (0.012–0.909) even when the DECODED TOKENS are identical. Validate
   transcript-first; use frame cosine ONLY for CoreML-vs-torch-on-identical-feeding (fp16 fidelity).
2. **Fixed-shape ANE can't honor a per-chunk true length.** NeMo's buffer feeds variable sizes (105/57);
   right-padding them to F with length=F corrupts output (0.645). The fix is a **uniform** feeding at the
   exact traced structure, NOT padding NeMo's variable chunks.
3. **att_context is model-specific:** EN streaming 0.6B uses left=70; the 3.5 multilingual used 56.

## Swift integration — BUILT + ON-ANE MECHANICS VALIDATED (M1 Max, 2026-06-09)
- `NemotronCoreMLStreamingEncoder` (new): loads the 4-in/4-out functional `.mlpackage`, threads the 3
  caches (feed in → read `new_*` → feed back), reads `encoded` stride-aware → `[1, dModel, T']`.
- `NemotronASRModel.cacheAwareStreamEncodeCoreML` (new): replicates the uniform-121 feeding (concat
  `[pre-zeros, prev-mel, new-mel, tail-zeros]`), crops to `ceil(realNew/sf)`, transposes, `applyPrompt`,
  `onChunk` — same contract as the MLX path; wired into `generateStream` (no silent fallback mid-stream).
- `enableCoreMLStreamingEncoder(modelURL:)` enable hook; `streamingCoreMLEncoder` property.
- Build green; 3 CI-safe tests pass. **On-ANE probe `nemotron-stream-probe` PASS**: loads, finite
  `[1,1024,14]`, caches thread (chunk2 ≠ chunk1), `reset()` exact (maxDiff 0.0).
- **BUG caught by the probe — the stride trap, again:** after a `step()` the cache vars point at the
  model's **stride-padded** output; the first `reset()` zeroed them *sequentially* (skipping padded gaps)
  → leftover non-zero cache → maxDiff 0.28. Fix: `reset()` allocates **fresh contiguous** zero arrays.
  (Feeding a strided array *back as input* is fine — CoreML honors its strides; only manual zero didn't.)

## 3.5 MATCHED-WEIGHT e2e — PASS (M1 Max, Swift CLI, 2026-06-09)
- Converted the **3.5 multilingual** streaming encoder on NeMo **2.8** prerelease (`uv pip install
  --prerelease=allow "nemo-toolkit[asr]>=2.8.0rc0"` — has `EncDecRNNTBPEModelWithPrompt`; stable 2.7.3
  doesn't). att_context **[56,13]** (3.5 left=56, EN was 70) → F=121, attnCache 56. `_remote_35*.sh`.
- `mlx-audio-swift-stt --model mlx-community/nemotron-3.5-asr-streaming-0.6b --stream` (MLX) vs same
  `+ --coreml-stream-encoder nemotron_35_stream_func.mlpackage` (ANE) on conversational_a.wav (13s):
  **transcripts word-for-word identical**, only diff = trailing "." — the final-chunk zero-pad subtly
  perturbs the last real frame's conv (fixed-shape can't feed a variable final size). Cosmetic.
- Swift derives all stream params from the loaded config (`attnCache = defaultAttContextSize.first` = 56),
  so the same code matches both EN (70) and 3.5 (56). `--coreml-stream-encoder` CLI flag added.

## ANE residency — FIXED (the critical catch, M1 Max anemll-profile 0.4.1)
- With `MLComputeUnits.all` the stream model runs **1.9% ANE / focus GPU** — the int32 mask /
  cache-length ops (`pad_mask`, `logical_and`, casts on `cache_last_channel_len`) nudge `.all` to place
  the WHOLE graph on the GPU → **no power win** (silent). The offline model dodges this (no cache-len masks).
- With **`MLComputeUnits.cpuAndNeuralEngine`**: **100% ANE** (cost-weighted; 1585 ANE / 15 CPU ops,
  1 negligible mask island ~0.005 ms), ~25 ms/chunk (~45× RTF). → streaming encoder default changed to
  `.cpuAndNeuralEngine`. Forced-ANE transcript: word-identical to MLX, punctuation at the fp16 floor.

## HF upload + --ane streaming auto-download — DONE
- Uploaded https://huggingface.co/beshkenadze/nemotron-3.5-asr-streaming-0.6b-coreml-ane-stream (1.1G).
- `defaultANEStreamingEncoderRepo` + `enableCoreMLStreamingEncoder(repo:)` (reuses the offline downloader).
- `mlx-audio-swift-stt --stream --ane` auto-downloads + runs 100% on ANE — **validated live**
  (downloaded from HF, transcribed conversational_a.wav, words identical to MLX).

## Cold/hot start — compiled-model caching (M1 Max, 13s clip)
- `MLModel.compileModel` compiles the 1.1G package to a THROWAWAY temp `.mlmodelc` every launch →
  every run paid the full ~34s ANE compile. Cache the `.mlmodelc` next to the source + reuse when newer.
- **Cold start (compile once): 46s · Hot start (reuse): 3.0s** (MLX baseline 1.9s). Hot per-chunk ≈ 0.1s
  (~1.6× MLX), not the 8× the cold runs implied. Transcript cold==hot.

## Pure-MLX vs Hybrid (CoreML/ANE) — comparison
| | Pure MLX | Hybrid (ANE encoder) |
|---|---|---|
| Conformer encoder on | GPU (Metal) | **ANE, 100% resident** (`.cpuAndNeuralEngine`) |
| GPU freed during encode | no | **yes** |
| GPU power (offline sibling A/B) | 17.3 W | **3.0 W (÷5.8)** |
| Cold start (13s clip) | ~1.9 s | ~46 s (one-time ANE compile, cached) |
| Hot start (13s clip) | ~1.9 s | ~3.0 s |
| Transcript | reference | word-identical (punctuation @ fp16 floor) |

## Remaining
- Open the PR (mirror offline #13).
- Per-chunk MLX↔CoreML marshaling overhead (~0.1s vs MLX 0.06s) — IOSurface zero-copy is the upstream fix.
- Apply the same `.mlmodelc` caching to the offline `ParakeetCoreMLEncoder` (#199 code; follow-up).
