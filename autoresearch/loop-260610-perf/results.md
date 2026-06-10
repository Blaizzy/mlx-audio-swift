# autoresearch results — offline/stream encoder performance

Metric: encoder ms/step (lower_is_better) + e2e load/size. Guard: encoder output checksum stable (374.1449)
+ transcript correct. Env: M1 Max, release build, `nemotron-stream-probe --bench`.

## Baseline + profiling (iter 0) — the bottleneck is NOT in the code
| phase | ms/step | share |
|---|---|---|
| marshal (window→fp16 MLMultiArray) | 0.049 | 0.2% |
| provider (MLDictionaryFeatureProvider) | 0.016 | 0.1% |
| **predict (model.prediction, ANE compute)** | **26.585** | **99.6%** |
| read (stride-read encoded) | 0.041 | 0.2% |
| **total median** | **26.75** | |

**Finding:** the streaming (and offline — same single-prediction pattern) encoder is CoreML/ANE
**compute-bound**. All Swift integration work is ~0.1 ms/step → no Swift micro-opt (buffer reuse, custom
provider, copy elimination) can move the metric (≤0.5% headroom). 26.6 ms ≈ anemll's 25 ms steady = ANE floor.
MLX-GPU encoder is already faster (project thesis: hybrid is a POWER feature, not speed).

## Real levers (model-level, conversion-side — not Swift)
1. **Palettization** (fp16→6/4-bit weights): smaller model → faster `--ane` download + `.mlmodelc` load
   (the e2e bottleneck is *load*, not per-chunk compute). Maybe faster ANE compute (if LUT-native).
2. ANE/GPU pipelining (overlap encoder/decoder) — breaks incremental streaming.

iteration log:

## Iteration log — palettization (conversion-side, the real lever)
| iter | change | ms/step | size | transcript | decision |
|---|---|---|---|---|---|
| 0 | fp16 baseline | 26.8 | 1.1 GB | reference | baseline |
| 1 | 6-bit uniform | 22.3 | 424 MB | **degraded** ("begins in Ethiopia" dropped, "felt invigorated"→"vigorated") | **discard** (accuracy) |
| 2 | **8-bit uniform** | **19.3** | **561 MB** | **word-identical ✅** | **KEEP** |

**Win: 8-bit uniform palettization → 2× smaller + ~28% faster ms/step, transcript word-identical.**
The speedup is real ANE compute (8-bit weights), not just size; the size halves download + `.mlmodelc` load.
(6-bit kmeans would beat 6-bit uniform on accuracy but OOM'd on alex-mac; 8-bit uniform already wins, no need.)

## Offline + Parakeet 8-bit (2026-06-10)
- Added `--palettize N` to the shared offline converter `convert_encoder.py` (+ ported the missing
  `aten::Int` patch — it broke on torch>=2.8 like the streaming one). `--palettize -1` = per-channel linear int8.
- **Offline Nemotron 3.5 @ 8-bit uniform: 564 MB, transcript word-identical** ("Caldi"→"Kaldi" floor) → SHIPPED
  (re-uploaded to the offline `--ane` repo).
- **Parakeet v3 — model-specific, more sensitive (TDT decoder):**
  - 8-bit **uniform** palettize → encoder cosine **0.21**, empty transcript. Uniform crushes Parakeet's
    outlier-heavy weights. **Don't use uniform for Parakeet.**
  - **Per-channel linear int8** → cosine 0.96 (synthetic), **word-identical per-window** (10s clip).
    Long audio (>10s) truncates the tail at the default 10s chunk (TDT + heavily-padded final chunk + quant);
    **`--chunk-duration 7` → word-identical** again.
  - Verdict: viable but NOT a zero-config drop-in like Nemotron — needs linear int8 (not uniform) AND smaller
    chunks for long audio. Not re-uploaded (default-chunk long-audio would regress). RNN-T (Nemotron) tolerates
    8-bit; TDT (Parakeet) is borderline.
