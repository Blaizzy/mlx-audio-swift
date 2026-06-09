# Nemotron streaming encoder on ANE — design (autoresearch:plan)

Date: 2026-06-09 · Branch base: `feat/nemotron-coreml-ane-encoder` (PR #13, offline already shipped)

## Goal
Run the Nemotron 3.5 FastConformer **streaming** encoder on the Apple Neural Engine
(CoreML), frame-identical to the existing pure-MLX cache-aware streaming path, for the
power/thermal win the offline path already gives (GPU power ÷~9).

## The pivot (key finding)
The half-built converter `convert_encoder_coreml_stream.py` wraps NeMo's **full**
`cache_aware_stream_step` = `pre_encode` (subsampling) + 24 conformer layers + native
caches (`cache_last_channel[56]`, `cache_last_time[8]`, `cache_last_channel_len`). Driving
that from Swift means **replicating NeMo's `CacheAwareStreamingAudioBuffer` feeding**:
variable chunk sizes (first 105, middle 121, last partial), a 9-frame pre_encode prepend,
`drop_extra_pre_encoded=2`. That is the source of the earlier mismatch (fed [105,57] ≠
converted-at-112) and of `cosine(stream, full-offline)=0.56` (the 0.56 is the expected
streaming-vs-**full**-context gap, not a bug — but the variable feeding is a real Swift
liability with silent-drift risk).

**Production Swift already solves feeding.** `NemotronASRStreaming.swift::cacheAwareStreamEncode`
does the subsampling itself in MLX (`win = [melCache(16) ++ chunkMel]` → `encoder.preEncode`),
runs the conformer layer loop with MLX caches (`attnCache` tail-56 of attn input;
`convCache` tail-8 of GLU output — same semantics as NeMo's two caches), then `applyPrompt`.
Its doc-contract: **frame-identical to the offline chunked encoder** → streamed transcript
equals `decode(...)`. This is the proven reference.

## Two integration options

| | **A — full stream_step (existing converter)** | **B — conformer-only (recommended)** |
|---|---|---|
| CoreML model wraps | pre_encode + conformer + caches | **only `encoder.layers`** (conformer stack) |
| Input | raw mel chunk (variable size) + 3 native caches | subsampled `h[1,14,d]` + attn/conv caches + pos_emb (all **fixed**) |
| Swift feeding | **rewrite** to NeMo buffer recipe (105/121/partial, prepend, drop) | **keep** MLX `melCache`+`preEncode`+`prompt`; swap only the layer loop |
| Fixed-shape ANE | needs pad variable→fixed + crop valid (fiddly) | clean: chunk = `right+1` = 14 subsampled frames, caches 56/8 |
| Correctness risk lives in | Swift feeding (Metal-gated, expensive to iterate, silent drift) | **Python converter** (cheap, observable, compares to MLX/torch layer outputs) |
| Swift change | large rewrite of streaming feeding | **localized**: replace `for li in encoder.layers {…}` with one CoreML call |

**Decision: Option B.** It moves the correctness-critical matching into Python (cheap,
visible) and reduces the Swift delta to a drop-in for the conformer loop, preserving the
already-proven feeding + the offline `ParakeetCoreMLEncoder` reuse pattern.

## Plan (autoresearch loop)
- **Metric**: aggregate cosine between the CoreML conformer-stack output and the
  torch/MLX conformer reference, over a real clip's streaming chunks. `higher_is_better`,
  target **≥ 0.999** (fp16 floor; offline full-encoder already hits 0.9974).
- **Inner (cheap) loop — Python**: new converter `convert_encoder_coreml_conformer_stream.py`
  wraps `encoder.layers` with explicit fixed caches + pos_emb; parity harness
  `parity_conformer_stream.py` drives a real clip through the MLX-equivalent feeding,
  runs CoreML per chunk, prints aggregate cosine vs torch.
- **Outer — Swift**: once Python cosine ≥ 0.999, add `enableCoreMLStreamingEncoder(...)`;
  in `cacheAwareStreamEncode`, when enabled, thread MLX caches into the CoreML call
  instead of the layer loop; validate frame-identity vs pure-MLX on Metal.
- **Guard**: CoreML model loads + runs (no crash); later `swift build` green.
- **Iterations**: ~12 (cache layout, pos_emb offset, fp16 tolerance, stride-padded output).

## Risks
- Extracting `encoder.layers` with explicit caches in torch must match the MLX cache
  semantics **exactly** (attn tail-56, conv tail-8, `posEnc` offset = cache length). The
  parity harness catches drift immediately (Python, per-chunk cosine).
- pos_emb length grows then saturates (cacheLen 0→56). Fixed-shape model must pad early
  chunks (NeMo's native model does this via `cache_last_channel_len`); B does it via a
  fixed cache + valid-length scalar or zero-pad-left.
