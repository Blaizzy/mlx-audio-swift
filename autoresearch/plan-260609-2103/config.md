# autoresearch config — Streaming CoreML (Nemotron conformer on ANE)

Derived by autoresearch:plan, 2026-06-09 21:03. Design: `docs/plans/2026-06-09-nemotron-streaming-coreml-design.md`.

**Decision (user, 2026-06-09):** Approach **A — full `cache_aware_stream_step`** (faithful to
NeMo native streaming; Swift will replicate `CacheAwareStreamingAudioBuffer` feeding).
Run mode: **local if env exists → it doesn't** (M1 Max here = nothing; alex-mac = coremltools
but no NeMo; pc.lan = NeMo but no ANE) → **prepared scripts, user runs on alex-mac**.

```
$autoresearch
Goal: Run the Nemotron streaming FastConformer encoder (full cache_aware_stream_step) on
      the ANE via CoreML, fp16-faithful to torch and with right-pad-to-fixed-F benign.
Scope: tools/coreml-ane/convert_encoder_coreml_stream.py
       tools/coreml-ane/parity_stream_func.py
       tools/coreml-ane/run_stream_parity.sh
       Sources/MLXAudioSTT/Models/NemotronASR/NemotronASRStreaming.swift  (later, Swift feeding)
Metric: min(CHECK1 fp16-fidelity, CHECK2 padding-benign) aggregate cosine over a real clip's NeMo-fed chunks
Direction: higher_is_better   # target >= 0.999 (fp16 floor; offline encoder already 0.9974)
Verify: uv run tools/coreml-ane/parity_stream_func.py --wav Tests/media/intention.wav | tail -1
Guard: convert + CoreML load/run without crash (Python); `swift build` once Swift integration starts
Iterations: 12
```

## The size fix (root cause of the earlier 112-vs-[105,57] mismatch)
NeMo's buffer feeds variable `processed_signal`: first=`chunk_size[0]` (105, no prepend),
middle=`pre_encode_cache_size[1]+chunk_size[1]` (9+112=121), last=partial. The old converter
traced at `chunk_size[-1]`=112 — wrong fixed shape. New converter traces at **F=121** (max
fed) and the harness/Swift **right-pads** each chunk to F. Two separated checks prove it:
- **CHECK1** CoreML(padded) vs torch(padded) → fp16 conversion error only.
- **CHECK2** torch(native) vs torch(padded) front frames → right-padding doesn't corrupt valid frames.

## Baseline (iteration 0)
Run `bash tools/coreml-ane/run_stream_parity.sh` (pushes to alex-mac, tmux). The metric is the
last line of `parity.log`. Expectation: CHECK2 should be ~1.0 if padding is benign; CHECK1
~0.997-0.999 (matches offline fp16). If CHECK2 < target → right-padding leaks into valid frames
→ iterate F / padding strategy (left-pad first chunk, or per-size conversion).

## Phased execution
1. **Python inner loop (Verify is cheap):** new converter wraps `encoder.layers` only,
   exposing fixed caches (attn tail-56, conv tail-8) + pos_emb; parity harness feeds a
   real clip via the MLX-equivalent recipe, runs CoreML per chunk, prints aggregate cosine.
   Iterate conversion until cosine ≥ 0.999.
2. **Swift outer loop:** add `enableCoreMLStreamingEncoder`; in `cacheAwareStreamEncode`
   thread MLX caches into one CoreML call instead of the layer loop; Guard = `swift build`;
   validate frame-identity vs pure-MLX on Metal (transcript-level + per-frame max-abs-diff).

## Env / heavy-run note (CLAUDE.md: prepare scripts, don't run heavy unattended)
- NeMo + coremltools conversion + CoreML execution must run on a Mac with the NeMo env
  (offline 3.5 conversion ran there; EN streaming model `nvidia/nemotron-speech-streaming-en-0.6b`
  loads in NeMo 2.7.3 on CPU). The autoresearch loop's Verify is heavy (~minutes/iter) →
  the loop should run **interactively / user-launched**, not as an unattended 12× auto-commit.
```
