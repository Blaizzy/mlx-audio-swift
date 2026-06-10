---
license: cc-by-4.0
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
tags:
  - coreml
  - ane
  - apple-neural-engine
  - asr
  - nemotron
  - fastconformer
  - streaming
library_name: coreml
---

# Nemotron 3.5 ASR — cache-aware **streaming** FastConformer encoder on the Apple Neural Engine (CoreML)

A fixed-shape **CoreML** conversion of the **cache-aware streaming** FastConformer encoder from
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
so the streaming encoder (≈95 % of the compute) runs on the **Apple Neural Engine** while the prompt MLP
and RNN-T decoder stay in MLX on the GPU.

Pairs with the MLX model [`mlx-community/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b)
in [mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift). For the **offline** encoder see
[`beshkenadze/nemotron-3.5-asr-streaming-0.6b-coreml-ane`](https://huggingface.co/beshkenadze/nemotron-3.5-asr-streaming-0.6b-coreml-ane).

## Use

```bash
# auto-downloads this encoder and streams on the ANE:
mlx-audio-swift-stt --model mlx-community/nemotron-3.5-asr-streaming-0.6b \
    --audio in.wav --output-path out --stream --ane
```

From Swift: `try await nemotron.enableCoreMLStreamingEncoder(repo: NemotronASRModel.defaultANEStreamingEncoderRepo)`.

## How it works

NeMo's `cache_aware_stream_step` is converted **functionally** — the three streaming caches are explicit
inputs **and** outputs (no `MLState`), so Swift threads them across chunks (feed in → read `new_*` → feed
back). A fixed-shape ANE model can't honor a per-chunk true length, so the Swift wrapper feeds a **uniform**
window every chunk:

- **Input** `processed_signal` `[1, 128, 121]` = `[9 prev-mel ++ 112 new-mel]`, stride 112 ·
  `cache_last_channel [24,1,56,1024]` · `cache_last_time [24,1,1024,8]` · `cache_last_channel_len [1]` (i32).
- **Output** `encoded [1,1024,14]` fp16 + the three `new_*` caches. att_context `[56,13]`.

This uniform feeding is **transcript-identical** to NeMo's native variable-chunk streaming.

## Details

- **8-bit palettized weights** (`palettize_weights`, uniform): **561 MB** (≈2× smaller than fp16) and
  **~28 % faster** on the ANE (~19 ms/chunk vs ~27 ms fp16) — and the streamed transcript is **word-for-word
  identical** to fp16/MLX. (6-bit was faster still but degraded the transcript, so 8-bit is shipped.)
- **100 % ANE-resident** (cost-weighted, `MLComputeUnits.cpuAndNeuralEngine`) — the few int32 mask /
  cache-length ops drop to CPU as one negligible island. **Use CPU+ANE, not `.all`**: with `.all`,
  CoreML places the whole graph on the GPU (≈2 % ANE) — no power win.
- ~19 ms/chunk on the ANE (chunk = 112 mel ≈ 1.1 s audio → ~58× realtime), GPU freed. The Swift wrapper
  caches the compiled `.mlmodelc` (cold compile once ~33 s, then hot start ~2.3 s — within 0.4 s of pure MLX).
- Validated end-to-end (M1 Max) against the MLX streaming path: **word-for-word identical** transcript;
  punctuation differs at the int8/fp16-ANE vs bf16-MLX precision floor (same ~1 % agreement as the offline path).
- Public `MLModel` + `MLComputeUnits` only — no private APIs.

## Conversion

`tools/coreml-ane/convert_encoder_coreml_stream.py --model nvidia/nemotron-3.5-asr-streaming-0.6b --att-context 56 13`
(needs NeMo ≥ 2.8 for `EncDecRNNTBPEModelWithPrompt`; patches coremltools' `aten::Int` to convert on torch 2.10).
