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
library_name: coreml
---

# Nemotron 3.5 ASR — FastConformer encoder on the Apple Neural Engine (CoreML)

A fixed-shape **CoreML** conversion of the **FastConformer encoder** from
[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
so the encoder (≈95 % of the compute) runs on the **Apple Neural Engine** while the prompt MLP and
RNN-T decoder stay in MLX on the GPU.

Pairs with the MLX model [`mlx-community/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/mlx-community/nemotron-3.5-asr-streaming-0.6b)
in [mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift).

## Use

```bash
# auto-downloads this encoder and runs it on the ANE for the offline path:
mlx-audio-swift-stt --model mlx-community/nemotron-3.5-asr-streaming-0.6b \
    --audio in.wav --output-path out --ane
```

From Swift: `try await nemotron.enableCoreMLEncoder(repo: NemotronASRModel.defaultANEEncoderRepo)`.

## Details

- **8-bit palettized weights** (`palettize_weights`, uniform): **564 MB** (≈2× smaller than fp16) and
  faster to download + load; the transcript stays **word-identical** to fp16/MLX (only the int8/fp16-vs-bf16
  floor — e.g. a single proper-noun grapheme).
- **Input** `features` `[1, 128, 1000]` (mel, 10 ms hop → 10 s) · **Output** `encoded` `[1, 1024, 126]` fp16.
- Fixed shape (ANE requires it); the Swift wrapper pads each chunk and crops the output, and
  `generate()` auto-clamps `chunkDuration` to ≤ 10 s and overlap-merges longer audio.
- **99 % ANE-native** ops; encoder fidelity **cosine 0.9974** (fp16) vs the fp32 reference.
- Measured (M1 Max): encoder ≈ **2×** faster than MLX-fp32, **~1.31×** end-to-end offline, GPU
  power **÷~9** (encoder off the GPU).
- Public `MLModel` + `MLComputeUnits` only — no private APIs.

## Conversion

`tools/coreml-ane/convert_encoder.py --model nvidia/nemotron-3.5-asr-streaming-0.6b --frames 1000 --palettize 8`
(needs NeMo ≥ 2.8 for the multilingual `EncDecRNNTBPEModelWithPrompt` class).
