# Parakeet Conformer encoder on CoreML / Apple Neural Engine

Convert a NeMo Parakeet **Conformer encoder** to a CoreML `.mlpackage` so the Swift STT
pipeline can run it on the **ANE** while the TDT decoder stays in MLX:

```bash
mlx-audio-swift-stt --model <parakeet-repo> --audio in.wav --output-path out \
    --coreml-encoder path/to/parakeet_enc.mlpackage --chunk-duration 9.95
```

The encoder is fixed-shape (a fixed mel-frame count) — a dynamic shape drops ANE
residency to 0% — so keep `--chunk-duration` at or below the converted length
(`frames * 10 ms`, i.e. ≤ ~10 s for the default 1000 frames). The Swift wrapper pads
each chunk to the fixed length and crops the output back. Only public `MLModel` +
`MLComputeUnits` APIs are used.

## Converting the encoder

Needs a PyTorch/NeMo environment (`nemo-toolkit[asr]`, `torch`). The trace and the
CoreML conversion are split because **coremltools 9.0 + numpy ≥ 2** fails on a folded
`aten::Int` constant (`only 0-dimensional arrays …`); the convert step runs in an
isolated `numpy<2` env.

```bash
# 1. Trace the encoder (NeMo env) -> saves <out>.traced.pt + manifest.
python convert_encoder.py --model nvidia/parakeet-tdt-0.6b-v3 --frames 1000 \
    --out out/parakeet_enc.mlpackage
#    (the in-process ct.convert may fail on numpy>=2 — that's fine, the .pt is saved)

# 2. Convert the traced model in a numpy<2 env -> .mlpackage (fp16 MLProgram, CPU_AND_NE).
uv venv ctenv && source ctenv/bin/activate
uv pip install "numpy<2" coremltools==9.0 "torch==2.10.*" --torch-backend=cpu
python convert_traced.py --traced out/parakeet_enc.traced.pt \
    --feat-in 128 --frames 1000 --fp16-io --out out/parakeet_enc.mlpackage
```

`--fp16-io` makes the model 100% ANE-resident (no fp32 boundary cast). `--feat-in` is
the encoder's mel dimension (128 for parakeet-tdt-0.6b-v3, 80 for the 1.1b variants;
printed by `convert_encoder.py`).

The resulting `.mlpackage` is what you pass to `--coreml-encoder`.
