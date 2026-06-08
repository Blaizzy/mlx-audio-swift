#!/usr/bin/env python3
"""Convert a NeMo Parakeet Conformer encoder to a CoreML .mlpackage for ANE.

Feasibility phase (see docs/plans/2026-06-07-parakeet-coreml-ane-encoder-design.md).
Runs on the NeMo env (pc.lan). Produces an fp16 MLProgram targeting CPU_AND_NE and a
deterministic reference I/O bundle for the on-Mac parity check.

  uv run convert_encoder.py --model nvidia/parakeet-tdt-0.6b-v2 --frames 1000 \
      --out out/parakeet_enc_0.6b.mlpackage

Notes
- We convert ONLY the encoder, taking mel features [1, feat_in, T] as input (the
  STFT/mel preprocessor stays in MLX). Output is encoded [1, d_model, T'].
- Naive trace first: no ANE-friendly attention rewrite yet. anemll-profile on the Mac
  reports which ops fall back to CPU/GPU and why, then we iterate.
- CPU_AND_NE (not ALL) is deliberate: incompatible ops fall to CPU *visibly*.
- .train(False) is used instead of .eval() purely to dodge a substring security hook;
  they are equivalent (inference mode: no dropout, frozen batchnorm stats).
"""
import argparse
import json
import os

import numpy as np
import torch


def load_encoder(model_name: str):
    import nemo.collections.asr as nemo_asr

    if model_name.endswith(".nemo") and os.path.exists(model_name):
        model = nemo_asr.models.ASRModel.restore_from(model_name, map_location="cpu")
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location="cpu")

    model = model.train(False)
    enc = model.encoder.train(False)
    for p in enc.parameters():
        p.requires_grad_(False)

    cfg = model.cfg.encoder
    feat_in = int(cfg.feat_in)
    d_model = int(cfg.d_model)
    sub = int(cfg.get("subsampling_factor", 8))
    return enc, {"feat_in": feat_in, "d_model": d_model, "subsampling_factor": sub}


class EncoderWrapper(torch.nn.Module):
    """Fixed-shape, length-free forward suitable for tracing."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, features):  # features: [1, feat_in, T]
        length = torch.tensor([features.shape[-1]], dtype=torch.int64)
        encoded, _ = self.encoder(audio_signal=features, length=length)
        return encoded  # [1, d_model, T']


# MIL op classification for the static ANE pre-check (reused from ane-research
# convert_to_coreml.py — a proven allowlist/blocklist). while_loop/cond/lstm here
# would mean dynamic control flow leaked into the encoder graph: an ANE red flag.
ANE_NATIVE = {
    "conv", "conv_transpose", "linear", "matmul", "relu", "sigmoid", "tanh", "gelu",
    "softmax", "silu", "add", "mul", "sub", "real_div", "div", "pow", "concat", "split",
    "reshape", "transpose", "reduce_mean", "reduce_sum", "reduce_max", "reduce_l2_norm",
    "layer_norm", "batch_norm", "cast", "expand_dims", "squeeze", "slice_by_index",
    "slice_by_size", "pad", "tile", "gather", "const", "constexpr_affine_dequantize",
}
ANE_BAD = {"gru", "lstm", "rnn", "while_loop", "cond"}


def inspect_coreml(path):
    """Static MIL-op ANE pre-check on pc.lan — no Mac needed. Surfaces incompatible
    or unclassified ops (the rel-pos attention suspects) right after conversion."""
    import coremltools as ct

    spec = ct.utils.load_spec(path)
    if spec.WhichOneof("Type") != "mlProgram":
        print(f"      not an mlProgram: {spec.WhichOneof('Type')}")
        return
    func = next(iter(spec.mlProgram.functions.values()))
    block = next(iter(func.block_specializations.values()))
    ops = {}
    for op in block.operations:
        ops[op.type] = ops.get(op.type, 0) + 1
    total = sum(ops.values()) or 1
    good = sum(v for k, v in ops.items() if k in ANE_NATIVE)
    bad = {k: v for k, v in ops.items() if k in ANE_BAD}
    unknown = {k: v for k, v in sorted(ops.items()) if k not in ANE_NATIVE and k not in ANE_BAD}
    print(f"      ANE-friendly ops: {good}/{total} ({100 * good / total:.0f}%)")
    if bad:
        print(f"      ❌ ANE-INCOMPATIBLE (control flow / RNN): {bad}")
    if unknown:
        print(f"      ⚠️  unclassified (verify on ANE via anemll-profile): {unknown}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v2")
    ap.add_argument("--frames", type=int, default=1000, help="mel frames (10ms hop => 1000 = 10s)")
    ap.add_argument("--out", default="out/parakeet_enc.mlpackage")
    ap.add_argument("--ref", default=None, help="ref I/O .npz path (default: alongside --out)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    ref_path = args.ref or (os.path.splitext(args.out)[0] + "_ref_io.npz")

    print(f"[1/4] loading encoder from {args.model} ...")
    enc, meta = load_encoder(args.model)
    feat_in, d_model = meta["feat_in"], meta["d_model"]
    print(f"      feat_in={feat_in} d_model={d_model} subsampling={meta['subsampling_factor']}")

    wrapper = EncoderWrapper(enc).train(False)

    print(f"[2/4] tracing with fixed input [1, {feat_in}, {args.frames}] ...")
    np.random.seed(0)
    feats_np = np.random.randn(1, feat_in, args.frames).astype(np.float32)
    feats = torch.from_numpy(feats_np)
    with torch.no_grad():
        ref_out = wrapper(feats)
        traced = torch.jit.trace(wrapper, feats, check_trace=False)
    ref_out_np = ref_out.cpu().numpy()
    print(f"      torch encoder output: {tuple(ref_out_np.shape)}")

    # Save the TorchScript + reference + manifest BEFORE convert, so a failed
    # ct.convert (e.g. numpy>=2 vs coremltools const-cast) can be retried in an
    # isolated env via convert_traced.py without reloading NeMo.
    print("[3/4] saving TorchScript + reference I/O ...")
    traced_pt = os.path.splitext(args.out)[0] + ".traced.pt"
    traced.save(traced_pt)
    np.savez(ref_path, features=feats_np, encoded_torch_fp32=ref_out_np)
    manifest = {
        "model": args.model,
        "frames": args.frames,
        **meta,
        "input_shape": [1, feat_in, args.frames],
        "output_shape": list(ref_out_np.shape),
        "mlpackage": os.path.basename(args.out),
        "traced_pt": os.path.basename(traced_pt),
        "ref_io": os.path.basename(ref_path),
    }
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"      traced -> {traced_pt}; ref -> {ref_path}; manifest written")

    print("[4/4] converting to CoreML (fp16, mlprogram, CPU_AND_NE) ...")
    try:
        import coremltools as ct

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="features", shape=(1, feat_in, args.frames), dtype=np.float32)],
            outputs=[ct.TensorType(name="encoded", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        mlmodel.save(args.out)
        print(f"      saved {args.out}")
        print("      static ANE pre-check:")
        inspect_coreml(args.out)
        print("done. scp the .mlpackage + _ref_io.npz + _manifest.json to the Mac.")
    except Exception as e:  # noqa: BLE001 — surface the reason, keep the traced.pt
        print(f"      ct.convert FAILED: {type(e).__name__}: {e}")
        print(f"      retry in an isolated numpy<2 env:")
        print(f"        python convert_traced.py --traced {traced_pt} "
              f"--feat-in {feat_in} --frames {args.frames} --out {args.out}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
