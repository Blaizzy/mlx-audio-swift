#!/usr/bin/env python3
"""Convert a saved TorchScript encoder (.traced.pt) to a CoreML .mlpackage.

Standalone: needs only torch + coremltools (+ numpy<2). No NeMo. Run this in an
isolated env when convert_encoder.py's in-line ct.convert fails on the host env
(e.g. coremltools 9.0 + numpy>=2 raising "only 0-dimensional arrays ..." on a folded
int const). Splits the heavy NeMo trace from the light, version-sensitive convert.

  python convert_traced.py --traced enc.traced.pt --feat-in 128 --frames 1000 \
      --out out/parakeet_enc_0.6b_v3.mlpackage
"""
import argparse

import numpy as np
import torch

ANE_NATIVE = {
    "conv", "conv_transpose", "linear", "matmul", "relu", "sigmoid", "tanh", "gelu",
    "softmax", "silu", "add", "mul", "sub", "real_div", "div", "pow", "concat", "split",
    "reshape", "transpose", "reduce_mean", "reduce_sum", "reduce_max", "reduce_l2_norm",
    "layer_norm", "batch_norm", "cast", "expand_dims", "squeeze", "slice_by_index",
    "slice_by_size", "pad", "tile", "gather", "const", "constexpr_affine_dequantize",
}
ANE_BAD = {"gru", "lstm", "rnn", "while_loop", "cond"}


def inspect_coreml(path):
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
    ap.add_argument("--traced", required=True, help="path to .traced.pt")
    ap.add_argument("--feat-in", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True, help="trace/default time length")
    ap.add_argument("--range-min", type=int, default=0, help="if >0: RangeDim lower bound on time")
    ap.add_argument("--range-max", type=int, default=0, help="if >0: RangeDim upper bound on time")
    ap.add_argument("--fp16-io", action="store_true", help="fp16 input/output (drops the boundary cast)")
    ap.add_argument("--out", default="out/parakeet_enc.mlpackage")
    args = ap.parse_args()

    print(f"numpy={np.__version__} torch={torch.__version__}")
    print(f"[1/2] loading TorchScript {args.traced} ...")
    traced = torch.jit.load(args.traced, map_location="cpu").train(False)

    import coremltools as ct

    if args.range_max:
        time_dim = ct.RangeDim(lower_bound=args.range_min or 1, upper_bound=args.range_max,
                               default=args.frames)
        shape = (1, args.feat_in, time_dim)
        print(f"      flexible time axis: RangeDim({args.range_min or 1}..{args.range_max}, default {args.frames})")
    else:
        shape = (1, args.feat_in, args.frames)
    io_dtype = np.float16 if args.fp16_io else np.float32

    print(f"[2/2] converting (fp16 weights, mlprogram, CPU_AND_NE, io={'fp16' if args.fp16_io else 'fp32'}) ...")
    print(f"      coremltools={ct.__version__}")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="features", shape=shape, dtype=io_dtype)],
        outputs=[ct.TensorType(name="encoded", dtype=io_dtype)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    mlmodel.save(args.out)
    print(f"      saved {args.out}")
    print("      static ANE pre-check:")
    inspect_coreml(args.out)
    print("done.")


if __name__ == "__main__":
    main()
