#!/usr/bin/env python3
"""Convert a NeMo cache-aware streaming FastConformer encoder to a CoreML .mlpackage,
caches as EXPLICIT in/out tensors (functional, NOT MLState).

Why functional, not stateful: coremltools 9 cannot lower the cache-aware graph when the
caches are mutable buffers (jit.trace -> "No matching select or slice"; torch.export ->
"Unsupported fx node alias"). Exposing the caches as plain inputs+outputs avoids those ops
and converts cleanly (~98% ANE-native). The Swift side threads the caches manually
(feed in -> read updated out -> feed back).

THE SIZE FIX (2026-06-09)
-------------------------
The earlier version traced at `chunk = cfg.chunk_size[-1]` (=112). But NeMo's
`CacheAwareStreamingAudioBuffer` does NOT feed 112: it feeds
  first chunk  : chunk_size[0]                       (=105, no pre_encode prepend)
  middle chunks: pre_encode_cache_size[1] + chunk_size[1]   (=9 + 112 = 121)
  last chunk   : whatever frames remain (< 121, right-short)
A model traced at 112 is therefore the WRONG fixed shape for streaming. We trace at the
MAX fed size F = pre_encode_cache_size[1] + chunk_size[1] (=121) and the Swift/parity side
RIGHT-PADS every fed chunk to F. Right-padding is validated benign by parity_stream_func.py
(check 2: torch(native) valid frames == torch(padded-F) front frames). `--fixed-frames`
overrides F if a clip needs a different bound.

  uv pip install "nemo-toolkit[asr]" coremltools
  uv run convert_encoder_coreml_stream.py --model nvidia/nemotron-speech-streaming-en-0.6b \
      --att-context 56 13 --out out/nemotron_stream_func.mlpackage

State per call (att-ctx [56,13]):
  cache_last_channel [L,1,Cattn,d]  cache_last_time [L,1,d,Cconv]  cache_last_channel_len [1]
"""
import argparse
import json
import os

import numpy as np
import torch
import coremltools as ct

from convert_encoder import load_encoder  # DRY: same NeMo extraction


def _patch_aten_int():
    """coremltools' built-in aten::Int handler does mb.const(val=int(x.val)), which throws
    'only 0-dimensional arrays can be converted to Python scalars' when NeMo traces
    int(cache_last_channel_len) as a (1,)-shaped const (torch >= 2.8). aten::Int semantically
    returns a Python scalar, so squeezing the const to a scalar is correct. Avoids the
    torch<=2.7 pin entirely."""
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs
    from coremltools.converters.mil import Builder as mb

    @register_torch_op(override=True, torch_alias=["int"])
    def Int(context, node):  # noqa: N802
        x = _get_inputs(context, node, expected=1)[0]
        if x.val is not None:
            v = int(np.array(x.val).reshape(-1)[0])
            context.add(mb.const(val=v, name=node.name))
        else:
            context.add(mb.cast(x=x, dtype="int32", name=node.name))


_patch_aten_int()


def fed_max_frames(cfg):
    """Max `processed_signal` length NeMo's streaming buffer ever feeds = the fixed CoreML
    trace size. middle chunk = pre_encode_cache_size[1] + chunk_size[1]."""
    def last(x):
        return x[-1] if isinstance(x, (list, tuple)) else x
    pre = cfg.pre_encode_cache_size
    pre1 = pre[-1] if isinstance(pre, (list, tuple)) else pre
    return int(pre1) + int(last(cfg.chunk_size))


class FuncWrapper(torch.nn.Module):
    """Caches as explicit in/out (no buffer mutation). Swift threads them across chunks.
    keep_all_outputs=True so the model returns every frame; the drop_extra_pre_encoded
    front-drop is a downstream (buffer-consumption) concern, not the model's."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, processed_signal, cache_last_channel, cache_last_time, cache_last_channel_len):
        length = torch.zeros(1, dtype=torch.int64) + processed_signal.shape[-1]
        eo, _, nch, nt, nl = self.encoder.cache_aware_stream_step(
            processed_signal=processed_signal, processed_signal_length=length,
            cache_last_channel=cache_last_channel, cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len, keep_all_outputs=True)
        return eo, nch, nt, nl.to(torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    ap.add_argument("--att-context", type=int, nargs=2, default=[70, 13])
    ap.add_argument("--fixed-frames", type=int, default=None,
                    help="processed_signal length to trace at (default: max NeMo-fed size)")
    ap.add_argument("--out", default="out/nemotron_stream_func.mlpackage")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    print(f"[1/3] loading {args.model} ...")
    enc, meta = load_encoder(args.model)
    enc.setup_streaming_params(att_context_size=args.att_context)
    cfg = enc.streaming_cfg
    fixed = args.fixed_frames or fed_max_frames(cfg)
    ch, t, clen = enc.get_initial_cache_state(batch_size=1)
    clen = clen.to(torch.int32)
    fi = meta["feat_in"]
    print(f"      fixed_frames={fixed} (chunk_size={cfg.chunk_size} "
          f"pre_encode_cache_size={cfg.pre_encode_cache_size}) feat_in={fi} "
          f"valid_out={cfg.valid_out_len} ch={tuple(ch.shape)} t={tuple(t.shape)}")

    w = FuncWrapper(enc).train(False)
    sig = torch.randn(1, fi, fixed)
    with torch.no_grad():
        ref = [x.shape for x in w(sig, ch, t, clen)]
        traced = torch.jit.trace(w, (sig, ch, t, clen), check_trace=False)
    print(f"[2/3] traced at {fixed}; outputs {[tuple(s) for s in ref]}")

    f16 = np.float16
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="processed_signal", shape=(1, fi, fixed), dtype=f16),
            ct.TensorType(name="cache_last_channel", shape=tuple(ch.shape), dtype=f16),
            ct.TensorType(name="cache_last_time", shape=tuple(t.shape), dtype=f16),
            ct.TensorType(name="cache_last_channel_len", shape=tuple(clen.shape), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="encoded", dtype=f16),
            ct.TensorType(name="new_cache_last_channel", dtype=f16),
            ct.TensorType(name="new_cache_last_time", dtype=f16),
            ct.TensorType(name="new_cache_last_channel_len", dtype=np.int32),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    mlmodel.save(args.out)
    print(f"[3/3] saved {args.out}")
    man = {
        "model": args.model, **meta, "att_context": args.att_context,
        "fixed_frames": fixed, "chunk_size": list(cfg.chunk_size),
        "pre_encode_cache_size": list(cfg.pre_encode_cache_size),
        "valid_out_len": cfg.valid_out_len,
        "cache_last_channel": list(ch.shape), "cache_last_time": list(t.shape),
        "inputs": ["processed_signal", "cache_last_channel", "cache_last_time", "cache_last_channel_len"],
        "outputs": ["encoded", "new_cache_last_channel", "new_cache_last_time", "new_cache_last_channel_len"],
    }
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"done. Trace size {fixed}; Swift/parity RIGHT-PADS each fed chunk to {fixed}.")


if __name__ == "__main__":
    main()
