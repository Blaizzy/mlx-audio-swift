#!/usr/bin/env python3
"""Parity for the Option-A streaming CoreML encoder (full cache_aware_stream_step),
using the VALIDATED uniform-F feeding recipe.

Recipe (Swift-replicable, validated transcript-identical to NeMo native streaming):
  F = pre_encode_cache_size[1] + chunk_size[1] (= 9 + 112 = 121); feed every chunk as
  [9 prev-mel ++ 112 new mel], stride 112 (zeros at the first prepend and last new-tail).
  cache_aware_stream_step already applies drop_extra_pre_encoded internally, so keep its
  output frames as-is (the last chunk keeps only its real valid frames).

Metric (LAST line) = CHECK1, the deployment-faithful number:
  CHECK1 fp16 fidelity : cosine(CoreML uniform-F frames, torch uniform-F frames).  >= 0.999 target.
  CHECK2 transcript    : decode(CoreML) tokens vs decode(torch) vs offline.  pass/fail gate.

  uv run parity_stream_func.py --wav Tests/media/conversational_a.wav \
      --mlpackage out/nemotron_stream_func.mlpackage [--torch-only]
"""
import argparse
import sys

import numpy as np
import torch
import librosa
import nemo.collections.asr as nemo_asr


def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    n = min(a.size, b.size); a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    ap.add_argument("--mlpackage", default="out/nemotron_stream_func.mlpackage")
    ap.add_argument("--att-context", type=int, nargs=2, default=[70, 13])
    ap.add_argument("--wav", default="Tests/media/conversational_a.wav")
    ap.add_argument("--torch-only", action="store_true")
    args = ap.parse_args()

    m = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cpu").train(False)
    enc = m.encoder

    wav, _ = librosa.load(args.wav, sr=16000, mono=True)
    audio = torch.tensor(wav, dtype=torch.float32)[None]
    alen = torch.tensor([audio.shape[1]], dtype=torch.int64)
    with torch.no_grad():
        mel, mlen = m.preprocessor(input_signal=audio, length=alen)
    T = int(mlen.item()); mel = mel[..., :T]

    enc.setup_streaming_params(att_context_size=args.att_context)
    cfg = enc.streaming_cfg
    last = lambda x: x[-1] if isinstance(x, (list, tuple)) else int(x)
    PRE, NEW = last(cfg.pre_encode_cache_size), last(cfg.chunk_size)
    F = PRE + NEW
    print(f"F={F} PRE={PRE} NEW={NEW} att={args.att_context}")

    ml = None
    if not args.torch_only:
        import coremltools as ct
        ml = ct.models.MLModel(args.mlpackage, compute_units=ct.ComputeUnit.CPU_AND_NE)

    def window(p):
        w = torch.zeros(1, mel.shape[1], F)
        if p > 0:
            s = max(0, p - PRE); w[..., PRE - (p - s):PRE] = mel[..., s:p]
        e = min(T, p + NEW); w[..., PRE:PRE + (e - p)] = mel[..., p:e]
        return w

    def decode(frames):
        eo = torch.as_tensor(np.concatenate(frames, axis=-1), dtype=torch.float32)
        el = torch.tensor([eo.shape[-1]], dtype=torch.int64)
        with torch.no_grad():
            h = m.decoding.rnnt_decoder_predictions_tensor(eo, el, return_hypotheses=False)
        return (h[0] if isinstance(h, (list, tuple)) else h)

    tstate = enc.get_initial_cache_state(batch_size=1)
    cch = ct_ = ccl = None
    t_fr, c_fr, c1, p = [], [], [], 0
    while p < T:
        win = window(p)
        with torch.no_grad():
            t_out, t_el, *tstate = enc.cache_aware_stream_step(
                processed_signal=win, processed_signal_length=torch.tensor([F]),
                cache_last_channel=tstate[0], cache_last_time=tstate[1],
                cache_last_channel_len=tstate[2], keep_all_outputs=True)
        real_new = min(NEW, T - p)
        keep = min(int(t_out.shape[-1]), max(1, (real_new + 7) // 8))
        t_fr.append(t_out[..., :keep].numpy())

        if ml is not None:
            if cch is None:
                ci = enc.get_initial_cache_state(batch_size=1)
                cch, ct_, ccl = (ci[0].numpy().astype(np.float16), ci[1].numpy().astype(np.float16),
                                 ci[2].numpy().astype(np.int32))
            out = ml.predict({"processed_signal": win.numpy().astype(np.float16),
                              "cache_last_channel": cch, "cache_last_time": ct_,
                              "cache_last_channel_len": ccl})
            cch, ct_, ccl = (out["new_cache_last_channel"], out["new_cache_last_time"],
                             out["new_cache_last_channel_len"].astype(np.int32))
            cf = out["encoded"][..., :keep]
            c_fr.append(cf)
            c1.append(cos(cf, t_fr[-1]))
        p += NEW

    txt_t = decode(t_fr)
    print(f"torch  stream: {txt_t.text!r}")
    if ml is not None:
        txt_c = decode(c_fr)
        print(f"coreml stream: {txt_c.text!r}")
        check1 = float(np.mean(c1))
        match = (list(txt_c.y_sequence) == list(txt_t.y_sequence))
        print(f"CHECK1 fp16-fidelity cosine (CoreML vs torch, uniform-{F}): mean={check1:.6f} min={min(c1):.6f}")
        print(f"CHECK2 transcript token-match (CoreML == torch): {match}")
        metric = check1 if match else min(check1, 0.0)
    else:
        metric = 1.0
    print(f"metric={metric:.6f}", file=sys.stderr)
    print(f"{metric:.6f}")


if __name__ == "__main__":
    main()
