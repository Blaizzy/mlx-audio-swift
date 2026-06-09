#!/usr/bin/env python3
"""Localize the streaming/offline mismatch before building the CoreML recipe.

Q1: does NeMo's OWN native buffer streaming match offline-chunked-limited? (validates the reference)
Q2: what output orientation + drop offset aligns one uniform-121 chunk to offline frames?
"""
import argparse
import numpy as np
import torch
import librosa
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer


def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    n = min(a.size, b.size); a, b = a[:n], b[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else 1.0


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
ap.add_argument("--att-context", type=int, nargs=2, default=[70, 13])
ap.add_argument("--wav", default="Tests/media/intention.wav")
a = ap.parse_args()

m = nemo_asr.models.ASRModel.from_pretrained(a.model, map_location="cpu").train(False)
enc = m.encoder
wav, _ = librosa.load(a.wav, sr=16000, mono=True)
audio = torch.tensor(wav, dtype=torch.float32)[None]
alen = torch.tensor([audio.shape[1]], dtype=torch.int64)
with torch.no_grad():
    mel, mlen = m.preprocessor(input_signal=audio, length=alen)
T = int(mlen.item()); mel = mel[..., :T]

# --- offline chunked-limited reference (BEFORE any streaming setup) ---
enc.set_default_att_context_size(a.att_context)
with torch.no_grad():
    off, off_len = enc(audio_signal=mel, length=mlen)
off = off[..., : int(off_len.item())]
print(f"offline shape={tuple(off.shape)}  frames={off.shape[-1]}  (axis -1 assumed time)")

# --- Q1: NeMo native buffer streaming vs offline ---
enc.setup_streaming_params(att_context_size=a.att_context)
cfg = enc.streaming_cfg
print(f"streaming_cfg chunk_size={cfg.chunk_size} pre_encode={cfg.pre_encode_cache_size} "
      f"valid_out={cfg.valid_out_len} drop_extra={getattr(cfg,'drop_extra_pre_encoded',None)}")
buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False)
buf.append_audio(audio[0].numpy())
st = enc.get_initial_cache_state(batch_size=1)
nat_frames, fed_sizes = [], []
for fed, plen in buf:
    fed_sizes.append(fed.shape[-1])
    last = buf.is_buffer_empty()
    with torch.no_grad():
        eo, el, *st = enc.cache_aware_stream_step(
            processed_signal=fed, processed_signal_length=plen,
            cache_last_channel=st[0], cache_last_time=st[1],
            cache_last_channel_len=st[2], keep_all_outputs=last)
    n = int(el.flatten()[0].item())
    nat_frames.append(eo[..., :n].numpy())
nat = np.concatenate(nat_frames, axis=-1)
print(f"native buffer: fed_sizes={fed_sizes}  eo.shape={tuple(eo.shape)}  total_frames={nat.shape[-1]}")
print(f"Q1 cosine(native-buffer-stream, offline) = {cos(nat[..., :off.shape[-1]], off.numpy()):.6f}")

# also test the OTHER orientation in case eo is [1, T', d]
if eo.ndim == 3 and eo.shape[1] != off.shape[1]:
    nat_T = np.concatenate([f.transpose(0, 2, 1) for f in nat_frames], axis=-1)
    print(f"   (transposed) cosine = {cos(nat_T[..., :off.shape[-1]], off.numpy()):.6f}")

# --- Q2: one uniform-121 chunk0, scan drop offset ---
PRE = cfg.pre_encode_cache_size[-1] if isinstance(cfg.pre_encode_cache_size, (list, tuple)) else int(cfg.pre_encode_cache_size)
NEW = cfg.chunk_size[-1] if isinstance(cfg.chunk_size, (list, tuple)) else int(cfg.chunk_size)
win = torch.zeros(1, mel.shape[1], PRE + NEW)
win[..., PRE:PRE + min(NEW, T)] = mel[..., :min(NEW, T)]
st0 = enc.get_initial_cache_state(batch_size=1)
with torch.no_grad():
    eo0, el0, *_ = enc.cache_aware_stream_step(
        processed_signal=win, processed_signal_length=torch.tensor([PRE + NEW]),
        cache_last_channel=st0[0], cache_last_time=st0[1], cache_last_channel_len=st0[2],
        keep_all_outputs=True)
print(f"Q2 uniform-121 chunk0 eo.shape={tuple(eo0.shape)} el={int(el0.flatten()[0])}")
for d in range(0, 6):
    print(f"   drop={d}: cosine(chunk0[{d}:{d+14}], offline[0:14]) = "
          f"{cos(eo0.numpy()[..., d:d+14], off.numpy()[..., 0:14]):.6f}")
