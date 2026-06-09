#!/usr/bin/env python3
"""Decisive transcript test: does fixed-shape uniform-121 feeding reproduce NeMo's native
cache-aware streaming TRANSCRIPT? (Frame-cosine-vs-offline was a mirage; text is ground truth.)

Decodes three encoder outputs through the same RNN-T:
  REF   : NeMo native CacheAwareStreamingAudioBuffer streaming (variable 105/57 chunks)
  UNI   : uniform-121 feeding [9 prev-mel ++ 112 new], stride 112, drop=0 (Swift-replicable)
  OFF   : offline (sanity)
"""
import argparse
import numpy as np
import torch
import librosa
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

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


def decode(enc_out, n):
    eo = torch.as_tensor(enc_out, dtype=torch.float32)
    el = torch.tensor([n], dtype=torch.int64)
    with torch.no_grad():
        hyps = m.decoding.rnnt_decoder_predictions_tensor(eo, el, return_hypotheses=False)
    return hyps[0] if isinstance(hyps, (list, tuple)) else hyps


# OFF (offline sanity)
enc.set_default_att_context_size(a.att_context)
with torch.no_grad():
    off, off_len = enc(audio_signal=mel, length=mlen)
print(f"OFF: {decode(off, int(off_len.item()))!r}")

# REF: native buffer streaming
enc.setup_streaming_params(att_context_size=a.att_context)
cfg = enc.streaming_cfg
buf = CacheAwareStreamingAudioBuffer(model=m, online_normalization=False)
buf.append_audio(audio[0].numpy())
st = enc.get_initial_cache_state(batch_size=1)
ref_fr = []
for fed, plen in buf:
    last = buf.is_buffer_empty()
    with torch.no_grad():
        eo, el, *st = enc.cache_aware_stream_step(
            processed_signal=fed, processed_signal_length=plen,
            cache_last_channel=st[0], cache_last_time=st[1],
            cache_last_channel_len=st[2], keep_all_outputs=last)
    ref_fr.append(eo[..., : int(el.flatten()[0])].numpy())
ref = np.concatenate(ref_fr, axis=-1)
print(f"REF native-stream ({ref.shape[-1]} fr): {decode(ref, ref.shape[-1])!r}")

# UNI: uniform-121 feeding
PRE = cfg.pre_encode_cache_size[-1] if isinstance(cfg.pre_encode_cache_size, (list, tuple)) else int(cfg.pre_encode_cache_size)
NEW = cfg.chunk_size[-1] if isinstance(cfg.chunk_size, (list, tuple)) else int(cfg.chunk_size)
F = PRE + NEW
st = enc.get_initial_cache_state(batch_size=1)
uni_fr = []
p = 0
while p < T:
    win = torch.zeros(1, mel.shape[1], F)
    if p > 0:
        s = max(0, p - PRE); win[..., PRE - (p - s):PRE] = mel[..., s:p]
    e = min(T, p + NEW); win[..., PRE:PRE + (e - p)] = mel[..., p:e]
    with torch.no_grad():
        eo, el, *st = enc.cache_aware_stream_step(
            processed_signal=win, processed_signal_length=torch.tensor([F]),
            cache_last_channel=st[0], cache_last_time=st[1],
            cache_last_channel_len=st[2], keep_all_outputs=True)
    real_new = min(NEW, T - p)
    keep = min(int(eo.shape[-1]), max(1, (real_new + 7) // 8))
    uni_fr.append(eo[..., :keep].numpy())
    p += NEW
uni = np.concatenate(uni_fr, axis=-1)
print(f"UNI uniform-{F} ({uni.shape[-1]} fr): {decode(uni, uni.shape[-1])!r}")
