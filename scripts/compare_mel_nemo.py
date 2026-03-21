"""
Compare log-mel spectrogram: NeMo reference vs our Swift-equivalent Python reimplementation.

Run on pc.lan (CUDA env):
  cd /mnt/d/Projects/nemo-test
  uv run python compare_mel_nemo.py --audio intention.wav

Produces numeric comparison for PR #108 review.
"""
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa


def swift_compatible_mel(audio_np: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Replicate our Swift ParakeetAudio.logMelSpectrogram exactly,
    matching NeMo's FilterbankFeatures pipeline step-by-step."""
    n_fft = 512
    win_length = int(0.025 * sr)   # 400
    hop_length = int(0.01 * sr)    # 160
    n_mels = 80
    preemph = 0.97
    log_zero_guard = 2**-24

    # Pre-emphasis (same as NeMo forward)
    x = np.concatenate([audio_np[:1], audio_np[1:] - preemph * audio_np[:-1]])

    # Window: symmetric hann (periodic=False) — exactly NeMo's torch.hann_window(400, periodic=False)
    window = torch.hann_window(win_length, periodic=False)

    # STFT — match NeMo exactly: win_length=400, n_fft=512, center=True, pad_mode="constant"
    # PyTorch internally zero-pads the window from 400 to 512
    x_tensor = torch.from_numpy(x.astype(np.float32))
    stft_out = torch.stft(
        x_tensor, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, pad_mode="constant",
        return_complex=True
    )

    # Magnitude then power (NeMo: sqrt(re²+im²) then .pow(2))
    power = torch.abs(stft_out).pow(2).numpy().T  # [frames, freq_bins]

    # Mel filterbank: slaney scale + slaney norm (same librosa call as NeMo)
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mels,
        fmin=0, fmax=None,
        norm="slaney", htk=False
    ).T  # [freq_bins, n_mels]

    mel = power @ mel_basis
    mel = np.log(mel + log_zero_guard)

    # NeMo get_seq_len: floor_divide((audio_len + n_fft - n_fft), hop_length)
    seq_len = len(audio_np) // hop_length

    # Per-feature normalization with ddof=1, only over seq_len valid frames (matching NeMo)
    valid = mel[:seq_len]
    mean = valid.mean(axis=0, keepdims=True)
    std = np.sqrt(np.sum((valid - mean) ** 2, axis=0, keepdims=True) / (seq_len - 1))
    normalized = (mel - mean) / (std + 1e-5)

    return normalized, seq_len


def nemo_mel(audio_np: np.ndarray, sr: int = 16000):
    """Extract mel via NeMo's FilterbankFeatures."""
    from nemo.collections.asr.parts.preprocessing.features import FilterbankFeatures

    fb = FilterbankFeatures(
        sample_rate=sr,
        n_window_size=int(0.025 * sr),
        n_window_stride=int(0.01 * sr),
        window="hann",
        normalize="per_feature",
        n_fft=512,
        nfilt=80,
        dither=0.0,
        preemph=0.97,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        mel_norm="slaney",
        stft_conv=False,
    )

    audio_tensor = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
    length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long)

    with torch.no_grad():
        mel_out, mel_len = fb(audio_tensor, length)

    return mel_out.squeeze(0).numpy().T, mel_len.item()  # [frames, n_mels], seq_len


def main():
    parser = argparse.ArgumentParser(description="Compare mel: Swift-equiv vs NeMo")
    parser.add_argument("--audio", required=True, help="Path to .wav file (16kHz mono)")
    args = parser.parse_args()

    audio_np, sr = sf.read(args.audio, dtype="float32")
    if sr != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        sr = 16000

    print(f"Audio: {args.audio} ({len(audio_np)} samples, {len(audio_np)/sr:.2f}s)")
    print()

    swift_mel, swift_seq_len = swift_compatible_mel(audio_np, sr)
    ref_mel, nemo_seq_len = nemo_mel(audio_np, sr)

    print(f"Shape — Swift-equiv: {swift_mel.shape}, NeMo: {ref_mel.shape}")
    print(f"Seq len — Swift-equiv: {swift_seq_len}, NeMo: {nemo_seq_len}")

    # Compare over valid frames (seq_len)
    valid_frames = min(swift_seq_len, nemo_seq_len, swift_mel.shape[0], ref_mel.shape[0])
    s = swift_mel[:valid_frames]
    r = ref_mel[:valid_frames]

    diff = np.abs(s - r)
    max_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f"\n=== Log-Mel Comparison (valid frames={valid_frames}, mels=80) ===")
    print(f"Max absolute diff:  {diff.max():.8f}  (frame={max_idx[0]}, mel={max_idx[1]})")
    print(f"Mean absolute diff: {diff.mean():.8f}")
    print(f"Median abs diff:    {np.median(diff):.8f}")
    print(f"99th pct abs diff:  {np.percentile(diff, 99):.8f}")
    print()
    print(f"Swift-equiv — mean: {s.mean():.6f}, std: {s.std():.6f}, min: {s.min():.6f}, max: {s.max():.6f}")
    print(f"NeMo ref    — mean: {r.mean():.6f}, std: {r.std():.6f}, min: {r.min():.6f}, max: {r.max():.6f}")
    print()

    # Per-feature correlation
    corr = np.array([np.corrcoef(s[:, i], r[:, i])[0, 1] for i in range(80)])
    print(f"Per-feature correlation — min: {corr.min():.6f}, mean: {corr.mean():.6f}, max: {corr.max():.6f}")

    # Per-frame max diff
    frame_max = diff.max(axis=1)
    print(f"\nPer-frame max diff (first 5): {np.array2string(frame_max[:5], precision=8)}")
    print(f"Per-frame max diff (last 5):  {np.array2string(frame_max[-5:], precision=8)}")
    print(f"Per-frame max diff (all):     mean={frame_max.mean():.8f}, max={frame_max.max():.8f}")

    if diff.max() < 1e-5:
        print("\n✅ EXACT MATCH — bit-identical within float32 epsilon")
    elif diff.max() < 0.001:
        print("\n✅ MATCH — differences within float32 precision")
    elif diff.max() < 0.01:
        print("\n✅ CLOSE — minor numerical differences")
    elif diff.max() < 0.05:
        print("\n⚠️  ACCEPTABLE — small differences, functionally equivalent")
    else:
        print(f"\n❌ MISMATCH — max diff {diff.max():.4f}")


if __name__ == "__main__":
    main()
