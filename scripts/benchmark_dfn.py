#!/usr/bin/env python3
"""
DeepFilterNet benchmark script for mlx-audio-swift.

Measures speed (offline + streaming) and accuracy (vs reference output)
before and after refactoring. Saves results to JSON for comparison.

Usage:
    # Build first
    swift build -c release

    # Run benchmark (uses defaults)
    python3 scripts/benchmark_dfn.py

    # Custom model/audio paths
    python3 scripts/benchmark_dfn.py \
        --model ~/Developer/ML-Models/DeepFilterNet3-MLX \
        --audio Tests/media/noisy_audio_10s.wav \
        --reference Tests/media/noisy_audio_10s_dfn3_reference.wav

    # Compare two saved results
    python3 scripts/benchmark_dfn.py --compare results_before.json results_after.json

    # Save results with a label
    python3 scripts/benchmark_dfn.py --label "before-refactor"
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_BINARY = REPO_ROOT / ".build" / "release" / "mlx-audio-swift-sts"
DEFAULT_MODEL = Path.home() / "Developer" / "ML-Models" / "DeepFilterNet3-MLX"
DEFAULT_AUDIO_10S = REPO_ROOT / "Tests" / "media" / "noisy_audio_10s.wav"
DEFAULT_AUDIO_52S = REPO_ROOT / "Tests" / "media" / "noisy_audio_1min.wav"
DEFAULT_REFERENCE = REPO_ROOT / "Tests" / "media" / "noisy_audio_10s_dfn3_reference.wav"
DEFAULT_RUNS = 5


# ── Audio I/O ─────────────────────────────────────────────────────────────────

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as float32 numpy array. Returns (samples, sample_rate).

    Handles both PCM (int16/int32) and IEEE float WAV files.
    Python's wave module doesn't support float WAV, so we parse the header manually.
    """
    import struct

    with open(str(path), "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError(f"Not a WAV file: {path}")
        f.read(4)  # file size
        wave_id = f.read(4)
        if wave_id != b"WAVE":
            raise ValueError(f"Not a WAV file: {path}")

        fmt_tag = None
        nch = None
        sr = None
        bits = None
        data_bytes = None

        while True:
            chunk_header = f.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id = chunk_header[:4]
            chunk_size = struct.unpack("<I", chunk_header[4:8])[0]

            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                fmt_tag = struct.unpack("<H", fmt_data[0:2])[0]
                nch = struct.unpack("<H", fmt_data[2:4])[0]
                sr = struct.unpack("<I", fmt_data[4:8])[0]
                bits = struct.unpack("<H", fmt_data[14:16])[0]
            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
            else:
                f.seek(chunk_size, 1)  # skip unknown chunks

    if fmt_tag is None or data_bytes is None:
        raise ValueError(f"Invalid WAV file: {path}")

    if fmt_tag == 1:  # PCM
        if bits == 16:
            data = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif bits == 32:
            data = np.frombuffer(data_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif bits == 24:
            # 24-bit PCM: 3 bytes per sample, little-endian
            n_samples = len(data_bytes) // 3
            raw = np.frombuffer(data_bytes, dtype=np.uint8).reshape(-1, 3)
            i32 = (raw[:, 0].astype(np.int32)
                   | (raw[:, 1].astype(np.int32) << 8)
                   | (raw[:, 2].astype(np.int32) << 16))
            # Sign extend from 24-bit
            i32[i32 >= 0x800000] -= 0x1000000
            data = i32.astype(np.float32) / 8388608.0
        else:
            raise ValueError(f"Unsupported PCM bit depth: {bits}")
    elif fmt_tag == 3:  # IEEE float
        if bits == 32:
            data = np.frombuffer(data_bytes, dtype=np.float32).copy()
        elif bits == 64:
            data = np.frombuffer(data_bytes, dtype=np.float64).astype(np.float32)
        else:
            raise ValueError(f"Unsupported float bit depth: {bits}")
    else:
        raise ValueError(f"Unsupported WAV format tag: {fmt_tag}")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)  # mix to mono

    return data, sr


# ── Accuracy Metrics ──────────────────────────────────────────────────────────

def waveform_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two waveforms.

    Uses np.corrcoef which handles float64 conversion internally,
    avoiding a numpy/Python 3.14 float32 reduction bug.
    """
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def log_spectrogram_mae(a: np.ndarray, b: np.ndarray,
                        fft_len: int = 960, hop_len: int = 480) -> float:
    """Mean absolute error between log-magnitude spectrograms (dB)."""
    n = min(len(a), len(b))
    # Use float64 throughout to avoid numpy/Python 3.14 float32 reduction bugs
    a64 = a[:n].astype(np.float64)
    b64 = b[:n].astype(np.float64)

    def log_spec(x):
        pad = fft_len - (len(x) % hop_len)
        x = np.pad(x, (0, pad))
        n_frames = (len(x) - fft_len) // hop_len + 1
        frames = np.stack([x[i * hop_len : i * hop_len + fft_len] for i in range(n_frames)])
        window = np.hanning(fft_len).astype(np.float64)
        spec = np.fft.rfft(frames * window, n=fft_len)
        mag = np.abs(spec)
        return 20.0 * np.log10(np.maximum(mag, 1e-10))

    sa = log_spec(a64)
    sb = log_spec(b64)
    n_frames = min(sa.shape[0], sb.shape[0])
    return float(np.mean(np.abs(sa[:n_frames] - sb[:n_frames])))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum absolute sample difference."""
    n = min(len(a), len(b))
    diff = a[:n].astype(np.float64) - b[:n].astype(np.float64)
    return float(np.max(np.abs(diff)))


def rms_error(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean square error between two waveforms."""
    n = min(len(a), len(b))
    diff = a[:n].astype(np.float64) - b[:n].astype(np.float64)
    return float(np.sqrt(np.mean(diff**2)))


# ── CLI Runner ────────────────────────────────────────────────────────────────

def run_cli(model_dir: str, audio_path: str, output_path: str,
            mode: str = "short", env_overrides: dict = None) -> tuple[float, str]:
    """
    Run the mlx-audio-swift-sts CLI and return (elapsed_seconds, stdout).
    """
    if not CLI_BINARY.exists():
        print(f"ERROR: CLI binary not found at {CLI_BINARY}")
        print("       Run: swift build -c release")
        sys.exit(1)

    cmd = [
        str(CLI_BINARY),
        "--model", str(model_dir),
        "--audio", str(audio_path),
        "-o", str(output_path),
        "--mode", mode,
    ]

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"ERROR: CLI failed (exit {result.returncode})")
        print(result.stderr)
        sys.exit(1)

    return elapsed, result.stdout


# ── Benchmark Functions ───────────────────────────────────────────────────────

def benchmark_speed(model_dir: str, audio_path: str, mode: str,
                    runs: int = DEFAULT_RUNS, label: str = "") -> dict:
    """Run the CLI multiple times and collect timing stats."""
    audio_name = Path(audio_path).stem
    audio_data, sr = load_wav(audio_path)
    audio_length = len(audio_data) / sr

    print(f"\n  {label}{mode} mode — {audio_name} ({audio_length:.1f}s @ {sr}Hz)")
    print(f"  {'─' * 50}")

    times = []
    for i in range(runs):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            elapsed, stdout = run_cli(model_dir, audio_path, tmp.name, mode=mode)
            times.append(elapsed)
            rtf = audio_length / elapsed
            marker = " (warmup)" if i == 0 else ""
            print(f"  Run {i+1}/{runs}: {elapsed:.3f}s ({rtf:.1f}x RT){marker}")

    warm_times = times[1:] if len(times) > 1 else times
    best = min(warm_times)
    worst = max(warm_times)
    avg = sum(warm_times) / len(warm_times)
    median = sorted(warm_times)[len(warm_times) // 2]

    result = {
        "mode": mode,
        "audio_file": str(audio_path),
        "audio_length_s": round(audio_length, 3),
        "sample_rate": sr,
        "runs": runs,
        "first_run_s": round(times[0], 4),
        "warm_avg_s": round(avg, 4),
        "warm_median_s": round(median, 4),
        "warm_best_s": round(best, 4),
        "warm_worst_s": round(worst, 4),
        "warm_rtf_avg": round(audio_length / avg, 2),
        "warm_rtf_best": round(audio_length / best, 2),
        "all_times_s": [round(t, 4) for t in times],
    }

    if mode == "stream":
        result["warm_per_hop_ms"] = round(avg / (audio_length / (480 / sr)) * 1000, 3)

    print(f"  ────────────────────────────────────────────────")
    print(f"  Avg (warm): {avg:.3f}s  |  Best: {best:.3f}s  |  RTF: {audio_length/avg:.1f}x")

    return result


def benchmark_accuracy(model_dir: str, audio_path: str, reference_path: str,
                       mode: str = "short") -> dict:
    """Run the CLI once and compare output against reference."""
    print(f"\n  Accuracy — {mode} mode vs reference")
    print(f"  {'─' * 50}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        elapsed, stdout = run_cli(model_dir, audio_path, tmp_path, mode=mode)

        output_data, output_sr = load_wav(tmp_path)
        ref_data, ref_sr = load_wav(reference_path)

        corr = waveform_correlation(output_data, ref_data)
        spec_mae = log_spectrogram_mae(output_data, ref_data)
        max_diff = max_abs_diff(output_data, ref_data)
        rms = rms_error(output_data, ref_data)

        result = {
            "mode": mode,
            "audio_file": str(audio_path),
            "reference_file": str(reference_path),
            "output_samples": len(output_data),
            "reference_samples": len(ref_data),
            "waveform_correlation": round(corr, 6),
            "log_spectrogram_mae_db": round(spec_mae, 4),
            "max_abs_diff": round(max_diff, 6),
            "rms_error": round(rms, 6),
            "pass_correlation": corr > 0.99,
            "pass_spectral_mae": spec_mae < 8.0,
        }

        status = "PASS" if result["pass_correlation"] and result["pass_spectral_mae"] else "FAIL"
        print(f"  Correlation:     {corr:.6f}  (threshold: >0.99)  {'✓' if result['pass_correlation'] else '✗'}")
        print(f"  Spectral MAE:    {spec_mae:.4f} dB  (threshold: <8.0)  {'✓' if result['pass_spectral_mae'] else '✗'}")
        print(f"  Max abs diff:    {max_diff:.6f}")
        print(f"  RMS error:       {rms:.6f}")
        print(f"  Result:          {status}")

        return result
    finally:
        os.unlink(tmp_path)


def benchmark_streaming_accuracy(model_dir: str, audio_path: str,
                                 reference_path: str) -> dict:
    """Compare streaming output against offline reference."""
    print(f"\n  Streaming vs Offline accuracy")
    print(f"  {'─' * 50}")

    # Generate offline output as ground truth
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_offline:
        offline_path = tmp_offline.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_stream:
        stream_path = tmp_stream.name

    try:
        run_cli(model_dir, audio_path, offline_path, mode="short")
        run_cli(model_dir, audio_path, stream_path, mode="stream")

        offline_data, _ = load_wav(offline_path)
        stream_data, _ = load_wav(stream_path)

        corr = waveform_correlation(stream_data, offline_data)
        spec_mae = log_spectrogram_mae(stream_data, offline_data)
        max_diff = max_abs_diff(stream_data, offline_data)
        rms = rms_error(stream_data, offline_data)

        result = {
            "comparison": "streaming_vs_offline",
            "waveform_correlation": round(corr, 6),
            "log_spectrogram_mae_db": round(spec_mae, 4),
            "max_abs_diff": round(max_diff, 6),
            "rms_error": round(rms, 6),
        }

        print(f"  Correlation:     {corr:.6f}")
        print(f"  Spectral MAE:    {spec_mae:.4f} dB")
        print(f"  Max abs diff:    {max_diff:.6f}")
        print(f"  RMS error:       {rms:.6f}")

        return result
    finally:
        os.unlink(offline_path)
        os.unlink(stream_path)


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_results(before_path: str, after_path: str):
    """Compare two saved benchmark JSON files."""
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    print(f"\n{'═' * 70}")
    print(f"  COMPARISON: {before.get('label', before_path)}")
    print(f"          vs: {after.get('label', after_path)}")
    print(f"{'═' * 70}")

    # Speed comparison
    print(f"\n  SPEED")
    print(f"  {'─' * 60}")
    print(f"  {'Test':<35} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'─' * 60}")

    for b_test in before.get("speed", []):
        key = f"{b_test['mode']}_{Path(b_test['audio_file']).stem}"
        a_test = None
        for t in after.get("speed", []):
            if f"{t['mode']}_{Path(t['audio_file']).stem}" == key:
                a_test = t
                break
        if a_test:
            b_val = b_test["warm_avg_s"]
            a_val = a_test["warm_avg_s"]
            pct = ((a_val - b_val) / b_val) * 100
            sign = "+" if pct > 0 else ""
            label = f"{b_test['mode']} ({Path(b_test['audio_file']).stem})"
            print(f"  {label:<35} {b_val:>7.3f}s {a_val:>7.3f}s {sign}{pct:>6.1f}%")

    # Accuracy comparison
    print(f"\n  ACCURACY")
    print(f"  {'─' * 60}")
    print(f"  {'Metric':<35} {'Before':>8} {'After':>8} {'Delta':>8}")
    print(f"  {'─' * 60}")

    for mode in ["short", "stream"]:
        b_acc = None
        a_acc = None
        for t in before.get("accuracy", []):
            if t.get("mode") == mode:
                b_acc = t
        for t in after.get("accuracy", []):
            if t.get("mode") == mode:
                a_acc = t

        if b_acc and a_acc:
            for metric, fmt in [("waveform_correlation", ".6f"),
                                ("log_spectrogram_mae_db", ".4f"),
                                ("rms_error", ".6f")]:
                b_val = b_acc[metric]
                a_val = a_acc[metric]
                diff = a_val - b_val
                sign = "+" if diff > 0 else ""
                label = f"{mode} {metric.replace('_', ' ')}"
                print(f"  {label:<35} {b_val:>{fmt}} {a_val:>{fmt}} {sign}{diff:>{fmt}}")

    # Streaming vs offline
    b_svo = before.get("streaming_vs_offline")
    a_svo = after.get("streaming_vs_offline")
    if b_svo and a_svo:
        print(f"\n  STREAMING vs OFFLINE PARITY")
        print(f"  {'─' * 60}")
        for metric in ["waveform_correlation", "log_spectrogram_mae_db", "rms_error"]:
            b_val = b_svo[metric]
            a_val = a_svo[metric]
            diff = a_val - b_val
            sign = "+" if diff > 0 else ""
            label = metric.replace("_", " ")
            print(f"  {label:<35} {b_val:>8.6f} {a_val:>8.6f} {sign}{diff:>8.6f}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DeepFilterNet in mlx-audio-swift (speed + accuracy)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Path to DeepFilterNet model directory")
    parser.add_argument("--audio", type=str, default=str(DEFAULT_AUDIO_10S),
                        help="Path to 10s noisy test audio")
    parser.add_argument("--audio-long", type=str, default=str(DEFAULT_AUDIO_52S),
                        help="Path to longer test audio (52s)")
    parser.add_argument("--reference", type=str, default=str(DEFAULT_REFERENCE),
                        help="Path to reference enhanced audio (for accuracy)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Number of benchmark runs (default: {DEFAULT_RUNS})")
    parser.add_argument("--label", type=str, default="",
                        help="Label for this benchmark run (e.g. 'before-refactor')")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSON path (default: benchmark_dfn_<label>.json)")
    parser.add_argument("--skip-long", action="store_true",
                        help="Skip the 52s audio benchmark")
    parser.add_argument("--skip-streaming", action="store_true",
                        help="Skip streaming benchmarks")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="Skip accuracy tests")
    parser.add_argument("--speed-only", action="store_true",
                        help="Only run speed benchmarks")
    parser.add_argument("--accuracy-only", action="store_true",
                        help="Only run accuracy tests")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two saved benchmark JSON files")
    args = parser.parse_args()

    # Compare mode
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # Validate inputs
    model_dir = Path(args.model)
    audio_path = Path(args.audio)
    audio_long_path = Path(args.audio_long)
    reference_path = Path(args.reference)

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        sys.exit(1)
    if not CLI_BINARY.exists():
        print(f"ERROR: CLI binary not found at {CLI_BINARY}")
        print("       Run: swift build -c release")
        sys.exit(1)

    do_speed = not args.accuracy_only
    do_accuracy = not args.speed_only

    # ── Header ──
    git_hash = "unknown"
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, cwd=str(REPO_ROOT))
        if r.returncode == 0:
            git_hash = r.stdout.strip()
    except Exception:
        pass

    label = args.label or git_hash
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    print(f"{'═' * 70}")
    print(f"  DeepFilterNet Benchmark — mlx-audio-swift")
    print(f"  Label:     {label}")
    print(f"  Git:       {git_hash}")
    print(f"  Time:      {timestamp}")
    print(f"  Model:     {model_dir}")
    print(f"  Audio:     {audio_path.name} ({audio_path})")
    if not args.skip_long and audio_long_path.exists():
        print(f"  Audio 52s: {audio_long_path.name}")
    print(f"  Reference: {reference_path.name}")
    print(f"  Runs:      {args.runs}")
    print(f"{'═' * 70}")

    results = {
        "label": label,
        "git_hash": git_hash,
        "timestamp": timestamp,
        "model": str(model_dir),
        "speed": [],
        "accuracy": [],
    }

    # ── Speed Benchmarks ──
    if do_speed:
        print(f"\n{'─' * 70}")
        print(f"  SPEED BENCHMARKS")
        print(f"{'─' * 70}")

        # Offline 10s
        results["speed"].append(
            benchmark_speed(str(model_dir), str(audio_path),
                            mode="short", runs=args.runs, label="[10s] ")
        )

        # Offline 52s
        if not args.skip_long and audio_long_path.exists():
            results["speed"].append(
                benchmark_speed(str(model_dir), str(audio_long_path),
                                mode="short", runs=args.runs, label="[52s] ")
            )

        # Streaming 10s
        if not args.skip_streaming:
            results["speed"].append(
                benchmark_speed(str(model_dir), str(audio_path),
                                mode="stream", runs=args.runs, label="[10s] ")
            )

            # Streaming 52s
            if not args.skip_long and audio_long_path.exists():
                results["speed"].append(
                    benchmark_speed(str(model_dir), str(audio_long_path),
                                    mode="stream", runs=args.runs, label="[52s] ")
                )

    # ── Accuracy Benchmarks ──
    if do_accuracy:
        print(f"\n{'─' * 70}")
        print(f"  ACCURACY BENCHMARKS")
        print(f"{'─' * 70}")

        if not reference_path.exists():
            print(f"  WARNING: Reference file not found: {reference_path}")
            print(f"  Skipping accuracy tests.")
        else:
            # Offline vs reference
            results["accuracy"].append(
                benchmark_accuracy(str(model_dir), str(audio_path),
                                   str(reference_path), mode="short")
            )

            # Streaming vs reference
            if not args.skip_streaming:
                results["accuracy"].append(
                    benchmark_accuracy(str(model_dir), str(audio_path),
                                       str(reference_path), mode="stream")
                )

                # Streaming vs offline parity
                results["streaming_vs_offline"] = benchmark_streaming_accuracy(
                    str(model_dir), str(audio_path), str(reference_path)
                )

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY — {label}")
    print(f"{'═' * 70}")

    if results["speed"]:
        print(f"\n  {'Mode':<20} {'Audio':<15} {'Avg (warm)':>12} {'Best':>10} {'RTF':>8}")
        print(f"  {'─' * 65}")
        for s in results["speed"]:
            audio_name = Path(s["audio_file"]).stem[:14]
            print(f"  {s['mode']:<20} {audio_name:<15} {s['warm_avg_s']:>10.3f}s "
                  f"{s['warm_best_s']:>8.3f}s {s['warm_rtf_avg']:>7.1f}x")

    if results["accuracy"]:
        print(f"\n  {'Mode':<20} {'Correlation':>12} {'Spec MAE':>10} {'Status':>8}")
        print(f"  {'─' * 50}")
        for a in results["accuracy"]:
            status = "PASS" if a["pass_correlation"] and a["pass_spectral_mae"] else "FAIL"
            print(f"  {a['mode']:<20} {a['waveform_correlation']:>12.6f} "
                  f"{a['log_spectrogram_mae_db']:>8.4f} dB {status:>8}")

    # ── Save ──
    if args.output:
        out_path = args.output
    else:
        safe_label = label.replace(" ", "_").replace("/", "_")
        out_path = str(REPO_ROOT / f"benchmark_dfn_{safe_label}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"  Compare later:    python3 scripts/benchmark_dfn.py --compare {out_path} <after.json>")
    print()


if __name__ == "__main__":
    main()
