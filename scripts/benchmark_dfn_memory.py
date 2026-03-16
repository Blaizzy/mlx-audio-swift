#!/usr/bin/env python3
"""
DeepFilterNet memory usage benchmark for mlx-audio-swift.

Measures peak RSS and peak memory footprint for offline and streaming modes
using macOS /usr/bin/time -l.

Usage:
    python3 scripts/benchmark_dfn_memory.py
    python3 scripts/benchmark_dfn_memory.py --label "before-refactor"
    python3 scripts/benchmark_dfn_memory.py --compare memory_before.json memory_after.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_BINARY = REPO_ROOT / ".build" / "release" / "mlx-audio-swift-sts"
DEFAULT_MODEL = Path.home() / "Developer" / "ML-Models" / "DeepFilterNet3-MLX"
DEFAULT_AUDIO_10S = REPO_ROOT / "Tests" / "media" / "noisy_audio_10s.wav"
DEFAULT_AUDIO_1MIN = REPO_ROOT / "Tests" / "media" / "noisy_audio_1min.wav"


def measure_memory(model_dir: str, audio_path: str, mode: str) -> dict:
    """Run CLI via /usr/bin/time -l and parse memory stats."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp_path = tmp.name

    cmd = [
        "/usr/bin/time", "-l",
        str(CLI_BINARY),
        "--model", model_dir,
        "--audio", audio_path,
        "-o", tmp_path,
        "--mode", mode,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # /usr/bin/time writes to stderr
    output = result.stderr

    if result.returncode != 0:
        # Check if the error is from CLI, not from time
        if "ERROR" in output or "error" in result.stdout.lower():
            print(f"  ERROR: CLI failed (exit {result.returncode})")
            print(output[:500])
            return {}

    parsed = {"mode": mode, "audio_file": str(audio_path)}

    # Parse /usr/bin/time -l output
    m = re.search(r"([\d.]+)\s+real\s+([\d.]+)\s+user\s+([\d.]+)\s+sys", output)
    if m:
        parsed["real_s"] = float(m.group(1))
        parsed["user_s"] = float(m.group(2))
        parsed["sys_s"] = float(m.group(3))

    m = re.search(r"(\d+)\s+maximum resident set size", output)
    if m:
        rss_bytes = int(m.group(1))
        parsed["peak_rss_bytes"] = rss_bytes
        parsed["peak_rss_mb"] = round(rss_bytes / 1024 / 1024, 1)

    m = re.search(r"(\d+)\s+peak memory footprint", output)
    if m:
        footprint_bytes = int(m.group(1))
        parsed["peak_footprint_bytes"] = footprint_bytes
        parsed["peak_footprint_mb"] = round(footprint_bytes / 1024 / 1024, 1)

    m = re.search(r"(\d+)\s+page reclaims", output)
    if m:
        parsed["page_reclaims"] = int(m.group(1))

    m = re.search(r"(\d+)\s+page faults", output)
    if m:
        parsed["page_faults"] = int(m.group(1))

    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    return parsed


def run_memory_benchmark(model_dir: str, audio_path: str,
                         label: str, runs: int = 3) -> list:
    """Measure memory for both modes, multiple runs."""
    audio_name = Path(audio_path).stem
    results = []

    for mode in ["short", "stream"]:
        print(f"\n  {label}{mode} mode — {audio_name}")
        print(f"  {'─' * 55}")

        measurements = []
        for i in range(runs):
            m = measure_memory(model_dir, audio_path, mode)
            if not m:
                continue
            measurements.append(m)
            rss_mb = m.get("peak_rss_mb", 0)
            foot_mb = m.get("peak_footprint_mb", 0)
            marker = " (warmup)" if i == 0 else ""
            print(f"    Run {i+1}/{runs}: RSS={rss_mb:.1f}MB  "
                  f"Footprint={foot_mb:.1f}MB  "
                  f"({m.get('real_s', 0):.3f}s){marker}")

        if not measurements:
            continue

        warm = measurements[1:] if len(measurements) > 1 else measurements
        avg_rss = sum(m["peak_rss_mb"] for m in warm) / len(warm)
        avg_foot = sum(m["peak_footprint_mb"] for m in warm) / len(warm)

        entry = {
            "mode": mode,
            "audio_file": str(audio_path),
            "audio_name": audio_name,
            "runs": runs,
            "avg_peak_rss_mb": round(avg_rss, 1),
            "avg_peak_footprint_mb": round(avg_foot, 1),
            "all_rss_mb": [m["peak_rss_mb"] for m in measurements],
            "all_footprint_mb": [m["peak_footprint_mb"] for m in measurements],
        }
        results.append(entry)

        print(f"    Avg (warm): RSS={avg_rss:.1f}MB  Footprint={avg_foot:.1f}MB")

    return results


def compare_results(before_path: str, after_path: str):
    """Compare two memory benchmark JSON files."""
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    print(f"\n{'═' * 65}")
    print(f"  MEMORY COMPARISON")
    print(f"  Before: {before.get('label', before_path)}")
    print(f"  After:  {after.get('label', after_path)}")
    print(f"{'═' * 65}")

    print(f"\n  {'Test':<30} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'─' * 62}")

    for b_entry in before.get("memory", []):
        key = f"{b_entry['mode']}_{b_entry['audio_name']}"
        a_entry = None
        for e in after.get("memory", []):
            if f"{e['mode']}_{e['audio_name']}" == key:
                a_entry = e
                break
        if not a_entry:
            continue

        for metric, unit in [("avg_peak_rss_mb", "MB RSS"),
                             ("avg_peak_footprint_mb", "MB Foot")]:
            b_val = b_entry[metric]
            a_val = a_entry[metric]
            delta = a_val - b_val
            pct = (delta / b_val) * 100 if b_val else 0
            sign = "+" if delta > 0 else ""
            label = f"{b_entry['mode']} {b_entry['audio_name'][:12]} {unit}"
            print(f"  {label:<30} {b_val:>8.1f}MB {a_val:>8.1f}MB "
                  f"{sign}{delta:>6.1f}MB ({sign}{pct:.0f}%)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="DeepFilterNet memory usage benchmark")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--audio", type=str, default=str(DEFAULT_AUDIO_10S))
    parser.add_argument("--audio-long", type=str, default=str(DEFAULT_AUDIO_1MIN))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

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

    print(f"{'═' * 65}")
    print(f"  DeepFilterNet Memory Benchmark")
    print(f"  Label: {label}  Git: {git_hash}  Time: {timestamp}")
    print(f"{'═' * 65}")

    all_memory = []

    # 10s audio
    all_memory.extend(
        run_memory_benchmark(args.model, args.audio, label="[10s] ", runs=args.runs)
    )

    # 1min audio
    audio_long = Path(args.audio_long)
    if not args.skip_long and audio_long.exists():
        all_memory.extend(
            run_memory_benchmark(args.model, str(audio_long), label="[1m] ", runs=args.runs)
        )

    # Summary
    print(f"\n  {'═' * 55}")
    print(f"  MEMORY SUMMARY")
    print(f"  {'─' * 55}")
    print(f"  {'Test':<30} {'Peak RSS':>10} {'Footprint':>12}")
    print(f"  {'─' * 55}")
    for m in all_memory:
        label_str = f"{m['mode']} ({m['audio_name'][:14]})"
        print(f"  {label_str:<30} {m['avg_peak_rss_mb']:>8.1f}MB "
              f"{m['avg_peak_footprint_mb']:>10.1f}MB")

    results = {
        "label": label,
        "git_hash": git_hash,
        "timestamp": timestamp,
        "model": args.model,
        "memory": all_memory,
    }

    if args.output:
        out_path = args.output
    else:
        safe_label = label.replace(" ", "_").replace("/", "_")
        out_path = str(REPO_ROOT / f"memory_{safe_label}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out_path}")
    print(f"  Compare:  python3 scripts/benchmark_dfn_memory.py "
          f"--compare {out_path} <after.json>")
    print()


if __name__ == "__main__":
    main()
