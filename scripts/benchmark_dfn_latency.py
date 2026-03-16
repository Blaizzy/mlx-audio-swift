#!/usr/bin/env python3
"""
DeepFilterNet streaming latency benchmark for mlx-audio-swift.

Measures per-hop latency for live audio call suitability.
At 48kHz with hop=480 samples, each hop = 10ms of audio.
For real-time, processing must complete within 10ms per hop.

Usage:
    # Build first
    swift build -c release

    # Run latency benchmark
    python3 scripts/benchmark_dfn_latency.py

    # Save baseline before refactoring
    python3 scripts/benchmark_dfn_latency.py --label "before-refactor"

    # Compare before/after
    python3 scripts/benchmark_dfn_latency.py --compare latency_before-refactor.json latency_after-refactor.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_BINARY = REPO_ROOT / ".build" / "release" / "mlx-audio-swift-sts"
DEFAULT_MODEL = Path.home() / "Developer" / "ML-Models" / "DeepFilterNet3-MLX"
DEFAULT_AUDIO = REPO_ROOT / "Tests" / "media" / "noisy_audio_10s.wav"

HOP_SAMPLES = 480
SAMPLE_RATE = 48000
HOP_DURATION_MS = HOP_SAMPLES / SAMPLE_RATE * 1000  # 10ms


def run_stream_profiled(model_dir: str, audio_path: str,
                        stage_eval: bool = False) -> dict:
    """Run streaming CLI with profiling and parse the output."""
    if not CLI_BINARY.exists():
        print(f"ERROR: CLI binary not found at {CLI_BINARY}")
        print("       Run: swift build -c release")
        sys.exit(1)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
    tmp_path = tmp.name

    cmd = [
        str(CLI_BINARY),
        "--model", model_dir,
        "--audio", audio_path,
        "-o", tmp_path,
        "--mode", "stream",
    ]

    env = os.environ.copy()
    env["DFN_STREAM_PROFILE"] = "1"
    env["DFN_STREAM_PROFILE_STAGE_EVAL"] = "1" if stage_eval else "0"

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall_time = time.perf_counter() - start

    if result.returncode != 0:
        print(f"ERROR: CLI failed (exit {result.returncode})")
        print(result.stderr)
        sys.exit(1)

    output = result.stdout

    # Parse profiling output
    parsed = {"wall_time_s": round(wall_time, 4)}

    # "Stream profile: hops=1001 total=4.687s perHop=4.683ms"
    m = re.search(r"hops=(\d+)\s+total=([\d.]+)s\s+perHop=([\d.]+)ms", output)
    if m:
        parsed["hops"] = int(m.group(1))
        parsed["total_s"] = float(m.group(2))
        parsed["per_hop_ms"] = float(m.group(3))

    # "Stream loop profile: process=4.707s (99.5%), emit(mlx->host+wav)=0.025s (0.5%)"
    m = re.search(r"process=([\d.]+)s\s+\(([\d.]+)%\).*emit.*=([\d.]+)s\s+\(([\d.]+)%\)", output)
    if m:
        parsed["process_s"] = float(m.group(1))
        parsed["process_pct"] = float(m.group(2))
        parsed["emit_s"] = float(m.group(3))
        parsed["emit_pct"] = float(m.group(4))

    # Per-stage breakdown: "  analysis:    0.364s (7.8%)"
    stages = {}
    for stage_match in re.finditer(r"^\s{2}(\w[\w.]+):\s+([\d.]+)s\s+\(([\d.]+)%", output, re.MULTILINE):
        name = stage_match.group(1)
        stages[name] = {
            "time_s": float(stage_match.group(2)),
            "pct": float(stage_match.group(3)),
        }
    if stages:
        parsed["stages"] = stages

    # "Enhanced 480000 samples (10.0s at 48000Hz)"
    m = re.search(r"Enhanced\s+(\d+)\s+samples\s+\(([\d.]+)s\s+at\s+(\d+)Hz\)", output)
    if m:
        parsed["audio_samples"] = int(m.group(1))
        parsed["audio_length_s"] = float(m.group(2))
        parsed["sample_rate"] = int(m.group(3))

    return parsed


def run_latency_benchmark(model_dir: str, audio_path: str,
                          runs: int = 3) -> dict:
    """Run multiple profiled streaming passes and collect stats."""
    audio_name = Path(audio_path).stem

    print(f"\n  Streaming latency — {audio_name}")
    print(f"  Hop: {HOP_SAMPLES} samples = {HOP_DURATION_MS:.0f}ms @ {SAMPLE_RATE}Hz")
    print(f"  Real-time budget: {HOP_DURATION_MS:.1f}ms per hop")
    print(f"  {'─' * 55}")

    # ── Without stage eval (real-world latency) ──
    print(f"\n  [Lazy eval] — real-world latency (graph build only per hop)")
    hop_times_lazy = []
    for i in range(runs):
        r = run_stream_profiled(model_dir, audio_path, stage_eval=False)
        per_hop = r.get("per_hop_ms", 0)
        hop_times_lazy.append(per_hop)
        marker = " (warmup)" if i == 0 else ""
        print(f"    Run {i+1}/{runs}: {per_hop:.3f}ms/hop  "
              f"(wall {r['wall_time_s']:.3f}s){marker}")

    warm_lazy = hop_times_lazy[1:] if len(hop_times_lazy) > 1 else hop_times_lazy
    avg_lazy = sum(warm_lazy) / len(warm_lazy)

    # ── With stage eval (true per-hop GPU time) ──
    print(f"\n  [Stage eval] — true per-hop GPU time (eval forced each stage)")
    hop_times_eval = []
    stage_breakdowns = []
    for i in range(runs):
        r = run_stream_profiled(model_dir, audio_path, stage_eval=True)
        per_hop = r.get("per_hop_ms", 0)
        hop_times_eval.append(per_hop)
        if r.get("stages"):
            stage_breakdowns.append(r["stages"])
        marker = " (warmup)" if i == 0 else ""
        print(f"    Run {i+1}/{runs}: {per_hop:.3f}ms/hop  "
              f"(wall {r['wall_time_s']:.3f}s){marker}")

    warm_eval = hop_times_eval[1:] if len(hop_times_eval) > 1 else hop_times_eval
    avg_eval = sum(warm_eval) / len(warm_eval)

    # ── Stage breakdown (average of warm runs with stage eval) ──
    avg_stages = {}
    if stage_breakdowns:
        warm_stages = stage_breakdowns[1:] if len(stage_breakdowns) > 1 else stage_breakdowns
        all_keys = set()
        for s in warm_stages:
            all_keys.update(s.keys())
        for key in sorted(all_keys):
            vals = [s[key]["time_s"] for s in warm_stages if key in s]
            pcts = [s[key]["pct"] for s in warm_stages if key in s]
            if vals:
                hops = r.get("hops", 1)
                avg_time = sum(vals) / len(vals)
                avg_pct = sum(pcts) / len(pcts)
                avg_stages[key] = {
                    "time_s": round(avg_time, 4),
                    "pct": round(avg_pct, 1),
                    "per_hop_ms": round(avg_time / hops * 1000, 3),
                }

    # ── Summary ──
    print(f"\n  {'═' * 55}")
    print(f"  LATENCY SUMMARY")
    print(f"  {'─' * 55}")
    print(f"  Real-time budget:     {HOP_DURATION_MS:.1f}ms per hop")
    print(f"  Lazy eval (avg warm): {avg_lazy:.3f}ms/hop  "
          f"({'OK' if avg_lazy < HOP_DURATION_MS else 'TOO SLOW'})")
    print(f"  Stage eval (avg warm):{avg_eval:.3f}ms/hop  "
          f"({'OK' if avg_eval < HOP_DURATION_MS else 'TOO SLOW'})")
    print(f"  Headroom (lazy):      {HOP_DURATION_MS - avg_lazy:.3f}ms "
          f"({(1 - avg_lazy/HOP_DURATION_MS)*100:.1f}%)")
    print(f"  Headroom (stage eval):{HOP_DURATION_MS - avg_eval:.3f}ms "
          f"({(1 - avg_eval/HOP_DURATION_MS)*100:.1f}%)")

    if avg_stages:
        print(f"\n  Per-stage breakdown (avg warm, stage eval):")
        print(f"  {'Stage':<20} {'Total':>8} {'Per-hop':>10} {'%':>6}")
        print(f"  {'─' * 48}")
        for name, s in avg_stages.items():
            indent = "  " if "." in name else ""
            print(f"  {indent}{name:<18} {s['time_s']:>7.3f}s {s['per_hop_ms']:>8.3f}ms {s['pct']:>5.1f}%")

    result = {
        "audio_file": str(audio_path),
        "audio_length_s": r.get("audio_length_s", 0),
        "hops": r.get("hops", 0),
        "hop_samples": HOP_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "hop_duration_ms": HOP_DURATION_MS,
        "runs": runs,
        "lazy_eval": {
            "all_per_hop_ms": [round(t, 3) for t in hop_times_lazy],
            "warm_avg_per_hop_ms": round(avg_lazy, 3),
            "warm_best_per_hop_ms": round(min(warm_lazy), 3),
            "realtime_ok": avg_lazy < HOP_DURATION_MS,
            "headroom_ms": round(HOP_DURATION_MS - avg_lazy, 3),
        },
        "stage_eval": {
            "all_per_hop_ms": [round(t, 3) for t in hop_times_eval],
            "warm_avg_per_hop_ms": round(avg_eval, 3),
            "warm_best_per_hop_ms": round(min(warm_eval), 3),
            "realtime_ok": avg_eval < HOP_DURATION_MS,
            "headroom_ms": round(HOP_DURATION_MS - avg_eval, 3),
            "stages": avg_stages,
        },
    }
    return result


def compare_results(before_path: str, after_path: str):
    """Compare two latency benchmark JSON files."""
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    print(f"\n{'═' * 65}")
    print(f"  LATENCY COMPARISON")
    print(f"  Before: {before.get('label', before_path)}")
    print(f"  After:  {after.get('label', after_path)}")
    print(f"{'═' * 65}")

    for mode in ["lazy_eval", "stage_eval"]:
        b = before.get("latency", {}).get(mode, {})
        a = after.get("latency", {}).get(mode, {})
        if not b or not a:
            continue

        b_val = b["warm_avg_per_hop_ms"]
        a_val = a["warm_avg_per_hop_ms"]
        delta = a_val - b_val
        pct = (delta / b_val) * 100 if b_val else 0
        sign = "+" if delta > 0 else ""
        label = mode.replace("_", " ").title()

        print(f"\n  {label}:")
        print(f"    Before: {b_val:.3f}ms/hop  After: {a_val:.3f}ms/hop  "
              f"Delta: {sign}{delta:.3f}ms ({sign}{pct:.1f}%)")
        print(f"    RT ok:  {b.get('realtime_ok')} -> {a.get('realtime_ok')}")

    # Stage comparison (stage_eval only)
    b_stages = before.get("latency", {}).get("stage_eval", {}).get("stages", {})
    a_stages = after.get("latency", {}).get("stage_eval", {}).get("stages", {})
    if b_stages and a_stages:
        print(f"\n  Per-stage comparison:")
        print(f"  {'Stage':<20} {'Before':>10} {'After':>10} {'Delta':>10}")
        print(f"  {'─' * 52}")
        all_keys = sorted(set(list(b_stages.keys()) + list(a_stages.keys())))
        for key in all_keys:
            b_hop = b_stages.get(key, {}).get("per_hop_ms", 0)
            a_hop = a_stages.get(key, {}).get("per_hop_ms", 0)
            delta = a_hop - b_hop
            sign = "+" if delta > 0 else ""
            indent = "  " if "." in key else ""
            print(f"  {indent}{key:<18} {b_hop:>8.3f}ms {a_hop:>8.3f}ms {sign}{delta:>8.3f}ms")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="DeepFilterNet streaming latency benchmark")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--audio", type=str, default=str(DEFAULT_AUDIO))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--output", "-o", type=str, default=None)
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
    print(f"  DeepFilterNet Streaming Latency Benchmark")
    print(f"  Label: {label}  Git: {git_hash}  Time: {timestamp}")
    print(f"  Model: {args.model}")
    print(f"  Audio: {args.audio}")
    print(f"{'═' * 65}")

    latency = run_latency_benchmark(args.model, args.audio, runs=args.runs)

    results = {
        "label": label,
        "git_hash": git_hash,
        "timestamp": timestamp,
        "model": args.model,
        "latency": latency,
    }

    if args.output:
        out_path = args.output
    else:
        safe_label = label.replace(" ", "_").replace("/", "_")
        out_path = str(REPO_ROOT / f"latency_{safe_label}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {out_path}")
    print(f"  Compare:  python3 scripts/benchmark_dfn_latency.py "
          f"--compare {out_path} <after.json>")
    print()


if __name__ == "__main__":
    main()
