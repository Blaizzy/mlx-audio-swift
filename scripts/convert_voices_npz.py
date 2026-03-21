#!/usr/bin/env python3
"""Convert KittenTTS voices.npz to voices.safetensors for MLX Swift."""

import argparse
import numpy as np
from pathlib import Path

try:
    from safetensors.numpy import save_file
except ImportError:
    print("Install safetensors: pip install safetensors")
    raise


def convert(npz_path: Path, output_path: Path | None = None):
    if output_path is None:
        output_path = npz_path.with_suffix(".safetensors")

    voices = np.load(npz_path)
    tensors = {k: voices[k].astype(np.float32) for k in voices.files}

    print(f"Converting {npz_path} -> {output_path}")
    for name, arr in tensors.items():
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")

    save_file(tensors, str(output_path))
    print(f"Done. {len(tensors)} voices saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert voices.npz to safetensors")
    parser.add_argument("input", type=Path, help="Path to voices.npz")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path")
    args = parser.parse_args()
    convert(args.input, args.output)
