#!/usr/bin/env python3
"""Check if Swift model structure matches weight keys from safetensors."""
import sys
from pathlib import Path
from safetensors import safe_open

model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / ".cache/huggingface/hub/models--mlx-community--kitten-tts-nano-0.8/snapshots/f57e91b190ca3323aa94c7bbdde4565343d79588"

with safe_open(str(model_dir / "model.safetensors"), framework="numpy") as f:
    weight_keys = set(f.keys())

expected_prefixes = {
    "bert.embeddings.", "bert.encoder.", "bert.pooler.", "bert_encoder.",
    "text_encoder.embedding.", "text_encoder.cnn.", "text_encoder.lstm.",
    "predictor.text_encoder.lstms.", "predictor.lstm.", "predictor.duration_proj.",
    "predictor.shared.", "predictor.F0.", "predictor.N.",
    "predictor.F0_proj.", "predictor.N_proj.",
    "decoder.encode.", "decoder.decode.", "decoder.F0_conv.", "decoder.N_conv.",
    "decoder.asr_res.", "decoder.generator.",
}

unmatched = [k for k in sorted(weight_keys) if not any(k.startswith(p) for p in expected_prefixes)]

if unmatched:
    print(f"❌ {len(unmatched)} weight keys not matched:")
    for k in unmatched:
        print(f"  {k}")
    sys.exit(1)
else:
    print(f"✅ All {len(weight_keys)} weight keys match expected module structure")
