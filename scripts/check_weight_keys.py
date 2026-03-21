#!/usr/bin/env python3
"""Check if Swift model structure matches weight keys from safetensors."""
import sys
from pathlib import Path
from safetensors import safe_open

model_dir = Path.home() / ".cache/huggingface/hub/models--mlx-community--kitten-tts-nano-0.8/snapshots/f57e91b190ca3323aa94c7bbdde4565343d79588"

f = safe_open(str(model_dir / "model.safetensors"), framework="numpy")
weight_keys = set(f.keys())

# Build expected Swift module key patterns based on our @ModuleInfo structure
# These are the key prefixes that Swift Module.parameters().flattened() would produce
expected_prefixes = {
    # ALBERT
    "bert.embeddings.": True,
    "bert.encoder.": True,
    "bert.pooler.": True,
    "bert_encoder.": True,
    # TextEncoder
    "text_encoder.embedding.": True,
    "text_encoder.cnn.": True,  # cnn.0._0.weight_v (via KittenCNNBlock._0)
    "text_encoder.lstm.": True,
    # ProsodyPredictor
    "predictor.text_encoder.lstms.": True,
    "predictor.lstm.": True,
    "predictor.duration_proj.": True,
    "predictor.shared.": True,
    "predictor.F0.": True,
    "predictor.N.": True,
    "predictor.F0_proj.": True,
    "predictor.N_proj.": True,
    # Decoder
    "decoder.encode.": True,
    "decoder.decode.": True,
    "decoder.F0_conv.": True,
    "decoder.N_conv.": True,
    "decoder.asr_res.": True,
    "decoder.generator.": True,
}

unmatched = []
for k in sorted(weight_keys):
    matched = any(k.startswith(p) for p in expected_prefixes)
    if not matched:
        unmatched.append(k)

if unmatched:
    print(f"❌ {len(unmatched)} weight keys not matched by any expected prefix:")
    for k in unmatched:
        print(f"  {k}")
    sys.exit(1)
else:
    print(f"✅ All {len(weight_keys)} weight keys match expected module structure")

# Check specific Swift quirks
# KittenCNNBlock uses _0 and _1 as property names (matching Python's array indexing)
cnn_keys = [k for k in weight_keys if k.startswith("text_encoder.cnn.")]
print(f"\n  text_encoder.cnn keys ({len(cnn_keys)}):")
for k in sorted(cnn_keys):
    print(f"    {k}")
    # Verify: cnn.0.0.weight_v -> cnn.0._0.weight_v in Swift
    # Python key: text_encoder.cnn.0.0.weight_v
    # Swift key:  text_encoder.cnn.0._0.weight_v  (via KittenCNNBlock._0)
    # MISMATCH: Python uses .0.0, Swift uses .0._0
