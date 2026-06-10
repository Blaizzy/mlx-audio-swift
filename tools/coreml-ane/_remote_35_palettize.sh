#!/usr/bin/env bash
# Runs ON alex-mac in tmux (venv35g). Palettizes the 3.5 streaming encoder to N-bit weights
# (uniform mode = low memory; kmeans OOM'd). Usage: _remote_35_palettize.sh <nbits>
set -e
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
cd "$(dirname "$0")"
source .venv35g/bin/activate
NB="${1:-8}"

echo "=== palettize ${NB}-bit (uniform) ==="
PYTHONUNBUFFERED=1 python -u convert_encoder_coreml_stream.py \
  --model nvidia/nemotron-3.5-asr-streaming-0.6b --att-context 56 13 \
  --palettize "$NB" --palettize-mode uniform \
  --out "out/nemotron_35_stream_p${NB}.mlpackage" 2>&1 | grep -vE "compression pass|ops/s|Frontend ==>|MIL "
du -sh "out/nemotron_35_stream_p${NB}.mlpackage"
echo "=== DONE ==="
