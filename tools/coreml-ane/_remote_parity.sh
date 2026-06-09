#!/usr/bin/env bash
# Runs ON alex-mac inside tmux (pushed by run_stream_parity.sh). Sets up the env,
# converts the streaming encoder at the correct fixed size, runs the parity harness.
set -e
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH
cd "$(dirname "$0")"

echo "=== venv ==="
[ -d .venv ] || uv venv --python 3.11 .venv
source .venv/bin/activate
python -c "import nemo" 2>/dev/null     || uv pip install -q "nemo-toolkit[asr]"
python -c "import coremltools" 2>/dev/null || uv pip install -q coremltools
python -c "import librosa" 2>/dev/null   || uv pip install -q librosa

echo "=== convert (fixed F = pre_encode+chunk, not chunk) ==="
python convert_encoder_coreml_stream.py \
  --model nvidia/nemotron-speech-streaming-en-0.6b --att-context 56 13 \
  --out out/nemotron_stream_func.mlpackage

echo "=== parity (CHECK1 fp16-fidelity, CHECK2 padding-benign) ==="
python parity_stream_func.py --wav Tests/media/intention.wav \
  --mlpackage out/nemotron_stream_func.mlpackage

echo "=== DONE ==="
