#!/usr/bin/env bash
# Streaming-CoreML (Option A) parity runner — YOU run this from THIS Mac.
# Pushes the converter + harness + remote script + test wav to alex-mac (which has
# CoreML/ANE), then launches _remote_parity.sh in a tmux session there. Heavy work runs
# on alex-mac, not over the SSH channel.
#
#   bash tools/coreml-ane/run_stream_parity.sh
#   ssh alex-mac 'tmux attach -t streamparity'          # watch progress
#   ssh alex-mac 'tail -n 25 nemo-stream-coreml/parity.log'   # or paste the tail back
set -euo pipefail

HOST="${HOST:-alex-mac}"
REMOTE="${REMOTE:-nemo-stream-coreml}"
SESSION="streamparity"
HERE="$(cd "$(dirname "$0")/../.." && pwd)"   # repo root

echo "[push] -> $HOST:$REMOTE"
ssh "$HOST" "mkdir -p $REMOTE/out $REMOTE/Tests/media"
rsync -az \
  "$HERE/tools/coreml-ane/convert_encoder.py" \
  "$HERE/tools/coreml-ane/convert_encoder_coreml_stream.py" \
  "$HERE/tools/coreml-ane/parity_stream_func.py" \
  "$HERE/tools/coreml-ane/_remote_parity.sh" \
  "$HOST:$REMOTE/"
rsync -az "$HERE/Tests/media/intention.wav" "$HOST:$REMOTE/Tests/media/"

echo "[run] tmux '$SESSION' on $HOST (logs -> $REMOTE/parity.log)"
# Single, simple remote command: tmux runs the standalone script, tee to a log.
ssh "$HOST" "export PATH=/opt/homebrew/bin:\$HOME/.local/bin:\$PATH; \
  tmux kill-session -t $SESSION 2>/dev/null || true; \
  tmux new-session -d -s $SESSION \"bash $REMOTE/_remote_parity.sh 2>&1 | tee $REMOTE/parity.log\""

echo "[ok] running. Attach:  ssh $HOST 'tmux attach -t $SESSION'"
echo "     Result tail:       ssh $HOST 'tail -n 25 $REMOTE/parity.log'"
