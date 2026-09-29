#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE_ROOT="${TRAINMEE_STATE_DIR:-$ROOT/state}"
ENV_DIR="${TRAINMEE_ENV_DIR:-$STATE_ROOT/venv}"
MODEL_ID="${TRAINMEE_HF_MODEL:-Qwen/Qwen3-4B}"
DEST="$STATE_ROOT/base-model"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { print -u2 "TrainMee MLX model initialization requires Apple Silicon macOS."; exit 1; }
[[ -x "$ENV_DIR/bin/mlx_lm.convert" ]] || { print -u2 "MLX-LM not found at $ENV_DIR. Create the Python 3.12 environment and install requirements.lock.txt first."; exit 1; }
export HF_HOME="${HF_HOME:-$STATE_ROOT/huggingface}"
mkdir -p "$STATE_ROOT" "$(dirname "$HF_HOME")"
if [[ -f "$DEST/config.json" && -f "$DEST/model.safetensors" ]]; then
  print "Qwen3-4B is already initialized at $DEST"
  exit 0
fi
if [[ -e "$DEST" ]]; then
  print -u2 "Incomplete model directory exists: $DEST"
  print -u2 "Move it aside or remove it, then rerun this script."
  exit 1
fi
SNAPSHOT_PATH=$("$ENV_DIR/bin/python" -c 'from huggingface_hub import snapshot_download; import sys; print(snapshot_download(repo_id=sys.argv[1]))' "$MODEL_ID")
"$ENV_DIR/bin/mlx_lm.convert" --hf-path "$SNAPSHOT_PATH" --mlx-path "$DEST"
print "Qwen3-4B MLX model is ready at $DEST"
