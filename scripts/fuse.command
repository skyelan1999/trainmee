#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STATE_ROOT="${TRAINMEE_STATE_DIR:-$ROOT/state}"
ENV_DIR="${TRAINMEE_ENV_DIR:-$STATE_ROOT/venv}"
export HF_HOME="${HF_HOME:-$STATE_ROOT/huggingface}"
source "$ENV_DIR/bin/activate"
test -f "$STATE_ROOT/base-model/config.json" || { print -u2 "Base model is missing. Run scripts/init-model.command first."; exit 1; }
test -f "$STATE_ROOT/adapters/qwen3-4b-lora/adapters.safetensors" || { print -u2 "Adapter not found. Run scripts/train.command first."; exit 1; }
mkdir -p "$STATE_ROOT/models"
mlx_lm.fuse \
  --model "$STATE_ROOT/base-model" \
  --adapter-path "$STATE_ROOT/adapters/qwen3-4b-lora" \
  --save-path "$STATE_ROOT/models/qwen3-4b-trainmee"
