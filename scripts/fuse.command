#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STATE_ROOT="${TRAINMEE_STATE_DIR:-$HOME/Library/Application Support/TrainMee}"
ENV_DIR="${TRAINMEE_ENV_DIR:-$STATE_ROOT/venv}"
export HF_HOME="${HF_HOME:-$STATE_ROOT/huggingface}"
source "$ENV_DIR/bin/activate"
test -f "$STATE_ROOT/adapters/qwen25-3b-lora/adapters.safetensors" || { print -u2 "Adapter not found. Run scripts/train.command first."; exit 1; }
mkdir -p "$STATE_ROOT/models"
mlx_lm.fuse \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path "$STATE_ROOT/adapters/qwen25-3b-lora" \
  --save-path "$STATE_ROOT/models/qwen25-3b-trainmee"
