#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STATE_ROOT="${TRAINMEE_STATE_DIR:-$ROOT/state}"
ENV_DIR="${TRAINMEE_ENV_DIR:-$STATE_ROOT/venv}"
export HF_HOME="${HF_HOME:-$STATE_ROOT/huggingface}"
source "$ENV_DIR/bin/activate"
BASE_MODEL="$STATE_ROOT/base-model"
[[ -f "$BASE_MODEL/config.json" ]] || { print -u2 "Base model is missing. Run ./scripts/init-model.command first."; exit 1; }
python scripts/check_dataset.py
mkdir -p "$STATE_ROOT/adapters" "$STATE_ROOT/logs"
ADAPTER_PATH="$STATE_ROOT/adapters/qwen3-4b-lora"
if [[ -e "$ADAPTER_PATH" && -z "${TRAINMEE_RESUME_ADAPTER:-}" ]]; then
  print -u2 "Adapter already exists: $ADAPTER_PATH"
  print -u2 "Set TRAINMEE_RESUME_ADAPTER to its adapters.safetensors to continue, or move the old directory to keep it as a checkpoint."
  exit 1
fi
ARGS=(
  --model "$BASE_MODEL" \
  --train \
  --data "$ROOT/data" \
  --iters "${TRAINMEE_ITERS:-100}" \
  --batch-size 1 \
  --num-layers 8 \
  --max-seq-length 2048 \
  --grad-checkpoint \
  --mask-prompt \
  --adapter-path "$ADAPTER_PATH"
)
if [[ -n "${TRAINMEE_RESUME_ADAPTER:-}" ]]; then
  [[ -f "$TRAINMEE_RESUME_ADAPTER" ]] || { print -u2 "Resume adapter file not found: $TRAINMEE_RESUME_ADAPTER"; exit 1; }
  ARGS+=(--resume-adapter-file "$TRAINMEE_RESUME_ADAPTER")
fi
mlx_lm.lora "${ARGS[@]}" 2>&1 | tee "$STATE_ROOT/logs/train-$(date +%Y%m%d-%H%M%S).log"
