#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STATE_ROOT="${TRAINMEE_STATE_DIR:-$ROOT/state}"
ENV_DIR="${TRAINMEE_ENV_DIR:-$STATE_ROOT/venv}"
export HF_HOME="${HF_HOME:-$STATE_ROOT/huggingface}"
[[ -x "$ENV_DIR/bin/mlx_lm.generate" ]] || { print -u2 "MLX-LM not found. Create state/venv and install requirements.lock.txt first."; exit 1; }
FUSED_MODEL="$STATE_ROOT/models/qwen3-4b-trainmee"
BASE_MODEL="$STATE_ROOT/base-model"
if [[ -d "$FUSED_MODEL" ]]; then
  MODEL_PATH="$FUSED_MODEL"
  ADAPTER_ARGS=()
elif [[ -f "$BASE_MODEL/config.json" ]]; then
  MODEL_PATH="$BASE_MODEL"
  ADAPTER_PATH="$STATE_ROOT/adapters/qwen3-4b-lora"
  ADAPTER_ARGS=()
  if [[ -f "$ADAPTER_PATH/adapters.safetensors" ]]; then
    ADAPTER_ARGS=(--adapter-path "$ADAPTER_PATH")
  fi
else
  print -u2 "Qwen3-4B is not initialized. Run ./scripts/init-model.command first."
  exit 1
fi
QUESTION="$*"
if [[ -z "$QUESTION" ]]; then
  print -n "你："
  IFS= read -r QUESTION
fi
[[ -n "$QUESTION" ]] || { print -u2 "Usage: ./scripts/ask.command \"你的问题\""; exit 2; }
"$ENV_DIR/bin/mlx_lm.generate" \
  --model "$MODEL_PATH" \
  "${ADAPTER_ARGS[@]}" \
  --chat-template-config '{"enable_thinking":false}' \
  --max-tokens "${TRAINMEE_MAX_TOKENS:-256}" \
  --verbose False \
  --prompt "$QUESTION"
