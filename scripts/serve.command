#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STATE_ROOT="${TRAINMEE_STATE_DIR:-$ROOT/state}"
ENV_DIR="${TRAINMEE_ENV_DIR:-$STATE_ROOT/venv}"
export HF_HOME="${HF_HOME:-$STATE_ROOT/huggingface}"
source "$ENV_DIR/bin/activate"
FUSED_MODEL="$STATE_ROOT/models/qwen3-4b-trainmee"
if [[ -n "${TRAINMEE_MODEL:-}" ]]; then
  MODEL_PATH="$TRAINMEE_MODEL"
elif [[ -d "$FUSED_MODEL" ]]; then
  MODEL_PATH="$FUSED_MODEL"
elif [[ -f "$STATE_ROOT/base-model/config.json" ]]; then
  MODEL_PATH="$STATE_ROOT/base-model"
  print "No fine-tuned model yet; serving the project-local Qwen3-4B base model."
else
  print -u2 "No model found. Run ./scripts/init-model.command first."
  exit 1
fi
API_PORT="${TRAINMEE_PORT:-8080}"
UPSTREAM_PORT="${TRAINMEE_UPSTREAM_PORT:-8081}"
mkdir -p "$STATE_ROOT/logs"
mlx_lm.server --model "$MODEL_PATH" --host 127.0.0.1 --port "$UPSTREAM_PORT" \
  > "$STATE_ROOT/logs/mlx-server.log" 2>&1 &
MLX_PID=$!
cleanup() {
  if kill -0 "$MLX_PID" 2>/dev/null; then
    kill -INT "$MLX_PID" 2>/dev/null || true
    wait "$MLX_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
READY=0
for _ in {1..300}; do
  if curl -fsS "http://127.0.0.1:$UPSTREAM_PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$MLX_PID" 2>/dev/null; then
    cat "$STATE_ROOT/logs/mlx-server.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$READY" != 1 ]]; then
  print -u2 "MLX-LM server did not become ready. See $STATE_ROOT/logs/mlx-server.log"
  exit 1
fi
python "$ROOT/scripts/api_proxy.py" \
  --host "${TRAINMEE_HOST:-127.0.0.1}" \
  --port "$API_PORT" \
  --upstream "http://127.0.0.1:$UPSTREAM_PORT" \
  --model-id "${TRAINMEE_MODEL_ID:-trainmee}" \
  --upstream-model "$MODEL_PATH"
