#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE_ROOT="${TRAINMEE_STATE_DIR:-$ROOT/state}"
ENV_DIR="${TRAINMEE_ENV_DIR:-$STATE_ROOT/venv}"
export HF_HOME="${HF_HOME:-$STATE_ROOT/huggingface}"
[[ -x "$ENV_DIR/bin/python" ]] || { print -u2 "Python environment missing. Create state/venv and install requirements.lock.txt first."; exit 1; }
exec "$ENV_DIR/bin/python" "$ROOT/scripts/chat.py" "$@"
