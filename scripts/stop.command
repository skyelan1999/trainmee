#!/bin/zsh
set -euo pipefail
PORT="${TRAINMEE_PORT:-8080}"
PID="$(lsof -nP -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
if [[ -z "$PID" ]]; then
  print "No listener found on port $PORT."
  exit 0
fi
PROCESS="$(ps -p "$PID" -o command= 2>/dev/null || true)"
if [[ "$PROCESS" != *"mlx_lm.server"* && "$PROCESS" != *"api_proxy.py"* ]]; then
  print -u2 "Port $PORT belongs to another process; leaving it untouched: $PROCESS"
  exit 1
fi
kill -INT "$PID"
print "Sent stop signal to TrainMee API (PID $PID, port $PORT)."
