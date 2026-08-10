#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SERVICE_DIR/.." && pwd)"
PID_FILE="${VLLM_BRAIN_PID_FILE:-$SERVICE_DIR/vllm-brain.pid}"
LOG_FILE="${VLLM_BRAIN_LOG_FILE:-$ROOT_DIR/data/prod/logs/vllm-brain.log}"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(<"$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "vLLM brain is already running: pid=$old_pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

mkdir -p "$(dirname "$LOG_FILE")"
setsid bash -lc "cd '$ROOT_DIR' && exec '$SERVICE_DIR/run.sh'" > "$LOG_FILE" 2>&1 &
pid="$!"
printf '%s\n' "$pid" > "$PID_FILE"
echo "started vLLM brain: pid=$pid log=$LOG_FILE"
