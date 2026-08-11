#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SERVICE_DIR/.." && pwd)"
ENV_FILE="${VLLM_BRAIN_ENV_FILE:-$SERVICE_DIR/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

PID_FILE="${VLLM_BRAIN_PID_FILE:-$SERVICE_DIR/vllm-brain.pid}"
LOG_FILE="${VLLM_BRAIN_LOG_FILE:-$SERVICE_DIR/logs/vllm-brain.log}"

case "$PID_FILE" in
  /*) ;;
  *) PID_FILE="$SERVICE_DIR/$PID_FILE" ;;
esac

case "$LOG_FILE" in
  /*) ;;
  *) LOG_FILE="$SERVICE_DIR/$LOG_FILE" ;;
esac

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
