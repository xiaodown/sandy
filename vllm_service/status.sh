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
VLLM_BRAIN_HOST="${VLLM_BRAIN_HOST:-127.0.0.1}"
VLLM_BRAIN_PORT="${VLLM_BRAIN_PORT:-8000}"

case "$PID_FILE" in
  /*) ;;
  *) PID_FILE="$SERVICE_DIR/$PID_FILE" ;;
esac

case "$LOG_FILE" in
  /*) ;;
  *) LOG_FILE="$SERVICE_DIR/$LOG_FILE" ;;
esac

if [[ -f "$PID_FILE" ]]; then
  pid="$(<"$PID_FILE")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "running: pid=$pid"
    ps -o pid,ppid,sid,stat,etime,pcpu,pmem,rss,cmd -p "$pid"
  else
    echo "not running: stale pid file at $PID_FILE"
  fi
else
  echo "not running"
fi

if command -v curl >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
  if curl -fsS "http://$VLLM_BRAIN_HOST:$VLLM_BRAIN_PORT/v1/models" >/tmp/sandy-vllm-models.json 2>/dev/null; then
    echo "endpoint: ready"
    jq -r '.data[].id' /tmp/sandy-vllm-models.json
  else
    echo "endpoint: not ready"
  fi
fi

if [[ -f "$LOG_FILE" ]]; then
  echo "log: $LOG_FILE"
fi
