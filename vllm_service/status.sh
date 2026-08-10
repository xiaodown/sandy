#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SERVICE_DIR/.." && pwd)"
PID_FILE="${VLLM_BRAIN_PID_FILE:-$SERVICE_DIR/vllm-brain.pid}"
LOG_FILE="${VLLM_BRAIN_LOG_FILE:-$ROOT_DIR/data/prod/logs/vllm-brain.log}"

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
  if curl -fsS http://127.0.0.1:8000/v1/models >/tmp/sandy-vllm-models.json 2>/dev/null; then
    echo "endpoint: ready"
    jq -r '.data[].id' /tmp/sandy-vllm-models.json
  else
    echo "endpoint: not ready"
  fi
fi

if [[ -f "$LOG_FILE" ]]; then
  echo "log: $LOG_FILE"
fi
