#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${VLLM_BRAIN_ENV_FILE:-$SERVICE_DIR/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

PID_FILE="${VLLM_BRAIN_PID_FILE:-$SERVICE_DIR/vllm-brain.pid}"

case "$PID_FILE" in
  /*) ;;
  *) PID_FILE="$SERVICE_DIR/$PID_FILE" ;;
esac

if [[ ! -f "$PID_FILE" ]]; then
  echo "vLLM brain is not running: no pid file"
  exit 0
fi

pid="$(<"$PID_FILE")"
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$PID_FILE"
  echo "vLLM brain is not running: stale pid file removed"
  exit 0
fi

kill "$pid"
for _ in {1..30}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "stopped vLLM brain: pid=$pid"
    exit 0
  fi
  sleep 1
done

kill -KILL "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
echo "force-stopped vLLM brain: pid=$pid"
