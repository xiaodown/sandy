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

VLLM_PYTHON="${VLLM_PYTHON:-$SERVICE_DIR/.venv/bin/python}"
VLLM_BRAIN_MODEL="${VLLM_BRAIN_MODEL:-mistralai/Mistral-Small-4-119B-2603-NVFP4}"
VLLM_BRAIN_SERVED_MODEL_NAME="${VLLM_BRAIN_SERVED_MODEL_NAME:-sandy-brain}"
VLLM_BRAIN_PROFILE="${VLLM_BRAIN_PROFILE:-auto}"
VLLM_BRAIN_HOST="${VLLM_BRAIN_HOST:-127.0.0.1}"
VLLM_BRAIN_PORT="${VLLM_BRAIN_PORT:-8000}"
VLLM_BRAIN_MAX_MODEL_LEN="${VLLM_BRAIN_MAX_MODEL_LEN:-16384}"
VLLM_BRAIN_GPU_MEMORY_UTILIZATION="${VLLM_BRAIN_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_BRAIN_ATTENTION_BACKEND="${VLLM_BRAIN_ATTENTION_BACKEND:-}"
VLLM_BRAIN_REASONING_PARSER="${VLLM_BRAIN_REASONING_PARSER:-}"
VLLM_BRAIN_TOOL_PARSER="${VLLM_BRAIN_TOOL_PARSER:-}"
VLLM_BRAIN_GENERATION_CONFIG="${VLLM_BRAIN_GENERATION_CONFIG:-vllm}"
VLLM_BRAIN_CUDA_VISIBLE_DEVICES="${VLLM_BRAIN_CUDA_VISIBLE_DEVICES:-2}"
VLLM_BRAIN_ENABLE_AUTO_TOOL_CHOICE="${VLLM_BRAIN_ENABLE_AUTO_TOOL_CHOICE:-}"
VLLM_BRAIN_ENABLE_PREFIX_CACHING="${VLLM_BRAIN_ENABLE_PREFIX_CACHING:-}"
VLLM_BRAIN_ENABLE_CHUNKED_PREFILL="${VLLM_BRAIN_ENABLE_CHUNKED_PREFILL:-}"
VLLM_BRAIN_ASYNC_SCHEDULING="${VLLM_BRAIN_ASYNC_SCHEDULING:-}"
VLLM_BRAIN_LANGUAGE_MODEL_ONLY="${VLLM_BRAIN_LANGUAGE_MODEL_ONLY:-}"
VLLM_BRAIN_TRUST_REMOTE_CODE="${VLLM_BRAIN_TRUST_REMOTE_CODE:-}"
VLLM_BRAIN_ENABLE_MTP="${VLLM_BRAIN_ENABLE_MTP:-}"
VLLM_BRAIN_CUDA_HOME="${VLLM_BRAIN_CUDA_HOME:-}"
VLLM_BRAIN_SPECULATIVE_CONFIG="${VLLM_BRAIN_SPECULATIVE_CONFIG:-}"
VLLM_BRAIN_DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_BRAIN_DEFAULT_CHAT_TEMPLATE_KWARGS:-}"
VLLM_BRAIN_KV_CACHE_DTYPE="${VLLM_BRAIN_KV_CACHE_DTYPE:-}"
VLLM_BRAIN_LOAD_FORMAT="${VLLM_BRAIN_LOAD_FORMAT:-}"
VLLM_BRAIN_DTYPE="${VLLM_BRAIN_DTYPE:-}"
VLLM_BRAIN_TENSOR_PARALLEL_SIZE="${VLLM_BRAIN_TENSOR_PARALLEL_SIZE:-}"
VLLM_BRAIN_MAX_NUM_SEQS="${VLLM_BRAIN_MAX_NUM_SEQS:-}"
VLLM_BRAIN_MAX_NUM_BATCHED_TOKENS="${VLLM_BRAIN_MAX_NUM_BATCHED_TOKENS:-}"
VLLM_BRAIN_EXTRA_ARGS="${VLLM_BRAIN_EXTRA_ARGS:-}"

model_lc="${VLLM_BRAIN_MODEL,,}"
profile_lc="${VLLM_BRAIN_PROFILE,,}"

is_enabled() {
  [[ "${1,,}" =~ ^(true|1|yes|on)$ ]]
}

if [[ "$profile_lc" == "auto" ]]; then
  case "$model_lc" in
    *qwen3.6*nvfp4*)
      profile_lc="qwen3-nvfp4"
      ;;
    *qwen3.6*|*qwen3-*|*qwen/qwen3*)
      profile_lc="qwen3"
      ;;
    *mistral*)
      profile_lc="mistral"
      ;;
    *llama*)
      profile_lc="llama"
      ;;
    *)
      profile_lc="generic"
      ;;
  esac
fi

case "$profile_lc" in
  mistral)
    VLLM_BRAIN_ATTENTION_BACKEND="${VLLM_BRAIN_ATTENTION_BACKEND:-TRITON_MLA}"
    VLLM_BRAIN_REASONING_PARSER="${VLLM_BRAIN_REASONING_PARSER:-mistral}"
    VLLM_BRAIN_TOOL_PARSER="${VLLM_BRAIN_TOOL_PARSER:-mistral}"
    VLLM_BRAIN_ENABLE_AUTO_TOOL_CHOICE="${VLLM_BRAIN_ENABLE_AUTO_TOOL_CHOICE:-true}"
    ;;
  qwen3)
    VLLM_BRAIN_ATTENTION_BACKEND="${VLLM_BRAIN_ATTENTION_BACKEND:-TRITON_ATTN}"
    VLLM_BRAIN_REASONING_PARSER="${VLLM_BRAIN_REASONING_PARSER:-qwen3}"
    VLLM_BRAIN_TOOL_PARSER="${VLLM_BRAIN_TOOL_PARSER:-qwen3_xml}"
    VLLM_BRAIN_LANGUAGE_MODEL_ONLY="${VLLM_BRAIN_LANGUAGE_MODEL_ONLY:-true}"
    VLLM_BRAIN_DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_BRAIN_DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
    if [[ "$model_lc" == *qwen3.6* ]] && is_enabled "$VLLM_BRAIN_ENABLE_MTP"; then
      VLLM_BRAIN_SPECULATIVE_CONFIG="${VLLM_BRAIN_SPECULATIVE_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":1}}"
    fi
    ;;
  qwen3-nvfp4)
    VLLM_BRAIN_ATTENTION_BACKEND="${VLLM_BRAIN_ATTENTION_BACKEND:-TRITON_ATTN}"
    VLLM_BRAIN_REASONING_PARSER="${VLLM_BRAIN_REASONING_PARSER:-qwen3}"
    VLLM_BRAIN_TOOL_PARSER="${VLLM_BRAIN_TOOL_PARSER:-qwen3_xml}"
    VLLM_BRAIN_LANGUAGE_MODEL_ONLY="${VLLM_BRAIN_LANGUAGE_MODEL_ONLY:-true}"
    VLLM_BRAIN_TRUST_REMOTE_CODE="${VLLM_BRAIN_TRUST_REMOTE_CODE:-true}"
    VLLM_BRAIN_DEFAULT_CHAT_TEMPLATE_KWARGS="${VLLM_BRAIN_DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
    VLLM_BRAIN_KV_CACHE_DTYPE="${VLLM_BRAIN_KV_CACHE_DTYPE:-fp8}"
    VLLM_BRAIN_ENABLE_PREFIX_CACHING="${VLLM_BRAIN_ENABLE_PREFIX_CACHING:-true}"
    VLLM_BRAIN_ENABLE_CHUNKED_PREFILL="${VLLM_BRAIN_ENABLE_CHUNKED_PREFILL:-true}"
    VLLM_BRAIN_ASYNC_SCHEDULING="${VLLM_BRAIN_ASYNC_SCHEDULING:-true}"
    if is_enabled "$VLLM_BRAIN_ENABLE_MTP"; then
      VLLM_BRAIN_SPECULATIVE_CONFIG="${VLLM_BRAIN_SPECULATIVE_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":2}}"
    fi
    ;;
  llama)
    VLLM_BRAIN_TOOL_PARSER="${VLLM_BRAIN_TOOL_PARSER:-llama3_json}"
    ;;
  generic)
    ;;
  *)
    echo "unknown VLLM_BRAIN_PROFILE=$VLLM_BRAIN_PROFILE" >&2
    echo "supported profiles: auto, generic, mistral, qwen3, qwen3-nvfp4, llama" >&2
    exit 2
    ;;
esac

configure_mtp_cuda_toolkit() {
  if ! is_enabled "$VLLM_BRAIN_ENABLE_MTP"; then
    return
  fi

  local cuda_home="$VLLM_BRAIN_CUDA_HOME"
  if [[ -z "$cuda_home" ]]; then
    local candidate
    for candidate in "$SERVICE_DIR"/.venv/lib/python*/site-packages/nvidia/cu13; do
      if [[ -x "$candidate/bin/nvcc" ]]; then
        cuda_home="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$cuda_home" || ! -x "$cuda_home/bin/nvcc" ]]; then
    echo "VLLM_BRAIN_ENABLE_MTP=true requires venv-local CUDA nvcc; install nvidia-cuda-nvcc==13.0.88 or set VLLM_BRAIN_CUDA_HOME" >&2
    exit 2
  fi

  if [[ ! -e "$cuda_home/lib64" && -d "$cuda_home/lib" ]]; then
    ln -s lib "$cuda_home/lib64"
  fi
  if [[ ! -e "$cuda_home/lib/libcudart.so" && -e "$cuda_home/lib/libcudart.so.13" ]]; then
    ln -s libcudart.so.13 "$cuda_home/lib/libcudart.so"
  fi

  export CUDA_HOME="$cuda_home"
  export PATH="$SERVICE_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
}

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="$VLLM_BRAIN_CUDA_VISIBLE_DEVICES"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
configure_mtp_cuda_toolkit

cd "$ROOT_DIR"

args=(
  "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$VLLM_BRAIN_MODEL"
  --served-model-name "$VLLM_BRAIN_SERVED_MODEL_NAME"
  --host "$VLLM_BRAIN_HOST"
  --port "$VLLM_BRAIN_PORT"
  --max-model-len "$VLLM_BRAIN_MAX_MODEL_LEN"
  --gpu-memory-utilization "$VLLM_BRAIN_GPU_MEMORY_UTILIZATION"
  --generation-config "$VLLM_BRAIN_GENERATION_CONFIG"
)

append_arg() {
  local flag="$1"
  local value="$2"
  if [[ -n "$value" ]]; then
    args+=("$flag" "$value")
  fi
}

append_bool_arg() {
  local flag="$1"
  local value="${2,,}"
  case "$value" in
    true|1|yes|on)
      args+=("$flag")
      ;;
    ""|false|0|no|off)
      ;;
    *)
      echo "invalid boolean for $flag: $2" >&2
      exit 2
      ;;
  esac
}

append_arg --attention-backend "$VLLM_BRAIN_ATTENTION_BACKEND"
append_arg --reasoning-parser "$VLLM_BRAIN_REASONING_PARSER"
append_arg --tool-call-parser "$VLLM_BRAIN_TOOL_PARSER"
append_arg --speculative-config "$VLLM_BRAIN_SPECULATIVE_CONFIG"
append_arg --default-chat-template-kwargs "$VLLM_BRAIN_DEFAULT_CHAT_TEMPLATE_KWARGS"
append_arg --kv-cache-dtype "$VLLM_BRAIN_KV_CACHE_DTYPE"
append_arg --load-format "$VLLM_BRAIN_LOAD_FORMAT"
append_arg --dtype "$VLLM_BRAIN_DTYPE"
append_arg --tensor-parallel-size "$VLLM_BRAIN_TENSOR_PARALLEL_SIZE"
append_arg --max-num-seqs "$VLLM_BRAIN_MAX_NUM_SEQS"
append_arg --max-num-batched-tokens "$VLLM_BRAIN_MAX_NUM_BATCHED_TOKENS"

append_bool_arg --enable-auto-tool-choice "$VLLM_BRAIN_ENABLE_AUTO_TOOL_CHOICE"
append_bool_arg --enable-prefix-caching "$VLLM_BRAIN_ENABLE_PREFIX_CACHING"
append_bool_arg --enable-chunked-prefill "$VLLM_BRAIN_ENABLE_CHUNKED_PREFILL"
append_bool_arg --async-scheduling "$VLLM_BRAIN_ASYNC_SCHEDULING"
append_bool_arg --language-model-only "$VLLM_BRAIN_LANGUAGE_MODEL_ONLY"
append_bool_arg --trust-remote-code "$VLLM_BRAIN_TRUST_REMOTE_CODE"

if [[ -n "$VLLM_BRAIN_EXTRA_ARGS" ]]; then
  # Shell-word parsed on purpose so JSON flags can still be passed as one string.
  eval "extra_args=($VLLM_BRAIN_EXTRA_ARGS)"
  args+=("${extra_args[@]}")
fi

printf 'starting vLLM brain profile=%s model=%s served_model_name=%s\n' \
  "$profile_lc" "$VLLM_BRAIN_MODEL" "$VLLM_BRAIN_SERVED_MODEL_NAME" >&2

for private_var in "${!VLLM_BRAIN_@}"; do
  unset "$private_var"
done

exec "${args[@]}"
