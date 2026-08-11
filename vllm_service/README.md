# Sandy vLLM Brain Service

This directory contains the local vLLM runtime used when `BRAIN_PROVIDER=vllm`.
It is intentionally separate from Sandy's main `.venv`, similar to `tts_service/`.

The service exposes its loaded Hugging Face model under the stable OpenAI API
model name `sandy-brain` by default. Sandy can keep `BRAIN_MODEL=sandy-brain`
while `VLLM_BRAIN_MODEL` changes during backend model experiments.

## Setup

```bash
uv venv .venv --python 3.12
uv sync
```

The service has its own `pyproject.toml`; do dependency changes there instead
of copying long install commands around. CUDA 13.0 PyTorch wheels are selected
through the service-local uv index configuration.

The project includes venv-local CUDA compiler packages for Qwen3.6 MTP /
FlashInfer JIT without adding a system CUDA toolkit:

```bash
uv pip freeze --python .venv/bin/python | grep -E '^(vllm|torch|nvidia-cuda|nvidia-nvvm|ninja)=='
```

## Run In A Screen Or Tmux Pane

```bash
./run.sh
```

`Ctrl-C` stops vLLM cleanly.

## Detached Local Run

```bash
./start.sh
./status.sh
./stop.sh
```

The detached process writes logs to `./logs/vllm-brain.log` and stores its pid
in `vllm-brain.pid`.

## Local Configuration

```bash
cp .env.example .env
```

`run.sh`, `start.sh`, `status.sh`, and `stop.sh` load `.env` automatically.
The local `.env` file is gitignored; commit changes to `.env.example` only when
the default template should change.

By default logs stay inside this service directory:

```env
VLLM_BRAIN_LOG_FILE="./logs/vllm-brain.log"
VLLM_BRAIN_PID_FILE="./vllm-brain.pid"
```

## GPU Placement

By default `run.sh` pins vLLM to CUDA device index `0`:

```bash
VLLM_BRAIN_CUDA_VISIBLE_DEVICES=0 ./run.sh
```

Override `VLLM_BRAIN_CUDA_VISIBLE_DEVICES` locally if you want vLLM to use a
different device or device set. A GPU UUID works too, but keep machine-specific
UUIDs in the gitignored `.env`, not in committed files.

## Model Profiles

`run.sh` uses `VLLM_BRAIN_PROFILE=auto` by default and chooses a small set of
family-specific vLLM flags from the model name.

Supported profiles:

| Profile | Intended models | Defaults |
| ------- | --------------- | -------- |
| `generic` | Unknown or one-off models | No family-specific parser or feature flags |
| `mistral` | Mistral models | `TRITON_MLA`, Mistral reasoning/tool parsers, auto tool choice |
| `qwen3` | Qwen3 / Qwen3.6 safetensors models | Triton attention, Qwen3 reasoning/tool parsers, thinking disabled by default, text-only brain serving |
| `qwen3-nvfp4` | Qwen3.6 NVFP4 safetensors models | Triton attention, Qwen3 parser flags, thinking disabled by default, text-only serving, FP8 KV cache, prefix caching, chunked prefill, async scheduling |
| `llama` | Llama-family models | Llama JSON tool parser |

Every profile default can be overridden with env vars such as
`VLLM_BRAIN_REASONING_PARSER`, `VLLM_BRAIN_SPECULATIVE_CONFIG`,
`VLLM_BRAIN_ENABLE_MTP`, `VLLM_BRAIN_DEFAULT_CHAT_TEMPLATE_KWARGS`,
`VLLM_BRAIN_LANGUAGE_MODEL_ONLY`, or `VLLM_BRAIN_EXTRA_ARGS`.

Qwen profiles set `--default-chat-template-kwargs '{"enable_thinking":false}'`
so ordinary Sandy chat responses land in OpenAI `message.content` instead of
`message.reasoning`. A test client can still opt into reasoning per request via
`chat_template_kwargs`.

## Trying Another Brain Model

Edit `.env`, then restart the service:

```bash
./stop.sh
./start.sh
./status.sh
```

For Qwen3.6 NVFP4 with MTP:

```env
VLLM_BRAIN_MODEL="unsloth/Qwen3.6-27B-NVFP4"
VLLM_BRAIN_MAX_MODEL_LEN=65536
VLLM_BRAIN_ENABLE_MTP=true
```

For the Mistral fallback/test model:

```env
VLLM_BRAIN_MODEL="mistralai/Mistral-Small-4-119B-2603-NVFP4"
VLLM_BRAIN_MAX_MODEL_LEN=16384
VLLM_BRAIN_ENABLE_MTP=false
```

Qwen3.6 MTP can be enabled with `VLLM_BRAIN_ENABLE_MTP=true`. When MTP is
enabled, `run.sh` automatically exports `CUDA_HOME`, `PATH`, and
`LD_LIBRARY_PATH` from `.venv/lib/python*/site-packages/nvidia/cu13` and
`.venv/bin` so vLLM's FlashInfer JIT can find the venv-local `nvcc` and
`ninja`. It also creates venv-local compatibility symlinks for `lib64` and
`libcudart.so`, because FlashInfer's generated linker command expects the
system CUDA toolkit layout. Override with `VLLM_BRAIN_CUDA_HOME` if needed.

## Model Downloads

vLLM downloads Hugging Face models on first startup, so there is no Ollama-style
required pull step. To prefetch a model before restarting the service:

```bash
.venv/bin/huggingface-cli download unsloth/Qwen3.6-27B-NVFP4
```

For gated/private models, authenticate first:

```bash
.venv/bin/huggingface-cli login
```

Sandy can continue to use:

```env
BRAIN_PROVIDER="vllm"
BRAIN_BASE_URL="http://127.0.0.1:8000/v1"
BRAIN_MODEL="sandy-brain"
```

Override `VLLM_BRAIN_SERVED_MODEL_NAME` only if you intentionally want Sandy to
send a different model name to the vLLM OpenAI-compatible endpoint.
