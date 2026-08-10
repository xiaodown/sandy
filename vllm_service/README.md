# Sandy vLLM Brain Service

This directory contains the local vLLM runtime used when `BRAIN_PROVIDER=vllm`.
It is intentionally separate from Sandy's main `.venv`, similar to `tts_service/`.

## Setup

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python 'vllm==0.26.0' --torch-backend cu130
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

The detached process writes logs to `../data/prod/logs/vllm-brain.log` and stores
its pid in `vllm-brain.pid`.

## GPU Placement

By default `run.sh` pins vLLM to the RTX PRO 6000 Blackwell GPU UUID:

```bash
GPU-5b44da3f-50e1-6622-bd78-8382819e596d
```

Override with `VLLM_BRAIN_CUDA_VISIBLE_DEVICES` if the hardware layout changes.
