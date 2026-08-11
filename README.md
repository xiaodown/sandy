# Sandy

<p align="center">
  <img src="docs/images/v2.png" width="256" alt="Sandy" />
</p>

Sandy is a self-hosted Discord bot designed to behave like a recurring participant in a Discord server, not a utility assistant. She has long-term memory, opinions, casual internet-native speech patterns, and no obligation to be useful. The default deployment is local-first: conversations, model calls, memory, and search stay on infrastructure you control unless you intentionally configure external services.

The name is short for "sandbox." This is a rapid-prototyping playground for local LLM experimentation.

## What it actually does

When someone posts a message in a channel Sandy is in:

1. **Bouncer** decides if Sandy should respond, and whether she needs a tool (memory lookup, web search, etc.)
2. If a tool was recommended, it runs and the results get injected into context
3. **RAG** pulls semantically similar past messages from vector memory
4. **Brain** generates a reply using all of the above — personality prompt, conversation history, memories, tool results
5. In the background: the message gets tagged, optionally summarized, stored in Recall (SQLite), and embedded in ChromaDB

Sandy sees images too. Attachments now go through a two-stage vision path:

- a cheap routing caption before the bouncer decides whether Sandy should reply
- a richer detailed description only if Sandy is actually going to respond

That keeps non-reply image posts cheaper while still grounding real image replies properly.

The reference deployment runs on a single machine: LLM inference, embeddings, databases, and optional web search. The pieces are intentionally plain enough to split out later if your setup needs it.

## Architecture

### Text pipeline

```
Discord message arrives
  │
  ├─ Vision (if images attached)
  ├─ Bouncer (should Sandy respond? should she use a tool?)
  ├─ Tool dispatch (memory recall, web search, time check)
  ├─ RAG query (vector similarity search against past messages)
  ├─ Brain (generate response — no tool calling, just talks)
  └─ Memory pipeline (tag → summarize → store → embed) [background]
```

The brain model does not do tool calling. That was tried and it was bad. The bouncer (running at low temperature with structured JSON output) makes the tool decision, the bot executes it, and the results are handed to the brain as pre-fetched context. The brain just talks.

All configuration is centralized in `sandy/config.py` as frozen dataclasses, built from `.env` via `SandyConfig.from_env()`.

### Voice

Sandy can join voice channels. She listens via faster-whisper STT, generates short replies via the brain, and speaks via an external Qwen3-TTS service (in `tts_service/`).

```bash
!join [channel]    # join the specified (or your current) voice channel
!leave             # leave the voice channel
```

Requires voice admin permission, set via `python -m sandy.maintenance set-voice-admin`.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (for venv and package management)
- One local model backend:
  - [Ollama](https://ollama.ai/) for the original all-local setup
  - An OpenAI-compatible [vLLM](https://docs.vllm.ai/) server for the main brain model
- Docker with the compose plugin, if you want local SearXNG web search
- A GPU is strongly recommended. CPU or very small-model setups may work, but interactive latency depends heavily on the models you choose.

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd sandy
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Configure

Copy the example env file and fill it in:

```bash
cp .env.example .env
```

The `.env` is well-commented and broken into sections. The important bits:

- `DISCORD_API_KEY` — your bot token (see [Creating a Discord bot](#creating-a-discord-bot) below)
- `DB_DIR` — production database directory
- `TEST_DB_DIR` — database directory used automatically by `python -m sandy --test`
- `BRAIN_MODEL`, `BOUNCER_MODEL`, etc. — model names for each LLM role. By default, roles use Ollama tags unless a provider is configured otherwise.
- `BRAIN_PROVIDER` — set to `ollama` for the original all-Ollama path, or `vllm` to send only the main brain to an OpenAI-compatible vLLM server.
- `BRAIN_BASE_URL` / `BRAIN_API_KEY` — vLLM OpenAI-compatible endpoint settings when `BRAIN_PROVIDER=vllm`.
- `VISION_ROUTER_MODEL` — optional tiny multimodal model for cheap pre-bouncer image captions
- `EMBED_MODEL` — embedding model for ChromaDB (default: `mxbai-embed-large`)
- `OLLAMA_KEEP_ALIVE` — how long Ollama keeps a model in VRAM after the last request. Longer values reduce reload churn; shorter values free VRAM sooner.
- If multiple roles share one model tag, keep their `*_NUM_CTX` values aligned unless you want Ollama to spin up separate runners for the same model.

### 3. Install or start model backends

```bash
# Make sure ollama is running
systemctl status ollama

# Pull the model(s) you configured in .env
ollama pull <brain-model>
ollama pull mxbai-embed-large
```

Ollama *might* auto-pull on first use, but don't count on it. Pull explicitly.

If the brain is served by vLLM, keep vLLM in its own local virtualenv:

```bash
(cd vllm_service && uv venv .venv --python 3.12 && uv sync && cp .env.example .env)
```

Then start the brain endpoint before Sandy from the repo root:

```bash
vllm_service/start.sh
```

For service operations:

```bash
vllm_service/start.sh
vllm_service/status.sh
vllm_service/stop.sh
```

The service has its own `pyproject.toml`, `.env.example`, gitignored `.env`,
and gitignored `logs/` directory. It serves `http://127.0.0.1:8000/v1` and
exposes the stable OpenAI API model name `sandy-brain`; keep Sandy's
`BRAIN_MODEL` set to `sandy-brain` and switch backend models in
`vllm_service/.env`.

### 4. Start SearXNG (web search)

Sandy can search the web via a local [SearXNG](https://docs.searxng.org/) instance. It runs in Docker:

```bash
cd searxng
docker compose up -d
```

Verify it's working:

```bash
curl 'http://localhost:8888/search?q=test&format=json'
```

(Quote the URL. Unquoted `&` in bash backgrounds the second parameter and you'll stare at `curl` hanging forever wondering what's broken.)

### 5. Run the bot

```bash
source .venv/bin/activate
python -m sandy
```

That's it. Ctrl+C to stop.

## Running tests

Sandy uses `pytest`.

```bash
source .venv/bin/activate
pytest
```

For faster iteration while you're working on one area:

```bash
pytest tests/test_bot_pipeline.py
pytest -k memory
```

Tests are developer checks, not startup checks. Runtime sanity checks like
"is Ollama up" or "is SearXNG reachable" should live in a separate preflight
command later instead of being mixed into `pytest`.

## Health Checks

Sandy has a separate health/preflight command:

```bash
source .venv/bin/activate
python -m sandy.health
python -m sandy.health --test
python -m sandy.health --json
```

Startup now uses the same check logic with a hard/soft split:

- hard failures abort startup: missing Discord token, broken local state paths, Recall init/migration failure, registry DB failure, malformed numeric config
- soft failures warn only: Ollama reachability, missing configured models, vector-memory availability, SearXNG reachability

That split is deliberate. Sandy should not stay dead just because an optional dependency is temporarily unhealthy.

## Observability API

Sandy now exposes a small read-only local API for dashboards and operator views.
By default it listens on `127.0.0.1:8765`.

Endpoints:

```bash
open http://127.0.0.1:8765/
curl http://127.0.0.1:8765/api/status
curl http://127.0.0.1:8765/api/gpu
curl 'http://127.0.0.1:8765/api/turns/recent?limit=10'
curl http://127.0.0.1:8765/api/turns/<trace_id>
```

Current scope is intentionally small:

- `/` — local dashboard page with status, GPU cards, current turn track, recent turns, and trace drill-down
- `/api/status` — Discord connected state, current active turn/stage, memory-worker state, LLM busy/idle, last bouncer decision
- `/api/gpu` — GPU telemetry via `nvidia-smi` when available
- `/api/turns/recent` — recent turn summaries for list views
- `/api/turns/<trace_id>` — one trace with timeline + forensic artifacts for drill-down

Config lives in `.env`:

- `SANDY_API_ENABLED=true`
- `SANDY_API_HOST=127.0.0.1`
- `SANDY_API_PORT=8765`

## Inspecting logs

Sandy stores:

- compact turn/stage traces in `data/<mode>/logs/trace_events.db`
- full forensic artifacts in `data/<mode>/logs/sandy.jsonl`

Use the local CLI to inspect them:

```bash
source .venv/bin/activate
python -m sandy.logs recent
python -m sandy.logs show <trace_id>
python -m sandy.logs find --text "search phrase"
python -m sandy.logs failures
```

Use `--test` with the CLI to read `TEST_DB_DIR` instead of `DB_DIR`:

```bash
python -m sandy.logs --test recent
```

## Maintenance

For Recall/Chroma cleanup work:

```bash
source .venv/bin/activate
python -m sandy.maintenance --test recall-find --query "topic"
python -m sandy.maintenance --test delete-vector --discord-message-id <discord_message_id>
python -m sandy.maintenance --test purge-vector-from-recall --query "topic" --yes
```

`recall-find` is the safe first step. It shows Recall rows plus any stored Discord
message IDs so you can decide what to delete from vector memory.

## Data layout

```
data/
├── prod/           # production databases
│   ├── recall.db   # message archive (SQLite + FTS5)
│   ├── server.db   # server/channel/user registry
│   └── chroma/     # ChromaDB vector store
└── test/           # test databases (same structure)
```

Set `DB_DIR` for prod and `TEST_DB_DIR` for test in `.env`. `python -m sandy --test` swaps to `TEST_DB_DIR` before the bot imports, and both sets remain fully independent.

## Creating a Discord bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click **New Application**, give it a name
3. Go to **Bot** in the sidebar
4. Click **Reset Token** and copy it — this is your `DISCORD_API_KEY`
5. Under **Privileged Gateway Intents**, enable:
   - **Server Members Intent** (needed if you want member-aware behavior)
   - **Message Content Intent** (Sandy needs to read messages)
6. Go to **OAuth2 → URL Generator**
7. Under **Scopes**, check `bot`
8. Under **Bot Permissions**, check:
   - Send Messages
   - Read Message History
   - View Channels
9. Copy the generated URL, paste it in your browser, and invite the bot to your server

## Project structure

```
sandy/
├── sandy/                  # Python package
│   ├── __main__.py         # entry point (python -m sandy)
│   ├── config.py           # central config (SandyConfig.from_env())
│   ├── bot.py              # Discord client lifecycle and event glue
│   ├── prompt.py           # prompt factory (loads text from prompts/)
│   ├── memory.py           # tag + summarize + store pipeline
│   ├── tools.py            # tool definitions and dispatch
│   ├── registry.py         # server/channel/user lookup cache (SQLite)
│   ├── last10.py           # per-channel message cache
│   ├── vector_memory.py    # ChromaDB semantic memory
│   ├── logconf.py          # logging config
│   ├── trace.py            # per-turn trace dataclasses
│   ├── runtime_state.py    # thread-safe observability snapshot
│   ├── health.py           # startup health checks
│   ├── api.py              # local observability HTTP API
│   ├── paths.py            # path resolution helpers
│   ├── llm/                # ollama interface subpackage
│   │   ├── __init__.py     # OllamaInterface class + re-exports
│   │   ├── models.py       # structured output schemas (Pydantic)
│   │   └── coercion.py     # bouncer result coercion heuristics
│   ├── pipeline/           # text message-turn orchestration
│   │   ├── __init__.py     # build_pipeline() factory
│   │   ├── orchestrator.py # SandyPipeline coordinator
│   │   ├── bouncer.py      # bouncer stage
│   │   ├── brain.py        # brain stage
│   │   ├── reply.py        # Discord send (splits overlong)
│   │   ├── retrieval.py    # RAG query stage
│   │   ├── tool_dispatch.py# tool execution + result framing
│   │   ├── attachments.py  # vision processing
│   │   ├── memory_worker.py# supervised background queue
│   │   └── tracing.py      # pipeline trace wiring
│   ├── prompts/            # LLM prompt text files
│   │   ├── brain_system.txt
│   │   ├── bouncer_system.txt
│   │   └── ...             # tagger, summarizer, vision, voice
│   ├── recall/             # long-term message storage
│   │   ├── database.py     # SQLite + FTS5 operations
│   │   └── models.py       # Pydantic models
│   └── voice/              # voice channel subpackage
│       ├── manager.py      # VoiceManager (!join / !leave)
│       ├── capture.py      # raw audio from Discord
│       ├── stitching.py    # speech fragment assembly
│       ├── stt.py          # faster-whisper transcription
│       ├── tts.py          # Qwen3-TTS client
│       ├── response.py     # voice reply generation
│       └── history.py      # voice conversation history
├── tests/                  # pytest suite
├── tts_service/            # standalone Qwen3-TTS FastAPI service
├── web/dashboard/          # local observability dashboard
├── searxng/                # SearXNG docker-compose
├── data/                   # runtime databases (gitignored)
└── .env                    # configuration (gitignored)
```

## Tools

Sandy has eight tools. The bouncer decides when to use them — the brain never calls tools directly.

| Tool | What it does |
|------|-------------|
| `recall_recent` | Fetch recent messages from the archive |
| `recall_from_user` | Fetch messages from a specific person |
| `recall_by_topic` | Fetch messages by topic tag |
| `search_memories` | Full-text search across all archived messages |
| `search_web` | Search the internet via SearXNG |
| `steam_browse` | Browse Steam top sellers, specials, upcoming, and new releases |
| `get_current_time` | Current date and time |
| `dice_roll` | Roll one or more groups of dice |

## License

GPLv3. See [LICENSE](LICENSE).
