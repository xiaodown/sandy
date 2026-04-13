# AGENTS.md

Context for AI coding agents working on this project. If you're a human, the [README](README.md) is probably more useful.

## Project overview

Sandy is a Discord personality bot powered by local LLM inference via [ollama](https://ollama.ai/). She is not a helpful assistant — she's a persistent, opinionated presence in a Discord server. The "digital person" framing is the design goal. If she sounds like a customer support chatbot, something is wrong.

All inference runs locally. No cloud APIs. The stack is Python 3.12+, py-cord, ollama, ChromaDB, SQLite, and SearXNG (for web search). Voice uses faster-whisper for STT and an external Qwen3-TTS service.

## Dev environment

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Run
python -m sandy

# Test
pytest

# Verify ollama is up
systemctl status ollama

# Verify SearXNG is up
curl 'http://localhost:8888/search?q=test&format=json'
```

Package manager is `uv`. Virtual environment lives at `.venv/`. Install in editable mode with dev dependencies via `uv pip install -e ".[dev]"`. Tests run with `pytest`.

## Package structure

All source code lives in the `sandy/` Python package. There are no source files at the repo root.

```
sandy/
├── __init__.py             # package marker
├── __main__.py             # entry point: python -m sandy
├── config.py               # central config (frozen dataclasses, SandyConfig.from_env())
├── bot.py                  # Discord client lifecycle, event handlers, shutdown glue
├── paths.py                # project_root(), resolve_runtime_path(), resolve_db_dir()
├── prompt.py               # prompt factory — loads text from prompts/*.txt
├── memory.py               # tag → summarize → store pipeline (Recall + ChromaDB)
├── tools.py                # tool schemas, handlers, and dispatch
├── registry.py             # SQLite-backed server/channel/user lookup cache
├── last10.py               # per-channel rolling message cache (in-memory)
├── vector_memory.py        # ChromaDB semantic memory (RAG)
├── logconf.py              # logging config + get_logger()
├── trace.py                # TurnTrace / VoiceTurnTrace dataclasses, event/forensic payloads
├── runtime_state.py        # RuntimeState — thread-safe observability snapshot
├── api.py                  # read-only local HTTP API (aiohttp)
├── health.py               # startup health checks (hard/soft split)
├── logs.py                 # CLI: python -m sandy.logs
├── maintenance.py          # CLI: python -m sandy.maintenance
│
├── llm/                    # ollama interface subpackage
│   ├── __init__.py         # OllamaInterface class, _default_llm_config(), re-exports
│   ├── models.py           # BouncerResponse, TaggerResponse, SummarizerResponse, BrainResponse
│   └── coercion.py         # bouncer result coercion + Steam override heuristics
│
├── pipeline/               # text message-turn orchestration subpackage
│   ├── __init__.py         # build_pipeline() factory, exports SandyPipeline
│   ├── orchestrator.py     # SandyPipeline — top-level handle_message() coordinator
│   ├── bouncer.py          # bouncer stage (should Sandy respond? which tool?)
│   ├── brain.py            # brain stage (generate reply)
│   ├── reply.py            # Discord send — splits overlong replies
│   ├── retrieval.py        # RAG query stage
│   ├── tool_dispatch.py    # tool execution + result framing
│   ├── attachments.py      # image attachment processing (vision router + detail)
│   ├── memory_worker.py    # MemoryWorker — supervised background queue
│   └── tracing.py          # per-turn trace wiring for pipeline stages
│
├── prompts/                # LLM prompt text files (loaded by prompt.py)
│   ├── __init__.py         # package marker
│   ├── brain_system.txt    # Sandy's personality prompt
│   ├── bouncer_system.txt  # bouncer decision engine prompt (includes tool docs)
│   ├── tagger_system.txt   # tagger instructions
│   ├── summarizer_system.txt
│   ├── vision_router_system.txt
│   ├── vision_detail_system.txt
│   └── voice_addendum.txt  # appended to brain prompt in voice mode
│
├── recall/                 # long-term message storage subpackage
│   ├── __init__.py         # re-exports ChatDatabase, ChatMessageCreate, ChatMessageResponse
│   ├── database.py         # SQLite + FTS5 CRUD, schema migrations (v1→v2→v3→v4)
│   └── models.py           # Pydantic models for message records
│
└── voice/                  # voice channel subpackage
    ├── __init__.py         # exports VoiceManager, configure_voice, etc.
    ├── manager.py          # VoiceManager — session lifecycle, !join / !leave
    ├── models.py           # VoiceSession, PendingSpeakerTurn, configure_voice()
    ├── capture.py          # raw audio capture from Discord voice
    ├── stitching.py        # speech fragment stitching + silence detection
    ├── stt.py              # faster-whisper transcription
    ├── tts.py              # TTS service client (external Qwen3-TTS)
    ├── response.py         # voice reply generation (brain + TTS)
    ├── history.py          # voice conversation history
    └── tracing.py          # voice turn tracing
```

Other top-level directories:

```
tests/              # pytest suite (~180+ tests)
tts_service/        # standalone Qwen3-TTS FastAPI service (separate package)
web/dashboard/      # local observability dashboard (HTML/JS/CSS)
searxng/            # SearXNG docker-compose config
data/               # runtime databases (gitignored)
docs/               # architecture plans and images
```

### Import conventions

- All imports within `sandy/` are relative (`from .llm import OllamaInterface`, not `from sandy.llm import ...`)
- Subpackage re-exports keep external import paths stable — e.g. `from sandy.llm import BouncerResponse` works via `llm/__init__.py`
- Logging uses `get_logger()` from `logconf.py` everywhere except two intentional exceptions: `bot.py` line 36 (`logging.getLogger("discord").setLevel(...)`) and `__main__.py` startup logger
- Type hints use Python 3.12+ builtins: `list[x]`, `dict[x, y]`, `X | None` — no `Optional`, `List`, `Dict` from typing

## Runtime architecture

### Text pipeline

```
Discord message
  → bot.py                  on_message entry point
  → pipeline/orchestrator   handle_message
  → pipeline/attachments    cheap vision router caption (if images)
  → pipeline/bouncer        ask_bouncer: should Sandy respond? + tool recommendation
  → pipeline/attachments    detailed vision grounding (only if Sandy will reply)
  → pipeline/tool_dispatch  execute tool, frame results for brain injection
  → pipeline/retrieval      RAG: semantically similar past messages
  → pipeline/brain          ask_brain: generate reply (NO tool calling — just talks)
  → pipeline/reply          send reply to Discord (splits overlong messages)
  → pipeline/memory_worker  enqueue for background processing
  → memory.py               process_and_store (tag → summarize → Recall + ChromaDB)
```

### Voice pipeline

```
User speaks in voice channel
  → voice/capture           raw audio from Discord
  → voice/stitching         fragment assembly + silence detection
  → voice/stt               faster-whisper transcription
  → voice/response          brain reply generation (shorter, voice-tuned prompt)
  → voice/tts               Qwen3-TTS synthesis → playback
  → voice/history           conversation history tracking
```

Voice commands: `!join [channel]` and `!leave`. Requires voice admin permission (set via `python -m sandy.maintenance set-voice-admin`).

### Startup flow

1. `__main__.py`: parse `--test`, load `.env`, resolve `DB_DIR`
2. `health.py`: run health checks — hard failures abort, soft failures warn
3. `config.py`: `SandyConfig.from_env()` builds full config from env vars
4. `bot.py`: `setup(config)` → `build_pipeline()` wires all dependencies
5. `api.py`: start observability API if enabled
6. `llm/`: optional prewarm of bouncer model
7. `bot.start()` → Discord event loop

### Key design decisions

- **The brain model does NOT do tool calling.** Tool selection is handled entirely by the bouncer (low temperature, structured JSON via ollama's `format=` parameter). The bot executes the tool and injects results into the brain's system prompt. The brain just generates text. This was a deliberate architectural choice after tool calling via the brain model proved unreliable (deferral phrases, double responses, failed tool invocations).

- **Single asyncio.Lock in OllamaInterface.** ALL model calls (brain, bouncer, tagger, summarizer, vision) go through one lock. This ensures only one inference runs at a time on the GPU. Do not add a second lock or bypass it.

- **`format=` and `tools=` are mutually exclusive in the ollama API.** The bouncer uses `format=` for structured JSON output. The brain uses neither — it's a plain chat call.

- **Server isolation is enforced at dispatch time.** `tools.dispatch()` forcibly injects `server_id` into all server-scoped tool calls and strips any hallucinated snowflake IDs (`channel_id`, `author_id`). The model never sees server_id in tool schemas.

- **Sandy's own replies get stored and embedded.** Discord echoes bot messages back through `on_message`, and they go through the memory pipeline. This is intentional — her side of conversations is part of her memory.

- **`steam_browse` currently bypasses RAG.** Fresh Steam storefront data is more trustworthy than stale bot-authored vector memories about storefront state.

- **The bouncer has one deterministic Steam override in `llm/coercion.py`.** If the small bouncer model picks `search_web` for an obvious Steam storefront/category request, post-parse code rewrites it to `steam_browse`. Not philosophically pure, but more reliable than trusting soft prompt wording.

- **"Sandy" is hardcoded, not parameterized.** The bot name used to flow through a `bot_name` parameter chain. That was removed — all prompts now hardcode "Sandy" in the text files under `prompts/`.

## Configuration

All runtime configuration is centralized in `sandy/config.py`. The `SandyConfig.from_env()` classmethod reads every env var and produces a frozen, typed config object. Individual modules receive their config slice at construction time via `build_pipeline()`.

Env vars are read from `.env` (gitignored). See `.env.example` for the full list with comments.

### Config classes

`SandyConfig` composes six frozen slotted dataclasses:

| Class | Scope | Key fields |
|-------|-------|------------|
| `LlmConfig` | Model names, temperatures, context sizes, predict caps | `brain_model`, `bouncer_model`, `tagger_model`, `summarizer_model`, `vision_model`, `*_temperature`, `*_num_ctx`, `*_num_predict`, `keep_alive` |
| `VoiceConfig` | STT/TTS settings, capture params, reply limits | `stt_model`, `tts_service_url`, `capture_dir`, `stitch_*`, `reply_max_*` |
| `StorageConfig` | Database paths, embedding model, thresholds | `db_dir`, `recall_db_name`, `server_db_name`, `embed_model`, `vector_max_distance`, `summarize_threshold` |
| `SearchConfig` | SearXNG connection, Steam cache | `searxng_host`, `searxng_port`, `steam_cache_ttl_seconds` |
| `LogConfig` | Log rotation, trace retention | `rotate_bytes`, `backup_count`, `trace_retention_days` |
| `ApiConfig` | Observability API settings | `enabled`, `host`, `port` |

Top-level `SandyConfig` also holds: `discord_api_key`, `prewarm_enabled`, `prewarm_model_name`, `test_mode`.

### Key env vars

| Variable | Purpose | Default |
|----------|---------|---------|
| `DISCORD_API_KEY` | Bot token | — (required) |
| `DB_DIR` | Production database directory | `data/prod/` |
| `TEST_DB_DIR` | Test database directory used by `--test` | `data/test/` |
| `BRAIN_MODEL` | Main personality model | `qwen2.5:14b` |
| `BOUNCER_MODEL` | Decision engine model | `qwen2.5:14b` |
| `TAGGER_MODEL` | Tag generation model | (Llama 3.2 3B GGUF) |
| `SUMMARIZER_MODEL` | Summarization model | (Llama 3.2 3B GGUF) |
| `EMBED_MODEL` | Embedding model (ChromaDB) | `mxbai-embed-large` |
| `BRAIN_TEMPERATURE` | Brain creativity | `1.1` |
| `BOUNCER_TEMPERATURE` | Bouncer determinism | `0.1` |
| `BRAIN_NUM_PREDICT` | Max tokens per text reply | `512` |
| `VOICE_BRAIN_NUM_PREDICT` | Max tokens per voice reply | `80` |
| `BRAIN_NUM_CTX` | Brain context window | `8192` |
| `BOUNCER_NUM_CTX` | Bouncer context window | `8192` |
| `TAGGER_NUM_CTX` | Tagger context window | `4096` |
| `SUMMARIZER_NUM_CTX` | Summarizer context window | `4096` |
| `VISION_NUM_CTX` | Vision context window | `4096` |
| `VISION_NUM_PREDICT` | Vision output token cap | `224` |
| `VISION_TEMPERATURE` | Detailed vision temperature | `0.3` |
| `VISION_ROUTER_MODEL` | Cheap pre-bouncer vision caption model | unset |
| `VISION_ROUTER_NUM_CTX` | Router caption context window | `2048` |
| `VISION_ROUTER_NUM_PREDICT` | Router caption output cap | `48` |
| `VISION_ROUTER_TEMPERATURE` | Router caption determinism | `0.1` |
| `PREWARM_NUM_CTX` | Prewarm context window | `BOUNCER_NUM_CTX` |
| `OLLAMA_KEEP_ALIVE` | VRAM model retention | `1h` |
| `SUMMARIZE_THRESHOLD` | Chars before summarizing | `144` |
| `VECTOR_MAX_DISTANCE` | ChromaDB similarity threshold | `0.6` |
| `SERVER_DB_NAME` | Registry DB filename | `server.db` |
| `RECALL_DB_NAME` | Recall DB filename | `recall.db` |
| `SEARXNG_HOST` | SearXNG hostname | `127.0.0.1` |
| `SEARXNG_PORT` | SearXNG port | `8888` |
| `STEAM_BROWSE_CACHE_TTL_SECONDS` | Steam storefront cache TTL | `600` |
| `LOG_ROTATE_BYTES` | JSONL log rotation size | `20971520` |
| `LOG_BACKUP_COUNT` | Rotated JSONL files to keep | `10` |
| `TRACE_RETENTION_DAYS` | Trace SQLite retention window | `14` |
| `SANDY_API_ENABLED` | Enable observability API | `true` |
| `SANDY_API_HOST` | API listen address | `127.0.0.1` |
| `SANDY_API_PORT` | API listen port | `8765` |

Code-level defaults in `config.py` for model names are stale (`qwen2.5:14b`, old Llama 3B GGUF paths). They don't matter in practice because `.env` always overrides them, but be aware if you're reading the source.

## Database layout

```
data/
├── prod/           # production
│   ├── recall.db   # message archive (SQLite, FTS5)
│   ├── server.db   # server/channel/user registry (SQLite)
│   ├── chroma/     # ChromaDB vector store
│   └── logs/       # JSONL + trace SQLite
└── test/           # test (same structure, independent data)
```

Set `DB_DIR` for prod and `TEST_DB_DIR` for test in `.env`. `python -m sandy --test` swaps `DB_DIR` to `TEST_DB_DIR` before importing the bot. All databases follow the active `DB_DIR`.

### Recall database schema (v4)

- `chat_messages` — main table: id, discord_message_id, author_id, author_name, channel_id, channel_name, server_id, server_name, content, timestamp, summary
- `tags` — tag dictionary (id, name)
- `message_tags` — M2M join table
- `messages_fts` — FTS5 virtual table over content + summary
- `schema_version` — migration tracking

Migrations run automatically on `init_db()`. The database module (`recall/database.py`) handles v1→v2→v3→v4 upgrades.

## Tools

Eight tools, defined in `tools.py`. The bouncer selects them; `pipeline/tool_dispatch.py` executes them.

| Tool | Server-scoped | Parameters |
|------|:---:|------------|
| `recall_recent` | yes | hours_ago, minutes_ago, since, until, channel, limit |
| `recall_from_user` | yes | author (required), hours_ago, since, channel, limit |
| `recall_by_topic` | yes | tag (required), author, hours_ago, limit |
| `search_memories` | yes | query (required), author, hours_ago, channel, limit |
| `search_web` | no | query (required), n_results (default 5, max 10) |
| `steam_browse` | no | category (required), limit |
| `get_current_time` | no | (none) |
| `dice_roll` | no | dice (required) |

### Adding a new tool

1. Write `async def _handle_<name>(args: dict) -> str` in `tools.py`
2. Add its schema dict to `TOOL_SCHEMAS`
3. Register the name in `_HANDLERS`
4. If it queries per-server data, add the name to `_SERVER_SCOPED_TOOLS`
5. Update the bouncer prompt in `prompts/bouncer_system.txt` (name, params, when-to-use guidance)
6. If it needs custom result framing, add a case to `_format_tool_context()` in `pipeline/tool_dispatch.py`

### Parameter remapping

The bouncer sends tool parameters using natural names (`author`, `query`). The Recall database expects different names (`author_name`, `q`). This remapping happens centrally in `_recall_query()` in `tools.py` — you don't need to handle it per-handler.

## LLM roles

All roles use ollama's async Python client via `OllamaInterface` in `sandy/llm/`. Structured output roles (bouncer, tagger, summarizer) use `format=<PydanticModel>.model_json_schema()` for constrained decoding at the logit level — this is stronger than prompt-and-retry approaches.

| Role | Temperature | Structured output | Purpose |
|------|:-----------:|:-----------------:|---------|
| Brain | 1.1 | no | Personality, conversation |
| Bouncer | 0.1 | `BouncerResponse` | Respond decision + tool recommendation |
| Tagger | 0.1 | `TaggerResponse` | 1-3 tags per message |
| Summarizer | 0.1 | `SummarizerResponse` | Optional long-message summary |
| Vision | 0.3 | no | Image description (sterile, no personality) |

The bouncer has an incoherence detector: if it says `should_respond=False` but its `reason` contains phrases like "mentioned", "addressed", "directed at", etc., the boolean is flipped to True. This catches a known failure mode where the model reasons correctly but outputs the wrong boolean.

There is also a Pydantic `model_validator` on `BouncerResponse` that forces `use_tool=False` when `recommended_tool` is empty — prevents a `use_tool=True` / `recommended_tool=None` mismatch.

### LLM subpackage layout

- `llm/models.py` — Pydantic schemas: `BouncerResponse`, `TaggerResponse`, `SummarizerResponse`, `BrainResponse`
- `llm/coercion.py` — deterministic post-parse fixes: `_coerce_bouncer_tool_selection()`, `_infer_steam_browse_category()`, `_looks_like_direct_image_ask()`, `_extract_history_messages()`
- `llm/__init__.py` — `OllamaInterface` class with methods: `ask_bouncer()`, `ask_brain()`, `ask_tagger()`, `ask_summarizer()`, `ask_vision()`, `ask_vision_router()`, `warm_model()`, `is_running()`. Also re-exports everything from models and coercion.

### Context sizing matters more than expected

- Sandy sets explicit `num_ctx` values for brain, bouncer, tagger, summarizer, vision, and prewarm via `.env`. This is not optional polish; it materially changes VRAM use and runner behavior.
- Leaving non-brain calls without explicit `num_ctx` lets ollama inherit the model's huge default context, which can produce a 131072-token runner, ~20 GiB KV cache, and weight splitting across GPUs.
- If the same model is used for multiple roles with different `num_ctx` values, ollama may spin up separate runners per context size. Unify role context sizes for maximum runner reuse.

### Prewarm behavior

- `warm_model()` is async and uses the shared `AsyncClient`, the shared `asyncio.Lock`, explicit `keep_alive`, and explicit `PREWARM_NUM_CTX`.
- Prewarming happens in `__main__.py` before `bot.start(...)`, not inside `on_ready()`. This avoids blocking Discord startup callbacks on model loading.
- Default `keep_alive` is `1h`. Shorter values free VRAM but hurt cold-start latency without meaningfully reducing idle GPU power draw.

### Actual runtime shape

One user turn can hit ollama multiple times even without tools:
- bouncer chat → RAG embed query → brain chat → tagger chat → optional summarizer chat → embed for vector storage

The embed model (`mxbai-embed-large`) is a separate ollama runner and will appear as a small single-GPU load. That is expected.

Current recommended local split: keep the larger conversational model for `BRAIN_MODEL`, share a smaller instruct model across `BOUNCER_MODEL` / `TAGGER_MODEL` / `SUMMARIZER_MODEL`, and use a dedicated `VISION_MODEL` for attachment descriptions.

## Tests and observability

Run the test suite with `pytest`. ~180+ tests covering pipeline behavior, tools, memory/Recall/Chroma, logging/trace, voice manager, and shutdown/background behavior.

### Logging layers

- **Console**: colored, directional runtime logs
- **JSONL**: full forensic artifacts (`data/<mode>/logs/sandy.jsonl`)
- **SQLite**: compact trace timeline (`data/<mode>/logs/trace_events.db`)

### Observability API

Local read-only HTTP API (default `127.0.0.1:8765`):

- `/` — dashboard page with status, GPU cards, current turn, recent turns, trace drill-down
- `/api/status` — Discord state, active turn/stage, memory-worker state, LLM busy/idle, last bouncer decision
- `/api/gpu` — GPU telemetry via `nvidia-smi`
- `/api/turns/recent` — recent turn summaries
- `/api/turns/<trace_id>` — one trace with timeline + forensic artifacts

### Inspection CLIs

```bash
python -m sandy.logs recent
python -m sandy.logs show <trace_id>
python -m sandy.logs find --text "..."
python -m sandy.logs failures
python -m sandy.health [--test] [--json]
python -m sandy.maintenance recall-find --query "..."
python -m sandy.maintenance delete-vector --discord-message-id ...
python -m sandy.maintenance purge-vector-from-recall --query "..." --yes
python -m sandy.maintenance set-voice-admin --guild-id ... --user-id ...
```

Add `--test` to CLI commands to read `TEST_DB_DIR` instead of `DB_DIR`.

### Startup health policy

- Hard failures abort startup: missing Discord token, malformed config values, unwritable local state paths, Recall init failure, registry DB failure.
- Soft failures warn only: Ollama reachability, missing configured models, vector-memory availability, SearXNG reachability.
- Do not promote optional dependencies to startup gates. That turns degraded mode into unnecessary downtime.

## Known rough edges

- The bouncer is in a workable middle ground, but still prompt-tuned rather than robustly evaluated.
- The brain still occasionally overuses conversational follow-up questions; prompt guidance pushes against it, but this is not "solved."
- Tool-informed turns can still get polluted by old conversation history even when RAG is bypassed.
- If you see `done_reason=length` in brain response logs, the model hit `num_predict`; that is a generation-cap issue, not a Discord send-limit issue.

## Conventions

- Imports within `sandy/` are relative (`from .module import Thing`)
- Logging uses `get_logger()` from `logconf.py` (not `logging.getLogger()`)
- Type hints use Python 3.12+ builtins: `list[x]`, `dict[x, y]`, `X | None`
- All blocking I/O (sqlite3 calls, file ops) runs in `asyncio.to_thread()` to avoid blocking the event loop
- Recall database operations are synchronous sqlite3 under the hood, wrapped with `asyncio.to_thread()` at the call site
- No SVG, no ImageMagick, no Inkscape — security exclusion. Image attachments processed by Pillow only (JPEG/PNG/GIF/WebP). WebP is converted to JPEG in memory before vision inference because ollama's vision runner crashes on WebP.

## Things that aren't obvious

- `format=` and `tools=` are mutually exclusive in the ollama API. If you try to use both, you'll get an error.
- `PREWARM_MODEL_NAME="${BOUNCER_MODEL}"` in `.env` — python-dotenv expands shell variable references.
- `KNOWN_TOOLS` in `tools.py` is a `frozenset` derived from `_HANDLERS.keys()`. It stays in sync automatically.
- ChromaDB runs embedded in-process (no separate server). It writes to `<DB_DIR>/chroma/`.
- The bouncer prompt lives in `prompts/bouncer_system.txt` and contains inline tool documentation. If you add a tool, you must update it manually — it does not auto-generate from schemas.
- Tool results are not stored in RAG or Recall. They're injected into context for one turn only. This is intentional.
- Ollama's default context behavior is not something to trust blindly. Sandy sets explicit context sizes per role; if a new ollama call site is added without a `num_ctx`, expect trouble.
- `_registry` in `tools.py` is `None` by default; it gets wired at startup via `init_tools_config(registry=...)`. Do not instantiate `Registry()` at module level.
- `load_dotenv()` appears in many modules defensively. This is harmless and idempotent.
- Reply sending splits overlong brain replies into multiple Discord messages. Do not assume a single `channel.send()` is always safe.
- Background memory processing runs behind a supervised queue/worker in `pipeline/memory_worker.py`. Per-message failures are caught; the worker keeps draining.
