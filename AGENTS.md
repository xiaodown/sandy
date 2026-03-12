# AGENTS.md

Context for AI coding agents working on this project. If you're a human, the [README](README.md) is probably more useful.

## Project overview

Sandy is a Discord personality bot powered by local LLM inference via [ollama](https://ollama.ai/). She is not a helpful assistant — she's a persistent, opinionated presence in a Discord server. The "digital person" framing is the design goal. If she sounds like a customer support chatbot, something is wrong.

All inference runs locally. No cloud APIs. The stack is Python 3.12+, discord.py, ollama, ChromaDB, SQLite, and SearXNG (for web search).

## Dev environment

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -e .

# Run
python -m sandy

# Verify ollama is up
systemctl status ollama

# Verify SearXNG is up
curl 'http://localhost:8888/search?q=test&format=json'
```

Package manager is `uv`. Virtual environment lives at `.venv/`. The package is installed in editable mode (`uv pip install -e .`). There is no test suite yet.

## Package structure

All source code lives in the `sandy/` Python package. There are no source files at the repo root.

```
sandy/
├── __init__.py         # package marker
├── __main__.py         # entry point: python -m sandy
├── bot.py              # Discord event loop, full message pipeline
├── llm.py              # async ollama wrapper (brain, bouncer, tagger, summarizer, vision)
├── prompt.py           # all system prompts as static factory methods on SandyPrompt
├── memory.py           # tag → summarize → store pipeline (Recall + ChromaDB)
├── tools.py            # tool schemas, handlers, and dispatch
├── registry.py         # SQLite-backed server/channel/user lookup cache
├── last10.py           # per-channel rolling message cache (in-memory)
├── vector_memory.py    # ChromaDB semantic memory (RAG)
├── logconf.py          # logging config
└── recall/             # long-term message storage subpackage
    ├── __init__.py     # re-exports ChatDatabase, ChatMessageCreate, ChatMessageResponse
    ├── database.py     # SQLite + FTS5 CRUD, schema migrations (v1→v2→v3)
    └── models.py       # Pydantic models for message records
```

All imports within the package are relative (`from .llm import OllamaInterface`, not `from sandy.llm import ...`).

## Runtime architecture

```
Discord message
  → bot.py              on_message entry point
  → bot.py              _describe_attachments (vision model, if images)
  → llm.py              ask_bouncer: should Sandy respond? + tool recommendation
  → tools.py            dispatch (execute tool, if bouncer recommended one)
  → bot.py              _format_tool_context (frame results for brain injection)
  → vector_memory.py    query (RAG: semantically similar past messages)
  → llm.py              ask_brain: generate reply (NO tool calling — just talks)
  → bot.py              send reply to Discord
  → memory.py           process_and_store (background: tag → summarize → Recall + ChromaDB)
```

### Key design decisions

- **The brain model does NOT do tool calling.** Tool selection is handled entirely by the bouncer (low temperature, structured JSON via ollama's `format=` parameter). The bot executes the tool and injects results into the brain's system prompt. The brain just generates text. This was a deliberate architectural choice after tool calling via the brain model proved unreliable (deferral phrases, double responses, failed tool invocations).

- **Single asyncio.Lock in OllamaInterface.** ALL model calls (brain, bouncer, tagger, summarizer, vision) go through one lock. This ensures only one inference runs at a time on the GPU. Do not add a second lock or bypass it.

- **`format=` and `tools=` are mutually exclusive in the ollama API.** The bouncer uses `format=` for structured JSON output. The brain uses neither — it's a plain chat call.

- **Server isolation is enforced at dispatch time.** `tools.dispatch()` forcibly injects `server_id` into all server-scoped tool calls and strips any hallucinated snowflake IDs (`channel_id`, `author_id`). The model never sees server_id in tool schemas.

- **Sandy's own replies get stored and embedded.** Discord echoes bot messages back through `on_message`, and they go through the memory pipeline. This is intentional — her side of conversations is part of her memory.

## Configuration

All configuration is in `.env` (gitignored). See `.env.example` for the full list with comments.

Key env vars:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DISCORD_API_KEY` | Bot token | — (required) |
| `DB_DIR` | Production database directory | `data/prod/` |
| `TEST_DB_DIR` | Test database directory used by `--test` | `data/test/` |
| `BRAIN_MODEL` | Main personality model | `qwen2.5:14b` |
| `BOUNCER_MODEL` | Decision engine model | `qwen2.5:14b` |
| `TAGGER_MODEL` | Tag generation model | `qwen2.5:14b` |
| `SUMMARIZER_MODEL` | Summarization model | `qwen2.5:14b` |
| `EMBED_MODEL` | Embedding model (ChromaDB) | `mxbai-embed-large` |
| `BRAIN_TEMPERATURE` | Brain creativity | `1.1` |
| `BOUNCER_TEMPERATURE` | Bouncer determinism | `0.1` |
| `BRAIN_NUM_PREDICT` | Max tokens per reply | `512` |
| `BRAIN_NUM_CTX` | Context window size | `8192` |
| `BOUNCER_NUM_CTX` | Bouncer context window size | `8192` |
| `TAGGER_NUM_CTX` | Tagger context window size | `4096` |
| `SUMMARIZER_NUM_CTX` | Summarizer context window size | `4096` |
| `VISION_NUM_CTX` | Vision context window size | `8192` |
| `PREWARM_NUM_CTX` | Prewarm context window size | `BOUNCER_NUM_CTX` |
| `OLLAMA_KEEP_ALIVE` | VRAM model retention | `1h` |
| `SUMMARIZE_THRESHOLD` | Chars before summarizing | `450` |
| `SERVER_DB_NAME` | Registry DB filename | `server.db` |
| `RECALL_DB_NAME` | Recall DB filename | `recall.db` |
| `SEARXNG_HOST` | SearXNG hostname | `127.0.0.1` |
| `SEARXNG_PORT` | SearXNG port | `8888` |

Note: the code-level defaults in `llm.py` for model names are stale (`qwen2.5:14b`, old Llama 3B GGUF paths). They don't matter in practice because `.env` always overrides them, but be aware if you're reading the source.

## Database layout

```
data/
├── prod/           # production
│   ├── recall.db   # message archive (SQLite, FTS5)
│   ├── server.db   # server/channel/user registry (SQLite)
│   └── chroma/     # ChromaDB vector store
└── test/           # test (same structure, independent data)
```

Set `DB_DIR` for prod and `TEST_DB_DIR` for test in `.env`. `python -m sandy --test` swaps `DB_DIR` to `TEST_DB_DIR` before importing the bot. All three databases (recall, server, chroma) follow the active `DB_DIR`.

### Recall database schema (v3)

- `chat_messages` — main table: id, author_id, author_name, channel_id, channel_name, server_id, server_name, content, timestamp, summary
- `tags` — tag dictionary (id, name)
- `message_tags` — M2M join table
- `messages_fts` — FTS5 virtual table over content + summary
- `schema_version` — migration tracking

Migrations run automatically on `init_db()`. The database module (`recall/database.py`) handles v1→v2→v3 upgrades.

## Tools

Six tools, defined in `tools.py`. The bouncer selects them; `bot.py` dispatches them.

| Tool | Server-scoped | Parameters |
|------|:---:|------------|
| `recall_recent` | yes | hours_ago, minutes_ago, since, until, channel, limit |
| `recall_from_user` | yes | author (required), hours_ago, since, channel, limit |
| `recall_by_topic` | yes | tag (required), author, hours_ago, limit |
| `search_memories` | yes | query (required), author, hours_ago, channel, limit |
| `search_web` | no | query (required), n_results (default 5, max 10) |
| `get_current_time` | no | (none) |

### Adding a new tool

1. Write `async def _handle_<name>(args: dict) -> str` in `tools.py`
2. Add its schema dict to `TOOL_SCHEMAS`
3. Register the name in `_HANDLERS`
4. If it queries per-server data, add the name to `_SERVER_SCOPED_TOOLS`
5. Add tool documentation to `bouncer_prompt()` in `prompt.py` (name, params, when-to-use guidance)
6. If it needs custom result framing, add a case to `_format_tool_context()` in `bot.py`

### Parameter remapping

The bouncer sends tool parameters using natural names (`author`, `query`). The Recall database expects different names (`author_name`, `q`). This remapping happens centrally in `_recall_query()` in `tools.py` — you don't need to handle it per-handler.

## LLM roles

All roles use ollama's async Python client. Structured output roles (bouncer, tagger, summarizer) use `format=<PydanticModel>.model_json_schema()` for constrained decoding at the logit level — this is stronger than prompt-and-retry approaches.

| Role | Temperature | Structured output | Purpose |
|------|:-----------:|:-----------------:|---------|
| Brain | 0.8 | no | Personality, conversation |
| Bouncer | 0.1 | `BouncerResponse` | Respond decision + tool recommendation |
| Tagger | 0.1 | `TaggerResponse` | 1-3 tags per message |
| Summarizer | 0.1 | `SummarizerResponse` | Optional long-message summary |
| Vision | — | no | Image description (sterile, no personality) |

The bouncer has an incoherence detector: if it says `should_respond=False` but its `reason` contains phrases like "mentioned", "addressed", "directed at", etc., the boolean is flipped to True. This catches a known failure mode where the model reasons correctly but outputs the wrong boolean.

There is also a Pydantic `model_validator` on `BouncerResponse` that forces `use_tool=False` when `recommended_tool` is empty — prevents a `use_tool=True` / `recommended_tool=None` mismatch.

### Context sizing matters more than expected

- As of 2026-03-11, Sandy sets explicit `num_ctx` values for brain, bouncer, tagger, summarizer, vision, and prewarm via `.env`. This is not optional polish; it materially changes VRAM use and runner behavior.
- One debugging session on dual 3090s showed that leaving non-brain calls without explicit `num_ctx` let ollama inherit the model's huge default context, which produced:
  - `requested context size too large for model num_ctx=262144 n_ctx_train=131072`
  - a 131072-token runner
  - ~20 GiB KV cache
  - weight splitting across both GPUs
  - misleading "VRAM thrash" symptoms in `nvtop`
- After adding explicit per-role context limits, the same 24B Mistral model fit comfortably on a single 3090 for normal Sandy operation. Do not assume "it used both GPUs before, therefore it needs both GPUs." That assumption was wrong.
- If the same model is used for multiple roles with different `num_ctx` values, ollama may still spin up separate runners per context size. If you want maximum runner reuse while using one model for everything, unify the role context sizes.

### Prewarm behavior

- `warm_model()` is async now and uses the shared `AsyncClient`, the shared `asyncio.Lock`, explicit `keep_alive`, and explicit `PREWARM_NUM_CTX`.
- Prewarming now happens in `__main__.py` before `bot.start(...)`, not inside `on_ready()`. This avoids blocking Discord startup callbacks on model loading.
- If you ever see a fresh `POST /api/generate` followed by a giant-context warning in the ollama logs again, assume the prewarm path regressed first.

### Actual runtime shape

- One user turn can still hit ollama multiple times even without tools:
  - bouncer chat
  - RAG embed query
  - brain chat
  - tagger chat
  - optional summarizer chat
  - embed for vector storage
- The embed model (`mxbai-embed-large`) is a separate ollama runner and will continue to appear as a small single-GPU load. That is expected.
- The current "same big model for brain/bouncer/tagger/summarizer/vision" setup is intentionally suboptimal and was originally a desktop-VRAM compromise. On the homelab box, revisiting role-specific models is now practical.

### Reply handling and background work

- As of 2026-03-12, Sandy splits overlong brain replies into multiple Discord messages before send. Do not assume one `channel.send()` is always safe.
- Message persistence is no longer tied to reply delivery. Incoming messages are queued for memory processing before the reply send path, so a Discord send failure does not skip Recall/RAG storage for the triggering user message.
- Background memory processing now runs behind a single in-process queue/worker owned by `bot.py`. It is still deferred work, but it is no longer an unsupervised `asyncio.create_task(...)` fire-and-forget path.
- Brain responses now log Ollama's `done_reason`. If you see `done_reason=length`, the model hit `num_predict`; that is a generation-cap issue, not a Discord send-limit issue.

## Conventions

- Imports within the `sandy/` package are relative (`from .module import Thing`)
- Logging uses Python stdlib `logging`. `bot.py` uses a custom `get_logger()` from `logconf.py`; other modules use `logging.getLogger(__name__)`
- All blocking I/O (sqlite3 calls, file ops) runs in `asyncio.to_thread()` to avoid blocking the event loop
- The Recall database operations are synchronous sqlite3 under the hood, wrapped with `asyncio.to_thread()` at the call site
- No SVG, no ImageMagick, no Inkscape. These are excluded as image processing dependencies for security reasons. Image attachments are processed by Pillow only (JPEG/PNG/GIF/WebP). WebP is converted to JPEG in memory before vision inference because ollama's vision runner crashes on WebP.

## Things that aren't obvious

- `format=` and `tools=` are mutually exclusive in the ollama API. If you try to use both, you'll get an error.
- `PREWARM_MODEL_NAME="${BOUNCER_MODEL}"` in `.env` — python-dotenv expands shell variable references.
- Historical note: `warm_model()` used to call synchronous `ollama.generate()` from `on_ready()`, which could block startup and accidentally inherit a huge model default context. That was fixed on 2026-03-11.
- `KNOWN_TOOLS` in `tools.py` is a `frozenset` derived from `_HANDLERS.keys()`. It stays in sync automatically. `bot.py` checks incoming bouncer recommendations against it.
- ChromaDB runs embedded in-process (no separate server). It writes to `<DB_DIR>/chroma/`.
- `notes` in the repo root is a flat text file, not a directory.  Actually, nevermind, the human deleted it.  Ok, `notes` doesn't exist, ignore.
- The bouncer prompt in `prompt.py` contains inline tool documentation (names, params, when-to-use). If you add a tool, you must update the bouncer prompt manually — it does not auto-generate from schemas.
- Tool results are not stored in RAG or Recall. They're injected into context for one turn only. This is intentional.
- Ollama's default context behavior is not something to trust blindly. Sandy now sets explicit context sizes per role; if a new ollama call site is added without a `num_ctx`, expect trouble.

## Recommended Next Work

The old high-level plan in the root project notes still mostly stands, but the practical near-term order is clearer now:

1. Finish Phase 1 hardening around the text pipeline:
   reply-length handling, send-path robustness, and supervised background work.
2. Add per-turn trace IDs and stage timing/logging before chasing more features.
   The biggest remaining weakness is that Sandy is still not inspectable enough.
3. Revisit role-specific models now that the VRAM picture is understood.
   Keeping one 24B multimodal model for every role was a temporary compromise, not a good architecture.
4. Keep voice work out of scope until the text bot is observable and boring.

If another agent needs a single concrete next task, "add per-turn tracing and stage latency logs" is the best choice. It supports the project plan, teaches useful ops skills, and reduces future debugging by a lot.
