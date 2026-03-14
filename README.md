# Sandy

<p align="center">
  <img src="docs/sandy-hoodie.png" width="256" alt="Sandy" />
</p>

Sandy is a Discord bot that acts like a person who lives in your server, not a helpful assistant. She has long-term memory, opinions, casual internet-native speech patterns, and no obligation to be useful. She runs entirely on local hardware — no cloud APIs, no OpenAI, no sending your conversations anywhere.

The name is short for "sandbox." This is a rapid-prototyping playground for local LLM experimentation.

## What it actually does

When someone posts a message in a channel Sandy is in:

1. **Bouncer** decides if Sandy should respond, and whether she needs a tool (memory lookup, web search, etc.)
2. If a tool was recommended, it runs and the results get injected into context
3. **RAG** pulls semantically similar past messages from vector memory
4. **Brain** generates a reply using all of the above — personality prompt, conversation history, memories, tool results
5. In the background: the message gets tagged, optionally summarized, stored in Recall (SQLite), and embedded in ChromaDB

Sandy sees images too. Attachments get described by a vision model and the description is injected into context so she can react to what people post.

Everything runs locally on a single machine. The LLM inference, the embeddings, the databases, the web search — all of it.

## Architecture

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

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (for venv and package management)
- [ollama](https://ollama.ai/) (local LLM server)
- Docker with the compose plugin (for SearXNG web search)
- A GPU with enough VRAM to run your chosen model. Sandy was developed on a 3090 Ti (24GB). Smaller models exist but YMMV.

## Setup

### 1. Clone and install

```bash
git clone <repo-url> sandy
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
- `BRAIN_MODEL`, `BOUNCER_MODEL`, etc. — ollama model tags. Roles can share a model, but splitting them is often better once VRAM behavior is understood.
- `EMBED_MODEL` — embedding model for ChromaDB (default: `mxbai-embed-large`)
- `OLLAMA_KEEP_ALIVE` — how long ollama keeps a model in VRAM after the last request. `1h` is a better default when Sandy is the main local GPU workload; lowering it mainly buys back VRAM, not necessarily lower idle power.
- If multiple roles share one model tag, keep their `*_NUM_CTX` values aligned unless you want Ollama to spin up separate runners for the same model.

### 3. Install and pull models

```bash
# Make sure ollama is running
systemctl status ollama

# Pull whatever model(s) you configured in .env
ollama pull hf.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF:UD-Q4_K_XL
ollama pull mxbai-embed-large
```

Ollama *might* auto-pull on first use, but don't count on it. Pull explicitly.

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

## Inspecting logs

Sandy stores:

- compact turn/stage traces in `data/<mode>/logs/trace_events.db`
- full forensic artifacts in `data/<mode>/logs/sandy.jsonl`

Use the local CLI to inspect them:

```bash
source .venv/bin/activate
python -m sandy.logs recent
python -m sandy.logs show 1482258945799094444
python -m sandy.logs find --text "color coding"
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
python -m sandy.maintenance --test recall-find --query "steam"
python -m sandy.maintenance --test delete-vector --discord-message-id 1482282320600891422
python -m sandy.maintenance --test purge-vector-from-recall --query "Vault of the Vanquished" --yes
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
   - **Server Members Intent** (Sandy needs to see who's in the server)
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
│   ├── bot.py              # Discord event loop, message pipeline
│   ├── llm.py              # ollama interface (brain, bouncer, tagger, summarizer, vision)
│   ├── prompt.py           # all system prompts
│   ├── memory.py           # tag + summarize + store pipeline
│   ├── tools.py            # tool definitions and dispatch
│   ├── registry.py         # server/channel/user lookup cache (SQLite)
│   ├── last10.py           # per-channel message cache
│   ├── vector_memory.py    # ChromaDB semantic memory
│   ├── logconf.py          # logging config
│   └── recall/             # long-term message storage
│       ├── database.py     # SQLite + FTS5 operations
│       └── models.py       # Pydantic models
├── data/                   # runtime databases (gitignored)
├── searxng/                # SearXNG docker-compose
├── docs/                   # images, etc.
└── .env                    # configuration (gitignored)
```

## Tools

Sandy has seven tools. The bouncer decides when to use them — the brain never calls tools directly.

| Tool | What it does |
|------|-------------|
| `recall_recent` | Fetch recent messages from the archive |
| `recall_from_user` | Fetch messages from a specific person |
| `recall_by_topic` | Fetch messages by topic tag |
| `search_memories` | Full-text search across all archived messages |
| `search_web` | Search the internet via SearXNG |
| `steam_browse` | Browse Steam top sellers, specials, upcoming, and new releases |
| `get_current_time` | Current date and time |

## License

GPLv3. See [LICENSE](LICENSE).
