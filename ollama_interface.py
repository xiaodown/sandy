"""
Interface to a local ollama server.

Wraps the four LLM roles Sandy uses:

    Brain      — main personality model; responds to users in Discord
    Bouncer    — small model; given last10 context, decides if Sandy should respond
    Tagger     — small model; generates 1-3 recall tags for a message
    Summarizer — small model; optionally summarises long messages before recall

Two models are in use:

    Brain + Bouncer — qwen2.5:14b (~9.0 GB, 24 GB VRAM machine)
        Tool-capable model for personality / reasoning / tool-calling and
        bouncer decisions. Same weights serve both roles; ollama keeps one
        copy in VRAM. Replaced gemma3:12b-it-qat (no tool support) and
        the 3B bouncer (insufficient contextual reasoning).
        https://ollama.com/library/qwen2.5

    Tagger + Summarizer — Llama-3.2-3B-Instruct Q8_0 (~3.4 GB)
        Lightweight model for structured outputs only (tags, summaries).
        Q8_0 chosen over Q4_K_M for better boolean/JSON coherence.
        https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF

Model names are read from the root .env:
    BRAIN_MODEL       (large model)
    BOUNCER_MODEL     (small model)
    TAGGER_MODEL      (small model)
    SUMMARIZER_MODEL  (small model)

Structured outputs (bouncer / tagger / summarizer) use ollama's format= parameter
with a Pydantic schema, which is far more reliable than asking the model to emit
valid JSON by itself.
"""

import asyncio
import logging
import os
from typing import Optional

import ollama
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

from prompt import SandyPrompt
import tools as _tools

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model selection (from .env, with sensible defaults)
# ---------------------------------------------------------------------------

_BRAIN_MODEL = os.getenv("BRAIN_MODEL", "qwen2.5:14b")
_BOUNCER_MODEL = os.getenv("BOUNCER_MODEL", "qwen2.5:14b")
_TAGGER_MODEL = os.getenv("TAGGER_MODEL", "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0")
_SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0")

# Brain generation options.
# BRAIN_TEMPERATURE: creativity/randomness. 1.0 = neutral, >1.0 = wilder.
#   Default 1.1 — enough personality without going structurally wobbly.
# BRAIN_NUM_PREDICT: max tokens per response. Discord cap is ~2000 chars ≈ 1500
#   tokens in the worst case, but concise is better. Default 512.
# BRAIN_NUM_CTX: context window size. Ollama defaults to 2048 regardless of
#   model capability — you must set this explicitly to get more. 8192 is safe
#   on 24GB VRAM with this model (~1GB KV overhead) and ample for Discord.
#   Bump to 16384 or 32768 if you want more conversation history fed to brain.
_BRAIN_TEMPERATURE = float(os.getenv("BRAIN_TEMPERATURE", "1.1"))
_BRAIN_NUM_PREDICT = int(os.getenv("BRAIN_NUM_PREDICT", "512"))
_BRAIN_NUM_CTX     = int(os.getenv("BRAIN_NUM_CTX", "8192"))

# How long ollama keeps a model loaded in VRAM after the last request.
# Ollama default is 5 minutes — far too short for a chat bot that may go quiet
# for stretches. Set to "1h", "2h", or -1 to keep loaded indefinitely.
# Particularly important on multi-GPU machines with slow PCIe slots where
# reloading weights across the bus adds noticeable latency.
_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "1h")


# ---------------------------------------------------------------------------
# Structured output schemas (Pydantic → JSON Schema → ollama format=)
# ---------------------------------------------------------------------------

class BouncerResponse(BaseModel):
    """Structured output for the Bouncer role."""
    should_respond: bool
    reason: str  # brief explanation — useful for debugging / logging


class TaggerResponse(BaseModel):
    """Structured output for the Tagger role."""
    tags: list[str]

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, v: list[str]) -> list[str]:
        """Lowercase, strip whitespace, drop empties, cap at 3."""
        cleaned = [t.lower().strip() for t in v if t.strip()]
        return cleaned[:3]


class SummarizerResponse(BaseModel):
    """Structured output for the Summarizer role."""
    summary: str


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

class OllamaInterface:
    """
    Async interface to the local ollama server.

    Create one instance and reuse it for the lifetime of the bot.

        llm = OllamaInterface()
        if not await llm.is_running():
            logger.error("ollama is not available")

        should_respond = await llm.ask_bouncer(history.format())
        tags           = await llm.ask_tagger(message.content)
        summary        = await llm.ask_summarizer(long_message)
        reply          = await llm.ask_brain(conversation_messages)
    """

    def __init__(self) -> None:
        self._client = ollama.AsyncClient()
        # Single lock serialises all ollama calls.
        # High-priority callers (bouncer, brain) always wait for it.
        # Low-priority callers (tagger, summarizer) check locked() first and
        # skip immediately if it is held — they lose their tags/summary for
        # that message but never delay a response.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def is_running(self) -> bool:
        """Return True if the ollama service is reachable."""
        try:
            await self._client.list()
            return True
        except Exception as exc:
            logger.warning("ollama health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Bouncer — should Sandy respond?
    # ------------------------------------------------------------------

    async def ask_bouncer(self, context: str, bot_name: str = "Sandy") -> bool:
        """Ask the bouncer whether Sandy should respond given channel context.

        context  — a formatted ChannelHistory string (from history.format()).
        bot_name — the bot's Discord display name, forwarded to the prompt so
                   the bouncer can recognise Sandy's prior lines in the history.

        Returns True if Sandy should respond, False otherwise.
        On any error, returns False (fail closed — don't respond if unsure).
        """
        prompt = SandyPrompt.bouncer_prompt(context, bot_name=bot_name)
        try:
            async with self._lock:  # high-priority: always wait
                response = await self._client.chat(
                    model=_BOUNCER_MODEL,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user",   "content": prompt.user},
                    ],
                    format=BouncerResponse.model_json_schema(),
                    keep_alive=_KEEP_ALIVE,
                )
            result = BouncerResponse.model_validate_json(response.message.content)
            # If the model voted NO but its reason contains clear YES-signal phrases,
            # the boolean is almost certainly wrong — trust the reason and flip it.
            # This is a known failure mode: the model reasons correctly but then
            # outputs the wrong boolean. Phrase list is intentionally conservative
            # to avoid flipping legitimate NO decisions.
            if not result.should_respond and any(
                phrase in result.reason.lower()
                for phrase in ("named", "mentioned", "addressed", "directed at",
                               "asked sandy", "follow-up", "back-and-forth")
            ):
                logger.warning(
                    "Bouncer incoherence: flipping False → True based on reason — %r",
                    result.reason,
                )
                result = BouncerResponse(should_respond=True, reason=result.reason)
            logger.info(
                "Bouncer → should_respond=%s  reason=%r",
                result.should_respond, result.reason,
            )
            return result.should_respond
        except Exception as exc:
            logger.error("Bouncer error (defaulting to no-respond): %s", exc)
            return False

    # ------------------------------------------------------------------
    # Tagger — generate recall tags
    # ------------------------------------------------------------------

    async def ask_tagger(self, content: str) -> list[str]:
        """Generate 1-3 lowercase tags for a message.

        content — raw text of the Discord message.

        Returns a (possibly empty) list of tag strings.
        On any error, returns [] so the caller can still store the message
        untagged.
        """
        prompt = SandyPrompt.tagger_prompt(content)
        try:
            async with self._lock:
                response = await self._client.chat(
                    model=_TAGGER_MODEL,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user",   "content": prompt.user},
                    ],
                    format=TaggerResponse.model_json_schema(),
                    keep_alive=_KEEP_ALIVE,
                )
            result = TaggerResponse.model_validate_json(response.message.content)
            logger.debug("Tagger → tags=%r", result.tags)
            return result.tags
        except Exception as exc:
            logger.error("Tagger error (returning empty tags): %s", exc)
            return []

    # ------------------------------------------------------------------
    # Summarizer — optional long-message summarization
    # ------------------------------------------------------------------

    async def ask_summarizer(self, content: str) -> Optional[str]:
        """Summarise a (long) message in one or two sentences.

        content — raw text of the Discord message.

        Returns the summary string, or None on error.
        Intended for messages above a length threshold; the caller decides
        whether to invoke this.
        """
        prompt = SandyPrompt.summarize_prompt(content)
        try:
            async with self._lock:
                response = await self._client.chat(
                    model=_SUMMARIZER_MODEL,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user",   "content": prompt.user},
                    ],
                    format=SummarizerResponse.model_json_schema(),
                    keep_alive=_KEEP_ALIVE,
                )
            result = SummarizerResponse.model_validate_json(response.message.content)
            logger.debug("Summarizer → summary=%r", result.summary)
            return result.summary
        except Exception as exc:
            logger.error("Summarizer error (returning None): %s", exc)
            return None

    # ------------------------------------------------------------------
    # Brain — main personality / response generation
    # ------------------------------------------------------------------

    async def ask_brain(
        self,
        messages: list[dict],
        bot_name: str = "Sandy",
        server_name: str = "the server",
        channel_name: str = "general",
        server_id: int = 0,
        tools: Optional[list] = None,
    ) -> Optional[str]:
        """Generate a response from the Brain model.

        messages     — multi-turn history from ChannelHistory.to_ollama_messages().
                       The system prompt is prepended and the grounding user turn
                       is appended automatically.
        bot_name     — the bot's Discord display name.
        server_name  — the Discord server (guild) name.
        channel_name — the channel Sandy is responding in.
        server_id    — the Discord guild ID; injected into all Recall tool calls
                       so Sandy cannot access another server's message history.
        tools        — list of ollama tool schema dicts (from tools.TOOL_SCHEMAS).

        Returns the response text, or None on error.
        """
        prompt = SandyPrompt.brain_prompt(
            bot_name=bot_name,
            server_name=server_name,
            channel_name=channel_name,
        )
        # System prompt first, then conversation history, then the grounding
        # user turn (current time + location) so it's the last thing she sees
        # before generating a response.
        full_messages = (
            [{"role": "system", "content": prompt.system}]
            + messages
            + [{"role": "user", "content": prompt.user}]
        )
        try:
            kwargs: dict = {
                "model": _BRAIN_MODEL,
                "messages": full_messages,
                "keep_alive": _KEEP_ALIVE,
                "options": {
                    "temperature": _BRAIN_TEMPERATURE,
                    "num_predict": _BRAIN_NUM_PREDICT,
                    "num_ctx":     _BRAIN_NUM_CTX,
                },
            }
            if tools:
                kwargs["tools"] = tools

            async with self._lock:  # high-priority: always wait
                try:
                    response = await self._client.chat(**kwargs)
                except ollama.ResponseError as exc:
                    if exc.status_code == 400 and kwargs.get("tools"):
                        # Model doesn't support native tool calling — retry bare.
                        # Sandy can still respond; she just won't have memory access
                        # this turn. Log clearly so we know tools are disabled.
                        logger.warning(
                            "Brain: model does not support tools (400) — "
                            "retrying without tools. Switch BRAIN_MODEL to one "
                            "that supports tool calling (e.g. llama3.1, qwen2.5)."
                        )
                        kwargs.pop("tools")
                        response = await self._client.chat(**kwargs)
                    else:
                        raise

            # Tool call loop.
            # Each iteration: the model said "call this tool"; we call it,
            # append the result as a tool message, and re-ask the model.
            # The lock is released between rounds so the GPU is free while
            # we await the Recall HTTP call; ollama's KV cache means the
            # re-call only processes the new tokens, not the whole history.
            for _round in range(_tools.MAX_TOOL_ROUNDS):
                if not response.message.tool_calls:
                    break

                names = [tc.function.name for tc in response.message.tool_calls]
                logger.info("Brain tool calls (round %d): %s", _round + 1, names)

                # Append the assistant turn that contains the tool call(s).
                # ollama accepts Message objects alongside plain dicts.
                full_messages.append(response.message)

                # Execute each tool and append its result.
                for tc in response.message.tool_calls:
                    result = await _tools.dispatch(
                        tc.function.name,
                        dict(tc.function.arguments),
                        server_id=server_id,
                        server_name=server_name,
                    )
                    full_messages.append({"role": "tool", "content": result})

                # Re-ask the model with tool results appended.
                kwargs["messages"] = full_messages
                async with self._lock:
                    response = await self._client.chat(**kwargs)
            else:
                # for/else: loop exhausted without break — model kept calling
                # tools for MAX_TOOL_ROUNDS straight, which is pathological.
                logger.warning(
                    "Brain hit MAX_TOOL_ROUNDS (%d) without a final answer — "
                    "returning whatever content exists",
                    _tools.MAX_TOOL_ROUNDS,
                )

            return response.message.content
        except Exception as exc:
            logger.error("Brain error: %s", exc)
            return None
