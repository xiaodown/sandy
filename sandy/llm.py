"""
Interface to a local ollama server.

Wraps the four LLM roles Sandy uses:

    Brain      — main personality model; responds to users in Discord
    Bouncer    — decision engine; decides if Sandy should respond and
                 recommends tool calls when additional context would help
    Tagger     — small model; generates 1-3 recall tags for a message
    Summarizer — small model; optionally summarises long messages before recall

All four roles currently use the same model to avoid VRAM thrashing.
Model names are read from the root .env - see the .env for details and notes.

Structured outputs (bouncer / tagger / summarizer) use ollama's format= parameter
with a Pydantic schema, which is far more reliable than asking the model to emit
valid JSON by itself.

The brain model does not do tool calling — tool selection is handled by the
bouncer, and the caller (discord_handler) executes the tool and injects results
into the brain's context before asking it to respond.
"""

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import ollama
from pydantic import BaseModel, field_validator, model_validator
from dotenv import load_dotenv

from .prompt import SandyPrompt
from .logconf import emit_forensic_record, get_logger
from .trace import TurnTrace, forensic_payload

load_dotenv()

logger = get_logger(__name__)

_HISTORY_LINE_RE = re.compile(r"^\[[^\]]+\] \[[^\]]+\] (?P<content>.*)$")
_STEAM_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("specials", ("on sale", "sales", "sale", "discount", "discounts")),
    ("upcoming", ("coming soon", "upcoming")),
    ("new_releases", ("new releases", "new release", "just came out", "fresh release", "fresh releases")),
    ("top_sellers", ("top sellers", "top seller", "best sellers", "best seller", "what's good", "whats good", "what's hot", "whats hot", "selling well")),
)

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
_BOUNCER_NUM_CTX   = int(os.getenv("BOUNCER_NUM_CTX", "8192"))
_TAGGER_NUM_CTX    = int(os.getenv("TAGGER_NUM_CTX", "4096"))
_SUMMARIZER_NUM_CTX = int(os.getenv("SUMMARIZER_NUM_CTX", "4096"))
_VISION_NUM_CTX    = int(os.getenv("VISION_NUM_CTX", "8192"))
_VISION_NUM_PREDICT = int(os.getenv("VISION_NUM_PREDICT", "384"))
_PREWARM_NUM_CTX   = int(os.getenv("PREWARM_NUM_CTX", str(_BOUNCER_NUM_CTX)))

# Vision model — describes image attachments. Defaults to brain model so no extra
# VRAM is consumed. Override with VISION_MODEL in .env if you ever want a dedicated
# vision model (e.g. llava, minicpm-v, etc.).
_VISION_MODEL = os.getenv("VISION_MODEL", None)  # resolved below after _BRAIN_MODEL is set

# How long ollama keeps a model loaded in VRAM after the last request.
# Sandy defaults to 1 hour because measured idle power draw stayed low even
# while the models remained resident, while unloading after 30 minutes caused
# a noticeable cold-start latency penalty on the next message.
# Lower this only if you explicitly want the VRAM back for other workloads.
_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "1h")

# Bouncer, tagger, and summarizer use low temperatures for consistent
# structured output.  The brain temperature is intentionally higher for
# personality — don't conflate the two.
_BOUNCER_TEMPERATURE    = float(os.getenv("BOUNCER_TEMPERATURE", "0.1"))
_TAGGER_TEMPERATURE     = float(os.getenv("TAGGER_TEMPERATURE", "0.1"))
_SUMMARIZER_TEMPERATURE = float(os.getenv("SUMMARIZER_TEMPERATURE", "0.1"))


# ---------------------------------------------------------------------------
# Structured output schemas (Pydantic → JSON Schema → ollama format=)
# ---------------------------------------------------------------------------

class BouncerResponse(BaseModel):
    """Structured output for the Bouncer role.

    Also includes tool recommendation: which tool (if any) to call before
    the brain generates a response.  Tool fields are ignored when
    should_respond is False.
    """
    should_respond: bool
    reason: str
    use_tool: bool = False
    recommended_tool: Optional[str] = None
    tool_parameters: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _tool_fields_consistent(self) -> "BouncerResponse":
        """If the model set use_tool but forgot to name a tool, zero it out.

        This prevents a use_tool=True / recommended_tool=None mismatch from
        propagating — the discord_handler guard would catch it anyway, but
        fixing it here keeps the log line honest too.
        """
        if self.use_tool and not self.recommended_tool:
            logger.warning(
                "Bouncer set use_tool=True but recommended_tool is empty — forcing use_tool=False",
            )
            self.use_tool = False
            self.tool_parameters = None
        return self

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


@dataclass
class BrainResponse:
    """Brain generation result plus completion metadata from Ollama."""

    content: str
    done_reason: str | None = None
    eval_count: int | None = None


def _extract_history_messages(context: str) -> list[str]:
    """Return plain message text from Last10-formatted history lines."""
    messages: list[str] = []
    for line in context.splitlines():
        match = _HISTORY_LINE_RE.match(line.strip())
        if match:
            messages.append(match.group("content").strip())
    return messages


def _infer_steam_browse_category(context: str) -> str | None:
    """Infer the Steam storefront category implied by the latest turn."""
    messages = _extract_history_messages(context)
    if not messages:
        return None
    lowered_messages = [message.lower() for message in messages]
    latest = lowered_messages[-1]
    recent_window = lowered_messages[-4:]

    if not any("steam" in message for message in recent_window):
        return None

    for category, keywords in _STEAM_CATEGORY_KEYWORDS:
        if any(keyword in latest for keyword in keywords):
            return category

    # Follow-up turns like "check actual steam" need the most recent
    # storefront category from nearby history rather than a blind default.
    if "steam" in latest or "check actual" in latest or "check again" in latest:
        for message in reversed(lowered_messages[:-1]):
            for category, keywords in _STEAM_CATEGORY_KEYWORDS:
                if any(keyword in message for keyword in keywords):
                    return category

    if "steam" in latest:
        return "top_sellers"
    return None


def _coerce_bouncer_tool_selection(
    context: str,
    result: "BouncerResponse",
) -> "BouncerResponse":
    """Apply deterministic tool overrides for obvious storefront asks."""
    if not result.should_respond:
        return result
    if result.use_tool and result.recommended_tool not in {None, "search_web", "steam_browse"}:
        return result

    category = _infer_steam_browse_category(context)
    if category is None:
        return result

    if result.recommended_tool == "steam_browse":
        if result.tool_parameters is None:
            result.tool_parameters = {"category": category, "limit": 5}
        else:
            result.tool_parameters.setdefault("category", category)
            result.tool_parameters.setdefault("limit", 5)
        return result

    limit = 5
    if isinstance(result.tool_parameters, dict):
        raw_limit = result.tool_parameters.get("limit", result.tool_parameters.get("n_results"))
        if isinstance(raw_limit, int):
            limit = max(1, min(raw_limit, 10))

    logger.info(
        "Bouncer storefront override: forcing steam_browse(%s) instead of %s",
        category,
        result.recommended_tool or "none",
    )
    result.use_tool = True
    result.recommended_tool = "steam_browse"
    result.tool_parameters = {"category": category, "limit": limit}
    return result


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
        # Single lock serialises all ollama calls so only one model
        # inference runs at a time.  All callers (bouncer, brain, tagger,
        # summarizer, vision) acquire and wait.  In practice tagger and
        # summarizer run after the reply is sent, so they queue behind
        # the brain without delaying the user.
        self._lock = asyncio.Lock()

    def is_busy(self) -> bool:
        """Return whether an Ollama request currently holds the shared lock."""
        return self._lock.locked()

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

    # -----------------------------------------------------------------
    # Warmer - force ollama to load a model into vram before Discord connects
    # -----------------------------------------------------------------

    async def warm_model(self, model_name: str) -> bool:
        """
        Send a minimal generate request so ollama loads the model with a
        predictable keepalive and context size.
        """
        try:
            async with self._lock:
                await self._client.generate(
                    model=model_name,
                    prompt="",
                    keep_alive=_KEEP_ALIVE,
                    options={
                        "num_ctx": _PREWARM_NUM_CTX,
                        "num_predict": 0,
                    },
                )
            return True
        except Exception as e:
            logger.error("Error occured when warming %s: %s",
                         model_name, e)
            return False

    # ------------------------------------------------------------------
    # Vision — describe image attachments
    # ------------------------------------------------------------------

    async def ask_vision(self, image_bytes: bytes) -> Optional[str]:
        """Generate a plain factual description of an image.

        Uses VISION_MODEL (defaults to BRAIN_MODEL if not set in .env).
        System prompt is deliberately sterile — no Sandy personality leaks
        into descriptions. This is raw context, not Sandy's voice.

        image_bytes — raw bytes of the image (JPEG, PNG, GIF, WebP).

        Returns the description string, or None on error.
        """
        prompt = SandyPrompt.vision_prompt()
        # Resolve at call time in case _BRAIN_MODEL was overridden via env.
        model = _VISION_MODEL or _BRAIN_MODEL
        try:
            async with self._lock:
                response = await self._client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user",   "content": prompt.user, "images": [image_bytes]},
                    ],
                    keep_alive=_KEEP_ALIVE,
                    options={
                        "num_ctx": _VISION_NUM_CTX,
                        "num_predict": _VISION_NUM_PREDICT,
                    },
                )
            desc = (response.message.content or "").strip()
            logger.debug("Vision → %d chars", len(desc))
            return desc or None
        except Exception as exc:
            logger.error("Vision error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Bouncer — should Sandy respond?
    # ------------------------------------------------------------------

    async def ask_bouncer(
        self,
        context: str,
        bot_name: str = "Sandy",
        *,
        trace: TurnTrace | None = None,
    ) -> BouncerResponse:
        """Decide whether Sandy should respond, and optionally which tool to use.

        context  — a formatted ChannelHistory string (from history.format()).
        bot_name — the bot's Discord display name, forwarded to the prompt so
                   the bouncer can recognise Sandy's prior lines in the history.

        Returns a BouncerResponse with respond/tool decisions.
        On any error, returns a no-respond response (fail closed).
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
                    options={
                        "temperature": _BOUNCER_TEMPERATURE,
                        "num_ctx": _BOUNCER_NUM_CTX,
                    },
                )
            raw_response = response.message.content or ""
            result = BouncerResponse.model_validate_json(raw_response)
            result = _coerce_bouncer_tool_selection(context, result)
            # If she's not responding, tool fields are meaningless — zero them.
            if not result.should_respond:
                result.use_tool = False
                result.recommended_tool = None
                result.tool_parameters = None
            logger.debug(
                "Bouncer → respond=%s  tool=%s(%s)  reason=%r",
                result.should_respond,
                result.recommended_tool or "none",
                "yes" if result.use_tool else "no",
                result.reason,
            )
            if trace is not None:
                emit_forensic_record(
                    logger,
                    "FORENSIC bouncer_decision",
                    forensic_payload(
                        trace,
                        "bouncer_decision",
                        model=_BOUNCER_MODEL,
                        prompt_system=prompt.system,
                        prompt_user=prompt.user,
                        options={
                            "temperature": _BOUNCER_TEMPERATURE,
                            "num_ctx": _BOUNCER_NUM_CTX,
                        },
                        parsed_result=result.model_dump(),
                        raw_response=raw_response,
                    ),
                )
            return result
        except Exception as exc:
            logger.error("Bouncer error (defaulting to no-respond): %s", exc)
            return BouncerResponse(
                should_respond=False,
                reason=f"error: {exc}",
                use_tool=False,
            )



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
                    options={
                        "temperature": _TAGGER_TEMPERATURE,
                        "num_ctx": _TAGGER_NUM_CTX,
                    },
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
                    options={
                        "temperature": _SUMMARIZER_TEMPERATURE,
                        "num_ctx": _SUMMARIZER_NUM_CTX,
                    },
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
        rag_context: Optional[str] = None,
        tool_context: Optional[str] = None,
        trace: TurnTrace | None = None,
    ) -> Optional[BrainResponse]:
        """Generate a response from the Brain model.

        messages     — multi-turn history from ChannelHistory.to_ollama_messages().
                       The system prompt is prepended automatically.
        bot_name     — the bot's Discord display name.
        server_name  — the Discord server (guild) name.
        channel_name — the channel Sandy is responding in.
        rag_context  — optional pre-formatted block of semantically similar past
                       messages from VectorMemory.query().  Injected into the
                       system prompt as background awareness.
        tool_context — optional pre-formatted block of tool results (memory recall,
                       web search, etc.) retrieved by the bouncer's recommendation.
                       Injected into the system prompt so Sandy can reference the
                       information naturally in her response.

        Returns the response text plus completion metadata, or None on error.
        """
        prompt = SandyPrompt.brain_prompt(
            bot_name=bot_name,
            server_name=server_name,
            channel_name=channel_name,
        )
        # System prompt (with grounding appended) followed by conversation history.
        # The grounding line (current time + server/channel) is merged into the
        # system prompt rather than appended as a trailing user turn. Appending it
        # as a user turn creates two consecutive user-role messages whenever the
        # conversation history already ends with a user turn (i.e. always), which
        # causes models that expect strict user/assistant alternation to enter
        # "completion mode" and echo the entire context back in their reply.
        system_content = f"{prompt.system}\n\n{prompt.user}"
        if rag_context:
            system_content += (
                "\n\n## Fragments from your memory that may be relevant\n"
                + rag_context
            )
        if tool_context:
            system_content += "\n\n" + tool_context
        full_messages = (
            [{"role": "system", "content": system_content}]
            + messages
        )
        try:
            async with self._lock:
                response = await self._client.chat(
                    model=_BRAIN_MODEL,
                    messages=full_messages,
                    keep_alive=_KEEP_ALIVE,
                    options={
                        "temperature": _BRAIN_TEMPERATURE,
                        "num_predict": _BRAIN_NUM_PREDICT,
                        "num_ctx":     _BRAIN_NUM_CTX,
                    },
                )
            brain_response = BrainResponse(
                content=response.message.content or "",
                done_reason=response.done_reason,
                eval_count=response.eval_count,
            )
            if trace is not None:
                emit_forensic_record(
                    logger,
                    "FORENSIC brain_generation",
                    forensic_payload(
                        trace,
                        "brain_generation",
                        model=_BRAIN_MODEL,
                        prompt_system=prompt.system,
                        prompt_user=prompt.user,
                        system_content=system_content,
                        conversation_messages=messages,
                        rag_context=rag_context,
                        tool_context=tool_context,
                        options={
                            "temperature": _BRAIN_TEMPERATURE,
                            "num_predict": _BRAIN_NUM_PREDICT,
                            "num_ctx": _BRAIN_NUM_CTX,
                        },
                        raw_response=brain_response.content,
                        done_reason=brain_response.done_reason,
                        eval_count=brain_response.eval_count,
                    ),
                )
            return brain_response
        except Exception as exc:
            logger.error("Brain error: %s", exc)
            return None
