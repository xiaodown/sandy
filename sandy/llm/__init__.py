"""
Interface to a local ollama server.

Wraps the four LLM roles Sandy uses:

    Brain      — main personality model; responds to users in Discord
    Bouncer    — decision engine; decides if Sandy should respond and
                 recommends tool calls when additional context would help
    Tagger     — small model; generates 1-3 recall tags for a message
    Summarizer — small model; optionally summarises long messages before recall

Structured outputs (bouncer / tagger / summarizer) use ollama's format= parameter
with a Pydantic schema, which is far more reliable than asking the model to emit
valid JSON by itself.

The brain model does not do tool calling — tool selection is handled by the
bouncer, and the caller (discord_handler) executes the tool and injects results
into the brain's context before asking it to respond.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import ollama

from ..prompt import SandyPrompt
from ..logconf import emit_forensic_record, get_logger
from ..trace import TurnTrace, forensic_payload
from .models import (
    BouncerResponse,
    BrainResponse,
    SummarizerResponse,
    TaggerResponse,
)
from .coercion import (
    _coerce_bouncer_tool_selection,
    _extract_history_messages,
    _infer_steam_browse_category,
    _looks_like_direct_image_ask,
)

if TYPE_CHECKING:
    from ..config import LlmConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Fallback config built from env vars — used ONLY when no LlmConfig is passed
# to OllamaInterface (i.e. never in production, but keeps tests that
# monkeypatch env vars working during the migration).
# ---------------------------------------------------------------------------

def _default_llm_config() -> "LlmConfig":
    """Build an LlmConfig via the central SandyConfig loader."""
    from ..config import SandyConfig

    return SandyConfig.from_env().llm


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

    def __init__(self, config: "LlmConfig | None" = None) -> None:
        from ..config import LlmConfig
        self._cfg: LlmConfig = config if config is not None else _default_llm_config()
        self._client = ollama.AsyncClient()
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
    # Warmer
    # -----------------------------------------------------------------

    async def warm_model(self, model_name: str) -> bool:
        """Send a minimal generate request so ollama loads the model."""
        try:
            async with self._lock:
                await self._client.generate(
                    model=model_name,
                    prompt="",
                    keep_alive=self._cfg.keep_alive,
                    options={
                        "num_ctx": self._cfg.effective_prewarm_num_ctx,
                        "num_predict": 0,
                    },
                )
            return True
        except Exception as e:
            logger.error("Error occured when warming %s: %s", model_name, e)
            return False

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    async def _ask_vision(
        self,
        image_bytes: bytes,
        *,
        prompt,
        model: str,
        num_ctx: int,
        num_predict: int,
        temperature: float | None = None,
    ) -> str | None:
        try:
            async with self._lock:
                options = {
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                }
                if temperature is not None:
                    options["temperature"] = temperature
                response = await self._client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user",   "content": prompt.user, "images": [image_bytes]},
                    ],
                    think=False,
                    keep_alive=self._cfg.keep_alive,
                    options=options,
                )
            desc = (response.message.content or "").strip()
            logger.debug("Vision → %d chars", len(desc))
            return desc or None
        except Exception as exc:
            logger.error("Vision error: %s", exc)
            return None

    async def ask_vision_router(self, image_bytes: bytes) -> str | None:
        """Generate a very short routing caption for the bouncer path."""
        prompt = SandyPrompt.vision_router_prompt()
        model = self._cfg.vision_router_model or self._cfg.vision_model or self._cfg.brain_model
        return await self._ask_vision(
            image_bytes,
            prompt=prompt,
            model=model,
            num_ctx=self._cfg.vision_router_num_ctx,
            num_predict=self._cfg.vision_router_num_predict,
            temperature=self._cfg.vision_router_temperature,
        )

    async def ask_vision(self, image_bytes: bytes) -> str | None:
        """Generate a richer factual description of an image for brain grounding."""
        prompt = SandyPrompt.vision_detail_prompt()
        model = self._cfg.vision_model or self._cfg.brain_model
        return await self._ask_vision(
            image_bytes,
            prompt=prompt,
            model=model,
            num_ctx=self._cfg.vision_num_ctx,
            num_predict=self._cfg.vision_num_predict,
            temperature=self._cfg.vision_temperature,
        )

    # ------------------------------------------------------------------
    # Bouncer
    # ------------------------------------------------------------------

    async def ask_bouncer(
        self,
        context: str,
        *,
        trace: TurnTrace | None = None,
    ) -> BouncerResponse:
        """Decide whether Sandy should respond, and optionally which tool to use."""
        prompt = SandyPrompt.bouncer_prompt(context)
        try:
            async with self._lock:
                response = await self._client.chat(
                    model=self._cfg.bouncer_model,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user",   "content": prompt.user},
                    ],
                    format=BouncerResponse.model_json_schema(),
                    keep_alive=self._cfg.keep_alive,
                    options={
                        "temperature": self._cfg.bouncer_temperature,
                        "num_ctx": self._cfg.bouncer_num_ctx,
                    },
                )
            raw_response = response.message.content or ""
            result = BouncerResponse.model_validate_json(raw_response)
            result = _coerce_bouncer_tool_selection(context, result)
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
                        model=self._cfg.bouncer_model,
                        prompt_system=prompt.system,
                        prompt_user=prompt.user,
                        options={
                            "temperature": self._cfg.bouncer_temperature,
                            "num_ctx": self._cfg.bouncer_num_ctx,
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
    # Tagger
    # ------------------------------------------------------------------

    async def ask_tagger(self, content: str) -> list[str]:
        """Generate 1-3 lowercase tags for a message."""
        prompt = SandyPrompt.tagger_prompt(content)
        try:
            async with self._lock:
                response = await self._client.chat(
                    model=self._cfg.tagger_model,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user",   "content": prompt.user},
                    ],
                    format=TaggerResponse.model_json_schema(),
                    keep_alive=self._cfg.keep_alive,
                    options={
                        "temperature": self._cfg.tagger_temperature,
                        "num_ctx": self._cfg.tagger_num_ctx,
                    },
                )
            result = TaggerResponse.model_validate_json(response.message.content)
            logger.debug("Tagger → tags=%r", result.tags)
            return result.tags
        except Exception as exc:
            logger.error("Tagger error (returning empty tags): %s", exc)
            return []

    # ------------------------------------------------------------------
    # Summarizer
    # ------------------------------------------------------------------

    async def ask_summarizer(self, content: str) -> str | None:
        """Summarise a (long) message in one or two sentences."""
        prompt = SandyPrompt.summarize_prompt(content)
        try:
            async with self._lock:
                response = await self._client.chat(
                    model=self._cfg.summarizer_model,
                    messages=[
                        {"role": "system", "content": prompt.system},
                        {"role": "user",   "content": prompt.user},
                    ],
                    format=SummarizerResponse.model_json_schema(),
                    keep_alive=self._cfg.keep_alive,
                    options={
                        "temperature": self._cfg.summarizer_temperature,
                        "num_ctx": self._cfg.summarizer_num_ctx,
                    },
                )
            result = SummarizerResponse.model_validate_json(response.message.content)
            logger.debug("Summarizer → summary=%r", result.summary)
            return result.summary
        except Exception as exc:
            logger.error("Summarizer error (returning None): %s", exc)
            return None

    # ------------------------------------------------------------------
    # Brain
    # ------------------------------------------------------------------

    async def ask_brain(
        self,
        messages: list[dict],
        server_name: str = "the server",
        channel_name: str = "general",
        rag_context: str | None = None,
        tool_context: str | None = None,
        mode: str = "text",
        participant_names: list[str] | None = None,
        trace: TurnTrace | None = None,
    ) -> BrainResponse | None:
        """Generate a response from the Brain model."""
        if mode == "voice":
            prompt = SandyPrompt.voice_brain_prompt(
                server_name=server_name,
                channel_name=channel_name,
                participant_names=participant_names,
            )
            num_predict = self._cfg.voice_brain_num_predict
        else:
            prompt = SandyPrompt.brain_prompt(
                server_name=server_name,
                channel_name=channel_name,
            )
            num_predict = self._cfg.brain_num_predict
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
                    model=self._cfg.brain_model,
                    messages=full_messages,
                    keep_alive=self._cfg.keep_alive,
                    options={
                        "temperature": self._cfg.brain_temperature,
                        "num_predict": num_predict,
                        "num_ctx":     self._cfg.brain_num_ctx,
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
                        model=self._cfg.brain_model,
                        prompt_system=prompt.system,
                        prompt_user=prompt.user,
                        system_content=system_content,
                        conversation_messages=messages,
                        rag_context=rag_context,
                        tool_context=tool_context,
                        mode=mode,
                        participant_names=participant_names,
                        options={
                            "temperature": self._cfg.brain_temperature,
                            "num_predict": num_predict,
                            "num_ctx": self._cfg.brain_num_ctx,
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
