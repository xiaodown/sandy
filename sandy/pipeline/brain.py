"""Brain call, reply finalization, and truncation cleanup."""

import re
import time

from ..logconf import get_logger
from ..trace import TurnTrace
from .tracing import trace_event, forensic_event

logger = get_logger("sandy.bot")


def _trim_to_last_boundary(reply: str) -> str:
    sentence_matches = list(re.finditer(r'[.!?]["\')\]]?(?:\s|$)', reply))
    if sentence_matches:
        cut = sentence_matches[-1].end()
        trimmed = reply[:cut].strip()
        if trimmed:
            return trimmed

    split_at = reply.rfind("\n\n")
    if split_at != -1:
        trimmed = reply[:split_at].strip()
        if trimmed:
            return trimmed

    split_at = reply.rfind("\n")
    if split_at != -1:
        trimmed = reply[:split_at].strip()
        if trimmed:
            return trimmed

    return reply.strip()


def _trim_truncated_reply(reply: str) -> str:
    paragraph_matches = list(re.finditer(r"\n\s*\n", reply))
    sentence_matches = list(re.finditer(r'[.!?]["\')\]]?(?:\s|$)', reply))

    if paragraph_matches:
        paragraph_cut = paragraph_matches[-1].start()
        trimmed = reply[:paragraph_cut].strip()
        if trimmed and len(trimmed) >= max(80, len(reply) // 3):
            return trimmed

    if sentence_matches:
        sentence_cut = sentence_matches[-1].end()
        trimmed = reply[:sentence_cut].strip()
        if trimmed and len(trimmed) >= max(80, len(reply) // 3):
            return trimmed

    return _trim_to_last_boundary(reply)


def _looks_truncated(reply: str, done_reason: str | None = None) -> bool:
    text = reply.rstrip()
    if not text:
        return False

    if done_reason == "length":
        return True

    if text[-1] in ".!?\"')]}":
        return False

    if text[-1] in ",:;/-([{":
        return True

    last_word_match = re.search(r"([A-Za-z']+)\s*$", text)
    last_word = last_word_match.group(1).lower() if last_word_match else ""
    if last_word in {
        "a", "an", "and", "are", "as", "at", "but", "for", "from",
        "i", "if", "in", "is", "it", "like", "my", "of", "on", "or",
        "so", "that", "the", "to", "was", "with", "you", "your",
    }:
        return True

    return len(last_word) <= 2


def finalize_reply(reply: str | None, *, done_reason: str | None = None) -> str | None:
    if reply is None:
        return None

    cleaned = reply.strip()
    if not cleaned:
        return None

    if not _looks_truncated(cleaned, done_reason=done_reason):
        return cleaned

    trimmed = _trim_truncated_reply(cleaned)
    if trimmed and trimmed != cleaned:
        logger.warning(
            "Brain reply looked truncated; trimmed from %d to %d chars",
            len(cleaned),
            len(trimmed),
        )
        return trimmed

    return cleaned


async def run_brain(
    llm,
    *,
    ollama_history: list[dict],
    bot_user,
    message,
    rag_context: str,
    tool_context: str | None,
    trace: TurnTrace,
    runtime_state,
) -> str | None:
    """Run brain generation and return the finalized reply (or None)."""
    brain_started = time.perf_counter()
    runtime_state.update_turn_stage(trace, "brain")
    brain_response = await llm.ask_brain(
        ollama_history,
        server_name=message.guild.name,
        channel_name=message.channel.name,
        rag_context=rag_context,
        tool_context=tool_context,
        trace=trace,
    )
    trace_event(
        trace,
        "brain_completed",
        duration_ms=int((time.perf_counter() - brain_started) * 1000),
        done_reason=brain_response.done_reason if brain_response else None,
        reply_chars=len((brain_response.content if brain_response else "") or ""),
    )

    reply = finalize_reply(
        brain_response.content if brain_response else None,
        done_reason=brain_response.done_reason if brain_response else None,
    )
    forensic_event(
        trace,
        "reply_output",
        raw_reply=brain_response.content if brain_response else None,
        finalized_reply=reply,
        done_reason=brain_response.done_reason if brain_response else None,
    )

    if not reply:
        trace_event(trace, "reply_skipped", status="empty_reply")
        logger.warning(
            "Brain returned None for message in %s/%s — not sending",
            message.guild.name,
            message.channel.name,
        )

    return reply
