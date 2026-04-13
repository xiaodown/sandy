from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING

import discord

from ..logconf import get_logger
from .history import VoiceHistoryEntry
from .models import (
    CompletedVoiceTurn,
    VoiceSession,
    _VOICE_REPLY_MAX_WORDS,
    _sanitize_voice_reply,
    _truncate_words,
)
from .tracing import build_voice_trace, forensic_event, trace_event
from .tts import wav_bytes_to_audio_source

if TYPE_CHECKING:
    from .manager import VoiceManager

logger = get_logger("sandy.voice")


async def respond_to_session(manager: VoiceManager, session_id: str) -> None:
    logger.info("Voice response task entered: session_id=%s", session_id)
    session = manager._session
    if session is None or session.session_id != session_id:
        logger.warning("Voice response task exiting early: missing or replaced session")
        return
    if session.playback_active:
        logger.warning("Voice response task exiting early: playback already active")
        return

    while session.pending_response_needed or session.pending_response_turns:
        session.pending_response_needed = False
        if manager._bot_user is None:
            logger.warning("Voice response task exiting early: bot user not set")
            return
        completed_turns = list(session.pending_response_turns)
        session.pending_response_turns.clear()
        if not completed_turns:
            continue
        session.response_counter += 1
        trace = build_voice_trace(session, completed_turns=completed_turns)
        latest_user_text = completed_turns[-1].text
        combined_text = "\n".join(
            f"{turn.speaker_name}: {turn.text}"
            for turn in completed_turns
        )
        total_audio_ms = int(sum(turn.total_audio_seconds for turn in completed_turns) * 1000)
        total_stt_ms = int(sum(turn.total_stt_elapsed_seconds for turn in completed_turns) * 1000)
        total_fragments = sum(turn.fragment_count for turn in completed_turns)
        manager.runtime_state.update_voice_stage(
            stage="turn_ready",
            status="connected",
            current_trace_id=trace.trace_id,
            last_transcript=combined_text,
            last_error=None,
        )
        trace_event(
            trace,
            "voice_turn_received",
            author_is_bot=False,
            audio_duration_ms=total_audio_ms,
            stt_duration_ms=total_stt_ms,
            fragment_count=total_fragments,
            speaker_count=len({turn.speaker_id for turn in completed_turns}),
            transcript_chars=len(combined_text),
        )
        forensic_event(
            trace,
            "turn_input",
            raw_content=combined_text,
            resolved_content=combined_text,
            completed_turns=[
                {
                    "speaker_id": turn.speaker_id,
                    "speaker_name": turn.speaker_name,
                    "text": turn.text,
                    "fragment_count": turn.fragment_count,
                    "audio_duration_ms": int(turn.total_audio_seconds * 1000),
                    "stt_duration_ms": int(turn.total_stt_elapsed_seconds * 1000),
                    "transcripts": list(turn.transcripts),
                }
                for turn in completed_turns
            ],
            participant_names=list(session.participant_names),
            source="discord_voice",
        )

        history_messages = session.history.to_ollama_messages(manager._bot_user.id)
        retrieval_started = perf_counter()
        manager.runtime_state.update_voice_stage(
            stage="retrieval",
            status="connected",
            current_trace_id=trace.trace_id,
        )
        rag_context = await manager.vector_memory.query(latest_user_text, server_id=session.guild_id) if latest_user_text else ""
        trace_event(
            trace,
            "retrieval_completed",
            duration_ms=int((perf_counter() - retrieval_started) * 1000),
            context_chars=len(rag_context or ""),
        )
        forensic_event(
            trace,
            "retrieval",
            query_text=latest_user_text,
            rag_context=rag_context,
            ollama_history=history_messages,
        )
        logger.info(
            "Voice brain start: latest_user_text=%r participant_count=%s history_messages=%s",
            latest_user_text,
            len(session.participant_names),
            len(history_messages),
        )
        brain_started = perf_counter()
        manager.runtime_state.update_voice_stage(
            stage="brain",
            status="connected",
            current_trace_id=trace.trace_id,
        )
        try:
            brain = await manager.llm.ask_brain(
                history_messages,
                bot_name=manager._bot_user.display_name,
                server_name=session.guild_name,
                channel_name=session.channel_name,
                rag_context=rag_context,
                tool_context=None,
                mode="voice",
                participant_names=session.participant_names,
                trace=trace,
            )
        except Exception:
            trace_event(trace, "brain_completed", status="error")
            trace_event(
                trace,
                "turn_completed",
                status="error",
                duration_ms=int((perf_counter() - trace.started_at) * 1000),
                replied=False,
            )
            manager.runtime_state.update_voice_stage(
                stage="brain",
                status="error",
                current_trace_id=trace.trace_id,
                last_error="Voice brain call failed",
            )
            logger.exception("Voice brain call failed")
            continue
        trace_event(
            trace,
            "brain_completed",
            duration_ms=int((perf_counter() - brain_started) * 1000),
            done_reason=brain.done_reason if brain is not None else None,
            reply_chars=len((brain.content if brain is not None else "") or ""),
        )
        logger.info(
            "Voice brain completed: has_response=%s",
            bool(brain and getattr(brain, "content", "").strip()),
        )
        raw_reply = (brain.content if brain is not None else "").strip()
        reply = _sanitize_voice_reply(raw_reply)
        forensic_event(
            trace,
            "reply_output",
            raw_reply=raw_reply,
            finalized_reply=reply,
            delivery_mode="voice",
            done_reason=brain.done_reason if brain is not None else None,
        )
        if not reply:
            trace_event(
                trace,
                "turn_completed",
                status="empty_reply",
                duration_ms=int((perf_counter() - trace.started_at) * 1000),
                replied=False,
            )
            manager.runtime_state.update_voice_stage(
                stage="brain",
                status="idle",
                current_trace_id=trace.trace_id,
                last_reply="",
            )
            logger.warning("Voice brain returned empty reply")
            continue
        logger.info(
            "Voice reply generated: raw=%r sanitized=%r",
            raw_reply,
            reply,
        )
        manager.runtime_state.update_voice_stage(
            stage="tts",
            status="connected",
            current_trace_id=trace.trace_id,
            last_reply=reply,
            last_error=None,
        )

        session.playback_active = True
        delivered = False
        try:
            tts_started = perf_counter()
            try:
                wav_bytes = await manager._tts.synthesize_bytes(reply)
            except Exception:
                fallback_reply = _sanitize_voice_reply(_truncate_words(reply, max(8, _VOICE_REPLY_MAX_WORDS // 2)))
                if fallback_reply and fallback_reply != reply:
                    logger.warning(
                        "Voice TTS failed for primary reply; retrying shorter fallback: %r",
                        fallback_reply,
                    )
                    reply = fallback_reply
                    wav_bytes = await manager._tts.synthesize_bytes(reply)
                else:
                    raise
            trace_event(
                trace,
                "tts_completed",
                duration_ms=int((perf_counter() - tts_started) * 1000),
                wav_bytes=len(wav_bytes),
                reply_chars=len(reply),
            )
            forensic_event(
                trace,
                "voice_tts",
                request_text=reply,
                wav_bytes=len(wav_bytes),
            )
            source = wav_bytes_to_audio_source(wav_bytes)
            logger.info("Voice playback start: reply=%r wav_bytes=%s", reply, len(wav_bytes))
            playback_started = perf_counter()
            manager.runtime_state.update_voice_stage(
                stage="playback",
                status="connected",
                current_trace_id=trace.trace_id,
                last_reply=reply,
            )
            trace_event(trace, "playback_started", reply_chars=len(reply))
            await manager._play_source(session, source)
            trace_event(
                trace,
                "playback_completed",
                duration_ms=int((perf_counter() - playback_started) * 1000),
                reply_chars=len(reply),
            )
            forensic_event(
                trace,
                "reply_delivery",
                finalized_reply=reply,
                delivery_mode="voice",
                wav_bytes=len(wav_bytes),
            )
            logger.info("Voice playback finished cleanly")
            delivered = True
        except Exception:
            trace_event(trace, "tts_completed", status="error")
            manager.runtime_state.update_voice_stage(
                stage="tts",
                status="error",
                current_trace_id=trace.trace_id,
                last_error="Voice synthesis or playback failed",
            )
            logger.exception("Voice reply synthesis/playback failed in %s/%s", session.guild_name, session.channel_name)
        finally:
            session.playback_active = False

        session.reply_counter += 1
        entry = VoiceHistoryEntry(
            speaker_id=manager._bot_user.id,
            speaker_name=manager._bot_user.display_name,
            text=reply,
            created_at=datetime.now(UTC),
            is_bot=True,
        )
        session.history.add(entry)
        manager._create_task(
            store_voice_memory(
                manager,
                session,
                message_id=f"voice-bot:{session.session_id}:{session.reply_counter}",
                author_name=manager._bot_user.display_name,
                text=reply,
            ),
            name=f"voice-store-bot:{session.session_id}",
        )
        trace_event(
            trace,
            "turn_completed",
            status="ok" if delivered else "error",
            duration_ms=int((perf_counter() - trace.started_at) * 1000),
            replied=delivered,
        )
        manager.runtime_state.update_voice_stage(
            stage="idle_in_channel",
            status="connected" if delivered else "error",
            current_trace_id=trace.trace_id,
            last_reply=reply,
        )


async def play_source(session: VoiceSession, source: discord.AudioSource) -> None:
    voice_client = session.voice_client
    if voice_client is None or not voice_client.is_connected():
        raise RuntimeError("voice client is not connected")

    while voice_client.is_playing():
        await asyncio.sleep(0.05)

    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def _after_playback(error: Exception | None) -> None:
        if done.done():
            return
        if error is not None:
            loop.call_soon_threadsafe(done.set_exception, error)
        else:
            loop.call_soon_threadsafe(done.set_result, None)

    voice_client.play(source, after=_after_playback)
    await done


async def store_voice_memory(
    manager: VoiceManager,
    session: VoiceSession,
    *,
    message_id: str,
    author_name: str,
    text: str,
) -> None:
    try:
        await manager.vector_memory.add_message(
            message_id=message_id,
            content=text,
            author_name=author_name,
            server_id=session.guild_id,
            timestamp=datetime.now(UTC),
        )
    except Exception:
        logger.exception("Voice vector-memory store failed for %s", message_id)


async def warm_voice_models(manager: VoiceManager) -> None:
    with contextlib.suppress(Exception):
        await manager._transcriber.warmup()
    with contextlib.suppress(Exception):
        await manager._tts.warmup()
