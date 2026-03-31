from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
import os
from pathlib import Path
import re
from time import time
from uuid import uuid4

import discord

from ..logconf import get_logger
from ..paths import resolve_runtime_path
from ..runtime_state import RuntimeState
from .capture import CaptureJob, UtteranceCaptureSink
from .history import VoiceHistory, VoiceHistoryEntry
from .stt import FasterWhisperTranscriber
from .tts import TtsServiceClient, TtsServiceConfig, wav_bytes_to_audio_source

try:
    from discord.ext import voice_recv
except Exception:  # pragma: no cover - import failure depends on environment
    voice_recv = None

logger = get_logger("sandy.voice")

_VOICE_CAPTURE_DIR = resolve_runtime_path(os.getenv("VOICE_CAPTURE_DIR", "data/prod/voice_captures"))
_VOICE_PREROLL_MS = int(os.getenv("VOICE_PREROLL_MS", "250"))
_VOICE_STITCH_GAP_SECONDS = float(os.getenv("VOICE_STITCH_GAP_SECONDS", "1.0"))
_VOICE_STITCH_RELEASE_SECONDS = float(os.getenv("VOICE_STITCH_RELEASE_SECONDS", "1.35"))
_VOICE_HISTORY_MAXLEN = int(os.getenv("VOICE_HISTORY_MAXLEN", "12"))
_VOICE_IDLE_AUTO_LEAVE_SECONDS = int(os.getenv("VOICE_IDLE_AUTO_LEAVE_SECONDS", "300"))
_VOICE_FORCE_RELEASE_SECONDS = float(os.getenv("VOICE_FORCE_RELEASE_SECONDS", "3.25"))
_VOICE_REPLY_MAX_WORDS = int(os.getenv("VOICE_REPLY_MAX_WORDS", "32"))
_VOICE_REPLY_MAX_CHARS = int(os.getenv("VOICE_REPLY_MAX_CHARS", "220"))
_VOICE_REPLY_MAX_SENTENCES = int(os.getenv("VOICE_REPLY_MAX_SENTENCES", "2"))


def _normalize_name(value: str) -> str:
    return " ".join(value.lower().split())


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:-")


def _truncate_sentences(text: str, max_sentences: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(parts) <= max_sentences:
        return text
    return " ".join(parts[:max_sentences]).strip()


def _sanitize_voice_reply(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    cleaned = _truncate_sentences(cleaned, _VOICE_REPLY_MAX_SENTENCES)
    cleaned = _truncate_words(cleaned, _VOICE_REPLY_MAX_WORDS)
    if len(cleaned) > _VOICE_REPLY_MAX_CHARS:
        cleaned = cleaned[:_VOICE_REPLY_MAX_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def resolve_target_channel(
    guild: discord.Guild,
    *,
    query: str,
    author_voice_channel: discord.VoiceChannel | None,
) -> discord.VoiceChannel | None:
    cleaned_query = query.strip()
    if not cleaned_query:
        return author_voice_channel

    normalized_query = _normalize_name(cleaned_query)
    channels = list(guild.voice_channels)

    for channel in channels:
        if _normalize_name(channel.name) == normalized_query:
            return channel

    partial_matches = [
        channel for channel in channels
        if normalized_query in _normalize_name(channel.name)
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    return None


@dataclass(slots=True)
class PendingSpeakerTurn:
    speaker_id: int
    speaker_name: str
    text: str
    started_at: float
    ended_at: float
    release_task: asyncio.Task | None = None
    force_release_task: asyncio.Task | None = None


@dataclass(slots=True)
class VoiceSession:
    session_id: str
    guild_id: int
    guild_name: str
    channel_id: int
    channel_name: str
    requested_by_user_id: int
    requested_by_name: str
    participant_names: list[str]
    started_at: float
    voice_client: discord.VoiceProtocol | None = None
    history: VoiceHistory = field(default_factory=lambda: VoiceHistory(maxlen=_VOICE_HISTORY_MAXLEN))
    pending_by_speaker: dict[int, PendingSpeakerTurn] = field(default_factory=dict)
    pending_stt_counts: dict[int, int] = field(default_factory=dict)
    active_speakers: set[int] = field(default_factory=set)
    response_task: asyncio.Task | None = None
    playback_active: bool = False
    pending_response_needed: bool = False
    response_counter: int = 0
    reply_counter: int = 0
    last_activity_at: float = field(default_factory=time)
    idle_task: asyncio.Task | None = None
    sink: object | None = None


@dataclass(frozen=True, slots=True)
class VoiceCommandResult:
    handled: bool
    reply: str | None = None
    ok: bool = True


class VoiceManager:
    """Own Sandy's single active voice session and command handling."""

    _STOP = object()

    def __init__(
        self,
        *,
        registry,
        runtime_state: RuntimeState,
        llm,
        vector_memory,
        background_tasks=None,
    ) -> None:
        self.registry = registry
        self.runtime_state = runtime_state
        self.llm = llm
        self.vector_memory = vector_memory
        self.background_tasks = background_tasks
        self._session: VoiceSession | None = None
        self._bot_user: discord.ClientUser | discord.User | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stt_queue: asyncio.Queue[CaptureJob | object] = asyncio.Queue()
        self._stt_worker_task: asyncio.Task | None = None
        self._transcriber = FasterWhisperTranscriber(
            model_name=os.getenv("VOICE_STT_MODEL", "base.en"),
            device=os.getenv("VOICE_STT_DEVICE", "cuda"),
            compute_type=os.getenv("VOICE_STT_COMPUTE_TYPE", "float16"),
            language=os.getenv("VOICE_STT_LANGUAGE", "en").strip() or None,
        )
        self._tts = TtsServiceClient(
            TtsServiceConfig(
                base_url=os.getenv("VOICE_TTS_SERVICE_URL", "http://127.0.0.1:8777"),
                timeout_seconds=float(os.getenv("VOICE_TTS_SERVICE_TIMEOUT_SECONDS", "180")),
                default_instruct=(os.getenv("VOICE_TTS_INSTRUCT") or None),
                default_language=(os.getenv("VOICE_TTS_LANGUAGE") or "English"),
            ),
        )

    @property
    def active_session(self) -> VoiceSession | None:
        return self._session

    def is_active(self) -> bool:
        return self._session is not None

    def text_replies_paused(self) -> bool:
        return self.is_active()

    def is_voice_admin(self, member: discord.abc.User, *, guild_id: int) -> bool:
        return self.registry.is_voice_admin(user_id=member.id, server_id=guild_id)

    async def on_ready(self, bot: discord.Client) -> None:
        self._loop = asyncio.get_running_loop()
        self._bot_user = bot.user
        if self._stt_worker_task is None or self._stt_worker_task.done():
            self._stt_worker_task = self._create_task(
                self._run_stt_worker(),
                name="voice-stt-worker",
            )

    def _create_task(self, awaitable, *, name: str) -> asyncio.Task:
        if self.background_tasks is not None:
            task = self.background_tasks.create_task(awaitable, name=name)
        else:
            task = asyncio.create_task(awaitable, name=name)
        if task is not None:
            task.add_done_callback(self._log_task_completion)
        return task

    def _log_task_completion(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("Voice task cancelled: %s", task.get_name())
        except Exception:
            logger.exception("Voice task failed: %s", task.get_name())
        else:
            logger.info("Voice task completed: %s", task.get_name())

    async def handle_text_command(
        self,
        message: discord.Message,
        *,
        bot_user: discord.ClientUser | discord.User | None,
    ) -> VoiceCommandResult:
        self._bot_user = bot_user
        content = (message.content or "").strip()
        if not content.startswith("!"):
            return VoiceCommandResult(handled=False)

        command, _, remainder = content[1:].partition(" ")
        command = command.strip().lower()
        if command == "join":
            return await self.join_from_message(message, query=remainder.strip(), bot_user=bot_user)
        if command == "leave":
            return await self.leave_from_message(message)
        return VoiceCommandResult(handled=False)

    async def join_from_message(
        self,
        message: discord.Message,
        *,
        query: str,
        bot_user: discord.ClientUser | discord.User | None,
    ) -> VoiceCommandResult:
        if message.guild is None:
            return VoiceCommandResult(handled=True, reply="voice commands only work in servers", ok=False)

        if not self.is_voice_admin(message.author, guild_id=message.guild.id):
            logger.warning(
                "Denied voice join command from %s in %s",
                message.author.display_name,
                message.guild.name,
            )
            return VoiceCommandResult(handled=True, reply="you're not allowed to control voice", ok=False)

        if self._session is not None:
            return VoiceCommandResult(handled=True, reply="already in a voice chat. use !leave first", ok=False)

        author_voice = getattr(getattr(message.author, "voice", None), "channel", None)
        target = resolve_target_channel(message.guild, query=query, author_voice_channel=author_voice)
        if target is None:
            return VoiceCommandResult(
                handled=True,
                reply="couldn't resolve a voice channel. join one or name it exactly",
                ok=False,
            )

        try:
            if voice_recv is not None:
                voice_client = await target.connect(cls=voice_recv.VoiceRecvClient)
            else:  # pragma: no cover
                voice_client = await target.connect()
        except Exception as exc:
            logger.exception("Voice join failed for %s/%s", message.guild.name, target.name)
            return VoiceCommandResult(handled=True, reply=f"voice join failed: {exc}", ok=False)

        participant_names = self._participant_names(target, bot_user=bot_user)
        self._session = VoiceSession(
            session_id=uuid4().hex,
            guild_id=message.guild.id,
            guild_name=message.guild.name,
            channel_id=target.id,
            channel_name=target.name,
            requested_by_user_id=message.author.id,
            requested_by_name=message.author.display_name,
            participant_names=participant_names,
            started_at=time(),
            voice_client=voice_client,
        )
        self._attach_receive_probe(voice_client)
        self._arm_idle_timer(self._session)
        self._sync_runtime_state(status="connected")
        asyncio.create_task(self._warm_voice_models(), name="voice-warmup")
        logger.info(
            "Voice session started: guild=%s channel=%s requested_by=%s",
            message.guild.name,
            target.name,
            message.author.display_name,
        )
        return VoiceCommandResult(handled=True, reply=f"joined voice: {target.name}")

    async def leave_from_message(self, message: discord.Message) -> VoiceCommandResult:
        if message.guild is None:
            return VoiceCommandResult(handled=True, reply="voice commands only work in servers", ok=False)
        if not self.is_voice_admin(message.author, guild_id=message.guild.id):
            logger.warning(
                "Denied voice leave command from %s in %s",
                message.author.display_name,
                message.guild.name,
            )
            return VoiceCommandResult(handled=True, reply="you're not allowed to control voice", ok=False)
        if self._session is None:
            return VoiceCommandResult(handled=True, reply="not in a voice chat", ok=False)

        channel_name = self._session.channel_name
        await self._disconnect_active_voice_client()
        logger.info("Voice session ended: guild=%s channel=%s", message.guild.name, channel_name)
        return VoiceCommandResult(handled=True, reply=f"left voice: {channel_name}")

    async def shutdown(self) -> None:
        await self._disconnect_active_voice_client()
        if self._stt_worker_task is not None and not self._stt_worker_task.done():
            await self._stt_queue.put(self._STOP)

    def handle_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
        *,
        bot_user: discord.ClientUser | discord.User | None,
    ) -> None:
        self._bot_user = bot_user
        session = self._session
        if session is None:
            return
        if member.guild.id != session.guild_id:
            return

        before_id = before.channel.id if before.channel is not None else None
        after_id = after.channel.id if after.channel is not None else None
        if session.channel_id not in {before_id, after_id}:
            return

        if member.id == getattr(bot_user, "id", None) and after.channel is None:
            asyncio.create_task(self._disconnect_active_voice_client(), name="voice-disconnect-sync")
            return

        channel = after.channel or before.channel
        if channel is None:
            return
        session.participant_names = self._participant_names(channel, bot_user=bot_user)
        session.last_activity_at = time()
        self._arm_idle_timer(session)
        self._sync_runtime_state(status="connected")

    def _attach_receive_probe(self, voice_client: discord.VoiceProtocol) -> None:
        if voice_recv is None or not isinstance(voice_client, voice_recv.VoiceRecvClient):
            logger.warning("voice_recv unavailable; inbound voice capture is disabled")
            return
        if voice_client.is_listening():
            return

        sink = UtteranceCaptureSink(
            _VOICE_CAPTURE_DIR,
            preroll_ms=_VOICE_PREROLL_MS,
            on_capture_saved=self._enqueue_capture_job_from_thread,
            on_speaker_started=self._speaker_started,
            on_speaker_stopped=self._speaker_stopped,
        )
        voice_client.listen(sink, after=lambda exc: logger.info("Capture sink stopped: error=%r", exc))
        if self._session is not None:
            self._session.sink = sink

    def _enqueue_capture_job_from_thread(self, job: CaptureJob) -> None:
        session = self._session
        if session is None:
            return
        speaker_id = job.speaker_id
        if speaker_id is not None:
            session.pending_stt_counts[speaker_id] = session.pending_stt_counts.get(speaker_id, 0) + 1
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._stt_queue.put_nowait, job)

    def _speaker_started(self, speaker_id: int | None, _speaker_name: str) -> None:
        if speaker_id is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._speaker_started_on_loop, speaker_id)

    def _speaker_stopped(self, speaker_id: int | None, _speaker_name: str, _reason: str) -> None:
        if speaker_id is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._speaker_stopped_on_loop, speaker_id)

    def _speaker_started_on_loop(self, speaker_id: int) -> None:
        session = self._session
        if session is None:
            return
        session.active_speakers.add(speaker_id)
        session.last_activity_at = time()
        self._arm_idle_timer(session)
        logger.info("Voice speaker start: speaker_id=%s", speaker_id)

    def _speaker_stopped_on_loop(self, speaker_id: int) -> None:
        session = self._session
        if session is None:
            return
        session.active_speakers.discard(speaker_id)
        logger.info("Voice speaker stop: speaker_id=%s", speaker_id)
        self._maybe_schedule_release(session, speaker_id)

    async def _run_stt_worker(self) -> None:
        while True:
            job = await self._stt_queue.get()
            try:
                if job is self._STOP:
                    return
                assert isinstance(job, CaptureJob)
                result = await self._transcriber.transcribe_file(job.path)
                logger.info(
                    "Voice transcript completed: speaker=%s text=%r duration=%.2fs",
                    job.speaker_label,
                    result.text.strip(),
                    job.duration_seconds,
                )
                await self._handle_transcript(job, result.text.strip())
            except Exception:
                logger.exception("STT failed for capture=%s", getattr(job, "path", "?"))
            finally:
                self._stt_queue.task_done()

    async def _handle_transcript(self, job: CaptureJob, text: str) -> None:
        session = self._session
        if session is None or session.guild_id != job.guild_id:
            return
        speaker_id = job.speaker_id
        if speaker_id is not None:
            session.pending_stt_counts[speaker_id] = max(session.pending_stt_counts.get(speaker_id, 1) - 1, 0)
        if not text:
            if speaker_id is not None:
                self._maybe_schedule_release(session, speaker_id)
            return

        session.last_activity_at = time()
        self._arm_idle_timer(session)
        if speaker_id is None:
            # No stable speaker id; treat as immediate one-off turn.
            await self._emit_completed_turn(
                session,
                speaker_id=0,
                speaker_name=job.speaker_label,
                text=text,
            )
            return

        pending = session.pending_by_speaker.get(speaker_id)
        if pending is not None and (job.started_at - pending.ended_at) <= _VOICE_STITCH_GAP_SECONDS:
            pending.text = f"{pending.text} {text}".strip()
            pending.ended_at = job.ended_at
            if pending.release_task is not None:
                pending.release_task.cancel()
            if pending.force_release_task is not None:
                pending.force_release_task.cancel()
        else:
            if pending is not None:
                await self._release_pending_turn(session, speaker_id)
            pending = PendingSpeakerTurn(
                speaker_id=speaker_id,
                speaker_name=job.speaker_label,
                text=text,
                started_at=job.started_at,
                ended_at=job.ended_at,
            )
            session.pending_by_speaker[speaker_id] = pending

        self._maybe_schedule_release(session, speaker_id)
        self._arm_force_release(session, speaker_id)

    def _maybe_schedule_release(self, session: VoiceSession, speaker_id: int) -> None:
        pending = session.pending_by_speaker.get(speaker_id)
        if pending is None:
            return
        if session.pending_stt_counts.get(speaker_id, 0) > 0:
            return
        if speaker_id in session.active_speakers:
            return
        if pending.release_task is not None and not pending.release_task.done():
            pending.release_task.cancel()
        logger.info(
            "Voice release scheduled: speaker_id=%s in %.2fs",
            speaker_id,
            _VOICE_STITCH_RELEASE_SECONDS,
        )
        pending.release_task = asyncio.create_task(
            self._release_after_delay(session.session_id, speaker_id),
            name=f"voice-release:{speaker_id}",
        )

    def _arm_force_release(self, session: VoiceSession, speaker_id: int) -> None:
        pending = session.pending_by_speaker.get(speaker_id)
        if pending is None:
            return
        if pending.force_release_task is not None and not pending.force_release_task.done():
            pending.force_release_task.cancel()
        pending.force_release_task = asyncio.create_task(
            self._force_release_after_delay(session.session_id, speaker_id),
            name=f"voice-force-release:{speaker_id}",
        )

    async def _release_after_delay(self, session_id: str, speaker_id: int) -> None:
        await asyncio.sleep(_VOICE_STITCH_RELEASE_SECONDS)
        session = self._session
        if session is None or session.session_id != session_id:
            return
        if session.pending_stt_counts.get(speaker_id, 0) > 0 or speaker_id in session.active_speakers:
            self._maybe_schedule_release(session, speaker_id)
            return
        await self._release_pending_turn(session, speaker_id)

    async def _force_release_after_delay(self, session_id: str, speaker_id: int) -> None:
        await asyncio.sleep(_VOICE_FORCE_RELEASE_SECONDS)
        session = self._session
        if session is None or session.session_id != session_id:
            return
        pending = session.pending_by_speaker.get(speaker_id)
        if pending is None:
            return
        logger.warning(
            "Voice force-release triggered: speaker_id=%s active=%s pending_stt=%s",
            speaker_id,
            speaker_id in session.active_speakers,
            session.pending_stt_counts.get(speaker_id, 0),
        )
        session.active_speakers.discard(speaker_id)
        await self._release_pending_turn(session, speaker_id)

    async def _release_pending_turn(self, session: VoiceSession, speaker_id: int) -> None:
        pending = session.pending_by_speaker.pop(speaker_id, None)
        if pending is None:
            return
        if pending.release_task is not None and not pending.release_task.done():
            pending.release_task.cancel()
        if pending.force_release_task is not None and not pending.force_release_task.done():
            pending.force_release_task.cancel()
        logger.info(
            "Voice turn released: speaker=%s text=%r",
            pending.speaker_name,
            pending.text,
        )
        await self._emit_completed_turn(
            session,
            speaker_id=pending.speaker_id,
            speaker_name=pending.speaker_name,
            text=pending.text,
        )

    async def _emit_completed_turn(
        self,
        session: VoiceSession,
        *,
        speaker_id: int,
        speaker_name: str,
        text: str,
    ) -> None:
        entry = VoiceHistoryEntry(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=text,
            created_at=datetime.now(UTC),
            is_bot=False,
        )
        session.history.add(entry)
        session.pending_response_needed = True
        logger.info(
            "Voice turn appended: speaker=%s text=%r",
            speaker_name,
            text,
        )
        if session.response_task is None or session.response_task.done():
            logger.info(
                "Voice response task create: session_id=%s existing_done=%s pending_response_needed=%s",
                session.session_id,
                session.response_task.done() if session.response_task is not None else None,
                session.pending_response_needed,
            )
            session.response_task = self._create_task(
                self._respond_to_session(session.session_id),
                name=f"voice-respond:{session.session_id}",
            )
        self._create_task(
            self._store_voice_memory(
                session,
                message_id=f"voice-human:{session.session_id}:{session.response_counter}:{speaker_id}:{int(time() * 1000)}",
                author_name=speaker_name,
                text=text,
            ),
            name=f"voice-store-human:{session.session_id}",
        )

    async def _respond_to_session(self, session_id: str) -> None:
        logger.info("Voice response task entered: session_id=%s", session_id)
        session = self._session
        if session is None or session.session_id != session_id:
            logger.warning("Voice response task exiting early: missing or replaced session")
            return
        if session.playback_active:
            logger.warning("Voice response task exiting early: playback already active")
            return

        while session.pending_response_needed:
            session.pending_response_needed = False
            if self._bot_user is None:
                logger.warning("Voice response task exiting early: bot user not set")
                return

            history_messages = session.history.to_ollama_messages(self._bot_user.id)
            latest_user_text = ""
            for entry in reversed(session.history.entries()):
                if not entry.is_bot:
                    latest_user_text = entry.text
                    break
            rag_context = await self.vector_memory.query(latest_user_text, server_id=session.guild_id) if latest_user_text else ""
            logger.info(
                "Voice brain start: latest_user_text=%r participant_count=%s history_messages=%s",
                latest_user_text,
                len(session.participant_names),
                len(history_messages),
            )
            try:
                brain = await self.llm.ask_brain(
                    history_messages,
                    bot_name=self._bot_user.display_name,
                    server_name=session.guild_name,
                    channel_name=session.channel_name,
                    rag_context=rag_context,
                    tool_context=None,
                    mode="voice",
                    participant_names=session.participant_names,
                )
            except Exception:
                logger.exception("Voice brain call failed")
                raise
            logger.info(
                "Voice brain completed: has_response=%s",
                bool(brain and getattr(brain, "content", "").strip()),
            )
            raw_reply = (brain.content if brain is not None else "").strip()
            reply = _sanitize_voice_reply(raw_reply)
            if not reply:
                logger.warning("Voice brain returned empty reply")
                continue
            logger.info(
                "Voice reply generated: raw=%r sanitized=%r",
                raw_reply,
                reply,
            )

            session.playback_active = True
            try:
                try:
                    wav_bytes = await self._tts.synthesize_bytes(reply)
                except Exception:
                    fallback_reply = _sanitize_voice_reply(_truncate_words(reply, max(8, _VOICE_REPLY_MAX_WORDS // 2)))
                    if fallback_reply and fallback_reply != reply:
                        logger.warning(
                            "Voice TTS failed for primary reply; retrying shorter fallback: %r",
                            fallback_reply,
                        )
                        reply = fallback_reply
                        wav_bytes = await self._tts.synthesize_bytes(reply)
                    else:
                        raise
                source = wav_bytes_to_audio_source(wav_bytes)
                logger.info("Voice playback start: reply=%r wav_bytes=%s", reply, len(wav_bytes))
                await self._play_source(session, source)
                logger.info("Voice playback finished cleanly")
            except Exception:
                logger.exception("Voice reply synthesis/playback failed in %s/%s", session.guild_name, session.channel_name)
            finally:
                session.playback_active = False

            session.reply_counter += 1
            entry = VoiceHistoryEntry(
                speaker_id=self._bot_user.id,
                speaker_name=self._bot_user.display_name,
                text=reply,
                created_at=datetime.now(UTC),
                is_bot=True,
            )
            session.history.add(entry)
            self._create_task(
                self._store_voice_memory(
                    session,
                    message_id=f"voice-bot:{session.session_id}:{session.reply_counter}",
                    author_name=self._bot_user.display_name,
                    text=reply,
                ),
                name=f"voice-store-bot:{session.session_id}",
            )

    async def _play_source(self, session: VoiceSession, source: discord.AudioSource) -> None:
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

    async def _store_voice_memory(
        self,
        session: VoiceSession,
        *,
        message_id: str,
        author_name: str,
        text: str,
    ) -> None:
        try:
            await self.vector_memory.add_message(
                message_id=message_id,
                content=text,
                author_name=author_name,
                server_id=session.guild_id,
                timestamp=datetime.now(UTC),
            )
        except Exception:
            logger.exception("Voice vector-memory store failed for %s", message_id)

    async def _warm_voice_models(self) -> None:
        with contextlib.suppress(Exception):
            await self._transcriber.warmup()
        with contextlib.suppress(Exception):
            await self._tts.warmup()

    def _participant_names(
        self,
        channel: discord.VoiceChannel,
        *,
        bot_user: discord.ClientUser | discord.User | None,
    ) -> list[str]:
        bot_id = getattr(bot_user, "id", None)
        return [member.display_name for member in channel.members if member.id != bot_id]

    def _arm_idle_timer(self, session: VoiceSession) -> None:
        if session.idle_task is not None and not session.idle_task.done():
            session.idle_task.cancel()
        session.idle_task = asyncio.create_task(
            self._idle_auto_leave(session.session_id),
            name=f"voice-idle:{session.session_id}",
        )

    async def _idle_auto_leave(self, session_id: str) -> None:
        await asyncio.sleep(_VOICE_IDLE_AUTO_LEAVE_SECONDS)
        session = self._session
        if session is None or session.session_id != session_id:
            return
        if session.participant_names:
            return
        logger.info("Auto-leaving idle voice session in %s/%s", session.guild_name, session.channel_name)
        await self._disconnect_active_voice_client()

    async def _disconnect_active_voice_client(self) -> None:
        session = self._session
        if session is None:
            return
        self._session = None

        if session.idle_task is not None:
            session.idle_task.cancel()
        if session.response_task is not None:
            session.response_task.cancel()
        for pending in session.pending_by_speaker.values():
            if pending.release_task is not None:
                pending.release_task.cancel()

        voice_client = session.voice_client
        self.runtime_state.set_voice_state(active=False, status="idle")
        if voice_client is not None:
            with contextlib.suppress(Exception):
                if voice_recv is not None and isinstance(voice_client, voice_recv.VoiceRecvClient):
                    voice_client.stop_listening()
            with contextlib.suppress(Exception):
                await voice_client.disconnect(force=True)

    def _sync_runtime_state(self, *, status: str) -> None:
        session = self._session
        if session is None:
            self.runtime_state.set_voice_state(active=False, status="idle")
            return
        self.runtime_state.set_voice_state(
            active=True,
            status=status,
            session_id=session.session_id,
            guild_id=session.guild_id,
            guild_name=session.guild_name,
            channel_id=session.channel_id,
            channel_name=session.channel_name,
            participant_names=session.participant_names,
        )
