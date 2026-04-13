from __future__ import annotations

import asyncio
import contextlib
import os
from time import time
from typing import TYPE_CHECKING
from uuid import uuid4

import discord

from ..logconf import get_logger
from ..runtime_state import RuntimeState
from .capture import CaptureJob, UtteranceCaptureSink
from .models import (
    VoiceCommandResult,
    VoiceSession,
    _VOICE_CAPTURE_DIR,
    _VOICE_IDLE_AUTO_LEAVE_SECONDS,
    _VOICE_PREROLL_MS,
    resolve_target_channel,
)
from .response import respond_to_session, store_voice_memory, warm_voice_models
from .stitching import emit_completed_turn, handle_transcript, maybe_schedule_release
from .stt import FasterWhisperTranscriber
from .tts import TtsServiceClient, TtsServiceConfig

if TYPE_CHECKING:
    from ..config import VoiceConfig

try:
    from discord.ext import voice_recv
except Exception:  # pragma: no cover - import failure depends on environment
    voice_recv = None

logger = get_logger("sandy.voice")


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
        voice_config: VoiceConfig | None = None,
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

        if voice_config is not None:
            stt_model = voice_config.stt_model
            stt_device = voice_config.stt_device
            stt_compute_type = voice_config.stt_compute_type
            stt_language = voice_config.stt_language
            tts_url = voice_config.tts_service_url
            tts_timeout = voice_config.tts_service_timeout_seconds
            tts_instruct = voice_config.tts_instruct
            tts_language = voice_config.tts_language
        else:
            stt_model = os.getenv("VOICE_STT_MODEL", "base.en")
            stt_device = os.getenv("VOICE_STT_DEVICE", "cuda")
            stt_compute_type = os.getenv("VOICE_STT_COMPUTE_TYPE", "float16")
            stt_language = os.getenv("VOICE_STT_LANGUAGE", "en").strip() or None
            tts_url = os.getenv("VOICE_TTS_SERVICE_URL", "http://127.0.0.1:8777")
            tts_timeout = float(os.getenv("VOICE_TTS_SERVICE_TIMEOUT_SECONDS", "180"))
            tts_instruct = os.getenv("VOICE_TTS_INSTRUCT") or None
            tts_language = os.getenv("VOICE_TTS_LANGUAGE") or "English"

        self._transcriber = FasterWhisperTranscriber(
            model_name=stt_model,
            device=stt_device,
            compute_type=stt_compute_type,
            language=stt_language,
        )
        self._tts = TtsServiceClient(
            TtsServiceConfig(
                base_url=tts_url,
                timeout_seconds=tts_timeout,
                default_instruct=tts_instruct,
                default_language=tts_language,
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
        self._sync_runtime_state(status="connected", stage="idle_in_channel")
        self._create_task(self._warm_voice_models(), name="voice-warmup")
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
            self._create_task(self._disconnect_active_voice_client(), name="voice-disconnect-sync")
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
        self.runtime_state.update_voice_stage(stage="capturing", status="connected")
        logger.info("Voice speaker start: speaker_id=%s", speaker_id)

    def _speaker_stopped_on_loop(self, speaker_id: int) -> None:
        session = self._session
        if session is None:
            return
        session.active_speakers.discard(speaker_id)
        self.runtime_state.update_voice_stage(stage="stitch_wait", status="connected")
        logger.info("Voice speaker stop: speaker_id=%s", speaker_id)
        maybe_schedule_release(self, session, speaker_id)

    async def _run_stt_worker(self) -> None:
        while True:
            job = await self._stt_queue.get()
            try:
                if job is self._STOP:
                    return
                assert isinstance(job, CaptureJob)
                self.runtime_state.update_voice_stage(
                    stage="transcribing",
                    status="connected",
                )
                result = await self._transcriber.transcribe_file(job.path)
                logger.info(
                    "Voice transcript completed: speaker=%s text=%r audio=%.2fs stt=%.2fs",
                    job.speaker_label,
                    result.text.strip(),
                    job.duration_seconds,
                    result.elapsed_seconds,
                )
                await self._handle_transcript(job, result)
            except Exception:
                self.runtime_state.update_voice_stage(
                    stage="transcribing",
                    status="error",
                    last_error=f"STT failed for {getattr(job, 'speaker_label', '?')}",
                )
                logger.exception("STT failed for capture=%s", getattr(job, "path", "?"))
            finally:
                self._stt_queue.task_done()

    # ── Thin delegators to stitching.py / response.py ─────────────────────────
    # These keep the method signatures tests expect on VoiceManager while the
    # actual logic lives in the extracted modules.

    async def _handle_transcript(self, job, result):
        await handle_transcript(self, job, result)

    async def _emit_completed_turn(self, session, *, completed_turn):
        await emit_completed_turn(self, session, completed_turn=completed_turn)

    async def _respond_to_session(self, session_id):
        await respond_to_session(self, session_id)

    async def _play_source(self, session, source):
        from .response import play_source
        await play_source(session, source)

    async def _store_voice_memory(self, session, *, message_id, author_name, text):
        await store_voice_memory(self, session, message_id=message_id, author_name=author_name, text=text)

    async def _warm_voice_models(self):
        await warm_voice_models(self)

    # ── Remaining manager-owned methods ───────────────────────────────────────

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
            if pending.force_release_task is not None:
                pending.force_release_task.cancel()
        session.pending_by_speaker.clear()
        session.pending_stt_counts.clear()
        session.active_speakers.clear()

        voice_client = session.voice_client
        self.runtime_state.set_voice_state(active=False, status="idle")
        if voice_client is not None:
            with contextlib.suppress(Exception):
                if voice_recv is not None and isinstance(voice_client, voice_recv.VoiceRecvClient):
                    voice_client.stop_listening()
            with contextlib.suppress(Exception):
                await voice_client.disconnect(force=True)

    def _sync_runtime_state(self, *, status: str, stage: str | None = None) -> None:
        session = self._session
        if session is None:
            self.runtime_state.set_voice_state(active=False, status="idle")
            return
        self.runtime_state.set_voice_state(
            active=True,
            status=status,
            stage=stage,
            session_id=session.session_id,
            guild_id=session.guild_id,
            guild_name=session.guild_name,
            channel_id=session.channel_id,
            channel_name=session.channel_name,
            participant_names=session.participant_names,
            session_started_at=session.started_at,
        )
