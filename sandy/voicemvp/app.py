"""Isolated Discord voice MVP.

This module intentionally does not import Sandy's normal bot pipeline, logging
stack, Recall, Chroma, or trace infrastructure. The point is proving Discord
voice transport and media plumbing in isolation.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
from collections import deque
import contextlib
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import logging
import os
from pathlib import Path
import threading
import time
import wave
from typing import Callable, Iterable

import discord
from discord.ext import voice_recv
from dotenv import load_dotenv

from .stt import FasterWhisperTranscriber
from .tts import QwenTtsConfig, QwenVoiceDesignTts, pcm_bytes_to_audio_source

load_dotenv()
discord.opus._load_default()

logger = logging.getLogger("sandy.voicemvp")
CAPTURES_DIR = Path(__file__).resolve().parent / "captures"
TEST_AUDIO_PATH = Path(__file__).resolve().parent / "file_example_WAV_2MG.wav"
WAV_CHANNELS = 2
WAV_SAMPLE_WIDTH = 2
WAV_SAMPLE_RATE = 48_000


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


@dataclass(frozen=True, slots=True)
class VoiceMvpConfig:
    token: str
    prefix: str = "!"
    join_command: str = "join"
    leave_command: str = "leave"
    http_host: str = "0.0.0.0"
    http_port: int = 8765
    preroll_ms: int = 250
    stt_model: str = "base.en"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    stt_language: str | None = "en"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    tts_language: str = "English"
    tts_instruct: str = (
        "Voice Identity: warm, mid-range, female-coded, calm and slightly amused, "
        "natural and conversational, medium speaking pace, clear diction, gentle "
        "dry humor, not cutesy, not breathy, not theatrical."
    )
    tts_device_map: str = "cuda:0"
    tts_dtype: str = "bfloat16"
    tts_attn_implementation: str = "sdpa"
    tts_max_new_tokens: int = 2048

    @property
    def help_text(self) -> str:
        return (
            "Voice MVP commands:\n"
            f"  {self.prefix}{self.join_command} [voice channel name]\n"
            f"  {self.prefix}{self.leave_command}\n"
            f"  {self.prefix}playtest\n"
            f"  {self.prefix}say <text>\n\n"
            "If you omit the channel name, Sandy uses your current voice channel."
        )


def build_config(*, test_mode: bool = False) -> VoiceMvpConfig:
    token_var = "DISCORD_API_KEY_TEST" if test_mode else "DISCORD_API_KEY"
    token = os.getenv(token_var) or os.getenv("DISCORD_API_KEY")
    if not token:
        raise RuntimeError(
            f"{token_var} is not set and no DISCORD_API_KEY fallback is available",
        )
    prefix = os.getenv("VOICE_MVP_PREFIX", "!")
    http_host = os.getenv("VOICE_MVP_HTTP_HOST", "0.0.0.0")
    http_port = int(os.getenv("VOICE_MVP_HTTP_PORT", "8765"))
    preroll_ms = int(os.getenv("VOICE_MVP_PREROLL_MS", "250"))
    stt_model = os.getenv("VOICE_MVP_STT_MODEL", "base.en")
    stt_device = os.getenv("VOICE_MVP_STT_DEVICE", "cuda")
    stt_compute_type = os.getenv("VOICE_MVP_STT_COMPUTE_TYPE", "float16")
    stt_language = os.getenv("VOICE_MVP_STT_LANGUAGE", "en").strip() or None
    tts_model = os.getenv("VOICE_MVP_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    tts_language = os.getenv("VOICE_MVP_TTS_LANGUAGE", "English").strip() or "English"
    tts_instruct = os.getenv(
        "VOICE_MVP_TTS_INSTRUCT",
        (
            "Voice Identity: warm, mid-range, female-coded, calm and slightly amused, "
            "natural and conversational, medium speaking pace, clear diction, gentle "
            "dry humor, not cutesy, not breathy, not theatrical."
        ),
    ).strip()
    tts_device_map = os.getenv("VOICE_MVP_TTS_DEVICE_MAP", "cuda:0")
    tts_dtype = os.getenv("VOICE_MVP_TTS_DTYPE", "bfloat16")
    tts_attn_implementation = os.getenv("VOICE_MVP_TTS_ATTN_IMPLEMENTATION", "sdpa")
    tts_max_new_tokens = int(os.getenv("VOICE_MVP_TTS_MAX_NEW_TOKENS", "2048"))
    return VoiceMvpConfig(
        token=token,
        prefix=prefix,
        http_host=http_host,
        http_port=http_port,
        preroll_ms=preroll_ms,
        stt_model=stt_model,
        stt_device=stt_device,
        stt_compute_type=stt_compute_type,
        stt_language=stt_language,
        tts_model=tts_model,
        tts_language=tts_language,
        tts_instruct=tts_instruct,
        tts_device_map=tts_device_map,
        tts_dtype=tts_dtype,
        tts_attn_implementation=tts_attn_implementation,
        tts_max_new_tokens=tts_max_new_tokens,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sacrificial Sandy voice MVP")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use DISCORD_API_KEY_TEST if available.",
    )
    return parser


def _normalize_name(value: str) -> str:
    return " ".join(value.lower().split())


def _voice_channels(guild: discord.Guild) -> Iterable[discord.VoiceChannel]:
    return guild.voice_channels


def _slugify_capture_label(value: str) -> str:
    collapsed = "-".join(value.lower().split())
    sanitized = "".join(ch for ch in collapsed if ch.isalnum() or ch == "-")
    return sanitized.strip("-") or "unknown"


def _pcm_bytes_for_milliseconds(milliseconds: int) -> int:
    bytes_per_second = WAV_SAMPLE_RATE * WAV_CHANNELS * WAV_SAMPLE_WIDTH
    return int(bytes_per_second * (milliseconds / 1000))


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
    channels = list(_voice_channels(guild))

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
class ActiveUtterance:
    speaker_label: str
    ssrc: int
    started_at: float
    pcm_chunks: list[bytes]
    packet_count: int = 0


@dataclass(frozen=True, slots=True)
class CaptureJob:
    path: Path
    speaker_label: str
    ssrc: int
    duration_seconds: float
    packet_count: int
    saved_at: float


class CaptureHttpServer:
    def __init__(self, directory: Path, host: str, port: int) -> None:
        self.directory = directory
        self.host = host
        self.port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        handler = partial(SimpleHTTPRequestHandler, directory=str(self.directory))
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="voicemvp-captures-http",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Capture HTTP server started: directory=%s listen=http://%s:%s/",
            self.directory,
            self.host,
            self.port,
        )

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class UtteranceCaptureSink(voice_recv.AudioSink):
    def __init__(
        self,
        capture_dir: Path,
        *,
        preroll_ms: int,
        on_capture_saved: Callable[[CaptureJob], None] | None = None,
    ) -> None:
        super().__init__()
        self.capture_dir = capture_dir
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.preroll_ms = max(preroll_ms, 0)
        self._preroll_limit_bytes = _pcm_bytes_for_milliseconds(self.preroll_ms)
        self._active: dict[int, ActiveUtterance] = {}
        self._preroll_chunks: dict[int, deque[bytes]] = {}
        self._preroll_sizes: dict[int, int] = {}
        self._capture_index = self._next_capture_index()
        self._on_capture_saved = on_capture_saved

    def wants_opus(self) -> bool:
        return False

    def _next_capture_index(self) -> int:
        indices = []
        for path in self.capture_dir.glob("*.wav"):
            prefix, _, _ = path.name.partition("-")
            if prefix.isdigit():
                indices.append(int(prefix))
        return (max(indices) + 1) if indices else 1

    def _speaker_label(
        self,
        user: discord.User | discord.Member | None,
        data: voice_recv.VoiceData,
    ) -> str:
        return (
            getattr(user, "display_name", None)
            or getattr(data.source, "display_name", None)
            or getattr(user, "name", None)
            or getattr(data.source, "name", None)
            or "unknown"
        )

    def _ensure_active(
        self,
        *,
        ssrc: int,
        speaker_label: str,
    ) -> ActiveUtterance:
        active = self._active.get(ssrc)
        if active is None:
            preroll_chunks = list(self._preroll_chunks.get(ssrc, ()))
            active = ActiveUtterance(
                speaker_label=speaker_label,
                ssrc=ssrc,
                started_at=time.perf_counter(),
                pcm_chunks=preroll_chunks,
            )
            self._active[ssrc] = active
            self._preroll_chunks.pop(ssrc, None)
            self._preroll_sizes.pop(ssrc, None)
            logger.info(
                "Capture started lazily: speaker=%s ssrc=%s preroll_chunks=%s preroll_ms=%s",
                speaker_label,
                ssrc,
                len(preroll_chunks),
                self.preroll_ms,
            )
        return active

    def _append_preroll(self, ssrc: int, pcm: bytes) -> None:
        if self._preroll_limit_bytes <= 0:
            return

        chunks = self._preroll_chunks.setdefault(ssrc, deque())
        total = self._preroll_sizes.get(ssrc, 0) + len(pcm)
        chunks.append(pcm)

        while chunks and total > self._preroll_limit_bytes:
            total -= len(chunks.popleft())

        self._preroll_sizes[ssrc] = total

    def write(
        self,
        user: discord.User | discord.Member | None,
        data: voice_recv.VoiceData,
    ) -> None:
        pcm = data.pcm or b""
        if not pcm:
            return

        ssrc = getattr(data.packet, "ssrc", None)
        if ssrc is None:
            return

        speaker_label = self._speaker_label(user, data)
        active = self._active.get(ssrc)
        if active is None:
            self._append_preroll(ssrc, pcm)
            return

        active.pcm_chunks.append(pcm)
        active.packet_count += 1

        if active.packet_count == 1 or active.packet_count % 50 == 0:
            logger.info(
                "Capture packet: speaker=%s ssrc=%s utterance_packets=%s pcm_bytes=%s",
                active.speaker_label,
                ssrc,
                active.packet_count,
                len(pcm),
            )

    def _finalize_ssrc(self, ssrc: int, *, reason: str) -> None:
        active = self._active.pop(ssrc, None)
        if active is None:
            return

        pcm_bytes = b"".join(active.pcm_chunks)
        if not pcm_bytes:
            logger.info(
                "Capture dropped empty utterance: speaker=%s ssrc=%s reason=%s",
                active.speaker_label,
                ssrc,
                reason,
            )
            return

        duration_seconds = len(pcm_bytes) / (WAV_SAMPLE_RATE * WAV_CHANNELS * WAV_SAMPLE_WIDTH)
        speaker_slug = _slugify_capture_label(active.speaker_label)
        capture_name = (
            f"{self._capture_index:04d}-{speaker_slug}-ssrc{ssrc}-{int(time.time())}.wav"
        )
        self._capture_index += 1
        capture_path = self.capture_dir / capture_name

        with wave.open(str(capture_path), "wb") as wav_file:
            wav_file.setnchannels(WAV_CHANNELS)
            wav_file.setsampwidth(WAV_SAMPLE_WIDTH)
            wav_file.setframerate(WAV_SAMPLE_RATE)
            wav_file.writeframes(pcm_bytes)

        logger.info(
            "Capture saved: file=%s speaker=%s ssrc=%s duration=%.2fs packets=%s bytes=%s reason=%s",
            capture_path.name,
            active.speaker_label,
            ssrc,
            duration_seconds,
            active.packet_count,
            len(pcm_bytes),
            reason,
        )
        if self._on_capture_saved is not None:
            self._on_capture_saved(
                CaptureJob(
                    path=capture_path,
                    speaker_label=active.speaker_label,
                    ssrc=ssrc,
                    duration_seconds=duration_seconds,
                    packet_count=active.packet_count,
                    saved_at=time.perf_counter(),
                )
            )

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_start(self, member: discord.Member) -> None:
        ssrc = self.voice_client._get_ssrc_from_id(member.id)
        if ssrc is None:
            logger.info(
                "Capture speaking-start without ssrc: speaker=%s",
                member.display_name,
            )
            return

        if ssrc in self._active:
            logger.info(
                "Capture speaking-start ignored because utterance already active: speaker=%s ssrc=%s",
                member.display_name,
                ssrc,
            )
            return

        self._active[ssrc] = ActiveUtterance(
            speaker_label=member.display_name,
            ssrc=ssrc,
            started_at=time.perf_counter(),
            pcm_chunks=list(self._preroll_chunks.get(ssrc, ())),
        )
        preroll_chunks = len(self._active[ssrc].pcm_chunks)
        self._preroll_chunks.pop(ssrc, None)
        self._preroll_sizes.pop(ssrc, None)
        logger.info(
            "Capture speaking-start: speaker=%s ssrc=%s preroll_chunks=%s preroll_ms=%s",
            member.display_name,
            ssrc,
            preroll_chunks,
            self.preroll_ms,
        )

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member: discord.Member) -> None:
        ssrc = self.voice_client._get_ssrc_from_id(member.id)
        if ssrc is None:
            logger.info(
                "Capture speaking-stop without ssrc: speaker=%s",
                member.display_name,
            )
            return
        self._finalize_ssrc(ssrc, reason="speaking-stop")

    @voice_recv.AudioSink.listener()
    def on_voice_member_disconnect(
        self,
        member: discord.Member,
        ssrc: int | None,
    ) -> None:
        if ssrc is not None:
            self._finalize_ssrc(ssrc, reason="member-disconnect")

    def cleanup(self) -> None:
        for ssrc in list(self._active):
            self._finalize_ssrc(ssrc, reason="cleanup")


class VoiceMvpClient(discord.Client):
    def __init__(self, config: VoiceMvpConfig) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        intents.guilds = True
        intents.members = True
        super().__init__(intents=intents)
        self.config = config
        self.capture_server = CaptureHttpServer(
            CAPTURES_DIR,
            config.http_host,
            config.http_port,
        )
        self.transcriber = FasterWhisperTranscriber(
            model_name=config.stt_model,
            device=config.stt_device,
            compute_type=config.stt_compute_type,
            language=config.stt_language,
        )
        self.tts = QwenVoiceDesignTts(
            QwenTtsConfig(
                model_name=config.tts_model,
                language=config.tts_language,
                instruct=config.tts_instruct,
                device_map=config.tts_device_map,
                dtype=config.tts_dtype,
                attn_implementation=config.tts_attn_implementation,
                max_new_tokens=config.tts_max_new_tokens,
            )
        )
        self._stt_queue: asyncio.Queue[CaptureJob] = asyncio.Queue()
        self._stt_worker: asyncio.Task | None = None

    async def setup_hook(self) -> None:
        self._stt_worker = asyncio.create_task(
            self._run_stt_worker(),
            name="sandy-voicemvp-stt",
        )

    async def on_ready(self) -> None:
        logger.info(
            "Voice MVP connected as %s (%s) across %d guild(s)",
            self.user,
            self.user.id if self.user else "?",
            len(self.guilds),
        )
        logger.info(
            "Voice MVP is isolated from Sandy's text pipeline, logs DB, Recall, and Chroma",
        )

    def _build_test_audio_source(self) -> discord.AudioSource:
        if not TEST_AUDIO_PATH.exists():
            raise FileNotFoundError(f"Missing test audio file: {TEST_AUDIO_PATH}")

        with wave.open(str(TEST_AUDIO_PATH), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            pcm = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            raise ValueError(
                f"Unsupported test WAV sample width: expected 16-bit PCM, got {sample_width * 8}-bit",
            )

        # discord.py's raw PCM path wants 16-bit 48kHz stereo. This keeps the
        # spike self-contained for a local test WAV without forcing ffmpeg to be
        # installed just to prove outbound playback works.
        if channels == 1:
            pcm = audioop.tostereo(pcm, sample_width, 1, 1)
            channels = 2
        elif channels != 2:
            raise ValueError(f"Unsupported test WAV channel count: {channels}")

        if sample_rate != WAV_SAMPLE_RATE:
            pcm, _ = audioop.ratecv(
                pcm,
                sample_width,
                channels,
                sample_rate,
                WAV_SAMPLE_RATE,
                None,
            )

        return discord.PCMAudio(io.BytesIO(pcm))

    def _on_playback_finished(self, guild_name: str, channel_name: str, error: Exception | None) -> None:
        if error is not None:
            logger.error(
                "Playback finished with error: guild=%s channel=%s error=%r",
                guild_name,
                channel_name,
                error,
            )
            return
        logger.info(
            "Playback finished cleanly: guild=%s channel=%s",
            guild_name,
            channel_name,
        )

    def _attach_receive_probe(self, voice_client: voice_recv.VoiceRecvClient) -> None:
        if voice_client.is_listening():
            logger.info("Capture sink already attached in guild=%s", voice_client.guild.name)
            return

        sink = UtteranceCaptureSink(
            CAPTURES_DIR,
            preroll_ms=self.config.preroll_ms,
            on_capture_saved=self._enqueue_capture_job_from_thread,
        )
        voice_client.listen(
            sink,
            after=lambda exc: logger.info("Capture sink stopped: error=%r", exc),
        )
        logger.info(
            "Capture sink attached: guild=%s channel=%s capture_dir=%s serve=http://%s:%s/ preroll_ms=%s stt_model=%s",
            voice_client.guild.name,
            voice_client.channel.name if voice_client.channel else "?",
            CAPTURES_DIR,
            self.config.http_host,
            self.config.http_port,
            self.config.preroll_ms,
            self.config.stt_model,
        )

    def _enqueue_capture_job_from_thread(self, job: CaptureJob) -> None:
        self.loop.call_soon_threadsafe(self._stt_queue.put_nowait, job)

    async def _run_stt_worker(self) -> None:
        while True:
            job = await self._stt_queue.get()
            try:
                result = await self.transcriber.transcribe_file(job.path)
                queue_delay = max(
                    time.perf_counter() - job.saved_at - result.elapsed_seconds,
                    0.0,
                )
                logger.info(
                    "STT transcript: file=%s speaker=%s ssrc=%s duration=%.2fs packets=%s queue_delay=%.2fs stt_time=%.2fs device=%s compute=%s text=%r",
                    job.path.name,
                    job.speaker_label,
                    job.ssrc,
                    job.duration_seconds,
                    job.packet_count,
                    queue_delay,
                    result.elapsed_seconds,
                    result.device,
                    result.compute_type,
                    result.text,
                )
            except Exception:
                logger.exception("STT failed for capture=%s", job.path)
            finally:
                self._stt_queue.task_done()

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return

        content = message.content.strip()
        if not content.startswith(self.config.prefix):
            return

        command_line = content[len(self.config.prefix):].strip()
        if not command_line:
            return

        verb, _, remainder = command_line.partition(" ")
        verb = verb.lower()
        arg = remainder.strip()

        if verb == self.config.join_command:
            await self._handle_join(message, arg)
            return
        if verb == self.config.leave_command:
            await self._handle_leave(message)
            return
        if verb == "playtest":
            await self._handle_playtest(message)
            return
        if verb == "say":
            await self._handle_say(message, arg)
            return
        if verb == "voicehelp":
            await message.channel.send(self.config.help_text)

    async def _handle_join(self, message: discord.Message, channel_query: str) -> None:
        author_voice = (
            message.author.voice.channel
            if getattr(message.author, "voice", None) and message.author.voice.channel
            else None
        )
        target_channel = resolve_target_channel(
            message.guild,
            query=channel_query,
            author_voice_channel=author_voice,
        )
        if target_channel is None:
            await message.channel.send(
                "No matching voice channel found. Use your current voice channel or name one explicitly.",
            )
            return

        permissions = target_channel.permissions_for(message.guild.me)
        if not permissions.connect:
            await message.channel.send("I don't have permission to connect to that voice channel.")
            return
        if not permissions.speak:
            await message.channel.send("I don't have permission to speak in that voice channel.")
            return

        existing_voice_client = discord.utils.get(self.voice_clients, guild=message.guild)
        if existing_voice_client and existing_voice_client.is_connected():
            current_channel = getattr(existing_voice_client, "channel", None)
            if current_channel and current_channel.id == target_channel.id:
                if isinstance(existing_voice_client, voice_recv.VoiceRecvClient):
                    self._attach_receive_probe(existing_voice_client)
                await message.channel.send(f"Already in `{target_channel.name}`.")
                return
            logger.info(
                "Moving existing voice client in guild=%s from %s to %s",
                message.guild.name,
                current_channel.name if current_channel else "?",
                target_channel.name,
            )
            await existing_voice_client.move_to(target_channel)
            if isinstance(existing_voice_client, voice_recv.VoiceRecvClient):
                self._attach_receive_probe(existing_voice_client)
            await message.channel.send(f"Moved to `{target_channel.name}`.")
            return

        stale_voice_client = discord.utils.get(self.voice_clients, guild=message.guild)
        if stale_voice_client is not None:
            logger.info("Disconnecting stale voice client before reconnect in guild=%s", message.guild.name)
            await stale_voice_client.disconnect(force=True)
            await asyncio.sleep(1)

        logger.info(
            "Connecting to voice channel guild=%s channel=%s with VoiceRecvClient",
            message.guild.name,
            target_channel.name,
        )
        voice_client = await target_channel.connect(
            cls=voice_recv.VoiceRecvClient,
            timeout=30.0,
            reconnect=False,
        )
        logger.info(
            "Connected: guild=%s channel=%s client=%s",
            message.guild.name,
            target_channel.name,
            type(voice_client).__name__,
        )
        self._attach_receive_probe(voice_client)
        await message.channel.send(f"Joined `{target_channel.name}`.")

    async def _handle_leave(self, message: discord.Message) -> None:
        voice_client = discord.utils.get(self.voice_clients, guild=message.guild)
        if voice_client is None or not voice_client.is_connected():
            await message.channel.send("Not connected to a voice channel in this guild.")
            return

        channel_name = voice_client.channel.name if voice_client.channel else "unknown"
        logger.info("Disconnecting from guild=%s channel=%s", message.guild.name, channel_name)
        with contextlib.suppress(Exception):
            voice_client.stop_listening()
        await voice_client.disconnect(force=True)
        await message.channel.send(f"Left `{channel_name}`.")

    async def _handle_playtest(self, message: discord.Message) -> None:
        voice_client = discord.utils.get(self.voice_clients, guild=message.guild)
        if voice_client is None or not voice_client.is_connected():
            await message.channel.send("Not connected to a voice channel in this guild.")
            return
        if voice_client.channel is None:
            await message.channel.send("Voice client is connected but has no active channel.")
            return
        if voice_client.is_playing():
            await message.channel.send("Already playing audio.")
            return

        try:
            source = self._build_test_audio_source()
        except FileNotFoundError:
            logger.exception("Playtest audio file is missing")
            await message.channel.send("Test audio file is missing.")
            return
        except Exception:
            logger.exception("Failed to prepare playtest audio source")
            await message.channel.send("Failed to prepare the test audio source.")
            return

        logger.info(
            "Starting playtest: guild=%s channel=%s file=%s",
            message.guild.name,
            voice_client.channel.name,
            TEST_AUDIO_PATH.name,
        )
        try:
            voice_client.play(
                source,
                after=lambda error: self._on_playback_finished(
                    message.guild.name,
                    voice_client.channel.name if voice_client.channel else "unknown",
                    error,
                ),
            )
        except Exception:
            logger.exception("Playtest playback failed to start")
            await message.channel.send("Failed to start test audio playback.")
            return

        await message.channel.send(f"Playing `{TEST_AUDIO_PATH.name}` in `{voice_client.channel.name}`.")

    async def _handle_say(self, message: discord.Message, text: str) -> None:
        text = text.strip()
        if not text:
            await message.channel.send("Usage: `!say <text>`")
            return

        voice_client = discord.utils.get(self.voice_clients, guild=message.guild)
        if voice_client is None or not voice_client.is_connected():
            await message.channel.send("Not connected to a voice channel in this guild.")
            return
        if voice_client.channel is None:
            await message.channel.send("Voice client is connected but has no active channel.")
            return
        if voice_client.is_playing():
            await message.channel.send("Already playing audio.")
            return

        logger.info(
            "Starting TTS synthesis: guild=%s channel=%s text=%r model=%s language=%s",
            message.guild.name,
            voice_client.channel.name,
            text,
            self.config.tts_model,
            self.config.tts_language,
        )

        try:
            pcm_bytes, sample_rate = await asyncio.to_thread(self.tts.synthesize_bytes, text)
            source = pcm_bytes_to_audio_source(pcm_bytes)
        except Exception:
            logger.exception("TTS synthesis failed")
            await message.channel.send("TTS synthesis failed.")
            return

        logger.info(
            "Starting TTS playback: guild=%s channel=%s pcm_bytes=%s sample_rate=%s",
            message.guild.name,
            voice_client.channel.name,
            len(pcm_bytes),
            sample_rate,
        )
        try:
            voice_client.play(
                source,
                after=lambda error: self._on_playback_finished(
                    message.guild.name,
                    voice_client.channel.name if voice_client.channel else "unknown",
                    error,
                ),
            )
        except Exception:
            logger.exception("TTS playback failed to start")
            await message.channel.send("Failed to start TTS playback.")
            return

        await message.channel.send(f"Speaking in `{voice_client.channel.name}`.")


async def run_voice_mvp(*, test_mode: bool = False) -> int:
    config = build_config(test_mode=test_mode)
    client = VoiceMvpClient(config)
    client.capture_server.start()
    try:
        await client.start(config.token)
    except KeyboardInterrupt:
        logger.info("^C caught, shutting down voice MVP")
    finally:
        if client._stt_worker is not None:
            client._stt_worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await client._stt_worker
        client.capture_server.stop()
        await client.close()
    return 0
