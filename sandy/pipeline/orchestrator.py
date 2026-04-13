"""
Sandy's text message orchestrator.

This is the readable top-level message flow. Each step either happens inline
(if trivial) or calls into the corresponding pipeline module.
"""

import asyncio
import time

import discord

from ..last10 import Last10, resolve_mentions
from ..llm import OllamaInterface
from ..logconf import get_logger
from ..memory import MemoryClient
from ..recall import ChatDatabase
from ..registry import Registry
from ..runtime_state import RuntimeState
from ..trace import TurnTrace
from ..vector_memory import VectorMemory
from ..voice import VoiceManager
from .. import tools as tools_module_default

from .attachments import (
    AttachmentPreparationResult,
    AttachmentProcessingResult,
    build_cache_message,
    describe_attachments,
    describe_prepared_attachments,
    prepare_attachments,
)
from .bouncer import build_bouncer_context, run_bouncer
from .brain import finalize_reply, run_brain
from .memory_worker import MemoryWorker
from .reply import send_reply
from .retrieval import run_retrieval
from .tool_dispatch import run_tool_dispatch
from .tracing import trace_event as _default_trace_event, forensic_event

logger = get_logger("sandy.bot")


class SandyPipeline:
    """Application-side owner of Sandy's message pipeline."""

    def __init__(
        self,
        *,
        background_tasks,
        registry: Registry,
        cache: Last10,
        llm: OllamaInterface,
        vector_memory: VectorMemory,
        recall_db: ChatDatabase,
        memory: MemoryClient,
        memory_worker: MemoryWorker,
        runtime_state: RuntimeState,
        voice: VoiceManager,
        tools_module=tools_module_default,
        trace_event=_default_trace_event,
    ) -> None:
        self.background_tasks = background_tasks
        self.registry = registry
        self.cache = cache
        self.llm = llm
        self.vector_memory = vector_memory
        self.recall_db = recall_db
        self.memory = memory
        self.memory_worker = memory_worker
        self.runtime_state = runtime_state
        self.voice = voice
        self.tools_module = tools_module
        self.trace_event = trace_event
        self._cache_seeded = False
        self._memory_worker_task: asyncio.Task | None = None

    # -- Delegate methods for test monkeypatching compatibility --

    async def describe_attachments(self, message: discord.Message) -> AttachmentProcessingResult:
        return await describe_attachments(message, self.llm)

    async def prepare_attachments(self, message: discord.Message) -> AttachmentPreparationResult:
        return await prepare_attachments(message)

    async def describe_prepared_attachments(
        self,
        prepared: AttachmentPreparationResult,
        *,
        detail: bool,
    ) -> AttachmentProcessingResult:
        return await describe_prepared_attachments(prepared, self.llm, detail=detail)

    async def send_reply(self, message: discord.Message, reply: str) -> int:
        return await send_reply(message, reply)

    # -- Lifecycle --

    async def on_ready(self, bot: discord.Client) -> None:
        logger.info("Logged in as %s (%s)", bot.user.name, bot.user.id)
        if self._memory_worker_task is None or self._memory_worker_task.done():
            self._memory_worker_task = self.background_tasks.create_task(
                self.memory_worker.run(),
                name="memory-worker",
            )
        await self.voice.on_ready(bot)
        if not self._cache_seeded:
            seeded = await self.memory.seed_cache(self.cache)
            logger.info("Cache seeded with %d message(s) from Recall", seeded)
            self._cache_seeded = True
            ready_info = "       ###   BOT READY   ###\n\n"
            ready_info += f"      * bot logged in as {bot.user.name} ({bot.user.id})\n"
            guild_count = 0
            for guild in bot.guilds:
                ready_info += f"      * attached to {guild.name} ({guild.id})\n"
                guild_count += 1
            ready_info += f"      * {bot.user.name} is on {guild_count} servers\n"
            logger.warning("\n\n%s", ready_info)

    async def handle_control_message(self, message: discord.Message, *, bot_user) -> bool:
        result = await self.voice.handle_text_command(message, bot_user=bot_user)
        if not result.handled:
            return False
        if result.reply:
            await message.channel.send(result.reply)
        return True

    def handle_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
        *,
        bot_user,
    ) -> None:
        self.voice.handle_voice_state_update(member, before, after, bot_user=bot_user)

    # -- Helpers --

    def _log_message_received(self, message: discord.Message) -> None:
        logger.info(
            "[%s/%s] %s%s: %s",
            message.guild.name,
            message.channel.name,
            message.author.display_name,
            f" [{len(message.attachments)} attachment(s)]" if message.attachments else "",
            resolve_mentions(message.content, message.mentions),
        )

    def _add_message_to_cache(
        self,
        message: discord.Message,
        image_descriptions: list[str],
    ) -> str:
        cache_message = build_cache_message(message, image_descriptions)
        self.cache.add(cache_message)
        return cache_message.content

    # -- Main message flow --

    async def handle_message(self, message: discord.Message, *, bot_user) -> None:
        trace = TurnTrace.from_message(message)
        turn_started = time.perf_counter()
        replied = False
        self.runtime_state.begin_turn(trace, author_is_bot=message.author.bot)

        try:
            # 1. Turn intake / trace bootstrap
            self.trace_event(
                trace,
                "message_received",
                content_chars=len(message.content or ""),
                attachments=len(message.attachments),
                author_is_bot=message.author.bot,
            )
            forensic_event(
                trace,
                "turn_input",
                raw_content=message.content,
                resolved_content=resolve_mentions(message.content, message.mentions),
                attachments=[
                    {
                        "filename": attachment.filename,
                        "content_type": attachment.content_type,
                        "size_bytes": attachment.size,
                    }
                    for attachment in message.attachments
                ],
                mentions=[mention.display_name for mention in message.mentions],
            )

            # 2. Registry refresh + human-readable ingress log
            self.background_tasks.create_task(
                asyncio.to_thread(self.registry.ensure_seen, message),
                name=f"registry.ensure_seen:{message.id}",
            )
            self._log_message_received(message)

            # 3. Bot messages short-circuit: cache + memory enqueue, then done
            if message.author.bot:
                self.runtime_state.update_turn_stage(trace, "memory_enqueue")
                logger.debug("Bot message from %s — storing and skipping pipeline", message.author.display_name)
                self.cache.add(message)
                await self.memory_worker.enqueue(message)
                self.trace_event(trace, "memory_enqueued", source="bot_message")
                self.trace_event(
                    trace,
                    "turn_completed",
                    duration_ms=int((time.perf_counter() - turn_started) * 1000),
                    replied=False,
                    bot_message=True,
                )
                return

            # 4. Voice pause: cache + memory enqueue, skip reply
            if self.voice.text_replies_paused():
                self.runtime_state.update_turn_stage(trace, "memory_enqueue")
                paused_content = self._add_message_to_cache(message, [])
                await self.memory_worker.enqueue(message)
                self.trace_event(
                    trace,
                    "text_reply_paused_for_voice",
                    voice_channel=self.voice.active_session.channel_name if self.voice.active_session else None,
                    content_chars=len(paused_content),
                )
                self.trace_event(
                    trace,
                    "turn_completed",
                    duration_ms=int((time.perf_counter() - turn_started) * 1000),
                    replied=False,
                    paused_for_voice=True,
                )
                return

            # 5. Attachment prep + cheap vision router caption
            vision_started = time.perf_counter()
            self.runtime_state.update_turn_stage(trace, "vision_router")
            prepared_attachments = await self.prepare_attachments(message)
            attachment_result = await self.describe_prepared_attachments(prepared_attachments, detail=False)
            self.trace_event(
                trace,
                "vision_router_completed",
                duration_ms=int((time.perf_counter() - vision_started) * 1000),
                image_count=len(attachment_result.descriptions),
                fallback_images=attachment_result.fallback_count,
                fallback_reasons=attachment_result.fallback_reasons or None,
            )
            history = self.cache.get(message.guild.id, message.channel.id)
            bouncer_context = build_bouncer_context(history, message, attachment_result.descriptions)
            forensic_event(
                trace,
                "vision_router_artifacts",
                descriptions=attachment_result.descriptions,
                fallback_count=attachment_result.fallback_count,
                fallback_reasons=attachment_result.fallback_reasons or None,
                bouncer_context=bouncer_context,
            )

            # 6. Bouncer decision
            bouncer_result = await run_bouncer(
                self.llm,
                bouncer_context=bouncer_context,
                bot_user=bot_user,
                trace=trace,
                runtime_state=self.runtime_state,
            )

            # 7. Detailed vision (only if responding + has images)
            final_attachment_result = attachment_result
            if bouncer_result.should_respond and prepared_attachments.attachments:
                detail_started = time.perf_counter()
                self.runtime_state.update_turn_stage(trace, "vision_detail")
                final_attachment_result = await self.describe_prepared_attachments(
                    prepared_attachments,
                    detail=True,
                )
                self.trace_event(
                    trace,
                    "vision_detail_completed",
                    duration_ms=int((time.perf_counter() - detail_started) * 1000),
                    image_count=len(final_attachment_result.descriptions),
                    fallback_images=final_attachment_result.fallback_count,
                    fallback_reasons=final_attachment_result.fallback_reasons or None,
                )
                forensic_event(
                    trace,
                    "vision_detail_artifacts",
                    descriptions=final_attachment_result.descriptions,
                    fallback_count=final_attachment_result.fallback_count,
                    fallback_reasons=final_attachment_result.fallback_reasons or None,
                )
            else:
                self.trace_event(
                    trace,
                    "vision_detail_completed",
                    status="skipped",
                    skipped_reason="no_reply" if not bouncer_result.should_respond else "no_attachments",
                    image_count=0,
                )

            # 8. Cache + memory enqueue (always, before reply attempt)
            rag_query_text = self._add_message_to_cache(message, final_attachment_result.descriptions)
            history = self.cache.get(message.guild.id, message.channel.id)
            self.runtime_state.update_turn_stage(trace, "memory_enqueue")
            await self.memory_worker.enqueue(message, image_descriptions=final_attachment_result.descriptions)
            self.trace_event(trace, "memory_enqueued", source="user_message")

            # 9-12. Reply pipeline (only if bouncer said yes)
            if bouncer_result.should_respond:
                async with message.channel.typing():
                    # 9. Tool dispatch
                    tool_context = await run_tool_dispatch(
                        self.tools_module,
                        message=message,
                        bouncer_result=bouncer_result,
                        trace=trace,
                        runtime_state=self.runtime_state,
                    )

                    # 10. RAG retrieval
                    ollama_history = history.to_ollama_messages(bot_user.id)
                    rag_context = await run_retrieval(
                        self.vector_memory,
                        rag_query_text=rag_query_text,
                        server_id=message.guild.id,
                        ollama_history=ollama_history,
                        recommended_tool=bouncer_result.recommended_tool,
                        trace=trace,
                        runtime_state=self.runtime_state,
                    )

                    # 11. Brain generation + finalization
                    reply = await run_brain(
                        self.llm,
                        ollama_history=ollama_history,
                        bot_user=bot_user,
                        message=message,
                        rag_context=rag_context,
                        tool_context=tool_context,
                        trace=trace,
                        runtime_state=self.runtime_state,
                    )

                    # 12. Send reply
                    if reply:
                        try:
                            send_started = time.perf_counter()
                            self.runtime_state.update_turn_stage(trace, "reply_send")
                            self.trace_event(trace, "reply_send_started", reply_chars=len(reply))
                            sent_parts = await self.send_reply(message, reply)
                            self.trace_event(
                                trace,
                                "reply_send_completed",
                                duration_ms=int((time.perf_counter() - send_started) * 1000),
                                reply_chars=len(reply),
                                message_parts=sent_parts,
                            )
                            forensic_event(
                                trace,
                                "reply_delivery",
                                finalized_reply=reply,
                                message_parts=sent_parts,
                            )
                            logger.info(
                                "Brain replied in %s/%s (%d chars, %d message(s))",
                                message.guild.name,
                                message.channel.name,
                                len(reply),
                                sent_parts,
                            )
                            replied = True
                        except Exception:
                            self.trace_event(trace, "reply_send_completed", status="error")
                            logger.exception(
                                "Reply send failed in %s/%s after generation",
                                message.guild.name,
                                message.channel.name,
                            )

            # 13. Turn completion
            self.runtime_state.update_turn_stage(trace, "turn_completed")
            self.trace_event(
                trace,
                "turn_completed",
                duration_ms=int((time.perf_counter() - turn_started) * 1000),
                replied=replied,
            )
        finally:
            self.runtime_state.end_turn(trace.trace_id)

    async def shutdown(self) -> None:
        await self.voice.shutdown()
        await self.memory_worker.shutdown()
