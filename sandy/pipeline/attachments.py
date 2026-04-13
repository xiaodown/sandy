"""Attachment preparation, vision captioning, and augmented content building."""

import io
from dataclasses import dataclass

import discord
from PIL import Image

from ..last10 import (
    ChannelHistory,
    SyntheticMessage,
    _SyntheticAuthor,
    _SyntheticChannel,
    _SyntheticGuild,
    resolve_mentions,
)
from ..logconf import get_logger

logger = get_logger("sandy.bot")

_VISION_CONTENT_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
})
_MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass(slots=True)
class AttachmentProcessingResult:
    descriptions: list[str]
    fallback_count: int = 0
    fallback_reasons: list[str] | None = None


@dataclass(slots=True)
class PreparedAttachment:
    filename: str
    image_bytes: bytes | None = None
    fallback_description: str | None = None
    fallback_reason: str | None = None


@dataclass(slots=True)
class AttachmentPreparationResult:
    attachments: list[PreparedAttachment]
    fallback_count: int = 0
    fallback_reasons: list[str] | None = None


def _fallback_attachment_description(reason: str) -> str:
    return f"attached image could not be inspected because {reason}"


async def describe_attachments(message: discord.Message, llm) -> AttachmentProcessingResult:
    descriptions: list[str] = []
    fallback_reasons: list[str] = []
    for attachment in message.attachments:
        content_type = (attachment.content_type or "").split(";")[0].strip().lower()
        if content_type not in _VISION_CONTENT_TYPES:
            logger.debug(
                "Skipping attachment %s (type %s — not a supported image format)",
                attachment.filename, content_type or "unknown",
            )
            continue
        if attachment.size > _MAX_IMAGE_BYTES:
            logger.warning(
                "Skipping oversized image %s (%d MB)",
                attachment.filename, attachment.size // (1024 * 1024),
            )
            descriptions.append(_fallback_attachment_description("the file was too large"))
            fallback_reasons.append("oversized")
            continue
        try:
            image_bytes = await attachment.read()
        except Exception as exc:
            logger.error("Failed to download attachment %s: %s", attachment.filename, exc)
            descriptions.append(_fallback_attachment_description("it could not be downloaded"))
            fallback_reasons.append("download_failed")
            continue
        if content_type == "image/webp":
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=90)
                    image_bytes = buf.getvalue()
                logger.debug("Converted WebP→JPEG for %s", attachment.filename)
            except Exception as exc:
                logger.error("WebP conversion failed for %s: %s", attachment.filename, exc)
                descriptions.append(_fallback_attachment_description("it could not be processed"))
                fallback_reasons.append("conversion_failed")
                continue
        desc = await llm.ask_vision(image_bytes)
        if desc:
            descriptions.append(desc)
            logger.info(
                "Vision described %s: %s", attachment.filename, desc[:80] + ("…" if len(desc) > 80 else "")
            )
        else:
            logger.warning("Vision returned nothing for %s", attachment.filename)
            descriptions.append(_fallback_attachment_description("no description could be generated"))
            fallback_reasons.append("empty_description")
    return AttachmentProcessingResult(
        descriptions=descriptions,
        fallback_count=len(fallback_reasons),
        fallback_reasons=fallback_reasons,
    )


async def prepare_attachments(message: discord.Message) -> AttachmentPreparationResult:
    attachments: list[PreparedAttachment] = []
    fallback_reasons: list[str] = []
    for attachment in message.attachments:
        content_type = (attachment.content_type or "").split(";")[0].strip().lower()
        if content_type not in _VISION_CONTENT_TYPES:
            logger.debug(
                "Skipping attachment %s (type %s — not a supported image format)",
                attachment.filename, content_type or "unknown",
            )
            continue
        if attachment.size > _MAX_IMAGE_BYTES:
            logger.warning(
                "Skipping oversized image %s (%d MB)",
                attachment.filename, attachment.size // (1024 * 1024),
            )
            attachments.append(
                PreparedAttachment(
                    filename=attachment.filename,
                    fallback_description=_fallback_attachment_description("the file was too large"),
                    fallback_reason="oversized",
                )
            )
            fallback_reasons.append("oversized")
            continue
        try:
            image_bytes = await attachment.read()
        except Exception as exc:
            logger.error("Failed to download attachment %s: %s", attachment.filename, exc)
            attachments.append(
                PreparedAttachment(
                    filename=attachment.filename,
                    fallback_description=_fallback_attachment_description("it could not be downloaded"),
                    fallback_reason="download_failed",
                )
            )
            fallback_reasons.append("download_failed")
            continue
        if content_type == "image/webp":
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=90)
                    image_bytes = buf.getvalue()
                logger.debug("Converted WebP→JPEG for %s", attachment.filename)
            except Exception as exc:
                logger.error("WebP conversion failed for %s: %s", attachment.filename, exc)
                attachments.append(
                    PreparedAttachment(
                        filename=attachment.filename,
                        fallback_description=_fallback_attachment_description("it could not be processed"),
                        fallback_reason="conversion_failed",
                    )
                )
                fallback_reasons.append("conversion_failed")
                continue
        attachments.append(
            PreparedAttachment(
                filename=attachment.filename,
                image_bytes=image_bytes,
            )
        )
    return AttachmentPreparationResult(
        attachments=attachments,
        fallback_count=len(fallback_reasons),
        fallback_reasons=fallback_reasons or None,
    )


async def describe_prepared_attachments(prepared: AttachmentPreparationResult, llm, *, detail: bool) -> AttachmentProcessingResult:
    descriptions: list[str] = []
    fallback_reasons: list[str] = []
    ask = llm.ask_vision if detail else llm.ask_vision_router

    for prepared_attachment in prepared.attachments:
        if prepared_attachment.fallback_description is not None:
            descriptions.append(prepared_attachment.fallback_description)
            if prepared_attachment.fallback_reason is not None:
                fallback_reasons.append(prepared_attachment.fallback_reason)
            continue

        if prepared_attachment.image_bytes is None:
            descriptions.append(_fallback_attachment_description("it could not be processed"))
            fallback_reasons.append("missing_image_bytes")
            continue

        desc = await ask(prepared_attachment.image_bytes)
        if desc:
            descriptions.append(desc)
            logger.info(
                "Vision %s described %s: %s",
                "detail" if detail else "router",
                prepared_attachment.filename,
                desc[:80] + ("…" if len(desc) > 80 else ""),
            )
        else:
            logger.warning(
                "Vision %s returned nothing for %s",
                "detail" if detail else "router",
                prepared_attachment.filename,
            )
            descriptions.append(_fallback_attachment_description("no description could be generated"))
            fallback_reasons.append("empty_description")

    return AttachmentProcessingResult(
        descriptions=descriptions,
        fallback_count=len(fallback_reasons),
        fallback_reasons=fallback_reasons or None,
    )


def build_augmented_content(message: discord.Message, descriptions: list[str]) -> str:
    name = message.author.display_name
    original = resolve_mentions(message.content, message.mentions).strip()
    n = len(descriptions)

    if n == 1:
        desc = descriptions[0]
        if original:
            return f"{original}\n[{name} also attached an image: {desc}]"
        return f"[{name} pasted an image into the chat]\n[Image: {desc}]"

    image_lines = "\n".join(f"[Image {i}: {d}]" for i, d in enumerate(descriptions, 1))
    if original:
        return f"{original}\n[{name} also attached {n} images]\n{image_lines}"
    return f"[{name} pasted {n} images into the chat]\n{image_lines}"


def build_cache_message(
    message: discord.Message,
    descriptions: list[str],
) -> discord.Message | SyntheticMessage:
    if descriptions:
        return SyntheticMessage(
            content=build_augmented_content(message, descriptions),
            created_at=message.created_at,
            author=_SyntheticAuthor(
                id=message.author.id,
                display_name=message.author.display_name,
                bot=message.author.bot,
            ),
            guild=_SyntheticGuild(id=message.guild.id, name=message.guild.name),
            channel=_SyntheticChannel(id=message.channel.id, name=message.channel.name),
            mentions=message.mentions,
        )
    return message
