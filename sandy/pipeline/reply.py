"""Reply splitting, truncation cleanup, and Discord delivery."""

import discord

from ..logconf import get_logger

logger = get_logger("sandy.bot")

_DISCORD_MESSAGE_LIMIT = 2000


def split_reply(reply: str, limit: int = _DISCORD_MESSAGE_LIMIT) -> list[str]:
    if len(reply) <= limit:
        return [reply]

    chunks: list[str] = []
    remaining = reply.strip()

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        split_at = remaining.rfind("\n\n", 0, limit + 1)
        if split_at == -1:
            split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at == -1:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at == -1 or split_at < limit // 2:
            split_at = limit

        chunk = remaining[:split_at].strip()
        if not chunk:
            chunk = remaining[:limit]
            split_at = limit

        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    return chunks


async def send_reply(message: discord.Message, reply: str) -> int:
    parts = split_reply(reply)
    if len(parts) > 1:
        logger.warning(
            "Reply exceeded Discord limit (%d chars) - sending %d chunks",
            len(reply),
            len(parts),
        )

    for part in parts:
        await message.channel.send(part)
    return len(parts)
