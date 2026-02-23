"""
A discord interface for my AI chatbot
It should:
1.) come online and connect to the servers
2.) be able to receive and parse chat messages
3.) be able to send these messages elsewhere for decisions to be made about what to do with them
4.) be able to send meta info to trigger a "typing..." rich presence
5.) be able to send messages to a channel
6.) gracefully handle disconnects and reconnects
7.) all the above are required to be async

Probably more to come.

"""


import discord
import asyncio
import os
from dotenv import load_dotenv
import logging

from logconf import get_logger
from registry import Registry
from last10 import Last10, resolve_mentions
from memory import MemoryClient
from ollama_interface import OllamaInterface
import tools

load_dotenv()

logger = get_logger("discord_handler")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = discord.Client(intents=intents)
DISCORD_API_KEY = os.getenv("DISCORD_API_KEY")

# Suppress default discord logging
logging.getLogger("discord").propagate = False

registry = Registry()
cache = Last10(maxlen=10, registry=registry)
llm = OllamaInterface()
memory = MemoryClient(llm=llm)

# Guard so cache seeding only runs once, even if on_ready fires on reconnect.
_cache_seeded = False


@bot.event
async def on_ready():
    """Event handler for when the bot is ready."""
    global _cache_seeded
    logger.info("Logged in as %s (%s)", bot.user.name, bot.user.id)
    if not _cache_seeded:
        seeded = await memory.seed_cache(cache)
        logger.info("Cache seeded with %d message(s) from Recall", seeded)
        _cache_seeded = True


@bot.event
async def on_message(message: discord.Message):
    """Event handler for incoming messages."""
    # Ignore DMs — guild context is required for registry, cache, and memory.
    # DM support can be added later if needed.
    if message.guild is None:
        return

    # Update the registry (server / channel / user lookup cache)
    # Note: bot's own messages are included — they're part of the conversation context
    # registry.ensure_seen() uses sqlite3 (blocking I/O), so run it in a thread.
    asyncio.create_task(asyncio.to_thread(registry.ensure_seen, message))
    

    # Add to the short-term in-memory rolling cache
    cache.add(message)

    logger.info("[%s/%s] %s: %s", message.guild.name, message.channel.name, message.author.display_name, resolve_mentions(message.content, message.mentions))

    if message.author.bot:
        # Store bot messages (including Sandy's own replies) in Recall and last10
        # so they appear in conversation history, but skip bouncer/brain entirely.
        logger.debug("Bot message from %s — storing and skipping pipeline", message.author.display_name)
        asyncio.create_task(memory.process_and_store(message))
        return

    # --- Bouncer / Brain pipeline (awaited) ----------------------------
    # Ask the bouncer if Sandy should respond, given the recent context.
    # If yes, call the brain for a reply and send it.
    # Both calls hold the shared OllamaInterface lock, so they take priority
    # over the background tagging/summarization that fires afterward.
    history = cache.get(message.guild.id, message.channel.id)

    # Fast-path: if Sandy is directly named or @mentioned, skip the bouncer entirely.
    # This catches "hey Sandy", "sandy can you...", "@sandy-test", etc. deterministically
    # without relying on the small model to get it right.
    should_respond = await llm.ask_bouncer(history.format(), bot_name=bot.user.display_name)

    if should_respond:
        # Show "Sandy is typing..." for the entire duration of LLM generation.
        # channel.typing() is an async context manager that sends the indicator
        # and refreshes it every 5 seconds automatically until the block exits.
        async with message.channel.typing():
            ollama_history = history.to_ollama_messages(bot.user.id)
            reply = await llm.ask_brain(
                ollama_history,
                bot_name=bot.user.display_name,
                server_name=message.guild.name,
                channel_name=message.channel.name,
                server_id=message.guild.id,
                tools=tools.TOOL_SCHEMAS,
                send_fn=message.channel.send,
            )

            if reply:
                await message.channel.send(reply)
                logger.info("Brain replied in %s/%s (%d chars)",
                            message.guild.name, message.channel.name, len(reply))
            else:
                logger.warning("Brain returned None for message in %s/%s — not sending",
                               message.guild.name, message.channel.name)

    # --- Memory (fire-and-forget after pipeline) -----------------------
    # Now that the LLM is free, tag and store the message in the background.
    # process_and_store will await the lock, so if another message's pipeline
    # fires concurrently the tagger/summarizer will queue behind it cleanly.
    asyncio.create_task(memory.process_and_store(message))


if __name__ == "__main__":
    async def main():
        try:
            await bot.start(DISCORD_API_KEY)
        except discord.LoginFailure as e:
            print(f"Failed to log in: {e}")
        except discord.DiscordException as e:
            print(f"A Discord-related error occurred: {e}")
        finally:
            await bot.close()
            await memory.close()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Ctrl+C — clean shutdown already handled in main()'s finally block