
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .logconf import get_logger

_PACIFIC = ZoneInfo("America/Los_Angeles")
_PROMPTS_DIR = Path(__file__).parent / "prompts"

logger = get_logger(__name__)


def _load(name: str) -> str:
    """Load a prompt text file from the prompts/ directory."""
    return (_PROMPTS_DIR / name).read_text()


@dataclass
class OllamaPrompt:
    """Container for an ollama chat prompt.

    system  — the SYSTEM role message (instructions / persona)
    user    — the USER role message (the actual input to reason over)
    """
    system: str
    user: str


class SandyPrompt:
    """Factory for all prompts used by Sandy's LLM roles.

    Every method is a @staticmethod that returns an OllamaPrompt.
    The caller is responsible for inserting the prompt messages into the
    correct ollama chat roles.
    """

    @staticmethod
    def brain_prompt(
        server_name: str = "the server",
        channel_name: str = "general",
    ) -> OllamaPrompt:
        """Main personality prompt for the Brain model."""
        system = _load("brain_system.txt")
        now = datetime.now(_PACIFIC).strftime("%Y-%m-%d %H:%M %Z")
        user = (
            f"The current time is {now}.\n"
            f"You are in channel {channel_name} in server {server_name}.\n\n"
            "You have read the recent messages in this channel and have decided to say something.\n"
            "Below are the conversation history, memory fragments, and other information you need "
            "in order to formulate a response."
        )
        return OllamaPrompt(system=system, user=user)

    @staticmethod
    def voice_brain_prompt(
        server_name: str = "the server",
        channel_name: str = "voice",
        participant_names: list[str] | None = None,
    ) -> OllamaPrompt:
        base = SandyPrompt.brain_prompt(
            server_name=server_name,
            channel_name=channel_name,
        )
        voice_addendum = _load("voice_addendum.txt")
        participants = ", ".join(participant_names or []) or "no one else right now"
        system = f"{base.system}\n\n{voice_addendum}"
        user = (
            f"The current time is {datetime.now(_PACIFIC).strftime('%Y-%m-%d %H:%M %Z')}.\n"
            f"You are in the live voice channel {channel_name} in server {server_name}.\n"
            f"People currently in the call: {participants}.\n"
            "You have the recent voice-session context, any relevant long-term memories, "
            "and the latest completed human turns."
        )
        return OllamaPrompt(system=system, user=user)

    @staticmethod
    def bouncer_prompt(context: str) -> OllamaPrompt:
        """Prompt for the Bouncer model.

        context  — the output of ChannelHistory.format(), oldest → newest,
                   with the last line being the message under consideration.
        """
        system = _load("bouncer_system.txt")
        user = (
            "Here is the recent channel history (oldest first, most recent last):\n\n"
            f"{context}\n\n"
            "Should Sandy respond to the most recent message? "
            "If responding, would a tool help Sandy give a better answer?"
        )
        return OllamaPrompt(system=system, user=user)

    @staticmethod
    def tagger_prompt(content: str) -> OllamaPrompt:
        """Prompt for the Tagger model."""
        system = _load("tagger_system.txt")
        user = f"Generate 1-3 tags for this Discord message:\n\n{content}"
        return OllamaPrompt(system=system, user=user)

    @staticmethod
    def summarize_prompt(content: str) -> OllamaPrompt:
        """Prompt for the Summarizer model."""
        system = _load("summarizer_system.txt")
        user = f"Summarise this Discord message in one sentence:\n\n{content}"
        return OllamaPrompt(system=system, user=user)

    @staticmethod
    def vision_router_prompt() -> OllamaPrompt:
        """Prompt for the cheap pre-bouncer vision caption."""
        system = _load("vision_router_system.txt")
        user = (
            "Give a terse caption for this image in about 12 to 24 words. "
            "If there is obvious readable text that matters, include it briefly."
        )
        return OllamaPrompt(system=system, user=user)

    @staticmethod
    def vision_detail_prompt() -> OllamaPrompt:
        """Prompt for the detailed vision grounding path."""
        system = _load("vision_detail_system.txt")
        user = (
            "Describe this image in enough detail for a Discord personality bot to understand what is "
            "happening, what stands out visually, and what text or small details might "
            "matter. Keep it compact, grounded, and under roughly 120 words."
        )
        return OllamaPrompt(system=system, user=user)


