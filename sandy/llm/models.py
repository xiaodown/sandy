"""Structured output schemas for Sandy's LLM roles.

These Pydantic models define the JSON shapes that ollama produces via
constrained decoding (format= parameter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, field_validator, model_validator

from ..logconf import get_logger

logger = get_logger(__name__)


class BouncerResponse(BaseModel):
    """Structured output for the Bouncer role.

    Also includes tool recommendation: which tool (if any) to call before
    the brain generates a response.  Tool fields are ignored when
    should_respond is False.
    """
    should_respond: bool
    reason: str
    use_tool: bool = False
    recommended_tool: str | None = None
    tool_parameters: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _tool_fields_consistent(self) -> "BouncerResponse":
        """If the model set use_tool but forgot to name a tool, zero it out."""
        if self.use_tool and not self.recommended_tool:
            logger.warning(
                "Bouncer set use_tool=True but recommended_tool is empty — forcing use_tool=False",
            )
            self.use_tool = False
            self.tool_parameters = None
        return self


class TaggerResponse(BaseModel):
    """Structured output for the Tagger role."""
    tags: list[str]

    @field_validator("tags")
    @classmethod
    def normalise_tags(cls, v: list[str]) -> list[str]:
        """Lowercase, strip whitespace, drop empties, cap at 3."""
        cleaned = [t.lower().strip() for t in v if t.strip()]
        return cleaned[:3]


class SummarizerResponse(BaseModel):
    """Structured output for the Summarizer role."""
    summary: str


@dataclass
class BrainResponse:
    """Brain generation result plus completion metadata from Ollama."""

    content: str
    done_reason: str | None = None
    eval_count: int | None = None
