"""Canonical model-layer types.

Internally woyo speaks ONE message format (≈ OpenAI shape). Providers convert
at their boundary. Tool arguments stay raw JSON strings until the ToolRegistry
validates them against the tool's pydantic model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: str  # raw JSON object


class Message(BaseModel):
    role: str  # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # role=tool: which call this answers
    name: str | None = None  # role=tool: tool name


class ToolSpec(BaseModel):
    """Tool description handed to the model (provider-converted)."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls


@dataclass(slots=True)
class ModelResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    model: str | None = None


def estimate_tokens(text: str | None) -> int:
    """Rough token estimate (chars/4). Only used for budget heuristics."""
    if not text:
        return 0
    return len(text) // 4


class ModelProvider(Protocol):
    """Everything the router needs from a model backend."""

    name: str

    async def complete(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse: ...
