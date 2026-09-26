"""Deterministic mock provider — makes the whole agent testable without keys."""

from __future__ import annotations

import itertools
import json
from copy import deepcopy
from typing import Any

from woyo.models.base import Message, ModelResponse, ToolCall, ToolSpec, Usage

_ids = itertools.count(1)


def text_response(text: str, usage: tuple[int, int] = (100, 50)) -> ModelResponse:
    return ModelResponse(
        content=text, usage=Usage(*usage), finish_reason="stop", model="mock"
    )


def tool_response(calls: list[tuple[str, dict[str, Any]]]) -> ModelResponse:
    """Build a model response that requests the given tool calls."""
    return ModelResponse(
        content=None,
        tool_calls=[
            ToolCall(id=f"call-{next(_ids)}", name=name, arguments=json.dumps(args))
            for name, args in calls
        ],
        usage=Usage(120, 60),
        finish_reason="tool_calls",
        model="mock",
    )


class MockProvider:
    """Pops scripted responses in order; records every request for assertions."""

    name = "mock"

    def __init__(self, responses: list[ModelResponse | Exception]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    @property
    def exhausted(self) -> bool:
        return not self._responses

    async def complete(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSpec],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "tool_names": [t.name for t in tools],
                "model": model,
            }
        )
        if not self._responses:
            return text_response("(mock exhausted: no more scripted responses)")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
