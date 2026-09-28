"""Serializable run state: what the agent loop needs to resume mid-task.

The loop keeps everything it needs to continue (messages, counters,
budgets spent, observed URLs, loop-detection table) in one dataclass
that round-trips through JSON. A checkpoint written by a dead process
is rehydrated by the next one — Phase 3's "task survives restart".
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from woyo.models.base import Message

_MAX_MSG_CHARS = 60_000  # cap per message when serializing (tool outputs cap first)


def _messages_to_json(messages: list[Message]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        out.append(
            {
                "role": m.role,
                "content": (m.content or "")[:_MAX_MSG_CHARS],
                "tool_calls": [tc.model_dump() for tc in m.tool_calls] if m.tool_calls else None,
                "tool_call_id": m.tool_call_id,
                "name": m.name,
            }
        )
    return out


def _messages_from_json(raw: list[dict[str, Any]]) -> list[Message]:
    from woyo.models.base import ToolCall

    out = []
    for item in raw:
        tool_calls = None
        if item.get("tool_calls"):
            tool_calls = [ToolCall(**tc) for tc in item["tool_calls"]]
        out.append(
            Message(
                role=item["role"],
                content=item.get("content") or "",
                tool_calls=tool_calls,
                tool_call_id=item.get("tool_call_id"),
                name=item.get("name"),
            )
        )
    return out


class RunState:
    """Mutable loop state; to_dict/from_dict give the checkpoint format."""

    __slots__ = (
        "task", "system_prompt", "plan_text",
        "messages", "steps", "tool_calls", "search_calls",
        "consecutive_failures", "text_only_strikes",
        "repeat_counter", "observed_urls", "elapsed_s",
    )

    def __init__(
        self,
        task: str,
        *,
        system_prompt: str = "",
        plan_text: str = "",
        messages: list[Message] | None = None,
        steps: int = 0,
        tool_calls: int = 0,
        search_calls: int = 0,
        consecutive_failures: int = 0,
        text_only_strikes: int = 0,
        repeat_counter: Counter[tuple[str, str]] | None = None,
        observed_urls: set[str] | None = None,
        elapsed_s: float = 0.0,
    ):
        self.task = task
        self.system_prompt = system_prompt
        self.plan_text = plan_text
        self.messages: list[Message] = messages if messages is not None else []
        self.steps = steps
        self.tool_calls = tool_calls
        self.search_calls = search_calls
        self.consecutive_failures = consecutive_failures
        self.text_only_strikes = text_only_strikes
        self.repeat_counter: Counter[tuple[str, str]] = repeat_counter or Counter()
        self.observed_urls: set[str] = observed_urls or set()
        self.elapsed_s = elapsed_s

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "v": 1,
            "task": self.task,
            "system_prompt": self.system_prompt,
            "plan_text": self.plan_text,
            "messages": _messages_to_json(self.messages),
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "search_calls": self.search_calls,
            "consecutive_failures": self.consecutive_failures,
            "text_only_strikes": self.text_only_strikes,
            "repeat_counter": [[name, args, count]
                               for (name, args), count in self.repeat_counter.items()],
            "observed_urls": sorted(self.observed_urls),
            "elapsed_s": round(self.elapsed_s, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        counter: Counter[tuple[str, str]] = Counter()
        for name, args, count in data.get("repeat_counter", []):
            counter[(str(name), str(args))] = int(count)
        return cls(
            task=data["task"],
            system_prompt=data.get("system_prompt", ""),
            plan_text=data.get("plan_text", ""),
            messages=_messages_from_json(data.get("messages", [])),
            steps=int(data.get("steps", 0)),
            tool_calls=int(data.get("tool_calls", 0)),
            search_calls=int(data.get("search_calls", 0)),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            text_only_strikes=int(data.get("text_only_strikes", 0)),
            repeat_counter=counter,
            observed_urls=set(data.get("observed_urls", [])),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, raw: str) -> RunState:
        return cls.from_dict(json.loads(raw))
