"""Structured events: the observability backbone.

Everything the agent does becomes an Event. Subscribers (CLI live view,
future web UI over SSE) and a jsonl sink consume the same stream.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Canonical event kinds (stable contract for subscribers).
EVENT_KINDS = {
    "run_started",
    "plan_created",
    "plan_fallback",
    "llm_call",
    "tool_call",
    "tool_result",
    "approval_requested",
    "approval_result",
    "injection_flagged",
    "notice",
    "limit_hit",
    "run_finished",
    "error",
}


@dataclass(slots=True)
class Event:
    kind: str
    ts: float
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ts": round(self.ts, 3), **self.data}

    def __str__(self) -> str:
        return f"[{self.kind}] {json.dumps(self.data, default=str)[:400]}"


class EventBus:
    """Fan-out event distribution with an optional jsonl file sink."""

    def __init__(self, sink_path: str | Path | None = None):
        self._subscribers: list[Callable[[Event], None]] = []
        self._sink_path = Path(sink_path) if sink_path else None
        self.events: list[Event] = []

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        self._subscribers.append(callback)

    def emit(self, kind: str, **data: Any) -> Event:
        if kind not in EVENT_KINDS:
            # unknown kinds are allowed but flagged in the payload for review
            data = {"_unknown_kind": True, **data}
        event = Event(kind=kind, ts=time.time(), data=data)
        self.events.append(event)
        if self._sink_path is not None:
            try:
                self._sink_path.parent.mkdir(parents=True, exist_ok=True)
                with self._sink_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event.to_dict(), default=str) + "\n")
            except OSError:
                pass  # observability must never break the run
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:  # noqa: BLE001 — subscriber bugs must not kill runs
                pass
        return event
