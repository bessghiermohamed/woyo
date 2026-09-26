"""Session transcripts: append-only jsonl under ~/.woyo/sessions/."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from woyo.models.base import Message


class SessionStore:
    def __init__(self, base_dir: str | Path = "~/.woyo/sessions"):
        self.base = Path(base_dir).expanduser()
        self.base.mkdir(parents=True, exist_ok=True)

    def new_session_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def events_path(self, session_id: str) -> Path:
        return self.base / f"{session_id}.events.jsonl"

    def transcript_path(self, session_id: str) -> Path:
        return self.base / f"{session_id}.transcript.jsonl"

    def append_events(self, session_id: str, events: list[dict]) -> Path:
        path = self.events_path(session_id)
        with path.open("a", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e, default=str) + "\n")
        return path

    def append_messages(self, session_id: str, messages: list[Message]) -> Path:
        path = self.transcript_path(session_id)
        with path.open("a", encoding="utf-8") as fh:
            for m in messages:
                fh.write(
                    json.dumps(
                        {
                            "role": m.role,
                            "content": (m.content or "")[:100_000],
                            "tool_calls": (
                                [tc.model_dump() for tc in m.tool_calls]
                                if m.tool_calls
                                else None
                            ),
                            "tool_call_id": m.tool_call_id,
                            "name": m.name,
                        },
                        default=str,
                    )
                    + "\n"
                )
        return path

    def list_sessions(self) -> list[dict]:
        out = []
        for path in sorted(self.base.glob("*.events.jsonl")):
            out.append(
                {
                    "session_id": path.name.removesuffix(".events.jsonl"),
                    "events_file": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        return out
