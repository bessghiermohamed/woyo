"""Chat session core shared by every frontend (Telegram, web).

A chat is a sequence of agent runs that share a rolling transcript:
each user message becomes a task framed with the recent conversation,
so tools, budgets and citation verification apply per message (ADR-10).

Design notes:
- The agent is rebuilt per message: budgets are per message, never
  inherited from a long-lived router that would exhaust mid-chat.
- Transcript history is persisted as JSON under ~/.woyo/chats/ so a
  restarted bot resumes conversations (a first slice of Phase 3).
- No approval channel exists in unattended chat, so external-write
  tools fail safe (denied) — same rule as headless `woyo run`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from woyo.agent import Agent
from woyo.config import Settings
from woyo.errors import AgentError
from woyo.events import EventBus
from woyo.memory.session import SessionStore
from woyo.models.router import ModelRouter
from woyo.tools.builtin import build_default_registry
from woyo.tools.builtin.file_transfer import FileSender

#: Tighter budget profile for interactive chat (applies per message).
CHAT_PROFILE: dict[str, object] = {
    "max_steps": 8,
    "max_tool_calls": 12,
    "max_time_s": 150.0,
    "max_tokens": 60_000,
    "max_cost_usd": 0.30,
    "max_search_calls": 6,
    "tool_output_cap_chars": 8_000,
    "direct_text_replies": True,  # conversational answers don't need the finish tool
}

_HISTORY_KEEP = 50  # turns persisted to disk (context window is smaller)


def chat_settings(base: Settings) -> Settings:
    """A copy of `base` with the interactive chat budget profile."""
    return base.model_copy(update=dict(CHAT_PROFILE))


def _safe_key(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", key)[:80] or "chat"


@dataclass(slots=True)
class ChatReply:
    """What a frontend receives for one user message."""

    answer: str
    verified: bool = False
    sources: list[dict[str, str]] = field(default_factory=list)
    sources_dropped: list[str] = field(default_factory=list)
    steps: int = 0
    tool_calls: int = 0
    duration_s: float = 0.0
    cost_usd_est: float = 0.0
    outcome: str = "completed"

    def render(self, *, with_sources: bool = True) -> str:
        """Plain-text rendering with an optional verified-sources footer."""
        text = self.answer.rstrip()
        if not with_sources:
            return text
        lines = [text]
        if self.sources:
            seen: list[str] = []
            for s in self.sources:
                url = s.get("url", "")
                if url and url not in seen:
                    seen.append(url)
            if seen:
                lines.append("")
                lines.append("Sources:")
                lines += [f"- {u}" for u in seen[:6]]
        if self.sources_dropped:
            lines.append(
                f"({len(self.sources_dropped)} cited source(s) could not be "
                "verified and were removed)"
            )
        return "\n".join(lines)


AgentFactory = Callable[[Settings, EventBus], Agent]


def _default_agent_factory(settings: Settings, bus: EventBus) -> Agent:
    from woyo.memory.longterm import build_memory_from_settings

    memory = build_memory_from_settings(settings)
    router = ModelRouter(settings, bus=bus)
    registry = build_default_registry(
        settings, bus=bus, memory=memory, router=router
    )
    return Agent(settings, router, registry, bus=bus, memory=memory)


class ChatSession:
    """One conversation: rolling transcript + per-message agent runs."""

    def __init__(
        self,
        key: str,
        settings: Settings,
        *,
        store: SessionStore | None = None,
        chats_dir: str | Path = "~/.woyo/chats",
        agent_factory: AgentFactory | None = None,
        file_sender: FileSender | None = None,
    ):
        self.key = key
        self.base_settings = settings
        self.settings = chat_settings(settings)
        self.store = store or SessionStore(settings.sessions_path())
        self.chats_dir = Path(chats_dir).expanduser()
        self.file_sender = file_sender
        if agent_factory is None and file_sender is not None:
            self._agent_factory = self._sender_aware_factory()
        else:
            self._agent_factory = agent_factory or _default_agent_factory

        self.history: list[tuple[str, str]] = []
        self.session_id = self.store.new_session_id()
        self.total_messages = 0
        self.total_cost_usd = 0.0
        self._day = ""
        self._day_count = 0
        self._load()
        self._roll_day()

    def _sender_aware_factory(self):
        """Default factory + the chat's file sender (enables send_file)."""
        sender = self.file_sender

        def factory(settings: Settings, bus: EventBus) -> Agent:
            from woyo.memory.longterm import build_memory_from_settings

            memory = build_memory_from_settings(settings)
            router = ModelRouter(settings, bus=bus)
            registry = build_default_registry(
                settings, bus=bus, memory=memory, router=router, file_sender=sender
            )
            return Agent(settings, router, registry, bus=bus, memory=memory)

        return factory

    # ------------------------------------------------------------------
    async def send(
        self, message: str, *, approval_cb=None, images: list[str] | None = None
    ) -> ChatReply:
        """Process one user message through a fresh agent run.

        `approval_cb` (optional, sync or async) is offered to the agent for
        tools that act externally — e.g. the Telegram frontend shows inline
        Approve/Deny buttons and waits for the press. Without a channel the
        loop fails safe (deny), same as headless runs.

        `images` (optional) are data URIs attached to this message for
        vision-capable models (the router picks one when configured).
        """
        message = message.strip()
        if not message:
            raise AgentError("empty message")
        if self._day_count >= self.settings.chat_daily_messages:
            raise AgentError(
                "daily message limit reached "
                f"({self.settings.chat_daily_messages}); the counter resets tomorrow"
            )

        bus = EventBus()
        agent = self._agent_factory(self.settings, bus)
        result = await agent.run(
            self._frame_task(message), approval_cb=approval_cb, images=images
        )
        self.store.append_events(self.session_id, result.events_log)

        self.history.append(("user", message))
        self.history.append(("woyo", result.final_answer))
        self.history = self.history[-_HISTORY_KEEP:]
        self.total_messages += 1
        self.total_cost_usd += result.usage.cost_usd_est
        self._day_count += 1
        self._save()
        return ChatReply(
            answer=result.final_answer,
            verified=result.verified,
            sources=result.sources,
            sources_dropped=result.sources_dropped,
            steps=result.steps,
            tool_calls=result.usage.tool_calls,
            duration_s=result.duration_s,
            cost_usd_est=result.usage.cost_usd_est,
            outcome=result.outcome,
        )

    def reset(self) -> str:
        """Start a fresh session (keeps nothing but the identity)."""
        self.history = []
        self.session_id = self.store.new_session_id()
        self._save()
        return self.session_id

    def status_line(self) -> str:
        return (
            f"messages: {self.total_messages} · today: {self._day_count}/"
            f"{self.settings.chat_daily_messages} · est. cost so far: "
            f"${self.total_cost_usd:.4f} · provider: {self.base_settings.provider} "
            f"({self.base_settings.model})"
        )

    # ------------------------------------------------------------------
    def _frame_task(self, message: str) -> str:
        preamble = (
            "You are woyo, a capable AI agent, chatting with the user. "
            "Reply to the newest message. Use your tools when they genuinely help "
            "(search, fetch pages, calculate); skip them for small talk. "
            "Keep replies compact, conversational and in the user's language."
        )
        if self.file_sender is not None:
            preamble += (
                " You can deliver files: create them in the workspace with "
                "write_file or python_exec, then call send_file with the "
                "workspace-relative path."
            )
        turns = self.history[-self.settings.chat_history_turns :]
        if not turns:
            return f"{preamble}\n\nUSER MESSAGE:\n{message}"
        transcript = "\n".join(
            f"{'User' if role == 'user' else 'woyo'}: {text[:2000]}" for role, text in turns
        )
        return (
            f"{preamble}\n\nTRANSCRIPT (oldest first):\n{transcript}\n\n"
            f"NEW MESSAGE:\n{message}"
        )

    # ------------------------------------------------------------------
    def _state_path(self) -> Path:
        return self.chats_dir / f"{_safe_key(self.key)}.json"

    def _load(self) -> None:
        try:
            data = json.loads(self._state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if data.get("key") != self.key:
            return
        self.history = [
            (str(r), str(t)) for r, t in data.get("history", []) if isinstance(t, str)
        ][-_HISTORY_KEEP:]
        if data.get("session_id"):
            self.session_id = str(data["session_id"])
        self.total_messages = int(data.get("total_messages", 0))
        self.total_cost_usd = float(data.get("total_cost_usd", 0.0))
        self._day = str(data.get("day", ""))
        self._day_count = int(data.get("day_count", 0))

    def _save(self) -> None:
        try:
            self.chats_dir.mkdir(parents=True, exist_ok=True)
            self._state_path().write_text(
                json.dumps(
                    {
                        "key": self.key,
                        "session_id": self.session_id,
                        "history": self.history,
                        "total_messages": self.total_messages,
                        "total_cost_usd": self.total_cost_usd,
                        "day": self._day,
                        "day_count": self._day_count,
                        "saved_at": datetime.now(ZoneInfo("UTC")).isoformat(),
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            pass  # persistence is best-effort; the chat still works

    def _roll_day(self) -> None:
        try:
            today = datetime.now(ZoneInfo(self.base_settings.timezone)).date().isoformat()
        except Exception:  # noqa: BLE001 — bad tz config must not kill the chat
            today = datetime.now(ZoneInfo("UTC")).date().isoformat()
        if today != self._day:
            self._day = today
            self._day_count = 0
