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
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from woyo.agent import Agent
from woyo.config import Settings
from woyo.errors import AgentError
from woyo.events import EventBus
from woyo.memory.session import SessionStore
from woyo.models.router import ModelRouter
from woyo.tools.base import ToolRegistry
from woyo.tools.builtin import build_default_registry
from woyo.tools.builtin.file_transfer import FileSender
from woyo.tools.builtin.telegram_tools import TelegramTransport

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


class _RecordingTransport:
    """Wraps a TelegramTransport so targeted sends land in `last_turn` facts.

    The next turn's prompt then answers "did you send it to the group?"
    from ground truth instead of from whatever the previous reply claimed.
    """

    def __init__(self, inner: TelegramTransport, session: ChatSession):
        self._inner = inner
        self._session = session

    @property
    def current_chat_id(self) -> int:
        return self._inner.current_chat_id

    async def send_message(self, chat_id: int, text: str) -> None:
        await self._inner.send_message(chat_id, text)
        self._session._sent_messages.append(
            f"chat {chat_id}: {text[:80]}"
        )

    async def send_document(
        self, chat_id: int, path: Path, caption: str
    ) -> None:
        await self._inner.send_document(chat_id, path, caption)
        self._session._sent_files.append(
            f"{Path(path).name} -> chat {chat_id}"
        )

    def known_chats(self) -> list[dict]:
        return self._inner.known_chats()

    async def chat_info(self, chat_id: int) -> dict:
        return await self._inner.chat_info(chat_id)

    def is_known(self, chat_id: int) -> bool:
        return self._inner.is_known(chat_id)


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
        environment: str | Callable[[], str] | None = None,
        telegram_transport: TelegramTransport | None = None,
        job_store=None,  # woyo.store.jobs.JobStore — enables the scheduler
        chat_id: int | None = None,  # numeric chat id for scheduler ownership
    ):
        self.key = key
        self.base_settings = settings
        self.settings = chat_settings(settings)
        self.store = store or SessionStore(settings.sessions_path())
        self.chats_dir = Path(chats_dir).expanduser()
        self.file_sender = file_sender
        self.environment = environment
        self.telegram_transport = telegram_transport
        self.job_store = job_store
        self.chat_id = chat_id
        if agent_factory is None and (
            file_sender is not None
            or telegram_transport is not None
            or job_store is not None
        ):
            self._agent_factory = self._chat_tools_factory()
        else:
            self._agent_factory = agent_factory or _default_agent_factory

        self.history: list[tuple[str, str]] = []
        self.session_id = self.store.new_session_id()
        self.total_messages = 0
        self.total_cost_usd = 0.0
        self._day = ""
        self._day_count = 0
        #: ground truth about the previous turn (kills "still working" hallucinations)
        self.last_turn: dict | None = None
        self._sent_files: list[str] = []  # files delivered during the current turn
        self._sent_messages: list[str] = []  # targeted sends during the current turn
        #: the REAL tool list, stashed by the factory for the environment block
        self._tool_names: list[str] | None = None
        self._load()
        self._roll_day()

    def _chat_tools_factory(self):
        """Default factory + the chat's wired tools (send_file, telegram,
        scheduler).

        Wrappers record deliveries so `last_turn` — the ground truth the
        next turn's prompt includes — knows what actually left the building.
        The registry's tool names are stashed on the session so the
        environment block can state the agent's REAL capabilities.
        """
        session = self

        recording_sender: FileSender | None = None
        if self.file_sender is not None:
            sender = self.file_sender

            async def recording_sender_impl(path: Path, caption: str) -> None:
                await sender(path, caption)
                session._sent_files.append(Path(path).name)

            recording_sender = recording_sender_impl

        recording_transport: TelegramTransport | None = None
        if self.telegram_transport is not None:
            recording_transport = _RecordingTransport(
                self.telegram_transport, self
            )

        def factory(settings: Settings, bus: EventBus) -> Agent:
            from woyo.memory.longterm import build_memory_from_settings

            memory = build_memory_from_settings(settings)
            router = ModelRouter(settings, bus=bus)
            registry: ToolRegistry = build_default_registry(
                settings, bus=bus, memory=memory, router=router,
                file_sender=recording_sender,
                telegram=recording_transport,
                job_store=session.job_store,
                chat_id=session.chat_id,
            )
            session._tool_names = registry.names()
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
        self._sent_files = []
        self._sent_messages = []

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
        self.last_turn = {
            "ts": datetime.now(UTC).isoformat(),
            "outcome": result.outcome,
            "tool_calls": result.usage.tool_calls,
            "duration_s": round(result.duration_s, 1),
            "files_sent": list(self._sent_files),
            "messages_sent": list(self._sent_messages),
            "reply_preview": (result.final_answer or "")[:200],
        }
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
    def _promises_rule(self) -> str:
        """The follow-through rule, adjusted to what is actually wired."""
        if self.job_store is not None:
            return (
                "CRITICAL — KEEPING PROMISES: nothing runs after you reply, "
                "EXCEPT tasks committed with the schedule_task tool — those "
                "run by themselves at their scheduled time and automatically "
                "report the result (or the failure) back in this chat. So \"I "
                "will do it later\" is ONLY real as a schedule_task call made "
                "in this same turn; words alone execute nothing. Either do "
                "the work now with tool calls, commit it with schedule_task, "
                "or explain exactly what you need to proceed."
            )
        return (
            "CRITICAL — NO BACKGROUND EXECUTION: nothing runs after you "
            "reply; the conversation simply waits for the user's next "
            "message. So NEVER promise to do something later ('I'll send it "
            "in a moment', 'I'm still working on it') unless you are doing "
            "it RIGHT NOW with tool calls in this turn. Either do the work "
            "now, or explain exactly what you need to proceed."
        )

    def _environment_text(self) -> str:
        """Facts about WHERE the agent runs — frontend-provided + real tools."""
        env = self.environment() if callable(self.environment) else self.environment
        parts = [str(env).strip()] if env and str(env).strip() else []
        if self._tool_names:
            parts.append(
                "Your complete, REAL tool list (these and only these exist — "
                f"{len(self._tool_names)} tools):\n"
                + ", ".join(sorted(self._tool_names))
            )
        if parts:
            return (
                "ENVIRONMENT (ground truth about where and what you are — "
                "trust this over any assumption or training memory):\n"
                + "\n\n".join(parts)
                + "\nNever claim you lack a capability that is in the list "
                "above; never claim you used a tool that is not. If a task "
                "needs something the list does not offer, say plainly what "
                "is missing instead of inventing it."
            )
        return ""

    def _frame_task(self, message: str) -> str:
        preamble = (
            "You are woyo, a capable AI agent, chatting with the user. "
            "Reply to the newest message. Use your tools when they genuinely "
            "help (search, fetch pages, calculate, create documents); skip "
            "them for small talk. Keep replies compact, conversational and "
            "in the user's language.\n"
            + self._promises_rule()
            + " When the user asks whether something finished, answer from "
              "the LAST TURN FACTS below (ground truth), never from promises "
              "in earlier replies — admit plainly if nothing actually ran."
        )
        if self.file_sender is not None:
            preamble += (
                " You can deliver files: create them with create_document "
                "(PDFs — it shapes Arabic/RTL and other scripts correctly), "
                "write_file or python_exec, then call send_file with the "
                "workspace-relative path (this chat) or send_document to any "
                "other chat you are in. Send the file BEFORE your final "
                "text reply, in the same turn."
            )
        environment = self._environment_text()
        if environment:
            preamble += f"\n\n{environment}"
        turns = self.history[-self.settings.chat_history_turns :]
        facts = self._last_turn_facts()
        if not turns:
            body = f"{preamble}\n\n{facts}\n\nUSER MESSAGE:\n{message}" if facts else (
                f"{preamble}\n\nUSER MESSAGE:\n{message}"
            )
            return body
        transcript = "\n".join(
            f"{'User' if role == 'user' else 'woyo'}: {text[:2000]}" for role, text in turns
        )
        middle = f"\n\n{facts}\n" if facts else ""
        return (
            f"{preamble}\n\nTRANSCRIPT (oldest first):\n{transcript}\n"
            f"{middle}\nNEW MESSAGE:\n{message}"
        )

    def _last_turn_facts(self) -> str:
        """Ground truth about the previous turn — injected into every task."""
        lt = self.last_turn
        if not lt:
            return ""
        try:
            ts = datetime.fromisoformat(lt["ts"])
            age_min = max(0.0, (datetime.now(UTC) - ts).total_seconds() / 60)
        except (KeyError, ValueError, TypeError):
            age_min = 0.0
        age = (
            f"{age_min:.0f} min" if age_min >= 1 else f"{age_min * 60:.0f}s"
        )
        files = lt.get("files_sent") or []
        files_txt = ", ".join(files) if files else "none"
        msgs = lt.get("messages_sent") or []
        msgs_txt = ("; ".join(msgs[:5])) if msgs else "none"
        preview = str(lt.get("reply_preview", "")).replace("\n", " ")[:120]
        facts = (
            f"LAST TURN FACTS (ground truth — no work is running right now, "
            f"nothing happens between messages except scheduled tasks):\n"
            f"- {age} ago your previous turn ended: "
            f"outcome={lt.get('outcome', '?')}, "
            f"{lt.get('tool_calls', 0)} tool call(s), "
            f"files sent: {files_txt}, messages sent elsewhere: {msgs_txt}.\n"
            f"- its reply began: \"{preview}\"\n"
            f"- If the user asks 'did you finish?': judge ONLY by these "
            f"facts. If you promised something and these facts show it "
            f"didn't happen, apologize briefly and DO IT NOW with tools."
        )
        if self.job_store is not None:
            try:
                active = self.job_store.active_for_chat(self.chat_id or 0)
            except Exception:  # noqa: BLE001 — facts must never break framing
                active = []
            if active:
                due_list = "; ".join(
                    f"#{j.id} '{j.title}' at {j.run_at}" for j in active[:6]
                )
                facts += (
                    f"\n- Scheduled commitments still pending (they WILL run "
                    f"on their own): {due_list}. If the user asks about "
                    f"them, they are real — unlike reply-text promises."
                )
        return facts

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
        lt = data.get("last_turn")
        self.last_turn = lt if isinstance(lt, dict) else None

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
                        "last_turn": self.last_turn,
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
