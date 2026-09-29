"""v0.8 proactivity: environment awareness, Telegram reach, scheduled jobs.

Covers the three pillars shipped together:
- awareness — the agent is told WHERE it runs and what its real tools are
- reach — addressable Telegram sends with an allowlist + approval gate
- follow-through — schedule_task commitments execute, report, and survive
  rotation via the jobs table in woyo.sqlite3
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from tests.conftest import PLAN_JSON, build_agent, make_settings
from woyo.chat.session import ChatReply, ChatSession
from woyo.chat.telegram import TelegramBot, _BotTransport
from woyo.errors import ErrorKind
from woyo.models.mock import text_response, tool_response
from woyo.store.db import connect, migrate
from woyo.store.jobs import JobStore, describe_run_at, now_iso
from woyo.tools.base import Permission
from woyo.tools.builtin import build_default_registry
from woyo.tools.builtin.agent_tool import _CHILD_TOOLS
from woyo.tools.builtin.schedule_tools import (
    CancelScheduledTaskTool,
    ListScheduledTasksTool,
    ScheduleTaskTool,
)
from woyo.tools.builtin.telegram_tools import (
    GetChatInfoTool,
    ListChatsTool,
    SendDocumentTool,
    SendMessageTool,
)

# --- fakes -------------------------------------------------------------------


class FakeTransport:
    """In-memory TelegramTransport double."""

    def __init__(self, known: dict[int, dict] | None = None, current: int = 111):
        self.current_chat_id = current
        self._known = known or {}
        self.messages: list[tuple[int, str]] = []
        self.documents: list[tuple[int, str, str]] = []
        self.fail = False

    async def send_message(self, chat_id: int, text: str) -> None:
        if self.fail:
            raise RuntimeError("telegram down")
        self.messages.append((chat_id, text))

    async def send_document(self, chat_id: int, path: Path, caption: str) -> None:
        if self.fail:
            raise RuntimeError("telegram down")
        self.documents.append((chat_id, path.name, caption))

    def known_chats(self) -> list[dict]:
        out = [{"chat_id": cid, **entry} for cid, entry in self._known.items()]
        if self.current_chat_id not in self._known:
            out.append({
                "chat_id": self.current_chat_id,
                "title": "this conversation",
                "type": "current",
                "last_seen": "now",
            })
        return out

    async def chat_info(self, chat_id: int) -> dict:
        if chat_id not in self._known and chat_id != self.current_chat_id:
            raise RuntimeError("chat not found")
        return {"id": chat_id, "title": "The Group", "type": "supergroup"}

    def is_known(self, chat_id: int) -> bool:
        return chat_id == self.current_chat_id or chat_id in self._known


def _store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.sqlite3")


# --- db migration -------------------------------------------------------------


def test_jobs_table_created_and_v1_upgrades(tmp_path):
    fresh = migrate(connect(tmp_path / "fresh.sqlite3"))
    assert fresh.execute("PRAGMA user_version").fetchone()[0] == 2
    assert fresh.execute(
        "SELECT name FROM sqlite_master WHERE name='jobs'"
    ).fetchone()

    # simulate a v1 database (pre-jobs): upgrade must add the table
    v1_path = tmp_path / "v1.sqlite3"
    conn = sqlite3.connect(v1_path)
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()
    upgraded = migrate(connect(v1_path))
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == 2
    assert upgraded.execute(
        "SELECT name FROM sqlite_master WHERE name='jobs'"
    ).fetchone()


# --- JobStore -----------------------------------------------------------------


def test_jobstore_lifecycle(tmp_path):
    store = _store(tmp_path)
    past = (datetime.now(UTC) - timedelta(minutes=5)).isoformat(timespec="seconds")
    future = (datetime.now(UTC) + timedelta(hours=2)).isoformat(timespec="seconds")
    due_job = store.create(111, "due thing", "do it", past)
    store.create(111, "later thing", "do it later", future)
    store.create(222, "other chat", "theirs", future)

    assert [j.id for j in store.due()] == [due_job.id]  # only the past one
    assert store.count_active(111) == 2

    store.mark_running(due_job.id)
    assert store.get(due_job.id).status == "running"
    assert store.get(due_job.id).attempts == 1
    assert store.get(due_job.id) not in store.due()  # running is not due

    store.finish(due_job.id, "done", result_preview="all good")
    done = store.get(due_job.id)
    assert done.status == "done" and done.result_preview == "all good"
    assert done.finished_at  # completion is timestamped


def test_jobstore_cancel_is_chat_scoped(tmp_path):
    store = _store(tmp_path)
    job = store.create(111, "mine", "do it", now_iso())

    ok, _ = store.cancel(job.id, 222)  # wrong chat
    assert not ok and store.get(job.id).status == "scheduled"

    ok, msg = store.cancel(job.id, 111)
    assert ok and "Cancelled" in msg
    assert store.get(job.id).status == "cancelled"

    ok, _ = store.cancel(job.id, 111)  # already finished
    assert not ok


def test_jobstore_reschedule_and_recover(tmp_path):
    store = _store(tmp_path)
    job = store.create(111, "retry me", "do it", now_iso())
    store.mark_running(job.id)

    # rotation interrupt: back to scheduled, run_at = now
    store.reschedule(job.id)
    again = store.get(job.id)
    assert again.status == "scheduled"
    assert [j.id for j in store.due()] == [job.id]

    # a killed host leaves 'running' rows — recover() sweeps them
    store.mark_running(job.id)
    for _ in range(2):
        store.mark_running(job.id)  # attempts now 4
    assert store.get(job.id).attempts == 4
    store.recover(max_attempts=4)  # too many rotations -> fail out
    assert store.get(job.id).status == "failed"
    assert "rotation" in (store.get(job.id).error or "")

    # under the cap, recover() re-queues instead of failing
    job2 = store.create(111, "one more", "x", now_iso())
    store.mark_running(job2.id)
    store.recover(max_attempts=3)
    assert store.get(job2.id).status == "scheduled"


def test_jobstore_prune(tmp_path):
    store = _store(tmp_path)
    job = store.create(111, "old done", "x", now_iso())
    store.finish(job.id, "done")
    store.conn.execute(
        "UPDATE jobs SET finished_at=? WHERE id=?",
        ((datetime.now(UTC) - timedelta(days=40)).isoformat(), job.id),
    )
    store.conn.commit()
    assert store.prune(days=30) == 1
    assert store.get(job.id) is None


def test_describe_run_at_formats():
    soon = (datetime.now(UTC) + timedelta(minutes=90)).isoformat(timespec="seconds")
    text = describe_run_at(soon, tz="UTC")
    import re
    assert re.search(r"in 1h (29|30)m", text), text
    past = (datetime.now(UTC) - timedelta(minutes=5)).isoformat(timespec="seconds")
    assert "m ago" in describe_run_at(past, tz="UTC")


# --- telegram tools -------------------------------------------------------------


async def test_send_message_known_current_and_other(tmp_path):
    t = FakeTransport(known={-100: {"title": "Group", "type": "supergroup"}})
    tool = SendMessageTool(t)
    ok = await tool.run(
        type(tool).Args(chat_id=111, text="hello there")
    )
    assert ok.ok and t.messages == [(111, "hello there")]

    ok = await tool.run(
        type(tool).Args(chat_id=-100, text="hi group")
    )
    assert ok.ok and t.messages[-1] == (-100, "hi group")


async def test_send_message_unknown_chat_refused(tmp_path):
    t = FakeTransport()
    result = await SendMessageTool(t).run(
        SendMessageTool.Args(chat_id=999, text="spam?")
    )
    assert not result.ok
    assert result.error_kind == ErrorKind.INVALID_INPUT.value
    assert "999" in result.content
    assert t.messages == []  # nothing left the building


async def test_send_message_failure_is_observation(tmp_path):
    t = FakeTransport()
    t.fail = True
    result = await SendMessageTool(t).run(
        SendMessageTool.Args(chat_id=111, text="hello?")
    )
    assert not result.ok and result.error_kind == ErrorKind.TOOL_FAILURE.value
    assert "telegram down" in result.content


async def test_list_chats_marks_current(tmp_path):
    t = FakeTransport(
        known={
            -100: {"title": "Study Group", "type": "supergroup", "last_seen": "x"},
        },
        current=111,
    )
    result = await ListChatsTool(t).run(ListChatsTool.Args())
    assert result.ok
    assert "Study Group" in result.content
    assert "this conversation" in result.content
    assert "<- this conversation" in result.content


async def test_get_chat_info_gated_and_fresh(tmp_path):
    t = FakeTransport(known={-100: {"title": "G", "type": "supergroup"}})
    result = await GetChatInfoTool(t).run(GetChatInfoTool.Args(chat_id=-100))
    assert result.ok and "The Group" in result.content

    denied = await GetChatInfoTool(t).run(GetChatInfoTool.Args(chat_id=31337))
    assert not denied.ok and denied.error_kind == ErrorKind.INVALID_INPUT.value


async def test_send_document_workspace_scoped(tmp_path, monkeypatch):

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setenv("WOYO_WORKSPACE_DIR", str(ws))
    (ws / "report.pdf").write_bytes(b"%PDF-1.4 fake")

    settings = make_settings(workspace_dir=str(ws))
    t = FakeTransport(known={-100: {"title": "G", "type": "supergroup"}})
    tool = SendDocumentTool(settings, t)

    ok = await tool.run(
        SendDocumentTool.Args(chat_id=-100, path="report.pdf", caption="here")
    )
    assert ok.ok and t.documents == [(-100, "report.pdf", "here")]

    missing = await tool.run(
        SendDocumentTool.Args(chat_id=-100, path="nope.pdf")
    )
    assert not missing.ok

    outside = await tool.run(
        SendDocumentTool.Args(chat_id=-100, path="../report.pdf")
    )
    assert not outside.ok

    stranger = await tool.run(
        SendDocumentTool.Args(chat_id=424242, path="report.pdf")
    )
    assert not stranger.ok and t.documents == [(-100, "report.pdf", "here")]


def test_child_agents_never_get_the_new_tools():
    # the fixed include-list is the structural guarantee: sub-agents cannot
    # send Telegram messages or schedule jobs on their own
    assert set(_CHILD_TOOLS).isdisjoint({
        "send_telegram_message", "send_document", "list_chats",
        "get_chat_info", "schedule_task", "list_scheduled_tasks",
        "cancel_scheduled_task",
    })


def test_send_tools_are_writes_external():
    # cross-chat destinations pass the approval gate (inline buttons);
    # read-only chat discovery never does
    assert SendMessageTool.permission is Permission.WRITES_EXTERNAL
    assert SendDocumentTool.permission is Permission.WRITES_EXTERNAL
    assert ListChatsTool.permission is Permission.READ_ONLY
    assert GetChatInfoTool.permission is Permission.READ_ONLY
    assert ScheduleTaskTool.permission is Permission.SANDBOXED
    assert ListScheduledTasksTool.permission is Permission.READ_ONLY
    assert CancelScheduledTaskTool.permission is Permission.SANDBOXED


async def test_send_telegram_message_requires_approval_in_loop(tmp_path):
    """The agent loop must gate send_telegram_message behind approval."""
    from woyo.tools.builtin.core_tools import FinishTool

    t = FakeTransport(known={-100: {"title": "G", "type": "supergroup"}})
    responses = [
        text_response(PLAN_JSON),
        tool_response([("send_telegram_message", {"chat_id": -100, "text": "yo"})]),
        tool_response([("finish", {"summary": "sent", "verified": True})]),
    ]
    agent, mock, bus = build_agent(
        responses, tools=[SendMessageTool(t), FinishTool()]
    )
    result = await agent.run("tell the group yo", approval_cb=lambda *a: False)
    assert t.messages == []  # denied -> nothing sent
    assert result.outcome == "completed"  # loop adapted and finished honestly

    t2 = FakeTransport(known={-100: {"title": "G", "type": "supergroup"}})
    responses2 = [
        text_response(PLAN_JSON),
        tool_response([("send_telegram_message", {"chat_id": -100, "text": "yo"})]),
        tool_response([("finish", {"summary": "sent", "verified": True})]),
    ]
    agent2, _, _ = build_agent(
        responses2, tools=[SendMessageTool(t2), FinishTool()]
    )
    await agent2.run("tell the group yo", approval_cb=lambda *a: True)
    assert t2.messages == [(-100, "yo")]


# --- scheduler tools -------------------------------------------------------------


async def test_schedule_task_with_delay(tmp_path):
    store = _store(tmp_path)
    settings = make_settings()
    tool = ScheduleTaskTool(settings, store, chat_id=111)
    result = await tool.run(
        ScheduleTaskTool.Args(title="remind me", prompt="say hi", delay_minutes=30)
    )
    assert result.ok
    job = store.get(1)
    assert job.chat_id == 111 and job.status == "scheduled"
    due_in = (
        datetime.fromisoformat(job.run_at) - datetime.now(UTC)
    ).total_seconds()
    assert 29 * 60 <= due_in <= 31 * 60
    assert "job #1" in result.content  # the agent can quote the commitment


async def test_schedule_task_absolute_time(tmp_path):
    from zoneinfo import ZoneInfo

    store = _store(tmp_path)
    settings = make_settings(timezone="Africa/Algiers")
    tool = ScheduleTaskTool(settings, store, chat_id=111)
    local = (
        (datetime.now(UTC) + timedelta(hours=1)).astimezone(ZoneInfo("Africa/Algiers"))
    ).strftime("%Y-%m-%d %H:%M")
    result = await tool.run(
        ScheduleTaskTool.Args(title="later", prompt="x", run_at=local)
    )
    assert result.ok
    delta = datetime.fromisoformat(store.get(1).run_at) - datetime.now(UTC)
    assert 55 * 60 <= delta.total_seconds() <= 65 * 60, delta


async def test_schedule_task_rejects_past_and_far_future(tmp_path):
    store = _store(tmp_path)
    tool = ScheduleTaskTool(make_settings(), store, chat_id=111)

    past = await tool.run(
        ScheduleTaskTool.Args(
            title="too late", prompt="x",
            run_at="2001-01-01 00:00",
        )
    )
    assert not past.ok and "past" in past.content

    far = await tool.run(
        ScheduleTaskTool.Args(title="too far", prompt="x", delay_minutes=60 * 24 * 45)
    )
    assert not far.ok and "horizon" in far.content


def test_schedule_task_needs_exactly_one_time():
    with pytest.raises(ValidationError):
        ScheduleTaskTool.Args(title="t", prompt="p")
    with pytest.raises(ValidationError):
        ScheduleTaskTool.Args(title="t", prompt="p", delay_minutes=5, run_at="2026-01-01")


async def test_schedule_task_respects_per_chat_cap(tmp_path):
    store = _store(tmp_path)
    settings = make_settings(max_scheduled_jobs=2)
    tool = ScheduleTaskTool(settings, store, chat_id=111)
    for i in range(2):
        assert (
            await tool.run(
                ScheduleTaskTool.Args(title=f"t{i}", prompt="x", delay_minutes=10)
            )
        ).ok
    capped = await tool.run(
        ScheduleTaskTool.Args(title="t3", prompt="x", delay_minutes=10)
    )
    assert not capped.ok and "limit" in capped.content
    # another chat is unaffected
    other = ScheduleTaskTool(settings, store, chat_id=222)
    assert (
        await other.run(
            ScheduleTaskTool.Args(title="theirs", prompt="x", delay_minutes=10)
        )
    ).ok


async def test_list_and_cancel_scheduled_tasks(tmp_path):
    store = _store(tmp_path)
    settings = make_settings(timezone="UTC")
    store.create(111, "pending thing", "x", now_iso())
    done = store.create(111, "finished thing", "y", now_iso())
    store.finish(done.id, "done", result_preview="it worked")
    store.create(222, "other chat thing", "z", now_iso())

    listing = await ListScheduledTasksTool(settings, store, 111).run(
        ListScheduledTasksTool.Args()
    )
    assert listing.ok
    assert "pending thing" in listing.content
    assert "it worked" in listing.content
    assert "other chat thing" not in listing.content  # chat-scoped

    # cancel from the wrong chat is refused
    denied = await CancelScheduledTaskTool(store, 222).run(
        CancelScheduledTaskTool.Args(job_id=1)
    )
    assert not denied.ok
    # from the owning chat it works
    ok = await CancelScheduledTaskTool(store, 111).run(
        CancelScheduledTaskTool.Args(job_id=1)
    )
    assert ok.ok and store.get(1).status == "cancelled"


# --- registry wiring -------------------------------------------------------------


def test_registry_wires_chat_tools_only_when_provided(tmp_path):
    settings = make_settings()
    base = build_default_registry(settings)
    for name in ("send_telegram_message", "schedule_task", "send_document"):
        assert name not in base.names()

    store = _store(tmp_path)
    wired = build_default_registry(
        settings,
        telegram=FakeTransport(),
        job_store=store,
        chat_id=111,
    )
    for name in (
        "send_telegram_message", "list_chats", "get_chat_info",
        "send_document", "schedule_task", "list_scheduled_tasks",
        "cancel_scheduled_task",
    ):
        assert name in wired.names()


# --- session framing: awareness + promises ----------------------------------------


def _fake_agent_factory(tasks, replies):
    from woyo.agent.results import RunResult

    class FakeAgent:
        async def run(self, task, *, approval_cb=None, images=None):
            tasks.append(task)
            spec = replies.pop(0)
            return RunResult(
                task=task, outcome="completed", final_answer=spec["answer"],
                usage=type("U", (), {"tool_calls": 1, "cost_usd_est": 0.001})(),
                duration_s=0.1,
            )

    return lambda s, b: FakeAgent()


def _session_with(tmp_path, *, env=None, job_store=None, chat_id=None,
                  transport=None, sender=None):
    settings = make_settings(sessions_dir=str(tmp_path / "sessions"))
    tasks: list[str] = []
    return (
        tasks,
        ChatSession(
            "telegram:111", settings,
            chats_dir=tmp_path / "chats",
            agent_factory=_fake_agent_factory(tasks, [{"answer": "ok"}]),
            file_sender=sender,
            environment=env,
            telegram_transport=transport,
            job_store=job_store,
            chat_id=chat_id,
        ),
    )


async def test_environment_block_in_framed_task(tmp_path):
    tasks, session = _session_with(
        tmp_path, env="You are the Telegram bot @Woyoclan_bot in chat 111."
    )
    session._tool_names = ["calculate", "send_file", "web_search"]
    await session.send("hello")
    framed = tasks[0]
    assert "ENVIRONMENT" in framed
    assert "@Woyoclan_bot" in framed
    assert "calculate, send_file, web_search" in framed
    assert "never claim you used a tool that is not" in framed.lower()


async def test_promises_rule_switches_with_scheduler(tmp_path):
    # without a scheduler: the plain no-background rule
    tasks, plain = _session_with(tmp_path)
    await plain.send("hello")
    assert "NO BACKGROUND EXECUTION" in tasks[0]

    # with a scheduler: promises must become schedule_task calls
    tasks2, with_jobs = _session_with(
        tmp_path, job_store=_store(tmp_path), chat_id=111
    )
    await with_jobs.send("hello")
    assert "KEEPING PROMISES" in tasks2[0]
    assert "schedule_task" in tasks2[0]


async def test_last_turn_facts_cover_targeted_sends_and_commitments(tmp_path):
    store = _store(tmp_path)
    store.create(111, "evening report", "compile it", now_iso())
    t = FakeTransport(known={-100: {"title": "G", "type": "supergroup"}})

    tasks, session = _session_with(
        tmp_path, job_store=store, chat_id=111, transport=t
    )
    # simulate what the recording wrapper does during a turn
    session._sent_messages.append("chat -100: progress note")
    session._sent_files.append("report.pdf -> chat -100")
    session.last_turn = {
        "ts": datetime.now(UTC).isoformat(),
        "outcome": "completed", "tool_calls": 3, "duration_s": 4.0,
        "files_sent": ["report.pdf -> chat -100"],
        "messages_sent": ["chat -100: progress note"],
        "reply_preview": "sent it",
    }
    await session.send("did you send it?")
    framed = tasks[-1]
    assert "chat -100: progress note" in framed
    assert "report.pdf -> chat -100" in framed
    assert "#1 'evening report'" in framed  # pending commitment is a fact


async def test_recording_transport_records_for_last_turn(tmp_path):
    import os
    import tempfile

    from woyo.chat.session import _RecordingTransport

    async def sender(path, caption):
        pass

    tasks, session = _session_with(tmp_path)
    session.file_sender = sender
    inner = FakeTransport(known={-100: {"title": "G", "type": "supergroup"}})
    rec = _RecordingTransport(inner, session)

    await rec.send_message(-100, "note to the group")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(b"%PDF")
        p = Path(fh.name)
    try:
        await rec.send_document(-100, p, "cap")
    finally:
        os.unlink(p)

    assert session._sent_messages == ["chat -100: note to the group"]
    assert any(".pdf -> chat -100" in f for f in session._sent_files)
    assert rec.is_known(-100) and not rec.is_known(999)


# --- TelegramBot: chat index, transport, jobs -------------------------------------


def _bot(tmp_path, *, owner=111, settings=None):
    bot = TelegramBot(
        settings or make_settings(
            sessions_dir=str(tmp_path / "sessions"),
            db_path=str(tmp_path / "bot.sqlite3"),  # isolated per test
            job_poll_s=0,  # tests drive the loop manually
        ),
        "T:test",
        set(),
        state_path=tmp_path / "state.json",
    )
    bot._owner = owner
    bot._send = _recorder(bot)
    return bot


def _recorder(bot):
    sent: list[tuple[int, str]] = []
    bot._sent_log = sent

    async def send(chat_id, text, *, markdown=False):
        sent.append((chat_id, text))

    return send


def _msg(chat_id: int, text: str, update_id: int = 1, **chat_extra) -> dict:
    chat = {"id": chat_id, "type": "private", **chat_extra}
    return {
        "update_id": update_id,
        "message": {"chat": chat, "text": text},
    }


async def test_note_chat_indexes_authorized_chats_only(tmp_path):
    bot = _bot(tmp_path)

    await bot._dispatch(_msg(111, "hi", 1, first_name="Mohamed"))
    await bot._dispatch(_msg(111, "again", 2))

    entry = bot._chats.get(111)
    assert entry and entry["title"] == "Mohamed" and entry["type"] == "private"

    # a stranger (owner claimed) is refused and NOT indexed
    await bot._dispatch(_msg(222, "let me in", 3, first_name="Stranger"))
    assert 222 not in bot._chats
    assert bot._sent_log[-1] == (222, "🔒 This bot is private.")

    # group titles are captured too — the OWNER speaking in a group they
    # added the bot to authorizes that chat (from.id is server-verified)
    await bot._dispatch(_msg(111, "/start", 4))
    bot._chats.clear()
    await bot._dispatch({
        "update_id": 5,
        "message": {
            "chat": {"id": -100, "type": "supergroup", "title": "Study Group"},
            "from": {"id": 111},
            "text": "bot here",
        },
    })
    assert bot._chats[-100]["title"] == "Study Group"

    # a STRANGER speaking in that same group is still refused
    await bot._dispatch({
        "update_id": 6,
        "message": {
            "chat": {"id": -100, "type": "supergroup", "title": "Study Group"},
            "from": {"id": 222},
            "text": "let me in too",
        },
    })
    assert bot._sent_log[-1] == (-100, "🔒 This bot is private.")

    # the index persists in the state file
    state = json.loads((tmp_path / "state.json").read_text())
    assert "-100" in state["chats"]


async def test_transport_send_message_uses_api_and_chunks(tmp_path):
    calls: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        calls.append((method, json.loads(request.read())))
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bot = TelegramBot(
        make_settings(sessions_dir=str(tmp_path / "sessions")),
        "T:x", set(), client=client,
        state_path=tmp_path / "state.json",
    )
    bot._owner = 111
    bot._chats = {-100: {"title": "G", "type": "supergroup"}}

    t = _BotTransport(bot, current_chat_id=111)
    await t.send_message(-100, "short one")
    long = "x" * 8000  # forces chunking at ~4000
    await t.send_message(111, long)

    sends = [b for m, b in calls if m == "sendMessage"]
    assert sends[0] == {"chat_id": -100, "text": "short one"}
    assert len(sends) == 3  # 1 + 2 chunks
    assert all(len(b["text"]) <= 4000 for b in sends)

    assert t.is_known(-100) and t.is_known(111) and not t.is_known(31337)
    chats = t.known_chats()
    assert {c["chat_id"] for c in chats} == {-100, 111}
    await client.aclose()


async def test_environment_provider_states_ground_truth(tmp_path):
    bot = _bot(tmp_path)
    bot._bot_username = "Woyoclan_bot"
    bot._chats = {
        -100: {"title": "Study Group", "type": "supergroup"},
        111: {"title": "Mohamed", "type": "private"},
    }
    env = bot._environment_provider(111)()
    assert "@Woyoclan_bot" in env
    assert "chat_id 111" in env
    assert "Study Group" in env and "-100" in env
    assert "INSIDE Telegram" in env
    assert "CANNOT" in env  # honest limits are stated too

    # and it reaches the framed prompt through the session
    session = bot._session(111)
    session._tool_names = ["send_telegram_message"]
    framed = session._frame_task("hello")
    assert "@Woyoclan_bot" in framed and "send_telegram_message" in framed


class _FakeSession:
    """Session double for job runs: scripted replies or failures."""

    def __init__(self, replies=None, error=None, block=None):
        self.replies = list(replies or [])
        self.error = error
        self.block = block
        self.prompts: list[str] = []

    async def send(self, message, *, approval_cb=None, images=None):
        self.prompts.append(message)
        if self.block is not None:
            await self.block.wait()
        if self.error is not None:
            raise self.error
        return self.replies.pop(0)


def _reply(text: str) -> ChatReply:
    return ChatReply(answer=text, steps=1, tool_calls=1)


async def test_due_job_runs_and_reports(tmp_path):
    bot = _bot(tmp_path)
    store = bot.jobs
    syncs = []
    bot.jobs_changed_cb = lambda: syncs.append(1)

    job = store.create(111, "evening report", "compile the report", now_iso())
    fake = _FakeSession(replies=[_reply("report done: 3 sections")])
    bot._sessions[111] = fake  # type: ignore[assignment]

    await bot._run_job(job.id)

    assert store.get(job.id).status == "done"
    assert "report done" in (store.get(job.id).result_preview or "")
    # the chat saw the ⏰ header AND the reply (the notification contract)
    texts = [t for _, t in bot._sent_log]
    assert any("⏰ Running your scheduled task: evening report" in t for t in texts)
    assert any("report done" in t for t in texts)
    # the prompt tells the agent this is its scheduled commitment
    assert "SCHEDULED task" in fake.prompts[0]
    assert "compile the report" in fake.prompts[0]
    assert syncs  # host sync hook fired on transitions


async def test_job_failure_is_reported_not_silent(tmp_path):
    bot = _bot(tmp_path)
    store = bot.jobs
    job = store.create(111, "doomed", "explode please", now_iso())
    bot._sessions[111] = _FakeSession(error=RuntimeError("model offline"))  # type: ignore[assignment]

    await bot._run_job(job.id)

    failed = store.get(job.id)
    assert failed.status == "failed" and "model offline" in failed.error
    texts = [t for _, t in bot._sent_log]
    assert any("failed" in t and "doomed" in t for t in texts)


async def test_job_rotation_cancel_reschedules(tmp_path):
    bot = _bot(tmp_path)
    store = bot.jobs
    job = store.create(111, "slow thing", "take your time", now_iso())
    block = asyncio.Event()
    bot._sessions[111] = _FakeSession(block=block)  # type: ignore[assignment]

    task = asyncio.create_task(bot._run_job(job.id))
    for _ in range(50):
        if store.get(job.id).status == "running":
            break
        await asyncio.sleep(0.01)
    assert store.get(job.id).status == "running"

    task.cancel()  # what _graceful_exit does to stragglers
    with pytest.raises(asyncio.CancelledError):
        await task

    again = store.get(job.id)
    assert again.status == "scheduled"  # back on the queue for the next host
    assert again.attempts == 1
    assert [j.id for j in store.due()] == [job.id]


async def test_run_due_jobs_spawns_tasks(tmp_path):
    bot = _bot(tmp_path)
    store = bot.jobs
    store.create(111, "quick", "x", now_iso())
    bot._sessions[111] = _FakeSession(replies=[_reply("done")])  # type: ignore[assignment]

    await bot._run_due_jobs()
    for _ in range(50):
        if not bot._inflight:
            break
        await asyncio.sleep(0.01)
    assert store.get(1).status == "done"


async def test_tasks_command_lists_pending(tmp_path):
    bot = _bot(tmp_path)
    job = bot.jobs.create(111, "water the plants", "x", now_iso())
    await bot._dispatch(_msg(111, "/tasks", 7))
    texts = [t for _, t in bot._sent_log]
    assert any("water the plants" in t for t in texts)

    bot.jobs.finish(job.id, "done")
    await bot._dispatch(_msg(111, "/tasks", 9))
    assert any("No scheduled tasks pending" in t for _, t in bot._sent_log)


async def test_session_for_chat_wires_everything(tmp_path):
    bot = _bot(tmp_path)
    session = bot._session(111)
    assert session.chat_id == 111
    assert session.job_store is bot.jobs
    assert session.telegram_transport is not None
    assert session.environment is not None
    assert session.file_sender is not None
