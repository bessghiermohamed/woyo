"""v0.7 reliability: pending journal, per-chat queueing, graceful rotation,
chat truthfulness (LAST TURN FACTS), and the respawn dispatch."""

from __future__ import annotations

import asyncio
import json

import httpx

from tests.conftest import make_settings
from woyo.chat.session import ChatReply, ChatSession
from woyo.chat.telegram import TelegramBot

# --- helpers -----------------------------------------------------------------------


def _text_update(chat_id: int, update_id: int, text: str) -> dict:
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id}, "text": text},
    }


class RecordingSession:
    """Stands in for ChatSession; can delay or fail."""

    def __init__(self, delay_s: float = 0.0, fail: bool = False):
        self.received: list[str] = []
        self.delay_s = delay_s
        self.fail = fail

    async def send(self, message, *, approval_cb=None, images=None) -> ChatReply:
        self.received.append(message)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("provider exploded")
        return ChatReply(answer=f"echo: {message[:30]}", verified=True)


def _make_bot(tmp_path, transport, **settings_kw) -> TelegramBot:
    client = httpx.AsyncClient(transport=transport)
    return TelegramBot(
        make_settings(
            sessions_dir=str(tmp_path / "sessions"),
            workspace_dir=str(tmp_path / "ws"),
            files_dir=str(tmp_path / "files"),
            **settings_kw,
        ),
        "T:x",
        set(),
        client=client,
        state_path=tmp_path / "state.json",
    )


def _api_transport(*updates: dict, sent: list | None = None):
    """getMe -> bot; getUpdates -> scripted batches; sendMessage recorded."""
    remaining = list(updates)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/getMe"):
            return httpx.Response(200, json={"ok": True, "result": {"username": "woyobot"}})
        if path.endswith("/getUpdates"):
            batch = [remaining.pop(0)] if remaining else []
            return httpx.Response(200, json={"ok": True, "result": batch})
        if path.endswith("/sendMessage") or path.endswith("/sendChatAction"):
            if sent is not None:
                sent.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
        return httpx.Response(200, json={"ok": True, "result": {}})

    return httpx.MockTransport(handler)


async def _drain():
    for task in [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
        await asyncio.wait([task], timeout=5.0)


# --- pending journal ---------------------------------------------------------------


async def test_journal_retains_unanswered_update_across_restart(tmp_path):
    """A host killed mid-turn must not lose the request: the update stays
    journaled in the state file and the next boot drains it."""
    state = tmp_path / "state.json"
    sent: list = []
    transport = _api_transport(_text_update(111, 500, "fix my pdf"), sent=sent)

    bot = _make_bot(tmp_path, transport)
    bot._owner = 111

    never = asyncio.Event()

    class NeverEndingSession(RecordingSession):
        async def send(self, message, *, approval_cb=None, images=None):
            self.received.append(message)
            await never.wait()  # simulates a turn still running at kill time
            return ChatReply(answer="too late")

    bot._sessions[111] = NeverEndingSession()  # type: ignore[assignment]

    updates = await bot._poll()
    assert updates and 500 in bot._journal  # journaled at fetch, pre-processing
    assert "500" in json.loads(state.read_text())["pending"]

    await bot._dispatch(updates[0])  # spawns the (never-ending) turn task
    await asyncio.sleep(0.05)
    assert 500 in bot._journal  # host "dies" here — turn unfinished, no reply

    # fresh host: new bot instance, journal restored from disk
    bot2 = _make_bot(tmp_path, transport)
    bot2._owner = 111
    working = RecordingSession()
    bot2._sessions[111] = working  # type: ignore[assignment]
    assert 500 in bot2._journal

    await bot2._drain_journal()
    await _drain()
    assert working.received == ["fix my pdf"]  # the lost request answered
    assert 500 not in bot2._journal

    never.set()  # let the original straggler settle, then clean up
    await _drain()
    await bot._client.aclose()
    await bot2._client.aclose()


async def test_failed_turn_still_replies_and_unjournals(tmp_path):
    """A turn that fails loudly sends the user an error message — that IS a
    reply, so the update is unjournaled (no silent retry loops)."""
    sent: list = []
    transport = _api_transport(_text_update(111, 503, "boom"), sent=sent)
    bot = _make_bot(tmp_path, transport)
    bot._owner = 111
    bot._sessions[111] = RecordingSession(fail=True)  # type: ignore[assignment]

    updates = await bot._poll()
    await bot._dispatch(updates[0])
    await _drain()
    assert 503 not in bot._journal
    errors = [s for s in sent if "Something went wrong" in str(s.get("text", ""))]
    assert errors, "user must see the failure, not silence"
    await bot._client.aclose()


async def test_journal_empties_after_successful_reply(tmp_path):
    sent: list = []
    transport = _api_transport(_text_update(111, 501, "hello"), sent=sent)
    bot = _make_bot(tmp_path, transport)
    bot._owner = 111
    session = RecordingSession()
    bot._sessions[111] = session  # type: ignore[assignment]

    updates = await bot._poll()
    await bot._dispatch(updates[0])
    await _drain()
    assert session.received == ["hello"]
    assert not bot._journal  # replied -> unjournaled
    data = json.loads((tmp_path / "state.json").read_text())
    assert data["pending"] == {}
    await bot._client.aclose()


async def test_journal_covers_commands_too(tmp_path):
    sent: list = []
    transport = _api_transport(_text_update(111, 502, "/status"), sent=sent)
    bot = _make_bot(tmp_path, transport)
    bot._owner = 111
    updates = await bot._poll()
    assert 502 in bot._journal
    await bot._dispatch(updates[0])
    assert 502 not in bot._journal  # synchronous path also un-journals
    await bot._client.aclose()


async def test_journal_bounded_at_50(tmp_path):
    bot = _make_bot(tmp_path, _api_transport())
    for i in range(60):
        bot._journal_update(_text_update(111, 1000 + i, "x"))
    assert len(bot._journal) == 50
    assert 1010 in bot._journal and 1009 not in bot._journal  # newest kept
    await bot._client.aclose()


# --- per-chat queueing ---------------------------------------------------------------


async def test_second_message_queues_instead_of_being_dropped(tmp_path):
    sent: list = []
    transport = _api_transport(sent=sent)
    bot = _make_bot(tmp_path, transport)
    bot._owner = 111
    session = RecordingSession(delay_s=0.2)
    bot._sessions[111] = session  # type: ignore[assignment]

    t1 = asyncio.create_task(bot._answer(111, "first"))
    await asyncio.sleep(0.02)  # let the first acquire the lock
    t2 = asyncio.create_task(bot._answer(111, "second"))
    await asyncio.gather(t1, t2)

    assert session.received == ["first", "second"]  # nothing dropped
    # the queued acknowledgement was sent to the user
    acks = [s for s in sent if "queued" in str(s.get("text", ""))]
    assert acks, "expected a queue acknowledgement"
    await bot._client.aclose()


# --- graceful rotation ----------------------------------------------------------------


async def test_run_forever_returns_on_runtime_budget(tmp_path):
    sent: list = []
    transport = _api_transport(sent=sent)
    bot = _make_bot(tmp_path, transport)
    # state file exists -> drain path (empty journal, no-op)
    (tmp_path / "state.json").write_text("{}")

    await asyncio.wait_for(
        bot.run_forever(max_runtime_s=0.5), timeout=30.0
    )
    # returned gracefully instead of looping forever
    await bot._client.aclose()


async def test_run_forever_returns_on_shutdown_event(tmp_path):
    transport = _api_transport()
    bot = _make_bot(tmp_path, transport)
    (tmp_path / "state.json").write_text("{}")
    stop = asyncio.Event()

    async def tripwire():
        await asyncio.sleep(0.2)
        stop.set()

    await asyncio.gather(tripwire(), bot.run_forever(shutdown_event=stop))
    await bot._client.aclose()


async def test_graceful_exit_notifies_busy_chat_and_keeps_journal(tmp_path):
    sent: list = []
    transport = _api_transport(sent=sent)
    bot = _make_bot(tmp_path, transport)
    bot._owner = 111

    release = asyncio.Event()

    class SlowSession(RecordingSession):
        async def send(self, message, *, approval_cb=None, images=None):
            self.received.append(message)
            await release.wait()
            return ChatReply(answer="late")

    bot._sessions[111] = SlowSession()  # type: ignore[assignment]
    update = _text_update(111, 700, "long job")
    bot._journal_update(update)  # _poll journals before dispatching
    await bot._dispatch(update)  # via _dispatch so the task is tracked
    await asyncio.sleep(0.05)

    await bot._graceful_exit(grace_s=0.3)  # times out the straggler
    release.set()
    await _drain()

    notices = [s for s in sent if "switching host runners" in str(s.get("text", ""))]
    assert notices, "busy chat should get a rotation notice"
    # cancellation kept the request journaled for the next host
    assert 700 in bot._journal
    await bot._client.aclose()


# --- chat truthfulness (LAST TURN FACTS) ----------------------------------------------


class FakeAgentResult:
    outcome = "completed"

    class usage:  # noqa: N801
        tool_calls = 3
        cost_usd_est = 0.0

    final_answer = "here is your file"
    verified = True
    sources: list = []
    sources_dropped: list = []
    steps = 2
    duration_s = 12.0
    events_log: list = []


class FakeAgent:
    def __init__(self, settings, router, registry, *, bus=None, memory=None):
        pass

    async def run(self, task, *, approval_cb=None, images=None):
        FakeAgent.seen_task = task
        return FakeAgentResult()


async def test_last_turn_facts_injected_into_next_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "woyo.chat.session._default_agent_factory",
        lambda settings, bus: FakeAgent(settings, None, None, bus=bus),
    )
    import woyo.chat.session as session_mod

    orig_factory = session_mod._default_agent_factory
    session_mod._default_agent_factory = lambda settings, bus: FakeAgent(
        settings, None, None, bus=bus
    )
    try:
        sess = ChatSession(
            "telegram:111",
            make_settings(sessions_dir=str(tmp_path / "sessions")),
            chats_dir=tmp_path / "chats",
        )
        await sess.send("make me a report")
        framed = sess._frame_task("did you finish?")
        assert "LAST TURN FACTS" in framed
        assert "3 tool call(s)" in framed
        assert "here is your file" in framed
        assert "NO BACKGROUND EXECUTION" in framed
        # persisted for the next host too
        data = json.loads(
            (tmp_path / "chats" / "telegram_111.json").read_text()
        )
        assert data["last_turn"]["tool_calls"] == 3
    finally:
        session_mod._default_agent_factory = orig_factory


async def test_last_turn_facts_reload_from_disk(tmp_path):
    import woyo.chat.session as session_mod

    orig_factory = session_mod._default_agent_factory
    session_mod._default_agent_factory = lambda settings, bus: FakeAgent(
        settings, None, None, bus=bus
    )
    try:
        sess = ChatSession(
            "telegram:111",
            make_settings(sessions_dir=str(tmp_path / "sessions")),
            chats_dir=tmp_path / "chats",
        )
        await sess.send("hello")
        sess2 = ChatSession(
            "telegram:111",
            make_settings(sessions_dir=str(tmp_path / "sessions")),
            chats_dir=tmp_path / "chats",
        )
        assert sess2.last_turn is not None
        assert sess2.last_turn["tool_calls"] == 3
        assert "LAST TURN FACTS" in sess2._frame_task("again?")
    finally:
        session_mod._default_agent_factory = orig_factory


# --- respawn dispatch (run_bot) ---------------------------------------------------------


def _load_run_bot(monkeypatch, env: dict | None = None):
    """Import deploy/gha-telegram/run_bot.py as a module (it's not a package)."""
    import importlib.util
    from pathlib import Path

    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    path = Path(__file__).resolve().parents[1] / "deploy" / "gha-telegram" / "run_bot.py"
    spec = importlib.util.spec_from_file_location("run_bot_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rotate_after_default(monkeypatch):
    run_bot = _load_run_bot(monkeypatch, {"BOT_STATE_REPO": "", "BOT_STATE_TOKEN": ""})
    monkeypatch.delenv("WOYO_ROTATE_AFTER_MIN", raising=False)
    assert run_bot._rotate_after_s() == 290 * 60.0


def test_rotate_after_env_overrides(monkeypatch):
    run_bot = _load_run_bot(
        monkeypatch,
        {"WOYO_ROTATE_AFTER_MIN": "10", "BOT_STATE_REPO": "", "BOT_STATE_TOKEN": ""},
    )
    assert run_bot._rotate_after_s() == 600.0


def test_rotate_after_zero_disables(monkeypatch):
    run_bot = _load_run_bot(
        monkeypatch,
        {"WOYO_ROTATE_AFTER_MIN": "0", "BOT_STATE_REPO": "", "BOT_STATE_TOKEN": ""},
    )
    assert run_bot._rotate_after_s() is None


def test_dispatch_next_run_posts_workflow_dispatch(monkeypatch):
    run_bot = _load_run_bot(
        monkeypatch,
        {
            "GITHUB_REPOSITORY": "owner/woyo",
            "GITHUB_REF_NAME": "main",
            "BOT_STATE_TOKEN": "pat-test",
        },
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/dispatches"):
            assert request.headers["Authorization"] == "Bearer pat-test"
            assert json.loads(request.content) == {"ref": "main"}
            return httpx.Response(204)
        raise AssertionError(f"unexpected call: {request.url}")

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        run_bot.httpx, "Client", lambda **kw: real_client(transport=transport, **kw)
    )
    assert run_bot._dispatch_next_run() is True
    assert len(calls) == 1
    assert calls[0].url.path.endswith("telegram.yml/dispatches")


def test_dispatch_next_run_without_repo_is_honest(monkeypatch, capsys):
    run_bot = _load_run_bot(
        monkeypatch,
        {
            "GITHUB_REPOSITORY": "",
            "WOYO_CODE_REPO": "",
            "BOT_STATE_TOKEN": "pat-test",
        },
    )
    assert run_bot._dispatch_next_run() is False
    assert "backstops" in capsys.readouterr().out

