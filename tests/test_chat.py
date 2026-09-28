"""Chat frontends: session core, Telegram bot, web server (all offline)."""

from __future__ import annotations

import asyncio
import json
import threading

import httpx
import pytest

from tests.conftest import make_settings
from woyo.agent.results import RunResult, UsageReport
from woyo.chat import web as webmod
from woyo.chat.session import ChatReply, ChatSession, chat_settings
from woyo.chat.telegram import TelegramBot, split_message, telegram_credentials
from woyo.errors import AgentError

# --- fakes -------------------------------------------------------------------


class FakeChatAgent:
    """Captures framed tasks; returns scripted results."""

    def __init__(self, tasks: list[str], replies: list[dict]):
        self.tasks = tasks
        self.replies = replies

    async def run(self, task: str, *, approval_cb=None) -> RunResult:
        self.tasks.append(task)
        spec = self.replies.pop(0)
        return RunResult(
            task=task,
            outcome="completed",
            final_answer=spec["answer"],
            verified=spec.get("verified", False),
            sources=spec.get("sources", []),
            sources_dropped=spec.get("sources_dropped", []),
            usage=UsageReport(
                llm_calls=2,
                tool_calls=spec.get("tool_calls", 1),
                cost_usd_est=0.001,
            ),
            duration_s=0.5,
            steps=spec.get("steps", 2),
        )


def make_chat_session(tmp_path, replies, *, key="test:1", **settings_kw):
    settings = make_settings(
        sessions_dir=str(tmp_path / "sessions"), **settings_kw
    )
    tasks: list[str] = []

    def factory(s, bus):
        return FakeChatAgent(tasks, replies)

    session = ChatSession(
        key, settings, chats_dir=tmp_path / "chats", agent_factory=factory
    )
    return session, tasks


# --- session core --------------------------------------------------------------


async def test_chat_frames_task_with_history(tmp_path):
    session, tasks = make_chat_session(
        tmp_path,
        [{"answer": "Paris"}, {"answer": "It is 15C"}],
    )
    await session.send("What is the capital of France?")
    await session.send("And what is the temperature there?")

    assert "USER MESSAGE" in tasks[0]
    assert "capital of France" in tasks[0]
    # second task carries the transcript of the first exchange
    assert "TRANSCRIPT" in tasks[1]
    assert "capital of France" in tasks[1]
    assert "Paris" in tasks[1]
    assert "NEW MESSAGE" in tasks[1]
    assert "temperature" in tasks[1]
    assert session.total_messages == 2


async def test_chat_history_window(tmp_path):
    session, tasks = make_chat_session(
        tmp_path,
        [{"answer": "a1"}, {"answer": "a2"}, {"answer": "a3"}],
        chat_history_turns=2,  # one user+assistant exchange kept
    )
    for i in range(1, 4):
        await session.send(f"question number {i}")

    assert "question number 1" not in tasks[2]
    assert "question number 2" in tasks[2]


async def test_chat_persistence_roundtrip(tmp_path):
    session, _ = make_chat_session(tmp_path, [{"answer": "hello there"}])
    await session.send("hi")
    first_id = session.session_id

    # a "restarted" session with the same key resumes the conversation
    session2, tasks = make_chat_session(tmp_path, [{"answer": "still here"}])
    assert session2.history[-1] == ("woyo", "hello there")
    assert session2.total_messages == 1
    await session2.send("do you remember me?")
    assert "hello there" in tasks[0]
    assert session2.session_id == first_id  # transcript continuity kept


async def test_chat_daily_limit(tmp_path):
    session, _ = make_chat_session(
        tmp_path, [{"answer": "one"}], chat_daily_messages=1
    )
    await session.send("first")
    with pytest.raises(AgentError, match="daily message limit"):
        await session.send("second")


def test_reply_render_sources():
    reply = ChatReply(
        answer="Answer text",
        verified=True,
        sources=[
            {"url": "https://a.example/x", "title": "A"},
            {"url": "https://a.example/x", "title": "A2"},  # duplicate url
            {"url": "https://b.example/y", "title": "B"},
        ],
        sources_dropped=["https://fake.example"],
    )
    text = reply.render()
    assert "Answer text" in text
    assert "Sources:" in text
    assert text.count("https://a.example/x") == 1  # dedup
    assert "https://b.example/y" in text
    assert "could not be verified" in text
    assert "Sources:" not in reply.render(with_sources=False)


def test_chat_settings_profile_isolated():
    base = make_settings(max_steps=40)
    profile = chat_settings(base)
    assert profile.max_steps == 8
    assert profile.max_search_calls == 6
    assert profile.direct_text_replies is True
    assert base.max_steps == 40  # untouched
    assert base.direct_text_replies is False


async def test_direct_text_replies_accepted_first_strike(tmp_path):
    """Chat mode: a prose reply becomes the answer without the finish tool."""
    from tests.conftest import PLAN_JSON, build_agent
    from woyo.models.mock import text_response

    agent, mock, _bus = build_agent(
        [text_response(PLAN_JSON), text_response("Hello! I am woyo.")],
        settings=make_settings(direct_text_replies=True),
    )
    result = await agent.run("say hi")
    assert result.outcome == "completed"
    assert result.final_answer == "Hello! I am woyo."
    assert result.steps == 1  # executor answered on the first try


async def test_task_mode_still_nudges_text_only():
    """Task mode (default): first prose reply triggers the nudge, not final."""
    from tests.conftest import PLAN_JSON, build_agent
    from woyo.models.mock import text_response

    agent, mock, _bus = build_agent(
        [
            text_response(PLAN_JSON),
            text_response("I will search now."),
            text_response("ok fine"),
        ],
        settings=make_settings(),
    )
    result = await agent.run("do the thing")
    assert result.steps == 2
    assert result.final_answer == "ok fine"
    # the nudge message was appended before the executor's second call
    assert any(
        "TOOL CALLS" in (m.content or "")
        for call in mock.calls
        for m in call["messages"]
    )


# --- telegram ------------------------------------------------------------------


def test_split_message():
    text = "x" * 9_000
    chunks = split_message(text)
    assert all(len(c) <= 4_000 for c in chunks)
    assert "".join(chunks) == text
    assert len(split_message("short")) == 1
    # prefers splitting at newlines
    lined = "\n".join(f"line {i}" for i in range(300))  # ~2.4k chars, many breaks
    chunks = split_message(lined)
    assert all(c.startswith("line") for c in chunks)


def _update(chat_id: int, text: str, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {"chat": {"id": chat_id}, "text": text},
    }


async def test_telegram_claim_and_refusal(tmp_path):
    sent: list[tuple[int, str]] = []

    async def recorder(chat_id, text, *, markdown=False):
        sent.append((chat_id, text))

    bot = TelegramBot(
        make_settings(), "T:test", set(), state_path=tmp_path / "state.json"
    )
    bot._send = recorder  # type: ignore[method-assign]

    await bot._dispatch(_update(111, "/start", 1))
    assert bot._owner == 111
    assert sent[0][0] == 111
    assert "woyo" in sent[0][1].lower()

    await bot._dispatch(_update(222, "hello?", 2))
    assert sent[-1] == (222, "🔒 This bot is private.")
    assert json.loads((tmp_path / "state.json").read_text())["owner"] == 111

    # after a restart the claim persists
    bot2 = TelegramBot(
        make_settings(), "T:test", set(), state_path=tmp_path / "state.json"
    )
    assert bot2._owner == 111


async def test_telegram_answer_with_markdown_fallback(tmp_path):
    calls: list[tuple[str, dict]] = []
    fail_markdown = [True]
    updates_queue: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.read())
        calls.append((method, body))
        if method == "getUpdates":
            return httpx.Response(
                200, json={"ok": True, "result": updates_queue.pop(0) if updates_queue else []}
            )
        if method == "sendMessage" and body.get("parse_mode") and fail_markdown[0]:
            fail_markdown[0] = False
            return httpx.Response(400, json={"ok": False, "error_code": 400})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bot = TelegramBot(
        make_settings(sessions_dir=str(tmp_path / "sessions")),
        "T:test",
        set(),
        client=client,
        state_path=tmp_path / "state.json",
    )
    bot._owner = 111

    class FakeSession:
        async def send(self, message: str, *, approval_cb=None) -> ChatReply:
            await asyncio.sleep(0)  # a real run yields to the loop (LLM call)
            return ChatReply(
                answer=f"**bold** answer to *{message}* with `code`",
                verified=True,
                sources=[{"url": "https://example.com/s", "title": "S"}],
                steps=3,
                tool_calls=2,
                duration_s=1.2,
                cost_usd_est=0.002,
            )

    bot._sessions[111] = FakeSession()  # type: ignore[assignment]
    await bot._dispatch(_update(111, "what is new?", 5))
    # _dispatch spawns the answer as a task (so button presses can arrive
    # mid-run) — let pending tasks run to completion before asserting
    for task in [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
        await task

    sends = [(m, b) for m, b in calls if m == "sendMessage"]
    assert sends, "no messages sent"
    # markdown attempt failed once, then plain-text retry delivered
    assert any(b.get("parse_mode") == "Markdown" for _, b in sends)
    assert sum(1 for _, b in sends if "parse_mode" not in b) == 1
    final_text = [b for _, b in sends if "parse_mode" not in b][0]["text"]
    assert "bold" in final_text and "Sources:" in final_text
    assert any(m == "sendChatAction" for m, _ in calls)
    await client.aclose()


async def test_telegram_offset_advances_and_persists(tmp_path):
    updates_queue = [[{"update_id": 10}, {"update_id": 11}]]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("getUpdates")
        params = json.loads(request.read())
        assert params["offset"] in (0, 12)
        return httpx.Response(
            200,
            json={"ok": True, "result": updates_queue.pop(0) if updates_queue else []},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = tmp_path / "state.json"
    bot = TelegramBot(make_settings(), "T:x", set(), client=client, state_path=state)
    await bot._poll()
    assert bot._offset == 12
    assert json.loads(state.read_text())["offset"] == 12
    await bot._poll()  # empty result keeps the offset
    assert bot._offset == 12
    await client.aclose()


async def test_telegram_409_is_explicit(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"ok": False})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bot = TelegramBot(make_settings(), "T:x", set(), client=client)
    with pytest.raises(RuntimeError, match="409"):
        await bot._api("getMe")
    await client.aclose()


async def test_telegram_409_stops_run_forever(tmp_path):
    """A 409 mid-poll stops the bot loudly instead of retrying forever."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        method = request.url.path.rsplit("/", 1)[-1]
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {"username": "woyo_bot"}})
        return httpx.Response(409, json={"ok": False})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bot = TelegramBot(
        make_settings(), "T:x", set(), client=client, state_path=tmp_path / "s.json"
    )
    with pytest.raises(RuntimeError, match="409"):
        await asyncio.wait_for(bot.run_forever(), timeout=2)
    await client.aclose()
    assert calls["n"] == 2  # getMe then one poll — no retry loop


async def test_telegram_skips_backlog_on_fresh_state(tmp_path):
    """Ephemeral host, no stored offset: pending updates are dropped, not answered."""
    seen_offsets: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        params = json.loads(request.read())
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {"username": "b"}})
        seen_offsets.append(params.get("offset"))
        if params.get("offset") == -1:  # backlog probe
            return httpx.Response(
                200,
                json={"ok": True, "result": [{"update_id": 41}, {"update_id": 99}]},
            )
        return httpx.Response(200, json={"ok": True, "result": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = tmp_path / "state.json"
    bot = TelegramBot(make_settings(), "T:x", set(), client=client, state_path=state)
    assert bot._had_state is False
    await bot._skip_backlog()
    assert bot._offset == 100
    assert json.loads(state.read_text())["offset"] == 100
    await bot._poll()  # normal polling continues from the new offset
    assert seen_offsets == [-1, 100]
    await client.aclose()

    # a restart WITH state must not probe the backlog again
    bot2 = TelegramBot(make_settings(), "T:x", set(), state_path=state)
    assert bot2._had_state is True
    assert bot2._offset == 100


def test_telegram_credentials_from_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_ALLOWED_CHAT_IDS", "42, 7; bad")
    token, allowed = telegram_credentials()
    assert token == "123:abc"
    assert allowed == {42, 7}


# --- web -------------------------------------------------------------------------


class FakeWebSession:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, message: str, *, approval_cb=None) -> ChatReply:
        self.sent.append(message)
        return ChatReply(
            answer=f"echo: {message}",
            verified=True,
            sources=[{"url": "https://example.com/p", "title": "P"}],
            steps=2,
            tool_calls=1,
            duration_s=0.05,
            cost_usd_est=0.0001,
        )


@pytest.fixture
def web_server(tmp_path, monkeypatch):
    def start(password="pw123", session=None):
        settings = make_settings(
            sessions_dir=str(tmp_path / "sessions"),
            chat_password=password,
        )
        fake = session or FakeWebSession()

        def factory(key, s):
            return fake

        server = webmod.build_server(
            settings, "127.0.0.1", 0, session_factory=factory
        )
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, fake

    return start


def _client(server):
    return httpx.Client(base_url=f"http://127.0.0.1:{server.server_address[1]}")


def test_web_requires_passcode_off_localhost(tmp_path):
    settings = make_settings(sessions_dir=str(tmp_path / "s"), chat_password=None)
    with pytest.raises(AgentError, match="passcode"):
        webmod.build_server(settings, "0.0.0.0", 0)


def test_web_page_health_and_passcode(web_server):
    server, fake = web_server()
    with _client(server) as c:
        page = c.get("/")
        assert page.status_code == 200
        assert "woyo" in page.text and "__VERSION__" not in page.text

        health = c.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True

        r = c.post(
            "/api/chat",
            json={"session_id": "abcdefgh1234", "message": "hi"},
        )
        assert r.status_code == 401

        r = c.post(
            "/api/chat",
            json={"session_id": "abcdefgh1234", "passcode": "wrong", "message": "hi"},
        )
        assert r.status_code == 401
        assert fake.sent == []  # nothing ran

        r = c.post(
            "/api/chat",
            json={
                "session_id": "abcdefgh1234",
                "passcode": "pw123",
                "message": "hello woyo",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert data["answer"] == "echo: hello woyo"
        assert data["sources"][0]["url"] == "https://example.com/p"
        assert fake.sent == ["hello woyo"]

        # invalid session ids are rejected before any agent work
        r = c.post(
            "/api/chat",
            json={"session_id": "nope!", "passcode": "pw123", "message": "hi"},
        )
        assert r.status_code == 400


def test_web_rate_limit(web_server, monkeypatch):
    monkeypatch.setattr(webmod, "_WINDOW_MESSAGES", 2)
    server, fake = web_server()
    with _client(server) as c:
        for _i in range(2):
            r = c.post(
                "/api/chat",
                json={"session_id": "abcdefgh1234", "passcode": "pw123", "message": "m"},
            )
            assert r.status_code == 200
        r = c.post(
            "/api/chat",
            json={"session_id": "abcdefgh1234", "passcode": "pw123", "message": "m"},
        )
        assert r.status_code == 429
        assert len(fake.sent) == 2
