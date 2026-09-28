"""v0.5 capabilities: sandbox (python/shell/files), sub-agents, chat approvals."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from tests.conftest import PLAN_JSON, build_agent, make_settings
from woyo.agent import Agent
from woyo.config import Settings
from woyo.events import EventBus
from woyo.models.mock import MockProvider, text_response, tool_response
from woyo.models.router import ModelRouter
from woyo.tools.base import Permission, Tool, ToolResult
from woyo.tools.builtin import build_default_registry
from woyo.tools.builtin.agent_tool import SpawnAgentArgs, SpawnAgentTool
from woyo.tools.builtin.core_tools import FinishTool
from woyo.tools.builtin.sandbox import (
    CodeArgs,
    ListDirArgs,
    ListDirTool,
    PythonExecTool,
    ReadFileArgs,
    ReadFileTool,
    ShellArgs,
    ShellExecTool,
    WriteFileArgs,
    WriteFileTool,
    resolve_ws_path,
    workspace_root,
)


def ws_settings(tmp_path, **kw) -> Settings:
    return make_settings(
        workspace_dir=str(tmp_path / "ws"), enable_code_exec=True, **kw
    )


# --- python_exec --------------------------------------------------------------


async def test_python_exec_workspace_persists(tmp_path):
    tool = PythonExecTool(ws_settings(tmp_path))
    r1 = await tool.run(CodeArgs(
        code="open('note.txt','w').write('hello workspace')\nprint('written')"
    ))
    assert r1.ok, r1.content
    r2 = await PythonExecTool(ws_settings(tmp_path)).run(
        CodeArgs(code="print(open('note.txt').read())"))
    assert r2.ok and "hello workspace" in r2.content


async def test_python_exec_env_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "SECRET-TOKEN-XYZ")
    monkeypatch.setenv("COHERE_API_KEY", "sk-SECRET")
    tool = PythonExecTool(ws_settings(tmp_path))
    r = await tool.run(CodeArgs(
        code="import os; print(sorted(os.environ))"
    ))
    assert r.ok
    assert "SECRET-TOKEN-XYZ" not in r.content and "sk-SECRET" not in r.content
    assert "TELEGRAM_BOT_TOKEN" not in r.content


async def test_python_exec_timeout_kills(tmp_path):
    tool = PythonExecTool(ws_settings(tmp_path))
    r = await tool.run(CodeArgs(
        code="while True: pass", timeout_s=2
    ))
    assert not r.ok and "killed" in r.content


async def test_python_exec_stderr_captured(tmp_path):
    tool = PythonExecTool(ws_settings(tmp_path))
    r = await tool.run(CodeArgs(
        code="import sys; print('boom', file=sys.stderr); sys.exit(3)"
    ))
    assert r.ok and "exit 3" in r.content and "boom" in r.content


# --- shell_exec ---------------------------------------------------------------


async def test_shell_exec_disabled_by_default(tmp_path):
    tool = ShellExecTool(make_settings(workspace_dir=str(tmp_path / "ws")))
    r = await tool.run(ShellArgs(command="echo hi"))
    assert not r.ok and "WOYO_ENABLE_SHELL" in r.content


async def test_shell_exec_requires_approval_level(tmp_path):
    assert ShellExecTool(make_settings()).permission == Permission.WRITES_EXTERNAL


async def test_shell_exec_runs_in_workspace(tmp_path):
    tool = ShellExecTool(ws_settings(tmp_path, enable_shell=True))
    r = await tool.run(ShellArgs(
        command="echo hello > out.txt && cat out.txt && pwd"
    ))
    assert r.ok and "hello" in r.content
    assert (tmp_path / "ws" / "out.txt").read_text().strip() == "hello"


async def test_shell_exec_env_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("BOT_STATE_TOKEN", "ghp-VERY-SECRET")
    tool = ShellExecTool(ws_settings(tmp_path, enable_shell=True))
    r = await tool.run(ShellArgs(command="env | sort"))
    assert r.ok
    assert "ghp-VERY-SECRET" not in r.content and "BOT_STATE_TOKEN" not in r.content


# --- workspace file tools -----------------------------------------------------


async def test_file_roundtrip_and_listing(tmp_path):
    s = ws_settings(tmp_path)
    w = await WriteFileTool(s).run(WriteFileArgs(
        path="reports/notes.md", content="# Notes\nline2"
    ))
    assert w.ok
    r = await ReadFileTool(s).run(ReadFileArgs(path="reports/notes.md"))
    assert r.ok and "# Notes" in r.content
    listing = await ListDirTool(s).run(ListDirArgs(path="."))
    assert listing.ok and "reports" in listing.content


async def test_file_traversal_blocked(tmp_path):
    s = ws_settings(tmp_path)
    root = workspace_root(s)
    (tmp_path / "outside.txt").write_text("secret")
    r = await ReadFileTool(s).run(ReadFileArgs(path="../outside.txt"))
    assert not r.ok and "escapes the workspace" in r.content
    with pytest.raises(ValueError):
        resolve_ws_path(root, "/etc/passwd")
    w = await WriteFileTool(s).run(WriteFileArgs(
        path="../../evil.txt", content="x"
    ))
    assert not w.ok


# --- spawn_agent --------------------------------------------------------------


def _make_spawn_stack(tmp_path, responses):
    settings = ws_settings(tmp_path)
    bus = EventBus()
    mock = MockProvider(list(responses))
    router = ModelRouter(settings, default_provider=mock, bus=bus)
    registry = build_default_registry(settings, bus=bus, router=router)
    agent = Agent(settings, router, registry, bus=bus)
    return agent, mock, bus, registry


async def test_spawn_agent_runs_child_and_returns_answer(tmp_path):
    # parent plan, parent executor calls spawn_agent, child flow, parent finish
    responses = [
        text_response(PLAN_JSON),  # parent planner
        tool_response([("spawn_agent", {
            "task": "Find the release year of Python 3.14 with a source."
        })]),
        text_response(PLAN_JSON),  # child planner
        tool_response([("finish", {
            "summary": "Python 3.14 was released in October 2025.",
            "verified": True, "open_questions": [],
            "sources": [{"title": "py", "url": "https://www.python.org/downloads/"}],
        })]),
        tool_response([("finish", {
            "summary": "Sub-agent says Python 3.14 (Oct 2025).",
            "verified": True, "open_questions": [],
            "sources": [{"title": "py", "url": "https://www.python.org/downloads/"}],
        })]),
    ]
    agent, mock, bus, registry = _make_spawn_stack(tmp_path, responses)
    result = await agent.run("When was Python 3.14 released?")
    assert result.outcome == "completed"
    assert "Oct" in result.final_answer or "2025" in result.final_answer
    # honesty property: nobody (parent or child) actually OBSERVED python.org
    # in this scripted run, so the claimed citation is dropped — even via a
    # sub-agent, unverified sources never reach the user
    assert result.sources == []
    assert result.sources_dropped or result.open_questions
    # events recorded the delegation
    kinds = [e.kind for e in bus.events]
    assert "subagent_started" in kinds and "subagent_finished" in kinds


async def test_spawn_agent_propagates_child_sources(tmp_path, monkeypatch):
    """A child that DID observe URLs hands them to the parent as observed."""
    from woyo.agent.results import RunResult

    class FakeAgent:
        def __init__(self, settings, router, registry, *, bus=None, memory=None):
            pass

        async def run(self, task, **kw):
            return RunResult(
                task=task, outcome="completed", outcome_detail="",
                final_answer="found it", verified=True,
                sources=[{"title": "py", "url": "https://www.python.org/downloads/"}],
            )

    import woyo.agent as agent_mod
    monkeypatch.setattr(agent_mod, "Agent", FakeAgent)

    settings = ws_settings(tmp_path)
    router = ModelRouter(settings, default_provider=MockProvider([]))
    tool = SpawnAgentTool(settings, router)
    result = await tool.run(SpawnAgentArgs(task="find the python downloads page"))
    assert result.ok
    from woyo.agent.citations import collect_observed_urls
    assert "https://www.python.org/downloads/" in collect_observed_urls(result)


async def test_spawn_agent_child_has_no_spawn_tool(tmp_path):
    # the include list used inside agent_tool is the structural guarantee:
    # a child literally cannot register spawn_agent or approval-gated tools
    from woyo.tools.builtin.agent_tool import _CHILD_TOOLS
    assert "spawn_agent" not in _CHILD_TOOLS
    assert "shell_exec" not in _CHILD_TOOLS
    assert "ask_user" not in _CHILD_TOOLS
    assert "write_file" not in _CHILD_TOOLS


async def test_spawn_agent_shares_router_cost(tmp_path):
    """Child + parent usage must land in ONE router account (budget unity)."""
    responses = [
        text_response(PLAN_JSON),
        tool_response([("spawn_agent", {"task": "do the sub thing"})]),
        text_response(PLAN_JSON),
        tool_response([("finish", {"summary": "sub answer", "verified": True})]),
        tool_response([("finish", {"summary": "done", "verified": True})]),
    ]
    agent, mock, bus, registry = _make_spawn_stack(tmp_path, responses)
    await agent.run("delegate please")
    # planner(2) + executors: parent 2 + child 2 = at least 5 model calls on one router
    assert len(mock.calls) >= 5


# --- approvals ----------------------------------------------------------------


async def test_async_approval_callback_supported(tmp_path):
    """The loop must await async approval channels (telegram buttons)."""

    class DangerArgs(BaseModel):
        pass

    class DangerTool(Tool):
        name = "danger_tool"
        description = "needs approval"
        permission = Permission.WRITES_EXTERNAL
        Args = DangerArgs
        ran = False

        async def run(self, args):
            DangerTool.ran = True
            return ToolResult.ok_result("did the dangerous thing")

    responses = [
        text_response(PLAN_JSON),
        tool_response([("danger_tool", {})]),
        tool_response([("finish", {"summary": "all done ok", "verified": True})]),
    ]
    agent, mock, bus = build_agent(responses, tools=[DangerTool(), FinishTool()])

    press = asyncio.Event()

    async def approval_channel(tool, args_json):
        await press.wait()  # simulate a human taking their time to press
        return True

    task = asyncio.create_task(agent.run("use the danger tool", approval_cb=approval_channel))
    await asyncio.sleep(0.05)
    assert not DangerTool.ran, "must not run before approval resolves"
    press.set()
    result = await task
    assert DangerTool.ran
    assert result.outcome == "completed"


# --- telegram inline approvals -------------------------------------------------


def _tg_bot(tmp_path, calls, updates_queue=None):
    from woyo.chat.telegram import TelegramBot

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.read())
        calls.append((method, body))
        if method == "getUpdates":
            queue = updates_queue or []
            return httpx.Response(
                200, json={"ok": True, "result": queue.pop(0) if queue else []}
            )
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 42}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    bot = TelegramBot(
        make_settings(
            sessions_dir=str(tmp_path / "sessions"),
            chat_approval_timeout_s=1,
        ),
        "T:test",
        set(),
        client=client,
        state_path=tmp_path / "state.json",
    )
    bot._owner = 111
    return bot, client


async def test_telegram_approval_approve_flow(tmp_path):
    calls: list[tuple[str, dict]] = []
    bot, client = _tg_bot(tmp_path, calls)

    class SessionWantsShell:
        async def send(self, message, *, approval_cb=None):
            assert approval_cb is not None
            outcome = approval_cb("shell_exec", '{"command": "ls"}')
            assert hasattr(outcome, "__await__")
            approved = await outcome
            from woyo.chat.session import ChatReply
            return ChatReply(answer="ran ls" if approved else "denied")

    bot._sessions[111] = SessionWantsShell()  # type: ignore[assignment]

    answer_task = asyncio.create_task(bot._answer(111, "list my files"))
    await asyncio.sleep(0.05)  # approval message goes out, future pending
    sent = [b for m, b in calls if m == "sendMessage"]
    assert sent and sent[0]["reply_markup"]["inline_keyboard"][0][0]["text"] == "✅ Approve"
    ap_id = sent[0]["reply_markup"]["inline_keyboard"][0][0]["callback_data"].split(":")[1]

    # the press arrives as a callback_query update while the run is waiting
    await bot._dispatch({"callback_query": {
        "id": 7, "data": f"woyo_apr:{ap_id}:1",
        "message": {"chat": {"id": 111}, "message_id": 42},
    }})
    await answer_task
    assert "ran ls" in [b.get("text") for m, b in calls if m == "sendMessage"][-1]
    # the button message gets edited to the verdict
    assert any(m == "editMessageText" for m, _ in calls)
    await client.aclose()


async def test_telegram_approval_timeout_denies(tmp_path):
    calls: list[tuple[str, dict]] = []
    bot, client = _tg_bot(tmp_path, calls)  # chat_approval_timeout_s=1

    class SessionWantsShell:
        async def send(self, message, *, approval_cb=None):
            approved = await approval_cb("shell_exec", '{"command": "rm -rf /"}')
            from woyo.chat.session import ChatReply
            return ChatReply(answer="ok:denied" if not approved else "ok:ran")

    bot._sessions[111] = SessionWantsShell()  # type: ignore[assignment]
    import time
    t0 = time.monotonic()
    await bot._answer(111, "delete everything")
    assert time.monotonic() - t0 >= 0.9, "must wait the timeout before denying"
    assert [b.get("text") for m, b in calls if m == "sendMessage"][-1] == "ok:denied"
    await client.aclose()


async def test_telegram_approval_wrong_chat_ignored(tmp_path):
    calls: list[tuple[str, dict]] = []
    bot, client = _tg_bot(tmp_path, calls)

    class SessionWantsShell:
        async def send(self, message, *, approval_cb=None):
            approved = await approval_cb("shell_exec", '{"command": "ls"}')
            from woyo.chat.session import ChatReply
            return ChatReply(answer="ran" if approved else "denied")

    bot._sessions[111] = SessionWantsShell()  # type: ignore[assignment]
    answer_task = asyncio.create_task(bot._answer(111, "list files"))
    await asyncio.sleep(0.05)
    ap_id = next(iter(bot._pending))

    # a different chat presses Approve — must be ignored
    await bot._dispatch({"callback_query": {
        "id": 9, "data": f"woyo_apr:{ap_id}:1",
        "message": {"chat": {"id": 999}, "message_id": 42},
    }})
    assert ap_id in bot._pending, "wrong-chat press must not consume the approval"
    # the rightful chat denies
    await bot._dispatch({"callback_query": {
        "id": 10, "data": f"woyo_apr:{ap_id}:0",
        "message": {"chat": {"id": 111}, "message_id": 42},
    }})
    await answer_task
    assert [b.get("text") for m, b in calls if m == "sendMessage"][-1] == "denied"
    await client.aclose()


# --- registry integration -------------------------------------------------------


async def test_registry_contains_new_tools_without_router(tmp_path):
    s = ws_settings(tmp_path)
    registry = build_default_registry(s)
    names = registry.names()
    for expected in ("python_exec", "shell_exec", "read_file", "write_file", "list_dir"):
        assert expected in names
    assert "spawn_agent" not in names, "no router -> no sub-agents"


async def test_registry_contains_spawn_agent_with_router(tmp_path):
    s = ws_settings(tmp_path)
    router = ModelRouter(s, default_provider=MockProvider([]))
    registry = build_default_registry(s, router=router)
    assert "spawn_agent" in registry.names()
