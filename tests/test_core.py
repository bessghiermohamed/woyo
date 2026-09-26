"""Config resolution, context compaction, session store."""

from __future__ import annotations

from woyo.config import Settings, resolve_api_key, resolve_base_url
from woyo.memory.session import SessionStore
from woyo.memory.working import compact_context
from woyo.models.base import Message, ToolCall

# --- config ---------------------------------------------------------------------

def test_provider_presets_resolution(monkeypatch):
    monkeypatch.delenv("WOYO_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gk-123")
    settings = Settings(provider="groq", model="llama-3.3-70b-versatile")
    assert settings.resolved_api_key() == "gk-123"
    assert settings.resolved_base_url() == "https://api.groq.com/openai/v1"


def test_role_model_overrides():
    settings = Settings(
        provider="mock",
        model="mock/base",
        planner_model="groq:llama-3.3-70b-versatile",
    )
    assert settings.model_for_role("planner") == "groq:llama-3.3-70b-versatile"
    assert settings.model_for_role("executor") == "mock/base"


def test_resolve_api_key_for_named_provider(monkeypatch):
    monkeypatch.delenv("WOYO_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-1")
    settings = Settings(provider="openai", model="x")
    assert resolve_api_key("openrouter", settings) == "or-1"
    assert resolve_base_url("openrouter", settings) == "https://openrouter.ai/api/v1"


# --- compaction -------------------------------------------------------------------

def _big_message_list():
    messages = [
        Message(role="system", content="sys " * 50),
        Message(role="user", content="Task: do things"),
        Message(role="assistant", tool_calls=[ToolCall(id="t1", name="web_search",
                                                       arguments='{}')]),
        Message(role="tool", tool_call_id="t1", name="web_search",
                content="BULK " * 3000),
        Message(role="assistant", tool_calls=[ToolCall(id="t2", name="fetch_url",
                                                       arguments='{}')]),
        Message(role="tool", tool_call_id="t2", name="fetch_url",
                content="MORE BULK " * 3000),
    ]
    return messages


def test_compaction_reduces_context_and_keeps_structure():
    messages = _big_message_list()
    before_total = sum(len(m.content or "") for m in messages)
    collapsed = compact_context(messages, soft_limit_tokens=500)
    after_total = sum(len(m.content or "") for m in messages)
    assert collapsed >= 1
    assert after_total < before_total
    # protocol structure preserved: every tool_call id still has a tool reply
    ids = {tc.id for m in messages if m.tool_calls for tc in m.tool_calls}
    replied = {m.tool_call_id for m in messages if m.role == "tool"}
    assert ids == replied
    # first two messages untouched
    assert messages[0].content.startswith("sys")
    assert messages[1].content.startswith("Task")


def test_compaction_noop_under_limit():
    messages = [Message(role="system", content="small"), Message(role="user", content="x")]
    assert compact_context(messages, soft_limit_tokens=10_000) == 0


# --- session store ------------------------------------------------------------------

def test_session_store_roundtrip(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session_id = store.new_session_id()
    store.append_events(session_id, [{"kind": "run_started", "ts": 1, "task": "t"}])
    store.append_messages(
        session_id,
        [Message(role="user", content="hello"),
         Message(role="assistant", tool_calls=[ToolCall(id="1", name="x", arguments="{}")])],
    )
    sessions = store.list_sessions()
    assert len(sessions) == 1 and sessions[0]["session_id"] == session_id
    events_file = tmp_path / "sessions" / f"{session_id}.events.jsonl"
    assert "run_started" in events_file.read_text()
    transcript = (tmp_path / "sessions" / f"{session_id}.transcript.jsonl").read_text()
    assert "hello" in transcript and "tool_calls" in transcript
