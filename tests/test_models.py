"""Model layer tests: providers (mocked HTTP), message conversion, costs."""

from __future__ import annotations

import json

import httpx
import pytest

from woyo.models.base import Message, ToolCall, ToolSpec, Usage
from woyo.models.costs import estimate_cost_usd
from woyo.models.mock import MockProvider, text_response, tool_response
from woyo.models.openai_compat import OpenAICompatProvider


def _spec(name="echo"):
    return ToolSpec(
        name=name,
        description="test tool",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
    )


def _openai_response_body(tool_call=False):
    message = (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                }
            ],
        }
        if tool_call
        else {"role": "assistant", "content": "Hello there"}
    )
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {"index": 0, "message": message, "finish_reason": "tool_calls" if tool_call else "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }


# --- openai-compat --------------------------------------------------------------

async def test_openai_provider_roundtrip():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_response_body(tool_call=True))

    provider = OpenAICompatProvider(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content=None,
                tool_calls=[ToolCall(id="c0", name="echo", arguments='{"text":"yo"}')]),
        Message(role="tool", tool_call_id="c0", name="echo", content="echo: yo"),
    ]
    resp = await provider.complete(
        messages=messages, tools=[_spec()], model="test-model"
    )
    assert captured["path"].endswith("/chat/completions")
    body = captured["body"]
    assert body["tools"][0]["function"]["name"] == "echo"
    assert body["messages"][3]["tool_call_id"] == "c0"
    assert resp.tool_calls[0].name == "echo"
    assert resp.tool_calls[0].arguments == '{"text": "hi"}'
    assert resp.usage.input_tokens == 11 and resp.usage.output_tokens == 7


async def test_openai_provider_retries_transient():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, json={"error": {"message": "rate limited", "type": "rate_limit"}}
            )
        return httpx.Response(200, json=_openai_response_body())

    provider = OpenAICompatProvider(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    resp = await provider.complete(messages=[Message(role="user", content="x")],
                                   tools=[], model="m")
    assert resp.content == "Hello there"
    assert calls["n"] == 2


async def test_openai_provider_does_not_retry_auth_errors():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(
            401, json={"error": {"message": "bad key", "type": "auth"}}
        )

    from woyo.errors import AgentError

    provider = OpenAICompatProvider(
        api_key="test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(AgentError, match="rejected the request"):
        await provider.complete(messages=[Message(role="user", content="x")],
                                tools=[], model="m")
    assert calls["n"] == 1


# --- anthropic -------------------------------------------------------------------

async def test_anthropic_conversion():
    pytest.importorskip("anthropic")
    from woyo.models.anthropic_provider import AnthropicProvider

    # the anthropic SDK vendors httpx as "httpx2" — use it for mocking
    try:
        import httpx2
    except ImportError:
        pytest.skip("httpx2 not available for anthropic mocking")

    captured = {}

    def handler(request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-test",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "thinking..."},
                    {"type": "tool_use", "id": "tu_1", "name": "echo",
                     "input": {"text": "hi"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 13, "output_tokens": 5},
            },
        )

    provider = AnthropicProvider(
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    messages = [
        Message(role="system", content="sys prompt"),
        Message(role="user", content="go"),
        Message(role="assistant", content=None,
                tool_calls=[ToolCall(id="tu_0", name="echo", arguments='{}')]),
        Message(role="tool", tool_call_id="tu_0", name="echo", content="result 0"),
        Message(role="tool", tool_call_id="tu_1", name="echo", content="result 1"),
    ]
    resp = await provider.complete(messages=messages, tools=[_spec()], model="claude-test")

    body = captured["body"]
    assert body["system"] == "sys prompt"
    assert body["tools"][0]["name"] == "echo"
    # consecutive tool results merged into ONE user turn
    tool_result_msgs = [m for m in body["messages"] if m["role"] == "user"
                        and any(b.get("type") == "tool_result" for b in m["content"])]
    assert len(tool_result_msgs) == 1
    assert len(tool_result_msgs[0]["content"]) == 2
    # tool_use parsed back
    assert resp.tool_calls[0].id == "tu_1"
    assert json.loads(resp.tool_calls[0].arguments) == {"text": "hi"}
    assert resp.usage.input_tokens == 13


# --- mock & costs ------------------------------------------------------------------

async def test_mock_provider_scripting():
    mock = MockProvider([text_response("a"), tool_response([("echo", {"text": "x"})])])
    r1 = await mock.complete(messages=[Message(role="user", content="1")], tools=[], model="m")
    r2 = await mock.complete(messages=[Message(role="user", content="2")],
                             tools=[_spec()], model="m")
    assert r1.content == "a"
    assert r2.tool_calls[0].name == "echo"
    assert len(mock.calls) == 2
    assert mock.calls[1]["tool_names"] == ["echo"]
    assert mock.exhausted


def test_cost_estimates():
    assert estimate_cost_usd("gpt-4o-mini", Usage(input_tokens=1000, output_tokens=1000)) > 0
    assert estimate_cost_usd("totally-unknown-model", Usage(1000, 1000)) == 0.0


def test_split_model_ref():
    from woyo.models.router import split_model_ref

    assert split_model_ref("groq:llama-3.3-70b") == ("groq", "llama-3.3-70b")
    assert split_model_ref("gpt-4o-mini") == (None, "gpt-4o-mini")
    assert split_model_ref("http://localhost:1/x") == (None, "http://localhost:1/x")


# --- router -------------------------------------------------------------------------

async def test_router_role_models_and_usage():
    from woyo.config import Settings
    from woyo.models.router import ModelRouter

    settings = Settings(
        provider="mock", model="mock/base",
        planner_model="mock/planner-x",
    )
    mock = MockProvider([text_response("plan"), text_response("answer")])
    router = ModelRouter(settings, default_provider=mock)
    await router.complete("planner", messages=[Message(role="user", content="p")])
    await router.complete("executor", messages=[Message(role="user", content="e")])
    assert mock.calls[0]["model"] == "mock/planner-x"
    assert mock.calls[1]["model"] == "mock/base"
    summary = router.usage_summary()
    assert summary["total"]["calls"] == 2


class TestCitationMarkupStripping:
    def _resp(self, content):
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )],
            usage=None,
        )

    def test_cohere_co_markers_stripped(self):
        from woyo.models.openai_compat import _from_openai_response

        raw = ("Node.js LTS is v24.21.0</co: 5:[0]>, with EOL "
               "September 2026</co: 5:[0]> and v22</co: 3>.")
        out = _from_openai_response(self._resp(raw), "command-a-03-2025")
        assert "</co" not in out.content
        assert "Node.js LTS is v24.21.0, with EOL September 2026 and v22." == out.content

    def test_plain_text_untouched(self):
        from woyo.models.openai_compat import _from_openai_response

        raw = "A plain answer with <b>html tags</b> kept as-is."
        out = _from_openai_response(self._resp(raw), "gpt-4o-mini")
        assert out.content == raw
