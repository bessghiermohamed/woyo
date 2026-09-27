"""Provider preset resolution + g4f adapter tests (no network)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from woyo.config import PROVIDER_PRESETS, Settings, resolve_api_key, resolve_base_url
from woyo.errors import AgentError
from woyo.models.base import Message, ToolSpec

# --- new presets ---------------------------------------------------------------

def test_new_presets_have_base_urls():
    expected = {
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        "xai": "https://api.x.ai/v1",
        "cohere": "https://api.cohere.ai/compatibility/v1",
        "huggingface": "https://router.huggingface.co/v1",
    }
    for name, url in expected.items():
        assert name in PROVIDER_PRESETS
        assert resolve_base_url(name, Settings()) == url


def test_gemini_key_falls_back_to_google_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    assert resolve_api_key("gemini", Settings()) == "g-key"
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    assert resolve_api_key("gemini", Settings()) == "gem-key"


def test_huggingface_key_prefers_hf_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "alt-key")
    assert resolve_api_key("huggingface", Settings()) == "hf-key"
    monkeypatch.delenv("HF_TOKEN")
    assert resolve_api_key("huggingface", Settings()) == "alt-key"


def test_g4f_needs_no_key():
    assert resolve_api_key("g4f", Settings()) == "g4f"


# --- g4f provider ----------------------------------------------------------------

def test_g4f_import_guard(monkeypatch):
    monkeypatch.setitem(sys.modules, "g4f", None)  # forces ImportError
    monkeypatch.setitem(sys.modules, "g4f.client", None)
    from woyo.models.g4f_provider import G4fProvider

    with pytest.raises(AgentError, match="woyo\\[g4f\\]"):
        G4fProvider()


async def test_g4f_adapter_roundtrip(monkeypatch):
    """Fake the g4f client; verify message/response conversion is correct."""
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="hi from g4f",
                        tool_calls=[
                            SimpleNamespace(
                                id="c1",
                                function=SimpleNamespace(
                                    name="web_search",
                                    arguments='{"query": "python"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
        )

    from woyo.models import g4f_provider

    provider = object.__new__(g4f_provider.G4fProvider)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )

    resp = await provider.complete(
        messages=[Message(role="user", content="go")],
        tools=[
            ToolSpec(
                name="web_search",
                description="search",
                parameters={"type": "object", "properties": {}},
            )
        ],
        model="fake-model",
    )

    assert captured["model"] == "fake-model"
    assert captured["messages"] == [{"role": "user", "content": "go"}]
    assert captured["tools"][0]["function"]["name"] == "web_search"
    assert resp.content == "hi from g4f"
    assert resp.tool_calls[0].name == "web_search"
    assert resp.tool_calls[0].arguments == '{"query": "python"}'
    assert resp.usage.input_tokens == 5 and resp.usage.output_tokens == 3


def test_router_builds_g4f_provider(monkeypatch):
    """Router dispatches the name 'g4f' to G4fProvider (mocked construction)."""
    from woyo.models import g4f_provider
    from woyo.models.router import ModelRouter

    class _FakeG4f(g4f_provider.G4fProvider):
        def __init__(self) -> None:  # skip real client init
            self._client = None

    monkeypatch.setattr(g4f_provider, "G4fProvider", _FakeG4f)
    # router imports G4fProvider from the module, so patch the module symbol
    router = ModelRouter(Settings(provider="g4f", model="fake-model"))
    built = router._build_named_provider("g4f")
    assert isinstance(built, _FakeG4f)
