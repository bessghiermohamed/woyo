"""Tool-system unit tests: registry validation, builtins, fetch SSRF, search."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from pydantic import BaseModel

from tests.conftest import EchoTool, SlowTool
from woyo.tools.base import Tool, ToolRegistry, ToolResult
from woyo.tools.builtin.core_tools import CalculateTool, NowTool
from woyo.tools.builtin.fetch_url import FetchURLTool
from woyo.tools.builtin.web_search import WebSearchTool

# --- registry -----------------------------------------------------------------

async def test_registry_unknown_tool():
    registry = ToolRegistry()
    result = await registry.execute("nope", "{}")
    assert not result.ok
    assert result.error_kind == "invalid_input"


async def test_registry_bad_json_arguments():
    registry = ToolRegistry()
    registry.register(EchoTool())
    result = await registry.execute("echo", "{not json")
    assert not result.ok
    assert result.error_kind == "invalid_input"


async def test_registry_schema_validation():
    registry = ToolRegistry()
    registry.register(EchoTool())
    result = await registry.execute("echo", '{"bogus": 1}')
    assert not result.ok
    assert result.error_kind == "invalid_input"
    assert "text" in result.content  # pydantic error names the field


async def test_registry_timeout_is_transient():
    registry = ToolRegistry()
    registry.register(SlowTool())
    result = await registry.execute("slow_tool", "{}")
    assert not result.ok
    assert result.error_kind == "transient"
    assert "timed out" in result.content


async def test_registry_caps_oversized_output():
    class BigArgs(BaseModel):
        pass

    class BigTool(Tool):
        name, description, Args = "big", "huge output", BigArgs

        async def run(self, args):
            return ToolResult.ok_result("x" * 50_000)

    registry = ToolRegistry(cap_chars=1000)
    registry.register(BigTool())
    result = await registry.execute("big", "{}")
    assert len(result.content) < 1200
    assert "truncated" in result.content


# --- calculate / now -----------------------------------------------------------

async def test_calculate_arithmetic():
    from woyo.tools.builtin.core_tools import CalculateArgs

    tool = CalculateTool()
    result = await tool.run(CalculateArgs(expression="2+2*3"))
    assert result.ok
    assert "8" in result.content


async def test_calculate_rejects_imports():
    from woyo.tools.builtin.core_tools import CalculateArgs

    tool = CalculateTool()
    result = await tool.run(CalculateArgs(expression="__import__('os').system('id')"))
    assert not result.ok
    assert result.error_kind == "invalid_input"


async def test_now_reports_utc():
    from woyo.tools.builtin.core_tools import NowArgs

    tool = NowTool(tz="UTC")
    result = await tool.run(NowArgs())
    assert result.ok
    assert "UTC" in result.content


# --- fetch_url SSRF ------------------------------------------------------------

async def test_fetch_url_blocks_private_ip():
    registry = ToolRegistry()
    registry.register(FetchURLTool())
    result = await registry.execute("fetch_url", '{"url": "http://127.0.0.1/x"}')
    assert not result.ok
    assert result.error_kind == "invalid_input"
    assert "SSRF" in result.content or "local" in result.content


async def test_fetch_url_blocks_bad_scheme_and_port():
    registry = ToolRegistry()
    registry.register(FetchURLTool())
    r1 = await registry.execute("fetch_url", '{"url": "ftp://example.com/f"}')
    assert r1.error_kind == "invalid_input"
    r2 = await registry.execute("fetch_url", '{"url": "http://127.0.0.1:8080/f"}')
    assert "Port" in r2.content


@pytest.fixture
def local_http_server():
    """A real local server; the test tool allows private hosts explicitly."""
    handler = type(
        "H",
        (BaseHTTPRequestHandler,),
        {
            "do_GET": lambda self: (
                self.send_response(200),
                self.send_header("Content-Type", "text/html; charset=utf-8"),
                self.end_headers(),
                self.wfile.write(
                    b"<html><title>Test Page</title><body><p>woyo fetch test content</p></body></html>"
                ),
            ),
            "log_message": lambda *a: None,
        },
    )
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/page.html"
    server.shutdown()


async def test_fetch_url_extracts_content(local_http_server):
    tool = FetchURLTool(allow_private_hosts=True, allowed_ports=set())  # test config
    registry = ToolRegistry()
    registry.register(tool)
    result = await registry.execute(
        "fetch_url", f'{{"url": "{local_http_server}"}}'
    )
    assert result.ok
    assert result.untrusted
    assert "woyo fetch test content" in result.content
    assert result.content.startswith("<untrusted")
    assert result.data["url"] == local_http_server


# --- web_search backends --------------------------------------------------------

async def test_web_search_tavily_backend(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    requests_seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        requests_seen.append(_json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "Result A", "url": "https://a.example", "content": "about a"},
                    {"title": "Result B", "url": "https://b.example", "content": "about b"},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    tool = WebSearchTool(backend="tavily", client=client)
    registry = ToolRegistry()
    registry.register(tool)
    result = await registry.execute("web_search", '{"query": "test query"}')
    await client.aclose()
    assert result.ok
    assert result.untrusted
    assert "Result A" in result.content and "https://b.example" in result.content
    assert requests_seen[0]["query"] == "test query"


async def test_web_search_no_backend_available(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("WOYO_SEARCH_BACKEND", raising=False)
    import sys

    monkeypatch.setitem(sys.modules, "ddgs", None)  # force ImportError
    tool = WebSearchTool(backend="ddg")
    registry = ToolRegistry()
    registry.register(tool)
    result = await registry.execute("web_search", '{"query": "anything"}')
    assert not result.ok
    assert result.error_kind == "config"
    assert "woyo[search]" in result.content
