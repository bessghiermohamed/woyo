"""Phase 2 tests: response cache, crawl_site, citation verification, search budget.

All offline: httpx.MockTransport serves a tiny fake site; the agent loop is
driven by the scripted MockProvider.
"""

from __future__ import annotations

import time

import httpx
import pytest
from pydantic import BaseModel

from woyo.agent import Agent
from woyo.agent.citations import (
    collect_observed_urls,
    normalize_url,
    verify_sources,
)
from woyo.events import EventBus
from woyo.models.mock import MockProvider, text_response, tool_response
from woyo.models.router import ModelRouter
from woyo.tools.base import Tool, ToolRegistry, ToolResult
from woyo.tools.builtin.core_tools import FinishTool
from woyo.tools.builtin.crawl_site import CrawlArgs, CrawlSiteTool
from woyo.tools.builtin.fetch_url import FetchArgs, FetchURLTool, extract_links
from woyo.tools.http_cache import HttpCache

# --- fake site ---------------------------------------------------------------

_SITE = {
    "https://docs.example.com/": (
        "<html><title>Docs Home</title><body>"
        "<a href='/guide'>Guide</a> <a href='/api'>API</a> "
        "<a href='https://other.example.com/x'>external</a> "
        "<a href='/img/logo.png'>logo</a>"
        "Welcome to the docs. Chapter one explains the basics of the system."
        "</body></html>"
    ),
    "https://docs.example.com/guide": (
        "<html><title>Guide</title><body>The guide covers installation "
        "and configuration in ten steps, plus troubleshooting.</body></html>"
    ),
    "https://docs.example.com/api": (
        "<html><title>API</title><body>The API reference lists every tool "
        "with typed arguments and examples.</body></html>"
    ),
}


def _site_client() -> tuple[httpx.AsyncClient, dict[str, int]]:
    hits: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        hits[url] = hits.get(url, 0) + 1
        body = _SITE.get(url)
        if body is None:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, text=body, headers={"content-type": "text/html"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), hits


@pytest.fixture
def no_ssrf_guard(monkeypatch):
    """The fake hosts don't resolve; disable DNS-based checks for these tests."""
    import socket

    def fake_getaddrinfo(host, port, proto=None):  # noqa: ANN001
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"93.184.216.{abs(hash(host)) % 254 + 1}", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


# --- http cache ----------------------------------------------------------------

def test_cache_roundtrip_and_ttl(tmp_path):
    cache = HttpCache(tmp_path / "c.db", ttl_s=60)
    assert cache.get("fetch", "u1") is None
    cache.put("fetch", "u1", {"text": "hello"})
    assert cache.get("fetch", "u1") == {"text": "hello"}
    # different kind or key misses
    assert cache.get("search", "u1") is None
    assert cache.get("fetch", "u2") is None
    # per-call TTL override expires immediately
    assert cache.get("fetch", "u1", ttl_s=0) is None
    cache.close()


def test_cache_evicts_oldest_beyond_entry_budget(tmp_path):
    cache = HttpCache(tmp_path / "c.db", ttl_s=60, max_entries=3)
    for i in range(5):
        cache.put("fetch", f"u{i}", {"i": i})
        time.sleep(0.01)
    stats = cache.stats()
    assert stats["entries"] == 3
    # oldest two evicted, newest survive
    assert cache.get("fetch", "u0") is None
    assert cache.get("fetch", "u1") is None
    assert cache.get("fetch", "u4") == {"i": 4}
    cache.close()


def test_cache_skips_oversized_values(tmp_path):
    cache = HttpCache(tmp_path / "c.db", ttl_s=60)
    cache.put("fetch", "big", {"blob": "x" * (600 * 1024)})
    assert cache.get("fetch", "big") is None
    cache.close()


# --- fetch_url with cache --------------------------------------------------------

async def test_fetch_url_uses_cache_on_second_call(no_ssrf_guard, tmp_path):
    client, hits = _site_client()
    cache = HttpCache(tmp_path / "c.db")
    tool = FetchURLTool(client=client, cache=cache)

    r1 = await tool.run(FetchArgs(url="https://docs.example.com/"))
    assert r1.ok and "docs" in r1.content.lower()
    assert r1.data["cached"] is False

    r2 = await tool.run(FetchArgs(url="https://docs.example.com/"))
    assert r2.ok
    assert r2.data["cached"] is True
    assert r2.content.startswith("[from cache]")

    assert hits["https://docs.example.com/"] == 1  # network hit exactly once
    await client.aclose()
    cache.close()


# --- crawl_site -------------------------------------------------------------------

async def test_crawl_site_bounds_and_same_origin(no_ssrf_guard):
    client, hits = _site_client()
    tool = CrawlSiteTool(client=client)

    result = await tool.run(
        CrawlArgs(url="https://docs.example.com/", max_pages=3)
    )
    assert result.ok
    assert result.untrusted
    # start page + 2 followed pages = 3; external link ignored
    pages = result.data["pages"]
    assert len(pages) == 3
    assert all(p["url"].startswith("https://docs.example.com/") for p in pages)
    assert "https://other.example.com/x" not in {p["url"] for p in pages}
    # no binary links fetched
    assert not any("logo.png" in u for u in hits)
    assert "Guide" in result.content and "API" in result.content
    await client.aclose()


async def test_crawl_site_page_budget(no_ssrf_guard):
    client, hits = _site_client()
    tool = CrawlSiteTool(client=client)
    result = await tool.run(CrawlArgs(url="https://docs.example.com/", max_pages=1))
    assert result.ok
    assert len(result.data["pages"]) == 1  # start page only
    await client.aclose()


def test_extract_links_filters():
    html = (
        '<a href="/a">a</a><a href="https://docs.example.com/b">b</a>'
        '<a href="https://evil.com/c">c</a><a href="/d.png">d</a>'
        '<a href="mailto:x@y.z">m</a>'
    )
    links = extract_links(html, "https://docs.example.com/start")
    assert links == ["https://docs.example.com/a", "https://docs.example.com/b"]


# --- citation verification ----------------------------------------------------------

def test_normalize_url():
    assert normalize_url("HTTPS://WWW.Example.com/A/") == "https://example.com/A"
    assert normalize_url("https://example.com/page#frag") == "https://example.com/page"
    assert normalize_url("ftp://example.com/x") is None
    assert normalize_url("not a url") is None


def test_verify_sources_drops_invented():
    observed = {normalize_url("https://real.example.com/a")}
    claimed = [
        {"url": "https://real.example.com/a", "title": "real"},
        {"url": "https://invented.example.com/fake", "title": "fake"},
    ]
    verified, dropped = verify_sources(claimed, observed)
    assert len(verified) == 1 and verified[0]["url"] == "https://real.example.com/a"
    assert len(dropped) == 1 and dropped[0]["url"].startswith("https://invented")


def test_verify_sources_tolerates_cosmetic_differences():
    observed = {normalize_url("https://www.real.com/page/")}
    claimed = [{"url": "http://real.com/page"}]
    verified, dropped = verify_sources(claimed, observed)
    assert len(verified) == 1 and not dropped


def test_collect_observed_urls_shapes():
    r_search = ToolResult.ok_result(
        "x", data={"results": [{"url": "https://a.example/1"}]}
    )
    r_fetch = ToolResult.ok_result(
        "x", data={"url": "https://b.example/final", "requested_url": "https://b.example/orig"}
    )
    r_crawl = ToolResult.ok_result(
        "x", data={"url": "https://c.example/", "pages": [{"url": "https://c.example/p"}]}
    )
    urls = set()
    for r in (r_search, r_fetch, r_crawl):
        urls.update(collect_observed_urls(r))
    assert urls == {
        "https://a.example/1", "https://b.example/final", "https://b.example/orig",
        "https://c.example/", "https://c.example/p",
    }


# --- agent-loop integration -----------------------------------------------------------

class _FakeSearchTool(Tool):
    """Search tool that reports one real hit (for URL observation)."""

    name = "web_search"
    description = "test search"
    Args = type("A", (BaseModel,), {"__annotations__": {}})

    def __init__(self):
        self.calls = 0

    async def run(self, args) -> ToolResult:  # noqa: ANN001
        self.calls += 1
        return ToolResult.ok_result(
            "1. Result\n   URL: https://real.example.com/data\n   snippet",
            untrusted=True,
            data={
                "source": "web-search:fake",
                "results": [{"title": "Result", "url": "https://real.example.com/data",
                             "snippet": "s"}],
            },
        )


async def test_loop_verifies_citations_and_drops_invented():
    from tests.conftest import PLAN_JSON, make_settings

    settings = make_settings()
    bus = EventBus()
    mock = MockProvider(
        [
            text_response(PLAN_JSON),                       # planner
            tool_response([("web_search", {})]),            # executor: search
            tool_response(                                  # executor: finish
                [
                    (
                        "finish",
                        {
                            "summary": "The answer, cross-checked across sources.",
                            "verified": True,
                            "sources": [
                                {"url": "https://real.example.com/data", "title": "real"},
                                {"url": "https://made-up.example.com/nowhere", "title": "fake"},
                            ],
                        },
                    )
                ]
            ),
        ]
    )
    router = ModelRouter(settings, default_provider=mock, bus=bus)
    registry = ToolRegistry(bus=bus)
    registry.register(_FakeSearchTool())
    registry.register(FinishTool())
    agent = Agent(settings, router, registry, bus=bus)

    result = await agent.run("research something")

    assert result.outcome == "completed"
    assert [s["url"] for s in result.sources] == ["https://real.example.com/data"]
    assert result.sources_dropped == ["https://made-up.example.com/nowhere"]
    assert any("could not be verified" in q for q in result.open_questions)
    assert any(e.kind == "citation_flagged" for e in bus.events)


async def test_search_budget_enforced():
    from tests.conftest import PLAN_JSON, make_settings

    settings = make_settings(max_search_calls=1, max_steps=10)
    bus = EventBus()
    mock = MockProvider(
        [
            text_response(PLAN_JSON),
            tool_response([("web_search", {})]),   # allowed (call 1)
            tool_response([("web_search", {})]),   # blocked by budget
            tool_response(                          # model complies and finishes
                [
                    (
                        "finish",
                        {"summary": "Answer from the one search I was allowed.",
                         "verified": False, "sources": []},
                    )
                ]
            ),
        ]
    )
    router = ModelRouter(settings, default_provider=mock, bus=bus)
    registry = ToolRegistry(bus=bus)
    search = _FakeSearchTool()
    search.calls = 0
    registry.register(search)
    registry.register(FinishTool())
    agent = Agent(settings, router, registry, bus=bus)

    result = await agent.run("search twice")

    assert result.outcome == "completed"
    assert search.calls == 1  # second call never executed
    assert any(
        e.kind == "limit_hit" and "search budget" in str(e.data.get("limit", ""))
        for e in bus.events
    )


def test_executor_prompt_has_cross_check_guidance():
    from woyo.agent.prompts import executor_system_prompt

    prompt = executor_system_prompt(
        today="2026-09-28", timezone_name="UTC", task="t", plan_text="p",
        budget_text="b",
    )
    assert "prefer two independent" in prompt
    assert "sources disagree" in prompt
