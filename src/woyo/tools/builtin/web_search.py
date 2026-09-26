"""web_search: pluggable search behind one tool (Tavily / Brave / DuckDuckGo).

The agent never knows which backend is active — availability and cost become
configuration, not architecture (ADR-6).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from woyo.errors import ErrorKind, ToolError
from woyo.tools.base import Tool, ToolResult


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=400)
    max_results: int = Field(default=5, ge=1, le=10)


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str


# --- backends ---------------------------------------------------------------

class TavilyBackend:
    """tavily.com — free tier 1,000 credits/month."""

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        self.api_key = api_key
        self._client = client

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        client = self._client or httpx.AsyncClient(timeout=20)
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                SearchHit(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                )
                for r in data.get("results", [])
            ]
        except httpx.HTTPError as exc:
            raise ToolError(ErrorKind.TOOL_FAILURE, f"Tavily search failed: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()


class BraveBackend:
    """brave.com/search/api — free tier available."""

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        self.api_key = api_key
        self._client = client

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        client = self._client or httpx.AsyncClient(timeout=20)
        try:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": max_results},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            hits = []
            for r in (data.get("web") or {}).get("results", []):
                hits.append(
                    SearchHit(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=r.get("description", ""),
                    )
                )
            return hits
        except httpx.HTTPError as exc:
            raise ToolError(ErrorKind.TOOL_FAILURE, f"Brave search failed: {exc}") from exc
        finally:
            if self._client is None:
                await client.aclose()


class DuckDuckGoBackend:
    """Keyless DuckDuckGo via the `ddgs` package (pip install woyo[search])."""

    def __init__(self) -> None:
        try:
            from ddgs import DDGS  # maintained fork of duckduckgo_search
        except ImportError as exc:
            raise ToolError(
                ErrorKind.CONFIG,
                "DuckDuckGo backend needs the 'ddgs' package: pip install 'woyo[search]'",
            ) from exc
        self._ddgs_cls = DDGS

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        import asyncio

        ddgs_cls = self._ddgs_cls

        def _run() -> list[dict[str, Any]]:
            with ddgs_cls() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        try:
            raw = await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001 — ddgs raises assorted errors
            raise ToolError(
                ErrorKind.TRANSIENT, f"DuckDuckGo search failed: {exc}"
            ) from exc
        return [
            SearchHit(title=r.get("title", ""), url=r.get("href", ""),
                      snippet=r.get("body", ""))
            for r in raw
        ]


# --- the tool ----------------------------------------------------------------

class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web. Returns a numbered list of results (title, URL, snippet). "
        "Use specific queries; refine and retry with different terms if results "
        "are poor. Snippets are untrusted external content."
    )
    timeout_s = 30.0
    Args = SearchArgs

    def __init__(self, backend: str | None = None, client: httpx.AsyncClient | None = None):
        self._client = client
        self._backend: Any = None
        self._missing: str | None = None
        chosen = backend or os.environ.get("WOYO_SEARCH_BACKEND") or None
        if not chosen:
            if os.environ.get("TAVILY_API_KEY"):
                chosen = "tavily"
            elif os.environ.get("BRAVE_API_KEY"):
                chosen = "brave"
            else:
                chosen = "ddg"
        self._backend_name = chosen.lower()
        try:
            if self._backend_name == "tavily":
                key = os.environ.get("TAVILY_API_KEY", "")
                self._backend: Any = TavilyBackend(key, client)
            elif self._backend_name == "brave":
                key = os.environ.get("BRAVE_API_KEY", "")
                self._backend = BraveBackend(key, client)
            elif self._backend_name == "ddg":
                self._backend = DuckDuckGoBackend()
            else:
                raise ToolError(
                    ErrorKind.CONFIG,
                    f"Unknown search backend '{chosen}' (expected tavily | brave | ddg)",
                )
        except ToolError as exc:
            # fail soft: tool registers but explains the fix at run time
            self._missing = exc.message

    async def run(self, args: SearchArgs) -> ToolResult:
        if self._missing:
            return ToolResult.error(ErrorKind.CONFIG, self._missing)
        hits = await self._backend.search(args.query, args.max_results)
        if not hits:
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"No results for '{args.query}'. Try different terms.",
            )
        lines = []
        for i, hit in enumerate(hits, 1):
            lines.append(f"{i}. {hit.title}\n   URL: {hit.url}\n   {hit.snippet}")
        return ToolResult.ok_result(
            "\n".join(lines),
            untrusted=True,
            data={
                "source": f"web-search:{self._backend_name}",
                "results": [h.model_dump() for h in hits],
            },
        )
