"""crawl_site: bounded, same-origin site crawl (Phase 2).

Lets the agent read a small coherent set of pages from one site — e.g. docs
sections or listing pages — without burning tool calls one fetch at a time.

Bounds (all enforced):
- max_pages (default 4, hard cap 10) including the start page
- per-page extraction cap; combined output capped by the registry
- same-origin only by default; SSRF checks apply to every page
- 250ms politeness delay between page fetches
- results go through the shared HTTP cache
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from woyo.errors import ErrorKind, ToolError
from woyo.tools.base import Tool, ToolResult
from woyo.tools.builtin.fetch_url import (
    _extract_text,
    extract_links,
    fetch_page_cached,
    hardened_fetch,
)

if TYPE_CHECKING:
    from woyo.tools.http_cache import HttpCache

_POLITENESS_S = 0.25


class CrawlArgs(BaseModel):
    url: str = Field(min_length=8, max_length=2000, description="Start page")
    max_pages: int = Field(default=4, ge=1, le=10)
    max_chars_per_page: int = Field(default=3_500, ge=500, le=12_000)


class CrawlSiteTool(Tool):
    name = "crawl_site"
    description = (
        "Crawl a few related pages from ONE website and return each page's "
        "main text. Starts at `url`, follows same-site links (breadth-first) "
        "up to `max_pages`. Use when a task needs several pages from the same "
        "site (docs sections, listings). Content is untrusted external data."
    )
    timeout_s = 150.0
    Args = CrawlArgs

    def __init__(
        self,
        *,
        allow_private_hosts: bool = False,
        client: httpx.AsyncClient | None = None,
        cache: HttpCache | None = None,
    ):
        self.allow_private_hosts = allow_private_hosts
        self._client = client
        self._cache = cache

    async def run(self, args: CrawlArgs) -> ToolResult:
        url = args.url.strip()
        client = self._client or httpx.AsyncClient(
            timeout=25, follow_redirects=False
        )
        owns_client = self._client is None
        try:
            # 1) start page: hardened fetch (need raw HTML for links)
            start_final, start_html = await hardened_fetch(
                url, client=client, allow_private_hosts=self.allow_private_hosts
            )
            origin = urlsplit(start_final)
            netloc = origin.netloc.lower()

            # 2) candidate links from the start page
            candidates = extract_links(start_html, start_final, same_origin=True)

            pages: list[dict[str, str]] = []
            sections: list[str] = []
            errors: list[str] = []

            def _section(final_url: str, title: str, text: str, cached: bool) -> None:
                tag = " [cached]" if cached else ""
                snippet = text[: args.max_chars_per_page]
                if len(text) > args.max_chars_per_page:
                    snippet += "\n[... truncated ...]"
                sections.append(
                    f"## {title or '(untitled)'}{tag}\nURL: {final_url}\n{snippet}"
                )
                pages.append({"url": final_url, "title": title})

            start_title, start_text = _extract_text(start_html)
            if self._cache is not None and start_text:
                self._cache.put(
                    "fetch", url,
                    {"url": start_final, "title": start_title, "text": start_text},
                )
            _section(start_final, start_title, start_text, False)

            # 3) breadth-first over same-site links, up to the page budget
            queue = list(candidates)
            visited = {start_final.rstrip("/")}
            while queue and len(pages) < args.max_pages:
                candidate = queue.pop(0)
                if candidate.rstrip("/") in visited:
                    continue
                visited.add(candidate.rstrip("/"))
                try:
                    final_url, title, text, cached = await fetch_page_cached(
                        candidate,
                        client=client,
                        cache=self._cache,
                        allow_private_hosts=self.allow_private_hosts,
                    )
                except ToolError as exc:
                    if exc.kind == ErrorKind.TRANSIENT:
                        raise  # network-level trouble should surface, not crawl on
                    errors.append(f"{candidate}: {exc.message[:120]}")
                    continue
                except httpx.HTTPError as exc:
                    raise ToolError(
                        ErrorKind.TRANSIENT, f"Network error crawling {candidate}: {exc}"
                    ) from exc
                if urlsplit(final_url).netloc.lower() != netloc:
                    continue  # redirect jumped off-site
                if not text:
                    continue
                _section(final_url, title, text, cached)
                await asyncio.sleep(_POLITENESS_S)

            header = (
                f"Crawled {len(pages)} page(s) from {netloc}"
                + (f"; {len(queue)} link(s) left unvisited (page budget reached)"
                   if queue else "")
                + (f"; {len(errors)} link(s) failed" if errors else "")
                + "."
            )
            body = header + "\n\n" + "\n\n---\n\n".join(sections)
            data: dict[str, Any] = {
                "source": f"web:{netloc}",
                "url": start_final,
                "requested_url": url,
                "pages": pages,
            }
            if errors:
                data["errors"] = errors[:5]
            return ToolResult.ok_result(body, untrusted=True, data=data)
        except ToolError as exc:
            if exc.kind == ErrorKind.TOOL_FAILURE:
                return ToolResult.error(ErrorKind.TOOL_FAILURE, exc.message)
            raise
        except httpx.HTTPStatusError as exc:
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"HTTP {exc.response.status_code} fetching {url}. "
                "The site may be gone, blocked, or bot-protected.",
            )
        except httpx.HTTPError as exc:
            raise ToolError(
                ErrorKind.TRANSIENT, f"Network error crawling {url}: {exc}"
            ) from exc
        finally:
            if owns_client:
                await client.aclose()
