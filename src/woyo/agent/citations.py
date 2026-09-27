"""Citation verification (Phase 2): sources claimed vs. sources observed.

The `finish` tool asks the model to cite "URLs you actually used" — but models
hallucinate URLs. This module is the enforcement layer:

1. Every web tool result carries the URLs it exposed:
   - web_search -> data["results"][*]["url"]
   - fetch_url  -> data["url"] (final) + data["requested_url"]
   - crawl_site -> data["url"] + data["pages"][*]["url"]
2. The agent loop collects them into an "observed" set.
3. At finish, claimed sources are matched against observed ones; anything
   unverifiable is DROPPED from the report and flagged — never silently kept.

URLs are normalized (case-insensitive scheme/host, www-stripped, fragment-
free, no trailing slash) so cosmetic differences don't fail a match.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from woyo.tools.base import ToolResult


def normalize_url(url: str) -> str | None:
    """Canonical form for matching, or None if not a plain http(s) URL."""
    if not url:
        return None
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    # citation matching treats http/https as equivalent (same resource)
    scheme = "https"
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, host, path, parts.query, ""))


def collect_observed_urls(result: ToolResult) -> list[str]:
    """URLs a tool result actually exposed to the model."""
    data = result.data
    if not data:
        return []
    urls: list[str] = []
    if isinstance(data.get("url"), str):
        urls.append(data["url"])
    if isinstance(data.get("requested_url"), str):
        urls.append(data["requested_url"])
    for hit in data.get("results") or []:
        if isinstance(hit, dict) and isinstance(hit.get("url"), str):
            urls.append(hit["url"])
    for page in data.get("pages") or []:
        if isinstance(page, dict) and isinstance(page.get("url"), str):
            urls.append(page["url"])
    return urls


def verify_sources(
    claimed: list[dict[str, Any]] | None,
    observed: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split claimed sources into (verified, dropped).

    A source is verified when its normalized URL was observed during the run
    (as a search hit, a fetched page, or a crawled page).
    """
    verified: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for source in claimed or []:
        raw_url = ""
        if isinstance(source, dict):
            raw_url = str(source.get("url") or "")
        normalized = normalize_url(raw_url)
        if normalized and normalized in observed:
            verified.append(source)
        else:
            dropped.append(source)
    return verified, dropped
