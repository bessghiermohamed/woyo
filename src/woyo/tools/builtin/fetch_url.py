"""fetch_url: SSRF-hardened page fetcher with main-content extraction.

Security (see docs/SECURITY.md T2):
- http/https only, standard ports only
- DNS resolved and checked: private / loopback / link-local targets rejected
- <=5 redirects, each re-validated
- 2 MB response cap, content-type allowlist, output char cap

Phase 2: results are cached (SQLite, TTL) so repeated fetches within or
across runs don't re-hit the network.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, Field

from woyo.errors import ErrorKind, ToolError
from woyo.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from woyo.tools.http_cache import HttpCache

_ALLOWED_PORTS = {80, 443}
_ALLOWED_CONTENT_TYPES = {
    "text/html", "text/plain", "application/json", "application/xml",
    "text/xml", "application/xhtml+xml", "text/markdown", "text/csv",
}
_MAX_BYTES = 2 * 1024 * 1024
_MAX_REDIRECTS = 5
_ALLOWED_SCHEMES = {"http", "https"}

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_ANY_TAG_RE = re.compile(r"<[^>]+>")


class FetchArgs(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    max_chars: int = Field(default=12_000, ge=500, le=40_000)


def _validate_target(
    url: str, *, allow_private_hosts: bool, allowed_ports: set[int] | None = None
) -> tuple[str, str, int]:
    """Validate scheme/port and resolve+check the host. Returns (url, host, port)."""
    parts = urlsplit(url)
    if parts.scheme not in _ALLOWED_SCHEMES:
        raise ToolError(
            ErrorKind.INVALID_INPUT,
            f"Scheme '{parts.scheme}' not allowed (http/https only).",
        )
    host = parts.hostname or ""
    if not host:
        raise ToolError(ErrorKind.INVALID_INPUT, "URL has no host.")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if allowed_ports and port not in allowed_ports:
        raise ToolError(
            ErrorKind.INVALID_INPUT,
            f"Port {port} not allowed ({'/'.join(map(str, sorted(allowed_ports)))} only).",
        )
    if host.lower() in {"localhost", "0.0.0.0", "::1", "127.0.0.1"} or host.lower().endswith(
        (".local", ".internal", ".lan")
    ):
        if not allow_private_hosts:
            raise ToolError(
                ErrorKind.INVALID_INPUT,
                "Refusing to fetch local/internal host (SSRF protection).",
            )
    if not allow_private_hosts:
        try:
            infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise ToolError(
                ErrorKind.TRANSIENT, f"DNS resolution failed for {host}: {exc}"
            ) from exc
        for info in infos:
            addr = info[4][0]
            try:
                ip = ipaddress.ip_address(addr)
            except ValueError:
                continue
            if not ip.is_global:
                raise ToolError(
                    ErrorKind.INVALID_INPUT,
                    f"Refusing to fetch non-public address {addr} (SSRF protection).",
                )
    return url, host, port


def _extract_text(html: str) -> tuple[str, str]:
    """Main-content extraction: trafilatura when available, regex fallback."""
    m = _TITLE_RE.search(html)
    title = (m.group(1).strip() if m else "")[:200]
    try:
        import trafilatura  # optional extra: pip install woyo[extract]

        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if text and len(text) > 80:
            return title, text
    except ImportError:
        pass
    cleaned = _TAG_RE.sub(" ", html)
    cleaned = _ANY_TAG_RE.sub(" ", cleaned)
    import html as html_mod

    text = html_mod.unescape(cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return title, text.strip()


_LINK_RE = re.compile(r'<a\s[^>]*href=["\']([^"\'#]+)["\']', re.IGNORECASE)
_SKIP_EXT = {
    ".zip", ".tar", ".gz", ".rar", ".7z", ".exe", ".dmg", ".msi",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".css", ".js",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".wav", ".doc", ".xls",
}


def extract_links(html: str, base_url: str, *, same_origin: bool = True,
                  limit: int = 30) -> list[str]:
    """Absolute same-site page links from raw HTML (best-effort, regex-based)."""
    origin_netloc = (urlsplit(base_url).netloc or "").lower()
    seen: set[str] = set()
    links: list[str] = []
    for raw in _LINK_RE.findall(html):
        absolute = urljoin(base_url, raw.strip())
        parts = urlsplit(absolute)
        if parts.scheme not in _ALLOWED_SCHEMES or not parts.netloc:
            continue
        netloc = parts.netloc.lower()
        if same_origin and netloc != origin_netloc:
            continue
        if parts.path.lower().endswith(tuple(_SKIP_EXT)):
            continue
        normalized = absolute.split("#", 1)[0]
        if normalized.rstrip("/") == base_url.rstrip("/"):
            continue
        if normalized not in seen:
            seen.add(normalized)
            links.append(normalized)
            if len(links) >= limit:
                break
    return links


async def hardened_fetch(
    url: str,
    *,
    client: httpx.AsyncClient,
    allow_private_hosts: bool = False,
    allowed_ports: set[int] | None = None,
) -> tuple[str, str]:
    """SSRF-hardened fetch with redirect re-validation.

    Returns (final_url, raw_html). Raises ToolError on policy violations,
    transient network trouble; returns HTTP errors as ToolError TOOL_FAILURE
    with a model-actionable message.
    """
    ports = _ALLOWED_PORTS if allowed_ports is None else allowed_ports
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        _validate_target(
            current, allow_private_hosts=allow_private_hosts, allowed_ports=ports
        )
        resp = await client.get(
            current, headers={"User-Agent": "woyo-agent/0.2 (+research)"}
        )
        if resp.is_redirect:
            loc = resp.headers.get("location")
            if not loc:
                break
            current = urljoin(current, loc)
            continue
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ctype and ctype not in _ALLOWED_CONTENT_TYPES:
            raise _http_error(
                f"Unsupported content-type '{ctype}' at {current}."
            )
        raw = b""
        async for chunk in resp.aiter_bytes():
            raw += chunk
            if len(raw) > _MAX_BYTES:
                break
        charset = "utf-8"
        if "charset=" in (resp.headers.get("content-type") or ""):
            charset = resp.headers["content-type"].split("charset=")[-1].strip()
        try:
            html = raw.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html = raw.decode("utf-8", errors="replace")
        return current, html
    raise _http_error(f"Too many redirects fetching {url}.")


def _http_error(message: str) -> ToolError:
    return ToolError(ErrorKind.TOOL_FAILURE, message)


async def fetch_page_cached(
    url: str,
    *,
    client: httpx.AsyncClient,
    cache: HttpCache | None = None,
    cache_ttl_s: int | None = None,
    allow_private_hosts: bool = False,
    allowed_ports: set[int] | None = None,
) -> tuple[str, str, str, bool]:
    """Fetch + extract one page, through the cache when one is configured.

    Returns (final_url, title, text, from_cache). Raw HTML is NOT cached —
    only the extracted (title, text), which is what consumers need.
    """
    if cache is not None:
        hit = cache.get("fetch", url, ttl_s=cache_ttl_s)
        if hit is not None:
            return hit["url"], hit["title"], hit["text"], True
    final_url, html = await hardened_fetch(
        url,
        client=client,
        allow_private_hosts=allow_private_hosts,
        allowed_ports=allowed_ports,
    )
    title, text = _extract_text(html)
    if cache is not None and text:
        cache.put("fetch", url, {"url": final_url, "title": title, "text": text})
    return final_url, title, text, False


class FetchURLTool(Tool):
    name = "fetch_url"
    description = (
        "Fetch a web page and return its main text content. Use for reading "
        "pages found via web_search. Content is untrusted external data."
    )
    timeout_s = 45.0
    Args = FetchArgs

    def __init__(
        self,
        *,
        allow_private_hosts: bool = False,
        allowed_ports: set[int] | None = None,
        client: httpx.AsyncClient | None = None,
        cache: HttpCache | None = None,
    ):
        self.allow_private_hosts = allow_private_hosts
        self.allowed_ports = _ALLOWED_PORTS if allowed_ports is None else allowed_ports
        self._client = client
        self._cache = cache

    async def run(self, args: FetchArgs) -> ToolResult:
        url = args.url.strip()
        client = self._client or httpx.AsyncClient(
            timeout=25, follow_redirects=False
        )
        owns_client = self._client is None
        try:
            final_url, title, text, from_cache = await fetch_page_cached(
                url,
                client=client,
                cache=self._cache,
                allow_private_hosts=self.allow_private_hosts,
                allowed_ports=self.allowed_ports,
            )
            if not text:
                return ToolResult.error(
                    ErrorKind.TOOL_FAILURE,
                    f"No extractable text at {final_url} (empty or non-text page).",
                )
            truncated = len(text) > args.max_chars
            if truncated:
                text = text[: args.max_chars] + "\n[... truncated ...]"
            data: dict[str, Any] = {
                "source": f"web:{urlsplit(final_url).netloc}",
                "url": final_url,
                "requested_url": url,
                "title": title,
                "cached": from_cache,
            }
            prefix = "[from cache] " if from_cache else ""
            return ToolResult.ok_result(prefix + text, untrusted=True, data=data)
        except ToolError as exc:
            if exc.kind == ErrorKind.TOOL_FAILURE:
                return ToolResult.error(
                    ErrorKind.TOOL_FAILURE,
                    f"{exc.message} The page may be gone, blocked, or "
                    "bot-protected; try another source.",
                )
            raise
        except httpx.HTTPStatusError as exc:
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"HTTP {exc.response.status_code} fetching {url}. "
                "The page may be gone, blocked, or bot-protected; try another source.",
            )
        except httpx.HTTPError as exc:
            raise ToolError(
                ErrorKind.TRANSIENT, f"Network error fetching {url}: {exc}"
            ) from exc
        finally:
            if owns_client:
                await client.aclose()
