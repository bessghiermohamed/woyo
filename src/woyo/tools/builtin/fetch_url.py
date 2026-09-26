"""fetch_url: SSRF-hardened page fetcher with main-content extraction.

Security (see docs/SECURITY.md T2):
- http/https only, standard ports only
- DNS resolved and checked: private / loopback / link-local targets rejected
- <=5 redirects, each re-validated
- 2 MB response cap, content-type allowlist, output char cap
"""

from __future__ import annotations

import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, Field

from woyo.errors import ErrorKind, ToolError
from woyo.tools.base import Tool, ToolResult

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
    ):
        self.allow_private_hosts = allow_private_hosts
        self.allowed_ports = _ALLOWED_PORTS if allowed_ports is None else allowed_ports
        self._client = client

    async def run(self, args: FetchArgs) -> ToolResult:
        url = args.url.strip()
        client = self._client or httpx.AsyncClient(
            timeout=25, follow_redirects=False
        )
        owns_client = self._client is None
        try:
            current = url
            for _ in range(_MAX_REDIRECTS + 1):
                _validate_target(
                    current,
                    allow_private_hosts=self.allow_private_hosts,
                    allowed_ports=self.allowed_ports,
                )
                resp = await client.get(
                    current, headers={"User-Agent": "woyo-agent/0.1 (+research)"}
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
                    return ToolResult.error(
                        ErrorKind.TOOL_FAILURE,
                        f"Unsupported content-type '{ctype}' at {current}.",
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
                title, text = _extract_text(html)
                if not text:
                    return ToolResult.error(
                        ErrorKind.TOOL_FAILURE,
                        f"No extractable text at {current} (empty or non-text page).",
                    )
                truncated = len(text) > args.max_chars
                if truncated:
                    text = text[: args.max_chars] + "\n[... truncated ...]"
                data: dict[str, Any] = {"source": f"web:{urlsplit(current).netloc}",
                                        "url": current, "title": title}
                return ToolResult.ok_result(text, untrusted=True, data=data)
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE, f"Too many redirects fetching {url}."
            )
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
