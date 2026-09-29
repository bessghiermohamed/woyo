"""Browser automation: a Playwright-driven page the agent can actually use.

Phase 5, following the browser-use accessibility-tree pattern: every page is
presented as a compact snapshot — title, URL, a text preview, and numbered
interactive elements (``[12] link "Docs"``). The model references elements by
that number; the tools resolve the number to a tagged element in the live DOM
(``data-woyo-ref``) and drive it with real Playwright events.

Tool set (one long-lived browser session per conversation / task run):

- ``browser_navigate``  — SSRF-validated navigation; returns the snapshot
- ``browser_click``     — click a numbered element (submit-classified elements
                          are refused here and MUST go through browser_submit)
- ``browser_type``      — fill a field, optionally press Enter
- ``browser_extract``  — page or element text (untrusted data)
- ``browser_screenshot``— PNG observation for vision models + workspace copy
- ``browser_submit``    — the "final click" on forms: WRITES_EXTERNAL, so the
                          runtime asks the user first (inline button in chat)
- ``browser_back``      — one step back in history
- ``browser_close``     — release the session (budget resets)

Security posture (allowlists, approvals, limits — see SECURITY.md T12):

- SSRF: same hardened validation as fetch_url (http/https, standard ports,
  DNS-resolved, private/loopback targets refused unless explicitly allowed).
  The browser then navigates by hostname, so re-resolution is a documented
  TOCTOU residual, identical in kind to any fetch-then-connect tool.
- ALLOWLIST: an optional domain allowlist (suffix match) is enforced before
  navigation AND re-checked after every action; violations bounce the page to
  about:blank with an honest error.
- APPROVAL: form submission (and any element classified submit-ish, including
  action-verb links like "delete account") requires browser_submit, which is
  WRITES_EXTERNAL — the approval gate fires before the click. Plain
  browser_click refuses those elements structurally.
- LIMITS: per-session action budget, session TTL, idle close, concurrent
  session cap, element/text/extract caps, downloads disabled, popups closed,
  service workers blocked.
"""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from woyo.config import Settings
from woyo.errors import ErrorKind, ToolError
from woyo.tools.base import Permission, Tool, ToolResult
from woyo.tools.builtin.fetch_url import _validate_target
from woyo.tools.builtin.sandbox import workspace_root

#: A plain desktop UA: the headless marker ("HeadlessChrome") is dropped so
#: pages render their normal variant. This is rendering parity, not an
#: authentication bypass — we make no claim to be a human anywhere else.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

#: Content signatures of bot-wall / challenge interstitials. Hitting one is
#: not an error to retry — the tool must say so and point at search+fetch.
_BOT_WALL_MARKERS = (
    "just a moment",                # cloudflare turnstile
    "checking your browser",        # generic js challenge
    "attention required",           # cloudflare block page
    "enable javascript and cookies",
    "verify you are human",
    "unusual traffic",              # google et al
    "access denied",                # aws/akamai
    "are you a robot",
    "bot verification",
)
_BOT_WALL_STATUSES = {403, 429, 503}

_MAX_ELEMENTS = 60          # rendered in a snapshot
_TEXT_PREVIEW_CHARS = 1_000  # page text preview inside a snapshot
_ANNOTATE_JS = r"""
() => {
  const out = {title: document.title || "", url: location.href, text: "",
               elements: []};
  document.querySelectorAll("[data-woyo-ref]").forEach(
    el => el.removeAttribute("data-woyo-ref"));
  if (!document.body) return out;
  out.text = (document.body.innerText || "").replace(/\s+\n/g, "\n")
    .slice(0, 4000);
  const selector = [
    "a[href]", "button", "input", "select", "textarea", "summary",
    "[role=button]", "[role=link]", "[role=checkbox]", "[role=radio]",
    "[role=textbox]", "[role=menuitem]", "[role=tab]", "[role=switch]",
    "[onclick]", "[contenteditable=true]",
  ].join(",");
  const verbs = /submit|send|post|publish|buy|pay|purchase|checkout|order|confirm|delete|remove|unsubscribe|cancel|approve|reject|accept|sign.?up|log.?in|register|vote|donate|transfer|withdraw|deploy/i;
  let n = 0;
  for (const el of document.querySelectorAll(selector)) {
    if (n >= 150) break;
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    if (style.display === "none" || style.visibility === "hidden") continue;
    if (rect.width === 0 && rect.height === 0) continue;
    if (el.getAttribute("aria-hidden") === "true") continue;
    n += 1;
    el.setAttribute("data-woyo-ref", String(n));
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute("type") || "").toLowerCase();
    let role = el.getAttribute("role") || "";
    if (!role) {
      if (tag === "a") role = "link";
      else if (tag === "button" || tag === "summary") role = "button";
      else if (tag === "select") role = "combobox";
      else if (tag === "textarea") role = "textbox";
      else if (tag === "input")
        role = ["password", "checkbox", "radio"].includes(type) ? type : "textbox";
    }
    const name = (
      el.getAttribute("aria-label") ||
      el.getAttribute("placeholder") ||
      el.getAttribute("title") ||
      el.getAttribute("alt") ||
      (el.innerText || el.value || "")
    ).trim().replace(/\s+/g, " ").slice(0, 120);
    const form = el.closest("form");
    const inForm = !!form;
    let isSubmit = type === "submit" || type === "image" ||
      (tag === "button" && inForm && type !== "button");
    if (!isSubmit && verbs.test(name || "")) {
      // action-verb buttons/links (delete/buy/send/...) — irreversible-ish
      if (tag !== "a" || /delete|remove|logout|buy|pay|checkout|unsubscribe/i
          .test(el.getAttribute("href") || "")) isSubmit = true;
    }
    out.elements.push({
      ref: n, tag, type, role, name,
      value: ((el.value !== undefined ? String(el.value) : "") || "")
        .slice(0, 80),
      checked: !!el.checked,
      href: tag === "a" ? (el.href || "") : "",
      form_action: isSubmit && form
        ? (form.getAttribute("action") || "") : "",
      is_submit: isSubmit,
    });
  }
  return out;
}
"""


def playwright_available() -> bool:
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Domain allowlist helpers
# ---------------------------------------------------------------------------

def _parse_allowlist(raw: str) -> list[str]:
    return [d.strip().lower().lstrip(".") for d in raw.split(",") if d.strip()]


def _host_allowed(host: str, allowlist: list[str]) -> bool:
    """Empty allowlist = the whole public web; otherwise suffix match."""
    host = (host or "").lower()
    if not allowlist:
        return True
    return any(host == d or host.endswith("." + d) for d in allowlist)


def _looks_like_bot_wall(status: int, title: str, text: str) -> str | None:
    if status in _BOT_WALL_STATUSES:
        return f"HTTP {status}"
    haystack = f"{title}\n{text[:800]}".lower()
    for marker in _BOT_WALL_MARKERS:
        if marker in haystack:
            return f"challenge page ({marker!r})"
    return None


def _bot_wall_error(url: str, why: str) -> ToolResult:
    return ToolResult.error(
        ErrorKind.TOOL_FAILURE,
        f"The page at {url} is bot-protected ({why}) — a real browser is "
        "not getting through either. Do NOT retry this site in the browser. "
        "Fall back to web_search and fetch_url for it (they often succeed "
        "where a browser is challenged), or pick a different source.",
    )


# ---------------------------------------------------------------------------
# Session + manager
# ---------------------------------------------------------------------------

class BrowserSession:
    """One chromium context + page, driven by the tools below."""

    def __init__(self, manager: BrowserManager, key: str, page: Any):
        self.manager = manager
        self.key = key
        self.page = page
        self.refs: dict[int, dict[str, Any]] = {}
        self.actions_used = 0
        self.created = time.monotonic()
        self.last_active = time.monotonic()
        self.closed = False
        self._last_url = "about:blank"

    # -- helpers -------------------------------------------------------
    def touch(self) -> None:
        self.last_active = time.monotonic()

    def check_budget(self) -> None:
        """Raise an honest ToolError when the session is out of budget."""
        if self.closed:
            raise ToolError(
                ErrorKind.INVALID_INPUT,
                "This browser session is closed. Start again with "
                "browser_navigate — a fresh session and budget.",
            )
        if time.monotonic() - self.created > self.manager.ttl_s:
            raise ToolError(
                ErrorKind.INVALID_INPUT,
                "This browser session has expired (time limit). Call "
                "browser_navigate to start a fresh one.",
            )
        if self.actions_used >= self.manager.max_actions:
            raise ToolError(
                ErrorKind.INVALID_INPUT,
                f"Browser action budget exhausted ({self.manager.max_actions} "
                "actions for this session). Call browser_close to reset it, "
                "then browser_navigate — or finish with what you have.",
            )

    def spend_action(self) -> None:
        self.actions_used += 1
        self.touch()

    def _validate_url(self, url: str) -> str:
        """Delegate to the manager (policy lives in one place)."""
        return self.manager.validate_url(url)

    async def annotate(self) -> dict[str, Any]:
        """Tag interactive elements and collect the page summary."""
        info = await self.page.evaluate(_ANNOTATE_JS)
        self.refs = {int(e["ref"]): e for e in info.get("elements", [])}
        self._last_url = info.get("url", self._last_url)
        return info

    async def settle(self) -> None:
        """Give navigations/SPAs a moment, without hanging on busy pages."""
        try:
            await self.page.wait_for_load_state(
                "domcontentloaded", timeout=5_000
            )
        except Exception:  # noqa: BLE001 — pages that never settle still snapshot
            pass
        await asyncio.sleep(0.3)

    async def enforce_allowlist(self) -> str | None:
        """After any action: if the page escaped the allowlist, bounce it."""
        if not self.manager.allowlist:
            return None
        host = urlsplit(self.page.url).hostname or ""
        if _host_allowed(host, self.manager.allowlist):
            return None
        blocked = self.page.url
        try:
            await self.page.goto("about:blank")
        except Exception:  # noqa: BLE001 — best-effort bounce
            pass
        return blocked

    def render_snapshot(self, info: dict[str, Any]) -> str:
        """The compact, model-facing page view (browser-use style)."""
        lines = [
            f"PAGE: {info.get('title', '')[:120]} — {info.get('url', '')}",
            f"ACTION BUDGET: {self.actions_used}/{self.manager.max_actions} "
            f"used, session age "
            f"{int(time.monotonic() - self.created)}s/{self.manager.ttl_s}s",
        ]
        text = (info.get("text") or "").strip()
        if text:
            preview = text[:_TEXT_PREVIEW_CHARS]
            more = f" (+{len(text) - _TEXT_PREVIEW_CHARS} more chars)" if len(
                text
            ) > _TEXT_PREVIEW_CHARS else ""
            lines.append("")
            lines.append("PAGE TEXT (preview — browser_extract for more):")
            lines.append(preview + more)
        elements = sorted(
            (e for e in self.refs.values()), key=lambda e: e["ref"]
        )
        lines.append("")
        if not elements:
            lines.append("INTERACTIVE ELEMENTS: none found on this page.")
        else:
            lines.append(
                "INTERACTIVE ELEMENTS (pass the [number] as `ref`):"
            )
            shown = 0
            for e in elements:
                if shown >= _MAX_ELEMENTS:
                    lines.append(
                        f"(… {len(elements) - shown} more elements not shown)"
                    )
                    break
                lines.append(_render_element(e))
                shown += 1
        lines.append(
            "\nNavigate with browser_navigate; act with browser_click / "
            "browser_type; browser_submit (asks the user) for forms and "
            "irreversible actions."
        )
        return "\n".join(lines)


def _render_element(e: dict[str, Any]) -> str:
    ref = e["ref"]
    role = e.get("role") or e.get("tag") or "element"
    name = (e.get("name") or "").strip()
    parts = [f"[{ref}] {role}"]
    if name:
        parts.append(f"\"{name}\"")
    value = e.get("value") or ""
    if value:
        shown = value if len(value) <= 40 else value[:37] + "…"
        parts.append(f"[value: {shown}]")
    if e.get("checked"):
        parts.append("[checked]")
    href = e.get("href") or ""
    if href:
        parts.append(f"-> {href[:160]}")
    if e.get("is_submit"):
        parts.append("— SUBMIT-CLASS: use browser_submit (needs approval)")
    return " ".join(parts)


class BrowserManager:
    """Owns the playwright lifecycle and the sessions keyed by caller."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.max_actions = settings.browser_max_actions
        self.ttl_s = settings.browser_session_ttl_s
        self.idle_close_s = settings.browser_idle_close_s
        self.max_sessions = settings.browser_max_sessions
        self.allow_private_hosts = settings.browser_allow_private_hosts
        self.allowlist = _parse_allowlist(settings.browser_allowlist)
        self._sessions: dict[str, BrowserSession] = {}
        self._playwright: Any = None
        self._browser: Any = None
        self._lock = asyncio.Lock()
        self._launch_attempts = 0

    def validate_url(self, url: str) -> str:
        """SSRF + allowlist validation; usable before any session exists."""
        cleaned, host, _port = _validate_target(
            url, allow_private_hosts=self.allow_private_hosts
        )
        if not _host_allowed(host, self.allowlist):
            allowed = ", ".join(self.allowlist) or "(none configured)"
            raise ToolError(
                ErrorKind.INVALID_INPUT,
                f"Domain '{host}' is not in the browser allowlist "
                f"({allowed}).",
            )
        return cleaned

    # -- lifecycle -----------------------------------------------------
    async def _ensure_browser(self) -> Any:
        if self._browser is not None and self._browser.is_connected():
            return self._browser
        async with self._lock:
            if self._browser is not None and self._browser.is_connected():
                return self._browser
            from playwright.async_api import async_playwright

            self._launch_attempts += 1
            if self._launch_attempts > 3:
                raise ToolError(
                    ErrorKind.CONFIG,
                    "The browser failed to launch repeatedly — giving up "
                    "for this process. Use web_search / fetch_url instead.",
                )
            self._playwright = await async_playwright().start()
            args: list[str] = []
            if self.settings.browser_no_sandbox:
                args.append("--no-sandbox")
            try:
                self._browser = await self._playwright.chromium.launch(
                    headless=True, args=args
                )
            except Exception as exc:  # noqa: BLE001 — launch problems are observations
                await self._stop_playwright()
                raise ToolError(
                    ErrorKind.CONFIG,
                    f"Chromium failed to launch ({type(exc).__name__}: "
                    f"{str(exc)[:200]}). If the host blocks the chromium "
                    "sandbox, set WOYO_BROWSER_NO_SANDBOX=true.",
                ) from exc
            return self._browser

    async def _new_page(self) -> Any:
        browser = await self._ensure_browser()
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent=_USER_AGENT,
            accept_downloads=False,
            service_workers="block",
        )
        # create the main page BEFORE registering the popup handler —
        # "page" fires for every new page, including this first one
        page = await context.new_page()

        async def _close_popups(p: Any) -> None:
            if p is page:
                return
            try:
                await p.close()
            except Exception:  # noqa: BLE001 — popup hygiene is best-effort
                pass

        context.on("page", lambda p: asyncio.create_task(_close_popups(p)))
        return page

    async def get_session(self, key: str) -> BrowserSession:
        """Create-or-reuse the caller's session; sweep stale ones first."""
        await self._sweep()
        session = self._sessions.get(key)
        if session is not None and not session.closed:
            session.touch()  # LRU refresh on use
            return session
        page = await self._new_page()
        session = BrowserSession(self, key, page)
        self._sessions[key] = session
        await self._enforce_cap()  # evict the least-recently-active others
        return session

    async def close_session(self, key: str) -> bool:
        session = self._sessions.pop(key, None)
        if session is None:
            return False
        await _close_session(session)
        return True

    async def _sweep(self) -> None:
        now = time.monotonic()
        stale = [
            key for key, s in self._sessions.items()
            if s.closed
            or now - s.created > self.ttl_s
            or now - s.last_active > self.idle_close_s
        ]
        for key in stale:
            await self.close_session(key)
        await self._enforce_cap()

    async def _enforce_cap(self) -> None:
        """Concurrent-session cap: evict the least recently used."""
        live = [s for s in self._sessions.values() if not s.closed]
        if len(live) <= self.max_sessions:
            return
        live.sort(key=lambda s: s.last_active)
        for session in live[: len(live) - self.max_sessions]:
            await self.close_session(session.key)

    async def close_all(self) -> None:
        for key in list(self._sessions):
            await self.close_session(key)
        await self._stop_browser()

    async def _stop_browser(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass
            self._browser = None
        await self._stop_playwright()

    async def _stop_playwright(self) -> None:
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                pass
            self._playwright = None


async def _close_session(session: BrowserSession) -> None:
    session.closed = True
    try:
        ctx = session.page.context
        await session.page.close()
        await ctx.close()
    except Exception:  # noqa: BLE001 — teardown is best-effort
        pass


# -- module-level default manager (process-wide; CLI + chat reuse it) ------

_default_manager: BrowserManager | None = None


def get_default_manager(settings: Settings) -> BrowserManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = BrowserManager(settings)
    return _default_manager


async def close_default_manager() -> None:
    global _default_manager
    if _default_manager is not None:
        await _default_manager.close_all()
        _default_manager = None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class _BrowserTool(Tool):
    """Shared plumbing: resolve the caller's session on every call."""

    permission = Permission.READ_ONLY

    def __init__(self, settings: Settings, manager: BrowserManager, key: str):
        self._settings = settings
        self._manager = manager
        self._key = key

    async def _session(self) -> BrowserSession:
        return await self._manager.get_session(self._key)

    async def _snapshot_result(
        self, session: BrowserSession, *, action: str
    ) -> ToolResult:
        info = await session.annotate()
        escaped = await session.enforce_allowlist()
        if escaped:
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"After '{action}' the page navigated to {escaped}, whose "
                "domain is not in the browser allowlist — it was blocked "
                "and the page reset. Stay on allowed domains.",
            )
        url = str(info.get("url") or "")
        host = urlsplit(url).hostname or ""
        return ToolResult.ok_result(
            session.render_snapshot(info),
            untrusted=True,
            data={"source": f"browser:{host}", "url": url},
        )


class NavigateArgs(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    wait_s: float = Field(default=1.0, ge=0.0, le=10.0,
                          description="Extra settle time for JS-rendered pages")


class BrowserNavigateTool(_BrowserTool):
    name = "browser_navigate"
    description = (
        "Open a web page in the bot's real browser and get its snapshot: "
        "title, URL, text preview, and numbered interactive elements. "
        "Use it when fetch_url fails or when you must click/type on a "
        "page (search forms, paginated lists, JS-only sites). The URL must "
        "be http(s) and public. If the page is bot-protected you get an "
        "honest error — then fall back to web_search/fetch_url."
    )
    timeout_s = 60.0
    Args = NavigateArgs

    async def run(self, args: NavigateArgs) -> ToolResult:
        # validate BEFORE opening a session: a refused URL must not even
        # spin up a browser
        try:
            url = self._manager.validate_url(args.url.strip())
        except ToolError as exc:
            return ToolResult.error(exc.kind, exc.message)
        session = await self._session()
        session.check_budget()
        session.spend_action()
        try:
            resp = await session.page.goto(
                url, wait_until="domcontentloaded", timeout=30_000
            )
        except Exception as exc:  # noqa: BLE001 — navigation problems are observations
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Navigation to {url} failed: {type(exc).__name__}: "
                f"{str(exc)[:200]}. The site may be down or unreachable — "
                "try fetch_url on it, or another source.",
            )
        if args.wait_s:
            await asyncio.sleep(min(args.wait_s, 10.0))
        status = getattr(resp, "status", 200) if resp is not None else 200
        title = await _safe_title(session)
        probe = ""
        try:
            probe = (title or "") + " " + (
                (await session.page.evaluate(
                    "() => document.body ? document.body.innerText.slice(0, 600) : ''"
                )) or ""
            )
        except Exception:  # noqa: BLE001 — sniffing is best-effort
            pass
        why = _looks_like_bot_wall(status, title, probe)
        if why:
            return _bot_wall_error(url, why)
        return await self._snapshot_result(session, action=f"navigate {url}")


class ClickArgs(BaseModel):
    ref: int = Field(ge=1, le=999,
                     description="Element number from the last snapshot")


class BrowserClickTool(_BrowserTool):
    name = "browser_click"
    description = (
        "Click a numbered element from the last browser snapshot (links, "
        "buttons, tabs, checkboxes). Submit-classified elements (forms, "
        "buy/delete/send buttons) are REFUSED here — use browser_submit "
        "for those so the user can approve first. If the page changed "
        "since the snapshot, you get an honest error; navigate again."
    )
    timeout_s = 45.0
    Args = ClickArgs

    async def run(self, args: ClickArgs) -> ToolResult:
        session = await self._session()
        session.check_budget()
        await session.annotate()  # refresh refs — pages mutate constantly
        el = session.refs.get(args.ref)
        if el is None:
            return _unknown_ref(session)
        if el.get("is_submit"):
            return ToolResult.error(
                ErrorKind.NEEDS_APPROVAL,
                f"Element [{args.ref}] is submit-classified "
                f"(\"{el.get('name', '')}\") — clicking it may act on the "
                "world irreversibly. Use browser_submit with the same ref: "
                "the user gets an approval button first.",
            )
        if not await self._click_ref(session, args.ref):
            return _click_failed()
        return await self._snapshot_result(session, action=f"click [{args.ref}]")

    async def _click_ref(self, session: BrowserSession, ref: int) -> bool:
        session.spend_action()
        try:
            await session.page.locator(
                f'[data-woyo-ref="{ref}"]'
            ).click(timeout=8_000)
            await session.settle()
            return True
        except Exception:  # noqa: BLE001 — clicks fail for living pages
            return False


class TypeArgs(BaseModel):
    ref: int = Field(ge=1, le=999,
                     description="Element number from the last snapshot")
    text: str = Field(min_length=1, max_length=4_000)
    press_enter: bool = Field(
        default=False,
        description="Press Enter after typing (submits search boxes and forms)",
    )


class BrowserTypeTool(_BrowserTool):
    name = "browser_type"
    description = (
        "Type text into a numbered input/textarea from the last snapshot "
        "(search boxes, form fields). Set press_enter=true to submit the "
        "field — fine for search boxes; for logins, purchases, posts and "
        "anything irreversible, leave it false and use browser_submit on "
        "the form's button so the user approves first. Password fields "
        "can be filled but never Enter-submitted."
    )
    timeout_s = 30.0
    Args = TypeArgs

    async def run(self, args: TypeArgs) -> ToolResult:
        session = await self._session()
        session.check_budget()
        await session.annotate()
        el = session.refs.get(args.ref)
        if el is None:
            return _unknown_ref(session)
        if args.press_enter and (el.get("type") == "password"):
            return ToolResult.error(
                ErrorKind.NEEDS_APPROVAL,
                "Pressing Enter in a password field submits the form — "
                "that needs approval. Fill the field, then use "
                "browser_submit on the form's submit button.",
            )
        session.spend_action()
        try:
            await session.page.locator(
                f'[data-woyo-ref="{args.ref}"]'
            ).fill(args.text, timeout=8_000)
            if args.press_enter:
                await session.page.keyboard.press("Enter")
            await session.settle()
        except Exception as exc:  # noqa: BLE001 — typing fails are observations
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Typing into [{args.ref}] failed: "
                f"{type(exc).__name__}: {str(exc)[:160]}. The element may "
                "not be a text field, or the page changed — take a new "
                "snapshot with browser_navigate.",
            )
        return await self._snapshot_result(
            session, action=f"type into [{args.ref}]"
        )


class ExtractArgs(BaseModel):
    ref: int | None = Field(
        default=None, ge=1, le=999,
        description="Optional element number to extract (page text otherwise)",
    )
    max_chars: int = Field(default=6_000, ge=500, le=20_000)


class BrowserExtractTool(_BrowserTool):
    name = "browser_extract"
    description = (
        "Read the full text of the current browser page (or of one "
        "numbered element) — more complete than the snapshot preview. "
        "Content is untrusted external data; cite the page URL when you "
        "use it as a source."
    )
    timeout_s = 30.0
    Args = ExtractArgs

    async def run(self, args: ExtractArgs) -> ToolResult:
        session = await self._session()
        session.check_budget()
        session.spend_action()
        try:
            if args.ref is None:
                text = await session.page.evaluate(
                    "() => document.body ? document.body.innerText : ''"
                )
            else:
                await session.annotate()
                if args.ref not in session.refs:
                    return _unknown_ref(session)
                text = await session.page.locator(
                    f'[data-woyo-ref="{args.ref}"]'
                ).inner_text(timeout=8_000)
        except Exception as exc:  # noqa: BLE001 — extraction problems are observations
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Extracting page text failed: {type(exc).__name__}: "
                f"{str(exc)[:160]}.",
            )
        text = str(text or "").strip()
        if not text:
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                "The page has no extractable text (empty, image-only, or "
                "still loading). Try browser_screenshot, or wait and "
                "extract again.",
            )
        truncated = len(text) > args.max_chars
        if truncated:
            text = text[: args.max_chars] + "\n[... truncated ...]"
        url = session.page.url
        host = urlsplit(url).hostname or ""
        return ToolResult.ok_result(
            text, untrusted=True,
            data={"source": f"browser:{host}", "url": url},
        )


class ScreenshotArgs(BaseModel):
    full_page: bool = Field(default=False,
                            description="Capture the whole scrollable page")


class BrowserScreenshotTool(_BrowserTool):
    name = "browser_screenshot"
    description = (
        "Take a PNG screenshot of the current browser page and attach it "
        "as an image observation (vision models see it directly). A copy "
        "is saved in the workspace under screenshots/ — deliver it with "
        "send_file if the user wants it. Use when text extraction misses "
        "layout/charts, or to verify what the page actually shows."
    )
    timeout_s = 45.0
    Args = ScreenshotArgs

    async def run(self, args: ScreenshotArgs) -> ToolResult:
        session = await self._session()
        session.check_budget()
        session.spend_action()
        try:
            png = await session.page.screenshot(
                type="png", full_page=args.full_page
            )
        except Exception as exc:  # noqa: BLE001 — screenshot problems are observations
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Screenshot failed: {type(exc).__name__}: {str(exc)[:160]}.",
            )
        if not png:
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE, "Screenshot returned no bytes."
            )
        # workspace copy (send_file-able proof of what was seen)
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        rel = f"screenshots/browser_{stamp}.png"
        try:
            path = workspace_root(self._settings) / "screenshots"
            path.mkdir(parents=True, exist_ok=True)
            (workspace_root(self._settings) / rel).write_bytes(png)
            saved = rel
        except OSError:
            saved = ""
        uri = "data:image/png;base64," + base64.b64encode(png).decode()
        url = session.page.url
        data: dict[str, Any] = {"url": url, "bytes": len(png)}
        if saved:
            data["path"] = saved
        return ToolResult.ok_result(
            f"Screenshot taken ({len(png) / 1024:.0f} KB"
            + (f", saved to {saved}" if saved else "")
            + "). It is attached below as an image — describe/use it from "
            "what you actually see.",
            images=[uri],
            data=data,
        )


class SubmitArgs(BaseModel):
    ref: int = Field(ge=1, le=999,
                     description="Element number to submit (the final click)")


class BrowserSubmitTool(_BrowserTool):
    name = "browser_submit"
    description = (
        "The FINAL click that acts on the world: submitting a form, "
        "confirming a purchase, posting a comment, deleting something. "
        "The user gets an approval button naming the target BEFORE the "
        "click happens. Use it for submit-classified elements (marked in "
        "snapshots) and for any action you would not take without "
        "permission. If denied, do not retry — report and move on."
    )
    permission = Permission.WRITES_EXTERNAL  # the irreversible click
    timeout_s = 60.0
    Args = SubmitArgs

    async def run(self, args: SubmitArgs) -> ToolResult:
        session = await self._session()
        session.check_budget()
        await session.annotate()
        el = session.refs.get(args.ref)
        if el is None:
            return _unknown_ref(session)
        # allowlist the form target too, when we can see it
        action = str(el.get("form_action") or "")
        if action:
            from urllib.parse import urljoin

            target = urljoin(session.page.url, action)
            host = urlsplit(target).hostname or ""
            if not _host_allowed(host, self._manager.allowlist):
                return ToolResult.error(
                    ErrorKind.INVALID_INPUT,
                    f"The form submits to '{host}', which is not in the "
                    "browser allowlist — refused.",
                )
        session.spend_action()
        try:
            await session.page.locator(
                f'[data-woyo-ref="{args.ref}"]'
            ).click(timeout=8_000)
            await session.settle()
        except Exception as exc:  # noqa: BLE001 — clicks fail on living pages
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Submitting [{args.ref}] failed: {type(exc).__name__}: "
                f"{str(exc)[:160]}. The element may have moved — take a "
                "new snapshot and retry once.",
            )
        result = await self._snapshot_result(
            session, action=f"submit [{args.ref}]"
        )
        if result.ok:
            result.content = (
                "Submitted (the approved click went through).\n"
                + result.content
            )
        return result


class BackArgs(BaseModel):
    pass


class BrowserBackTool(_BrowserTool):
    name = "browser_back"
    description = (
        "Go one step back in the browser's history (like the back button) "
        "and get a fresh snapshot. Useful after following a dead end."
    )
    timeout_s = 30.0
    Args = BackArgs

    async def run(self, args: BackArgs) -> ToolResult:
        session = await self._session()
        session.check_budget()
        session.spend_action()
        try:
            await session.page.go_back(
                wait_until="domcontentloaded", timeout=15_000
            )
            await session.settle()
        except Exception as exc:  # noqa: BLE001 — history problems are observations
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Going back failed: {type(exc).__name__}: {str(exc)[:160]}.",
            )
        return await self._snapshot_result(session, action="back")


class CloseArgs(BaseModel):
    pass


class BrowserCloseTool(_BrowserTool):
    name = "browser_close"
    description = (
        "Close this browser session (frees the browser, resets the action "
        "budget). Call it when browsing is done, or when the session is "
        "exhausted and you want a fresh one."
    )
    timeout_s = 20.0
    Args = CloseArgs

    async def run(self, args: CloseArgs) -> ToolResult:
        closed = await self._manager.close_session(self._key)
        if not closed:
            return ToolResult.ok_result(
                "No browser session was open for this conversation."
            )
        return ToolResult.ok_result(
            "Browser session closed. A fresh one (with a full action "
            "budget) starts on the next browser_navigate."
        )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_browser_tools(
    settings: Settings, manager: BrowserManager, key: str
) -> list[Tool]:
    """All browser tools, or [] when playwright/the feature is unavailable."""
    if not settings.browser_enabled or not playwright_available():
        return []
    return [
        BrowserNavigateTool(settings, manager, key),
        BrowserClickTool(settings, manager, key),
        BrowserTypeTool(settings, manager, key),
        BrowserExtractTool(settings, manager, key),
        BrowserScreenshotTool(settings, manager, key),
        BrowserSubmitTool(settings, manager, key),
        BrowserBackTool(settings, manager, key),
        BrowserCloseTool(settings, manager, key),
    ]


def _unknown_ref(session: BrowserSession) -> ToolResult:
    known = ", ".join(
        f"[{e['ref']}] {(e.get('name') or e.get('role') or '?')[:30]}"
        for e in sorted(session.refs.values(), key=lambda e: e["ref"])[:15]
    ) or "none found"
    return ToolResult.error(
        ErrorKind.INVALID_INPUT,
        f"That element number is not on the current page. Elements now: "
        f"{known}. Use the numbers from the LATEST snapshot.",
    )


def _click_failed() -> ToolResult:
    return ToolResult.error(
        ErrorKind.TOOL_FAILURE,
        "Clicking that element failed — it may be covered, off-screen, or "
        "the page changed since the snapshot. Take a fresh snapshot "
        "(browser_navigate to the same URL) and try once; if it still "
        "fails, use browser_extract or another path.",
    )


async def _safe_title(session: BrowserSession) -> str:
    try:
        return await session.page.title() or ""
    except Exception:  # noqa: BLE001 — titles are cosmetic
        return ""


__all__ = [
    "BrowserManager",
    "BrowserSession",
    "build_browser_tools",
    "get_default_manager",
    "close_default_manager",
    "playwright_available",
]
