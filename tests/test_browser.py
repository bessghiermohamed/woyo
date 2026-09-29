"""Browser tool tests (Phase 5).

Two layers, same contract:

- Mock-driven tests (default, run everywhere): a FakePage implementing the
  tiny playwright surface the tools use. No network, no chromium.
- Real-browser tests (opt-in via WOYO_TEST_REAL_BROWSER=1): a local http
  server + real headless chromium — they validate the DOM walker JS and the
  real event fidelity (clicks, fills, screenshots). Run before shipping.
"""

from __future__ import annotations

import asyncio
import base64
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest
from pydantic import BaseModel

from tests.conftest import PLAN_JSON, make_settings
from woyo.agent import Agent
from woyo.config import Settings
from woyo.events import EventBus
from woyo.models.mock import MockProvider, text_response, tool_response
from woyo.models.router import ModelRouter
from woyo.tools.base import Permission, Tool, ToolRegistry, ToolResult
from woyo.tools.builtin.browser import (
    BrowserManager,
    BrowserSubmitTool,
    build_browser_tools,
    playwright_available,
)
from woyo.tools.builtin.sandbox import workspace_root

# ---------------------------------------------------------------------------
# Fakes: the playwright surface the tools actually touch
# ---------------------------------------------------------------------------

ELEMENTS = [
    {"ref": 1, "tag": "a", "type": "", "role": "link", "name": "Next page",
     "value": "", "checked": False, "href": "https://example.com/page2",
     "form_action": "", "is_submit": False},
    {"ref": 2, "tag": "input", "type": "text", "role": "textbox", "name": "Search",
     "value": "", "checked": False, "href": "", "form_action": "",
     "is_submit": False},
    {"ref": 3, "tag": "input", "type": "password", "role": "password",
     "name": "Password", "value": "", "checked": False, "href": "",
     "form_action": "", "is_submit": False},
    {"ref": 4, "tag": "button", "type": "submit", "role": "button",
     "name": "Log in", "value": "", "checked": False, "href": "",
     "form_action": "/login", "is_submit": True},
    {"ref": 5, "tag": "a", "type": "", "role": "link",
     "name": "Delete my account", "value": "", "checked": False,
     "href": "https://example.com/account/delete", "form_action": "",
     "is_submit": True},
]


class FakeResponse:
    def __init__(self, status: int = 200):
        self.status = status


class FakeLocator:
    def __init__(self, page: FakePage, selector: str):
        self._page = page
        self._selector = selector

    async def click(self, timeout: float | None = None) -> None:
        if self._page.fail_clicks:
            raise RuntimeError("element is not visible")
        self._page.clicks.append(self._selector)

    async def fill(self, text: str, timeout: float | None = None) -> None:
        self._page.fills.append((self._selector, text))

    async def inner_text(self, timeout: float | None = None) -> str:
        return self._page.element_text


class FakeContext:
    def __init__(self, page: FakePage):
        self.page = page
        self.closed = False

    def on(self, _event: str, _handler) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class FakePage:
    """Canned playwright page: records actions, serves scripted content."""

    def __init__(
        self,
        *,
        url: str = "https://example.com/page",
        title: str = "Example Page",
        text: str = "Welcome to the example page.\nSecond line of content.",
        elements: list[dict] | None = None,
        goto_status: int = 200,
        element_text: str = "element-scoped text",
        fail_clicks: bool = False,
    ):
        self.url = url
        self._title = title
        self._text = text
        self._elements = elements if elements is not None else ELEMENTS
        self.goto_status = goto_status
        self.element_text = element_text
        self.fail_clicks = fail_clicks
        self.clicks: list[str] = []
        self.fills: list[tuple[str, str]] = []
        self.presses: list[str] = []
        self.goto_calls: list[str] = []
        self.closed = False
        self.context = FakeContext(self)

    async def goto(self, url: str, **_kw):
        self.goto_calls.append(url)
        self.url = url
        return FakeResponse(self.goto_status)

    async def evaluate(self, js: str):
        if "data-woyo-ref" in js:  # the DOM walker
            return {
                "title": self._title,
                "url": self.url,
                "text": self._text,
                "elements": [dict(e) for e in self._elements],
            }
        if "innerText" in js:  # extract / bot-wall probe
            return self._text
        return ""

    async def title(self) -> str:
        return self._title

    def locator(self, selector: str) -> FakeLocator:
        return FakeLocator(self, selector)

    @property
    def keyboard(self):
        class _Kb:
            def __init__(self, page):
                self._page = page

            async def press(self, key: str) -> None:
                self._page.presses.append(key)

        return _Kb(self)

    async def screenshot(self, **_kw) -> bytes:
        return b"\x89PNG\r\n\x1a\nfake-screenshot-bytes"

    async def go_back(self, **_kw):
        self.url = "https://example.com/page"
        return FakeResponse(200)

    async def wait_for_load_state(self, *_a, **_kw) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class FakeManager(BrowserManager):
    """A BrowserManager whose pages are FakePages (no chromium, no network).

    ``validate_url`` mirrors the real policy minus DNS resolution (CI runs
    offline); the DNS/private-IP layer is shared with fetch_url and covered
    by its tests + the real-browser layer here.
    """

    _LOCAL_HOSTS = {"localhost", "0.0.0.0", "::1", "127.0.0.1"}

    def __init__(self, settings, page_factory):
        super().__init__(settings)
        self._page_factory = page_factory
        self.pages: list[FakePage] = []

    def validate_url(self, url: str) -> str:
        from woyo.errors import ErrorKind, ToolError

        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise ToolError(ErrorKind.INVALID_INPUT, "bad scheme (fake)")
        host = (parts.hostname or "").lower()
        if (
            not host
            or host in self._LOCAL_HOSTS
            or host.endswith((".local", ".internal", ".lan"))
        ):
            raise ToolError(ErrorKind.INVALID_INPUT, "local target refused (fake)")
        if not _host_allowed_for_tests(host, self.allowlist):
            raise ToolError(ErrorKind.INVALID_INPUT, "not in allowlist (fake)")
        return url

    async def _new_page(self):
        page = self._page_factory()
        self.pages.append(page)
        return page


def _host_allowed_for_tests(host: str, allowlist: list[str]) -> bool:
    if not allowlist:
        return True
    return any(host == d or host.endswith("." + d) for d in allowlist)


def browser_settings(**kw) -> Settings:
    defaults = dict(
        provider="mock", model="mock/test", browser_enabled=True,
        browser_allow_private_hosts=False,
    )
    defaults.update(kw)
    return make_settings(**defaults)


def make_tools(manager: BrowserManager, key: str = "test", settings=None):
    settings = settings or manager.settings
    return build_browser_tools(settings, manager, key)


async def tool_run(tool, **args):
    return await tool.run(tool.Args(**args))


# ---------------------------------------------------------------------------
# availability + assembly
# ---------------------------------------------------------------------------

def test_no_browser_tools_when_disabled():
    s = browser_settings(browser_enabled=False)
    assert build_browser_tools(s, BrowserManager(s), "k") == []


@pytest.mark.skipif(not playwright_available(), reason="playwright not installed")
def test_eight_browser_tools_registered():
    s = browser_settings()
    manager = BrowserManager(s)
    tools = make_tools(manager)
    assert [t.name for t in tools] == [
        "browser_navigate", "browser_click", "browser_type",
        "browser_extract", "browser_screenshot", "browser_submit",
        "browser_back", "browser_close",
    ]
    # only the final click is approval-gated
    perms = {t.name: t.permission for t in tools}
    assert perms["browser_submit"] == Permission.WRITES_EXTERNAL
    assert all(
        v == Permission.READ_ONLY for k, v in perms.items() if k != "browser_submit"
    )


@pytest.mark.skipif(not playwright_available(), reason="playwright not installed")
def test_registry_includes_browser_tools(tmp_path):
    from woyo.tools.builtin import build_default_registry

    registry = build_default_registry(browser_settings())
    names = registry.names()
    assert "browser_navigate" in names and "browser_submit" in names
    # sub-agents keep their read/compute include-list: no browser there
    from woyo.tools.builtin.agent_tool import _CHILD_TOOLS

    assert not any(n.startswith("browser_") for n in _CHILD_TOOLS)


# ---------------------------------------------------------------------------
# navigate: SSRF, allowlist, bot walls, snapshots
# ---------------------------------------------------------------------------

async def test_navigate_snapshot_happy_path():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_navigate"], url="https://example.com/page")
    assert result.ok
    assert "PAGE: Example Page — https://example.com/page" in result.content
    assert "[1] link \"Next page\"" in result.content
    assert "[4] button \"Log in\" — SUBMIT-CLASS" in result.content
    assert "PAGE TEXT (preview" in result.content
    assert result.untrusted  # page content is external data
    assert result.data["url"] == "https://example.com/page"


async def test_navigate_refuses_local_targets():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    for url in (
        "http://localhost:8000/admin",
        "http://127.0.0.1/x",
        "file:///etc/passwd",
        "ftp://example.com/file",
    ):
        result = await tool_run(tools["browser_navigate"], url=url)
        assert not result.ok
        assert result.error_kind == "invalid_input"
    assert manager.pages == []  # nothing was ever fetched


async def test_navigate_allowlist_blocks_other_domains():
    s = browser_settings(browser_allowlist="wikipedia.org, github.com")
    manager = FakeManager(s, FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_navigate"], url="https://example.com/page")
    assert not result.ok
    assert "allowlist" in result.content
    # wikipedia.org itself is fine (suffix match)
    ok = await tool_run(tools["browser_navigate"], url="https://en.wikipedia.org/wiki/X")
    assert ok.ok


async def test_navigate_bot_wall_degrades_honestly():
    page = FakePage(title="Just a moment...", goto_status=403)
    manager = FakeManager(browser_settings(), lambda: page)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_navigate"], url="https://blocked.example/x")
    assert not result.ok
    assert "bot-protected" in result.content
    assert "web_search" in result.content and "fetch_url" in result.content
    assert "Do NOT retry" in result.content


async def test_navigate_failure_is_an_observation():
    page = FakePage(fail_clicks=True)

    async def boom(url, **kw):
        raise RuntimeError("net::ERR_NAME_NOT_RESOLVED")

    page.goto = boom
    manager = FakeManager(browser_settings(), lambda: page)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_navigate"], url="https://gone.example/")
    assert not result.ok
    assert "try fetch_url" in result.content


# ---------------------------------------------------------------------------
# click / type / submit: the approval story
# ---------------------------------------------------------------------------

async def test_click_plain_element_clicks_and_snapshots():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_click"], ref=1)
    assert result.ok
    assert manager.pages[0].clicks == ['[data-woyo-ref="1"]']
    assert "[1] link" in result.content  # fresh snapshot attached


async def test_click_refuses_submit_classified_elements():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    for ref in (4, 5):  # submit button + action-verb link
        result = await tool_run(tools["browser_click"], ref=ref)
        assert not result.ok
        assert result.error_kind == "needs_approval"
        assert "browser_submit" in result.content
    assert manager.pages[0].clicks == []  # nothing was clicked


async def test_click_unknown_ref_lists_current_elements():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_click"], ref=99)
    assert not result.ok
    assert "not on the current page" in result.content
    assert "[1] Next page" in result.content  # known refs listed


async def test_click_failure_is_honest():
    manager = FakeManager(browser_settings(), lambda: FakePage(fail_clicks=True))
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_click"], ref=1)
    assert not result.ok
    assert "Clicking that element failed" in result.content


async def test_type_fills_and_can_press_enter():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(
        tools["browser_type"], ref=2, text="woyo agent", press_enter=True
    )
    assert result.ok
    page = manager.pages[0]
    assert page.fills == [('[data-woyo-ref="2"]', "woyo agent")]
    assert page.presses == ["Enter"]


async def test_type_password_enter_needs_submit():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(
        tools["browser_type"], ref=3, text="hunter2", press_enter=True
    )
    assert not result.ok
    assert result.error_kind == "needs_approval"
    assert "browser_submit" in result.content
    assert manager.pages[0].presses == []  # Enter never pressed


async def test_submit_clicks_the_final_element():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_submit"], ref=4)
    assert result.ok
    assert result.content.startswith("Submitted (the approved click went through)")
    assert manager.pages[0].clicks == ['[data-woyo-ref="4"]']


async def test_submit_checks_form_action_allowlist():
    s = browser_settings(browser_allowlist="example.com")
    # submit target posts to evil.example — refused even though the page is allowed
    elements = [dict(e) for e in ELEMENTS]
    elements[3] = dict(elements[3], form_action="https://evil.example/login")
    manager = FakeManager(s, lambda: FakePage(elements=elements))
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_submit"], ref=4)
    assert not result.ok
    assert "allowlist" in result.content
    assert manager.pages[0].clicks == []


async def test_click_that_escapes_the_allowlist_bounces():
    s = browser_settings(browser_allowlist="example.com")
    page = FakePage(url="https://example.com/start")

    class EscapingLocator(FakeLocator):
        async def click(self, timeout=None):
            await super().click(timeout)
            page.url = "https://evil.example/landed"  # js redirect

    page.locator = lambda sel: EscapingLocator(page, sel)
    manager = FakeManager(s, lambda: page)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_click"], ref=1)
    assert not result.ok
    assert "not in the browser allowlist" in result.content
    assert page.goto_calls[-1] == "about:blank"  # bounced off the page


# ---------------------------------------------------------------------------
# extract / screenshot / back / close
# ---------------------------------------------------------------------------

async def test_extract_page_and_element():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_extract"])
    assert result.ok and result.untrusted
    assert "Welcome to the example page." in result.content
    assert result.data["url"].startswith("https://example.com")
    scoped = await tool_run(tools["browser_extract"], ref=2)
    assert scoped.ok
    assert scoped.content == "element-scoped text"


async def test_screenshot_returns_image_observation(tmp_path):
    s = browser_settings()
    s = s.model_copy(update={"workspace_dir": str(tmp_path)})
    manager = FakeManager(s, FakePage)
    tools = {t.name: t for t in make_tools(manager, settings=s)}
    result = await tool_run(tools["browser_screenshot"])
    assert result.ok
    assert result.images and result.images[0].startswith("data:image/png;base64,")
    decoded = base64.b64decode(result.images[0].split(",", 1)[1])
    assert decoded.startswith(b"\x89PNG")
    saved = tmp_path / result.data["path"]
    assert saved.read_bytes() == decoded  # workspace copy for send_file


async def test_back_and_close():
    manager = FakeManager(browser_settings(), FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    result = await tool_run(tools["browser_back"])
    assert result.ok and "PAGE:" in result.content
    closed = await tool_run(tools["browser_close"])
    assert closed.ok and manager.pages[0].closed
    again = await tool_run(tools["browser_close"])
    assert again.ok  # closing nothing is fine


# ---------------------------------------------------------------------------
# budgets + manager lifecycle
# ---------------------------------------------------------------------------

async def test_action_budget_exhausts_honestly():
    s = browser_settings(browser_max_actions=2)
    manager = FakeManager(s, FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    registry = ToolRegistry()
    for t in tools.values():
        registry.register(t)
    assert (await registry.execute("browser_navigate", '{"url": "https://example.com/a"}')).ok
    assert (await registry.execute("browser_extract", "{}")).ok
    spent = await registry.execute("browser_navigate", '{"url": "https://example.com/b"}')
    assert not spent.ok
    assert "budget exhausted" in spent.content
    assert "browser_close" in spent.content  # the recovery path is named


async def test_session_ttl_expiry():
    s = browser_settings(browser_session_ttl_s=0, browser_idle_close_s=999)
    manager = FakeManager(s, FakePage)
    tools = {t.name: t for t in make_tools(manager)}
    registry = ToolRegistry()
    for t in tools.values():
        registry.register(t)
    result = await registry.execute("browser_navigate", '{"url": "https://example.com/a"}')
    assert not result.ok
    assert "expired" in result.content


async def test_manager_sweeps_idle_sessions():
    s = browser_settings(browser_idle_close_s=0, browser_session_ttl_s=999,
                         browser_max_sessions=5)
    manager = FakeManager(s, FakePage)
    await manager.get_session("a")
    await manager.get_session("b")
    await asyncio.sleep(0.01)
    await manager._sweep()  # idle_close_s=0 -> both stale
    assert manager._sessions == {}
    assert all(p.closed for p in manager.pages)


async def test_manager_caps_concurrent_sessions_lru():
    s = browser_settings(browser_max_sessions=1, browser_idle_close_s=999,
                         browser_session_ttl_s=999)
    manager = FakeManager(s, FakePage)
    first = await manager.get_session("a")
    await asyncio.sleep(0.01)
    await manager.get_session("b")  # evicts 'a' (least recently active)
    assert first.closed
    assert list(manager._sessions) == ["b"]


# ---------------------------------------------------------------------------
# agent-loop integration: approval gate + image observations
# ---------------------------------------------------------------------------

class _SnapArgs(BaseModel):
    pass


class _SnapTool(Tool):
    name = "snap"
    description = "fake tool returning an image observation"
    Args = _SnapArgs

    async def run(self, args) -> ToolResult:
        return ToolResult.ok_result(
            "screenshot taken", images=["data:image/png;base64,QUJD"]
        )


def _agent_with(responses, tools, settings=None):
    settings = settings or make_settings(direct_text_replies=True)
    bus = EventBus()
    mock = MockProvider(list(responses))
    router = ModelRouter(settings, default_provider=mock, bus=bus)
    registry = ToolRegistry(bus=bus)
    for t in tools:
        registry.register(t)
    return Agent(settings, router, registry, bus=bus), mock


async def test_submit_requires_approval_in_the_loop():
    """Exit criterion: a form-submission flow pauses for approval first."""
    manager = FakeManager(browser_settings(), FakePage)
    submit = BrowserSubmitTool(manager.settings, manager, "test")
    from woyo.tools.builtin.core_tools import FinishTool

    agent, mock = _agent_with(
        [
            text_response(PLAN_JSON),
            tool_response([("browser_submit", {"ref": 4})]),
            tool_response([("finish", {"summary": "ok", "verified": False,
                                       "sources": [], "open_questions": []})]),
        ],
        [submit, FinishTool()],
    )
    result = await agent.run("log in", approval_cb=lambda name, args: False)
    assert result.outcome == "completed"
    # denial happens BEFORE the tool runs: no page was even created
    assert manager.pages == []
    # and the observation the model saw says exactly that
    all_tool_contents = [
        m.content
        for call in mock.calls
        for m in call["messages"]
        if m.role == "tool"
    ]
    assert any("did not approve" in c for c in all_tool_contents)

    # approved: the click goes through
    agent2, _ = _agent_with(
        [
            text_response(PLAN_JSON),
            tool_response([("browser_submit", {"ref": 4})]),
            tool_response([("finish", {"summary": "ok", "verified": False,
                                       "sources": [], "open_questions": []})]),
        ],
        [submit, FinishTool()],
    )
    await agent2.run("log in", approval_cb=lambda name, args: True)
    assert manager.pages[0].clicks == ['[data-woyo-ref="4"]']


async def test_screenshot_rides_as_image_observation():
    """Tool images become a user-turn image the model sees once, then strips."""
    from tests.conftest import EchoTool
    from woyo.tools.builtin.core_tools import FinishTool

    class _RecordingMock(MockProvider):
        def __init__(self, responses):
            super().__init__(responses)
            self.images_by_call: list[list[list[str] | None]] = []

        async def complete(self, **kw):
            self.images_by_call.append(
                [m.images for m in kw["messages"] if m.images]
            )
            return await super().complete(**kw)

    settings = make_settings(
        direct_text_replies=True, vision_model="mock:vision-model"
    )
    bus = EventBus()
    mock = _RecordingMock(
        [
            text_response(PLAN_JSON),
            tool_response([("snap", {})]),
            tool_response([("echo", {"text": "hi"})]),
            text_response("done, saw the screenshot"),
        ]
    )
    # register the scripted mock under its provider name so the vision-ref
    # route ("mock:vision-model") lands on the SAME scripted provider
    router = ModelRouter(
        settings, default_provider=mock, extra_providers={"mock": mock}, bus=bus
    )
    registry = ToolRegistry(bus=bus)
    registry.register(_SnapTool())
    registry.register(EchoTool())
    registry.register(FinishTool())
    agent = Agent(settings, router, registry, bus=bus)
    await agent.run("take a screenshot")

    # the executor call right after the screenshot carried the image
    # (call 0 = planner, call 1 = executor pre-screenshot, call 2 = executor
    #  that must SEE the screenshot)
    assert mock.images_by_call[2] == [["data:image/png;base64,QUJD"]]
    # and no call ever saw it twice
    assert sum(len(x) for x in mock.images_by_call) == 1


async def test_loop_strips_images_after_consumption():
    """One-shot screenshots: later executor calls don't re-carry them."""
    settings = make_settings(
        direct_text_replies=True, vision_model="mock:vision-model"
    )
    bus = EventBus()

    class _ImgMock(MockProvider):
        def __init__(self, responses):
            super().__init__(responses)
            self.image_counts: list[int] = []

        async def complete(self, **kw):
            self.image_counts.append(sum(1 for m in kw["messages"] if m.images))
            return await super().complete(**kw)

    mock = _ImgMock(
        [
            text_response(PLAN_JSON),
            tool_response([("snap", {})]),
            tool_response([("snap", {})]),
            text_response("done"),
        ]
    )
    router = ModelRouter(
        settings, default_provider=mock, extra_providers={"mock": mock}, bus=bus
    )
    registry = ToolRegistry(bus=bus)
    registry.register(_SnapTool())
    agent = Agent(settings, router, registry, bus=bus)
    await agent.run("two screenshots")

    # call0 = planner (0 images); call1 executor (0); call2 executor sees 1
    # image; call3 executor sees 1 image (the second), NOT 2 — the first was
    # stripped after being seen
    assert mock.image_counts[2] == 1
    assert mock.image_counts[3] == 1


# ---------------------------------------------------------------------------
# real-browser layer (opt-in): validates the DOM walker + real events
# ---------------------------------------------------------------------------

real_browser = pytest.mark.skipif(
    not os.environ.get("WOYO_TEST_REAL_BROWSER") or not playwright_available(),
    reason="set WOYO_TEST_REAL_BROWSER=1 with chromium installed",
)

_SUBMITTED: list[dict] = []


class _SiteHandler(BaseHTTPRequestHandler):
    PAGES = {
        "/": (
            "<html><head><title>Home</title></head><body>"
            "<h1>woyo test site</h1>"
            "<p>Real browser validation page.</p>"
            '<a href="/page2">Second page</a>\n'
            '<form action="/search" method="get">'
            '<input type="text" name="q" placeholder="Search the site">'
            '<button type="submit">Go</button></form>\n'
            '<form action="/submit" method="post">'
            '<input type="text" name="email" placeholder="Your email">'
            '<button type="submit">Sign up</button></form>'
            "</body></html>"
        ),
        "/page2": (
            "<html><head><title>Page Two</title></head><body>"
            "<h1>second page content</h1>"
            "<p>unique-token-12345 lives here.</p>"
            '<a href="/">Back home</a>'
            "</body></html>"
        ),
        "/wall": (
            "<html><head><title>Just a moment...</title></head><body>"
            "<p>Checking your browser before accessing the site.</p>"
            "</body></html>"
        ),
    }

    def _serve(self, body: str, status: int = 200) -> None:
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/search"):
            qs = self.path.split("?", 1)[-1]
            self._serve(
                f"<html><head><title>Results</title></head><body>"
                f"<h1>results for {qs}</h1><p>result-token-xyz</p></body></html>"
            )
            return
        if self.path.startswith("/submit"):
            length = int(self.headers.get("Content-Length") or 0)
            _SUBMITTED.append(
                {"path": self.path, "body": self.rfile.read(length).decode()}
            )
            self._serve(
                "<html><head><title>Posted</title></head><body>"
                "<h1>form received</h1></body></html>"
            )
            return
        body = self.PAGES.get(self.path.rstrip("/") or "/")
        if body is None:
            self._serve("<html><body><p>not found</p></body></html>", 404)
            return
        status = 403 if self.path == "/wall" else 200
        self._serve(body, status)

    def do_POST(self):  # noqa: N802
        self.do_GET()

    def log_message(self, *_args):  # silence the test server
        pass


@pytest.fixture()
def local_site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SiteHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@real_browser
async def test_real_browser_research_flow(local_site):
    """Exit criterion: multi-page research completes over real pages."""
    _SUBMITTED.clear()
    s = browser_settings(browser_allow_private_hosts=True)
    manager = BrowserManager(s)
    tools = {t.name: t for t in build_browser_tools(s, manager, "e2e")}
    try:
        home = await tool_run(tools["browser_navigate"], url=local_site + "/")
        assert home.ok, home.content
        assert "[1] link \"Second page\"" in home.content
        assert "woyo test site" in home.content

        # follow the link like a researcher
        page2 = await tool_run(tools["browser_click"], ref=1)
        assert page2.ok
        assert "PAGE: Page Two" in page2.content

        extracted = await tool_run(tools["browser_extract"])
        assert extracted.ok and "unique-token-12345" in extracted.content

        # back home, then search via the form: type + Enter (GET form)
        back = await tool_run(tools["browser_back"])
        assert back.ok and "PAGE: Home" in back.content
        session = await manager.get_session("e2e")
        search_ref = next(
            (e["ref"] for e in session.refs.values()
             if "Search the site" in (e.get("name") or "")),
            None,
        )
        assert search_ref, "walker must expose the search box"
        results = await tool_run(
            tools["browser_type"], ref=search_ref, text="unique", press_enter=True
        )
        assert results.ok, results.content
        assert "results for" in results.content
        assert "result-token-xyz" in results.content

        shot = await tool_run(tools["browser_screenshot"])
        assert shot.ok and shot.images[0].startswith("data:image/png;base64,")
        assert workspace_root(s).joinpath(shot.data["path"]).is_file()
    finally:
        await manager.close_all()


@real_browser
async def test_real_browser_form_submission_and_approval(local_site):
    """Exit criterion: the irreversible click pauses for approval."""
    _SUBMITTED.clear()
    s = browser_settings(browser_allow_private_hosts=True)
    manager = BrowserManager(s)
    tools = {t.name: t for t in build_browser_tools(s, manager, "e2e2")}
    try:
        home = await tool_run(tools["browser_navigate"], url=local_site + "/")
        assert home.ok, home.content
        # the Sign up button (POST form) is submit-classified -> plain click refuses
        session = await manager.get_session("e2e2")
        signup_ref = next(
            (e["ref"] for e in session.refs.values()
             if e.get("name") == "Sign up" and e.get("is_submit")),
            None,
        )
        assert signup_ref, "walker must classify the Sign up button as submit"
        refused = await tool_run(tools["browser_click"], ref=signup_ref)
        assert not refused.ok and refused.error_kind == "needs_approval"

        # fill the email field first (typing is not the irreversible part)
        email_ref = next(
            (e["ref"] for e in session.refs.values()
             if "Your email" in (e.get("name") or "")),
            None,
        )
        if email_ref:
            filled = await tool_run(
                tools["browser_type"], ref=email_ref, text="user@example.com"
            )
            assert filled.ok, filled.content

        # through the submit tool (approval gate sits in the loop) it lands
        result = await tool_run(tools["browser_submit"], ref=signup_ref)
        assert result.ok, result.content
        assert _SUBMITTED, "the POST must have reached the server"
        assert "email=user%40example.com" in _SUBMITTED[0]["body"] or \
            "user@example.com" in _SUBMITTED[0]["body"]
    finally:
        await manager.close_all()


@real_browser
async def test_real_browser_bot_wall_detection(local_site):
    _SUBMITTED.clear()
    s = browser_settings(browser_allow_private_hosts=True)
    manager = BrowserManager(s)
    tools = {t.name: t for t in build_browser_tools(s, manager, "e2e3")}
    try:
        result = await tool_run(tools["browser_navigate"], url=local_site + "/wall")
        assert not result.ok
        assert "bot-protected" in result.content
        assert "web_search" in result.content  # the graceful fallback is named
    finally:
        await manager.close_all()
