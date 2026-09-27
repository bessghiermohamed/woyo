"""Web chat frontend: one self-contained page + a tiny JSON API (ADR-10).

Stdlib-only HTTP server (ThreadingHTTPServer) bridging to a single
background asyncio loop that owns the agent. Security posture:

- A passcode (WOYO_CHAT_PASSWORD) is REQUIRED whenever the server is
  reachable beyond loopback; checked with a constant-time compare.
- Per-session sliding-window rate limit + global daily message cap
  protect the provider budget behind a public URL.
- The API key lives server-side only — the browser never sees it.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from woyo import __version__
from woyo.chat.session import ChatReply, ChatSession
from woyo.config import Settings
from woyo.errors import AgentError

log = logging.getLogger("woyo.web")

_MAX_BODY = 64 * 1024
_MAX_MESSAGE = 8_000
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_WINDOW_MESSAGES = 20  # per session
_WINDOW_S = 3_600
_GLOBAL_DAILY = 400
_SESSIONS_CAP = 100

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _agent_loop() -> asyncio.AbstractEventLoop:
    """The single background asyncio loop that runs all agent turns."""
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True, name="woyo-agent").start()
        return _loop


def _is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


class _ChatApp:
    """Mutable server state: sessions, rate limits, counters."""

    def __init__(self, settings: Settings, session_factory=None):
        self.settings = settings
        self.session_factory = session_factory or (
            lambda key, s: ChatSession(key, s)
        )
        self.sessions: dict[str, tuple[ChatSession, float]] = {}
        self.windows: dict[str, list[float]] = {}
        self.daily_day = time.strftime("%Y-%m-%d", time.gmtime())
        self.daily_count = 0

    # ------------------------------------------------------------------
    def check_rate(self, session_id: str) -> str | None:
        """Return a rejection reason, or None if allowed."""
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self.daily_day:
            self.daily_day, self.daily_count = today, 0
        if self.daily_count >= _GLOBAL_DAILY:
            return "daily global message limit reached — back tomorrow"
        now = time.monotonic()
        window = [t for t in self.windows.get(session_id, []) if now - t < _WINDOW_S]
        if len(window) >= _WINDOW_MESSAGES:
            self.windows[session_id] = window
            return f"rate limit: max {_WINDOW_MESSAGES} messages/hour per session"
        window.append(now)
        self.windows[session_id] = window
        self.daily_count += 1
        return None

    def session(self, session_id: str) -> ChatSession:
        found = self.sessions.get(session_id)
        if found is not None:
            self.sessions[session_id] = (found[0], time.monotonic())
            return found[0]
        session = self.session_factory(f"web:{session_id}", self.settings)
        if len(self.sessions) >= _SESSIONS_CAP:  # drop the idle-most session
            oldest = min(self.sessions, key=lambda k: self.sessions[k][1])
            del self.sessions[oldest]
        self.sessions[session_id] = (session, time.monotonic())
        return session


class _Handler(BaseHTTPRequestHandler):
    app: _ChatApp  # set by build_server

    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 — stdlib naming
        if self.path in ("/", "/index.html"):
            body = _PAGE.replace("__VERSION__", __version__).encode("utf-8")
            self._send(200, body, content_type="text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._json(
                200,
                {
                    "ok": True,
                    "version": __version__,
                    "provider": self.server.app.settings.provider,
                },
            )
        elif self.path == "/favicon.ico":
            self._send(204, b"")
        else:
            self._json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802 — stdlib naming
        if self.path != "/api/chat":
            self._json(404, {"ok": False, "error": "not_found"})
            return
        try:
            data = self._read_json()
        except ValueError as exc:
            self._json(400, {"ok": False, "error": "bad_request", "detail": str(exc)})
            return

        password = self.server.app.settings.chat_password
        if password is not None:
            supplied = str(data.get("passcode") or "")
            if not hmac.compare_digest(supplied, password):
                self._json(401, {"ok": False, "error": "unauthorized"})
                return

        session_id = str(data.get("session_id") or "")
        if not _SESSION_RE.match(session_id):
            self._json(400, {"ok": False, "error": "bad_session"})
            return
        message = str(data.get("message") or "").strip()
        if not message:
            self._json(400, {"ok": False, "error": "empty_message"})
            return
        if len(message) > _MAX_MESSAGE:
            self._json(400, {"ok": False, "error": "message_too_long"})
            return

        reason = self.server.app.check_rate(session_id)
        if reason is not None:
            self._json(429, {"ok": False, "error": "rate_limited", "detail": reason})
            return

        session = self.server.app.session(session_id)
        try:
            fut = asyncio.run_coroutine_threadsafe(session.send(message), _agent_loop())
            reply = fut.result(timeout=200)
        except AgentError as exc:
            self._json(429, {"ok": False, "error": "chat_refused", "detail": str(exc)})
            return
        except TimeoutError:
            self._json(504, {"ok": False, "error": "timeout"})
            return
        except Exception as exc:  # noqa: BLE001 — API must answer, never hang
            log.exception("chat turn failed")
            self._json(500, {"ok": False, "error": "internal", "detail": str(exc)})
            return
        self._json(200, _reply_json(reply))

    # ------------------------------------------------------------------
    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > _MAX_BODY:
            raise ValueError("invalid body size")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise ValueError("invalid JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:  # quiet stdlib chatter
        log.debug("%s - %s", self.address_string(), fmt % args)


def _reply_json(reply: ChatReply) -> dict:
    return {
        "ok": True,
        "answer": reply.answer,
        "verified": reply.verified,
        "sources": [
            {"url": s.get("url", ""), "title": s.get("title", "")} for s in reply.sources
        ],
        "sources_dropped": len(reply.sources_dropped),
        "steps": reply.steps,
        "tool_calls": reply.tool_calls,
        "duration_s": round(reply.duration_s, 1),
        "cost_usd_est": round(reply.cost_usd_est, 5),
        "outcome": reply.outcome,
    }


def build_server(
    settings: Settings,
    host: str = "127.0.0.1",
    port: int = 7860,
    session_factory=None,
) -> ThreadingHTTPServer:
    """Assemble the server (returned unstarted — handy for tests)."""
    if not _is_loopback(host) and not settings.chat_password:
        raise AgentError(
            "refusing to expose the web chat beyond localhost without a passcode — "
            "set WOYO_CHAT_PASSWORD first (see docs/SECURITY.md)"
        )
    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    server.app = _ChatApp(settings, session_factory)  # type: ignore[attr-defined]
    return server


def serve(settings: Settings, host: str = "127.0.0.1", port: int = 7860) -> None:
    """Run the web chat until interrupted."""
    server = build_server(settings, host, port)
    where = f"http://{host}:{port}"
    print(f"woyo web chat v{__version__} — listening on {where}")
    if settings.chat_password is None:
        print("  (no passcode set — local access only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()


def new_browser_session_id() -> str:
    return uuid.uuid4().hex


# ----------------------------------------------------------------------
# The page — vanilla HTML/CSS/JS, mobile-first, no external assets.
_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="dark">
<title>woyo — chat</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Ccircle cx='16' cy='16' r='14' fill='%2322d3ee'/%3E%3Ctext x='16' y='22' font-size='18' text-anchor='middle' fill='%23000' font-family='monospace' font-weight='bold'%3Ew%3C/text%3E%3C/svg%3E">
<style>
  :root{
    --bg:#0b0f14; --panel:#11161d; --panel2:#161d26; --line:#232c38;
    --text:#dbe4ee; --dim:#7d8da0; --accent:#22d3ee; --accent2:#0e7490;
    --user:#123a47; --ok:#34d399; --err:#f87171;
  }
  *{box-sizing:border-box; -webkit-tap-highlight-color:transparent}
  html,body{margin:0; height:100%; background:var(--bg); color:var(--text);
    font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
  #app{display:flex; flex-direction:column; height:100dvh; max-width:880px; margin:0 auto}
  header{display:flex; align-items:center; gap:10px; padding:12px 16px;
    border-bottom:1px solid var(--line); background:var(--panel); position:sticky; top:0; z-index:5}
  .logo{width:30px; height:30px; border-radius:9px; background:var(--accent); color:#001318;
    display:grid; place-items:center; font-weight:800; font-family:ui-monospace,monospace}
  header h1{font-size:17px; margin:0; font-weight:650}
  header h1 span{color:var(--dim); font-weight:400}
  #status{width:8px; height:8px; border-radius:50%; background:var(--dim); margin-left:auto}
  #status.on{background:var(--ok)}
  #msgs{flex:1; overflow-y:auto; padding:16px 14px calc(14px + env(safe-area-inset-bottom));
    display:flex; flex-direction:column; gap:12px; scroll-behavior:smooth}
  .msg{max-width:86%; padding:10px 14px; border-radius:16px; white-space:pre-wrap;
    overflow-wrap:anywhere; font-size:15.5px}
  .msg.user{align-self:flex-end; background:var(--user); border-bottom-right-radius:5px}
  .msg.bot{align-self:flex-start; background:var(--panel2); border:1px solid var(--line);
    border-bottom-left-radius:5px; white-space:normal}
  .msg.bot.error{border-color:#5b2226; color:#fca5a5}
  .meta{margin-top:8px; padding-top:6px; border-top:1px dashed var(--line); color:var(--dim);
    font-size:12px; display:flex; flex-wrap:wrap; gap:6px 12px; align-items:center}
  .badge{font-size:10.5px; padding:1px 8px; border-radius:99px; border:1px solid var(--line)}
  .badge.v{color:var(--ok); border-color:#14532d}
  .src{display:block; color:var(--accent); text-decoration:none; font-size:12.5px;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:100%}
  .msg.bot pre{background:#0a0e13; border:1px solid var(--line); border-radius:10px;
    padding:10px; overflow-x:auto; font-size:13px}
  .msg.bot code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13.5px;
    background:#0a0e13; padding:1px 5px; border-radius:6px}
  .msg.bot pre code{background:none; padding:0}
  .msg.bot a{color:var(--accent)}
  #thinking{display:none; align-self:flex-start; color:var(--dim); font-size:13.5px;
    padding:6px 14px}
  #thinking .dots::after{content:"…"; animation:d 1.2s infinite steps(4)}
  @keyframes d{0%{content:""} 25%{content:"."} 50%{content:".."} 75%{content:"..."}}
  form{display:flex; gap:10px; padding:12px 14px calc(12px + env(safe-area-inset-bottom));
    border-top:1px solid var(--line); background:var(--panel)}
  #inp{flex:1; resize:none; background:var(--panel2); color:var(--text);
    border:1px solid var(--line); border-radius:14px; padding:11px 14px; font:inherit;
    max-height:132px; outline:none}
  #inp:focus{border-color:var(--accent2)}
  #send{background:var(--accent); color:#001318; border:0; border-radius:14px; width:52px;
    font-size:21px; cursor:pointer; flex:none}
  #send:disabled{opacity:.45}
  #gate{position:fixed; inset:0; background:rgba(5,8,12,.9); display:none; place-items:center;
    z-index:10; padding:20px}
  #gate.show{display:grid}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:18px;
    padding:26px; width:min(360px,100%); text-align:center}
  .card .logo{width:44px; height:44px; border-radius:13px; margin:0 auto 12px; font-size:22px}
  .card h2{margin:0 0 6px; font-size:19px}
  .card p{margin:0 0 18px; color:var(--dim); font-size:14px}
  #pw{width:100%; background:var(--panel2); border:1px solid var(--line); color:var(--text);
    border-radius:12px; padding:12px; font:inherit; text-align:center; outline:none; margin-bottom:12px}
  #pw:focus{border-color:var(--accent2)}
  #go{width:100%; background:var(--accent); color:#001318; border:0; border-radius:12px;
    padding:12px; font:inherit; font-weight:700; cursor:pointer}
  #go:disabled{opacity:.5}
  .hint{margin-top:14px; font-size:12px; color:var(--dim)}
</style>
</head>
<body>
<div id="app">
  <header>
    <div class="logo">w</div>
    <h1>woyo <span>· agent chat</span></h1>
    <div id="status"></div>
  </header>
  <div id="msgs"></div>
  <div id="thinking">woyo is thinking<span class="dots"></span> <span id="elapsed"></span></div>
  <form id="f">
    <textarea id="inp" rows="1" placeholder="Ask woyo anything…" enterkeyhint="send"></textarea>
    <button id="send" type="submit" aria-label="Send">➤</button>
  </form>
</div>

<div id="gate">
  <div class="card">
    <div class="logo">w</div>
    <h2>woyo</h2>
    <p>This chat is private. Enter the passcode to continue.</p>
    <input id="pw" type="password" inputmode="text" autocomplete="current-password"
      placeholder="Passcode">
    <button id="go">Enter chat</button>
    <div class="hint">woyo agent · v__VERSION__</div>
  </div>
</div>

<script>
"use strict";
const $ = id => document.getElementById(id);
const msgs = $("msgs"), form = $("f"), inp = $("inp"), send = $("send");
const gate = $("gate"), pw = $("pw"), go = $("go"), statusDot = $("status");

const store = {
  get session(){ return localStorage.getItem("woyo_session"); },
  set session(v){ localStorage.setItem("woyo_session", v); },
  get pass(){ return localStorage.getItem("woyo_pass"); },
  set pass(v){ localStorage.setItem("woyo_pass", v); },
};
if (!store.session) store.session = crypto.randomUUID().replace(/-/g,"");

function esc(s){ const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
function mdLite(s){
  let html = esc(s);
  html = html.replace(/```([\s\S]*?)```/g, (_,c) => "<pre><code>"+c.replace(/^\n/,"")+"</code></pre>");
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>");
  html = html.replace(/(^|\s)\*([^*\n]+)\*(?=\s|$)/g, "$1<i>$2</i>");
  html = html.replace(/\[([^\]\n]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/^(\s*)[-•] /gm, "$1• ");
  return html;
}
function add(cls, html){
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.innerHTML = html;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
  return d;
}
function userMsg(t){ add("user", esc(t)); }

function botMsg(reply){
  let html = mdLite(reply.answer || "(no answer)");
  const bits = [];
  if (reply.verified) bits.push('<span class="badge v">✓ verified</span>');
  if (reply.steps) bits.push(reply.steps + " steps · " + (reply.tool_calls||0) + " tools");
  if (reply.duration_s) bits.push(reply.duration_s + "s");
  if (reply.cost_usd_est != null) bits.push("$" + reply.cost_usd_est.toFixed(4));
  if (reply.sources && reply.sources.length){
    bits.push('<span style="flex-basis:100%;display:block;height:2px"></span>');
    for (const s of reply.sources.slice(0,6))
      bits.push('<a class="src" href="'+esc(s.url)+'" target="_blank" rel="noopener">🔗 '+
        esc(s.title || s.url)+"</a>");
  }
  if (bits.length) html += '<div class="meta">'+bits.join("")+"</div>";
  return add("bot", html);
}

let busy = false, t0 = 0, ticker = null;
function setBusy(b){
  busy = b; send.disabled = b; inp.disabled = b;
  $("thinking").style.display = b ? "block" : "none";
  if (b){ t0 = Date.now();
    ticker = setInterval(()=> $("elapsed").textContent =
      Math.round((Date.now()-t0)/1000)+"s", 1000);
  } else { clearInterval(ticker); }
  if (!b) inp.focus();
}

async function call(msg){
  const r = await fetch("/api/chat", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({session_id: store.session, passcode: store.pass || "", message: msg}),
  });
  const data = await r.json().catch(()=>({ok:false, error:"bad_response"}));
  if (r.status === 401){ gate.classList.add("show"); pw.focus();
    throw new Error("unauthorized"); }
  if (!data.ok) throw new Error(data.detail || data.error || "error");
  return data;
}

form.addEventListener("submit", async e => {
  e.preventDefault();
  const t = inp.value.trim();
  if (!t || busy) return;
  inp.value = ""; inp.style.height = "auto";
  userMsg(t); setBusy(true);
  try {
    botMsg(await call(t));
  } catch (err) {
    if (err.message !== "unauthorized")
      add("bot error", "⚠️ " + esc(err.message));
  } finally { setBusy(false); }
});

inp.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey){ e.preventDefault(); form.requestSubmit(); }
});
inp.addEventListener("input", () => {
  inp.style.height = "auto";
  inp.style.height = Math.min(inp.scrollHeight, 132) + "px";
});

go.addEventListener("click", async () => {
  const v = pw.value.trim();
  if (!v) return;
  go.disabled = true; go.textContent = "Checking…";
  store.pass = v;
  try {
    const h = await fetch("/api/health");
    if (h.ok){ gate.classList.remove("show"); inp.focus(); }
    else throw new Error();
  } catch { go.textContent = "Retry"; setTimeout(()=> go.disabled = false, 600); }
  go.textContent = "Enter chat"; go.disabled = false;
});
pw.addEventListener("keydown", e => { if (e.key === "Enter") go.click(); });

fetch("/api/health").then(r => { statusDot.classList.toggle("on", r.ok); }).catch(()=>{});
if (!store.pass) gate.classList.add("show");
</script>
</body>
</html>
"""
