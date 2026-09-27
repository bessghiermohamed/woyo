"""Telegram frontend: chat with woyo from any phone (ADR-10).

Zero new dependencies — plain long-polling against the Bot API with the
httpx client woyo already ships. Safety properties:

- The bot token never leaves this process (server-side only).
- Access control: TELEGRAM_ALLOWED_CHAT_IDS (comma-separated) wins;
  when unset, the FIRST chat to send /start claims the bot — everyone
  else is refused. The claim persists in ~/.woyo/telegram_state.json.
- Replies are chunked to fit Telegram's 4096-char message limit and
  fall back from Markdown to plain text when parsing fails.
- One in-flight run per chat; a global 409 means another instance is
  polling with the same token (only one may run).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx

from woyo.chat.session import ChatReply, ChatSession
from woyo.config import Settings, env_value
from woyo.errors import AgentError

log = logging.getLogger("woyo.telegram")

_API = "https://api.telegram.org"
_CHUNK = 4000  # Telegram hard limit is 4096; leave headroom
_POLL_TIMEOUT = 50  # seconds the server holds the connection
_HTTP_TIMEOUT = httpx.Timeout(65.0, connect=15.0)

WELCOME = (
    "👋 I'm *woyo* — an AI agent you can talk to.\n\n"
    "Send me anything: questions to research, things to calculate, pages to read. "
    "I plan, use tools (web search, page fetch, math) and cite verified sources.\n\n"
    "Commands:\n"
    "/new — start a fresh conversation\n"
    "/status — usage so far\n"
    "/help — this message"
)


def telegram_credentials() -> tuple[str | None, set[int]]:
    """(token, allowed_chat_ids) from env / .env."""
    token = env_value("TELEGRAM_BOT_TOKEN", "WOYO_TELEGRAM_BOT_TOKEN")
    raw = env_value("TELEGRAM_ALLOWED_CHAT_IDS", "WOYO_TELEGRAM_ALLOWED_CHAT_IDS") or ""
    allowed: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            allowed.add(int(part))
        except ValueError:
            log.warning("ignoring non-numeric chat id in TELEGRAM_ALLOWED_CHAT_IDS: %r", part)
    return token, allowed


def split_message(text: str, limit: int = _CHUNK) -> list[str]:
    """Split long text into Telegram-sized chunks, preferring line breaks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    return [c for c in chunks if c]


class TelegramBot:
    """Long-polling bot bridging Telegram chats to ChatSessions."""

    def __init__(
        self,
        settings: Settings,
        token: str,
        allowed_chat_ids: set[int] | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        state_path: str | Path = "~/.woyo/telegram_state.json",
    ):
        self.settings = settings
        self.token = token
        self.allowed = set(allowed_chat_ids or ())
        self.state_path = Path(state_path).expanduser()
        self._client = client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        self._offset = 0
        self._owner: int | None = None
        self._sessions: dict[int, ChatSession] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._load_state()

    # ------------------------------------------------------------------
    async def run_forever(self) -> None:
        me = await self._api("getMe")
        bot_name = me.get("username", "?")
        log.info("logged in as @%s — polling for messages…", bot_name)
        print(f"woyo telegram bot @{bot_name} is up — Ctrl-C to stop")
        while True:
            try:
                updates = await self._poll()
                for update in updates:
                    await self._dispatch(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — the poller must survive
                log.error("poll cycle failed: %s", exc)
                await asyncio.sleep(3)

    async def _poll(self) -> list[dict]:
        # _api() already unwraps the "result" envelope
        result = await self._api("getUpdates", timeout=_POLL_TIMEOUT, offset=self._offset)
        updates = result if isinstance(result, list) else []
        if updates:
            self._offset = int(updates[-1]["update_id"]) + 1
            self._save_state()
        return updates

    # ------------------------------------------------------------------
    async def _dispatch(self, update: dict) -> None:
        msg = update.get("message") or update.get("edited_message") or {}
        chat_id = int((msg.get("chat") or {}).get("id", 0))
        if not chat_id:
            return
        text = (msg.get("text") or "").strip()
        if not text:
            await self._send(chat_id, "I can only read text messages for now 🙂")
            return
        if not self._authorized(chat_id):
            await self._send(chat_id, "🔒 This bot is private.")
            return
        if text.startswith("/"):
            await self._command(chat_id, text)
            return
        await self._answer(chat_id, text)

    def _authorized(self, chat_id: int) -> bool:
        if self.allowed:
            return chat_id in self.allowed
        return self._owner in (None, chat_id)

    async def _command(self, chat_id: int, text: str) -> None:
        cmd = text.split()[0].split("@")[0].lower()
        if cmd == "/start":
            if self.allowed:
                if chat_id not in self.allowed:
                    await self._send(chat_id, "🔒 This bot is private.")
                    return
            elif self._owner is None:
                self._owner = chat_id
                self._save_state()
                log.info("bot claimed by chat %s", chat_id)
            await self._send(chat_id, WELCOME, markdown=True)
        elif cmd == "/new":
            session = self._session(chat_id)
            session.reset()
            await self._send(chat_id, "🧹 Fresh conversation — I've forgotten the previous one.")
        elif cmd == "/status":
            await self._send(chat_id, self._session(chat_id).status_line())
        elif cmd == "/help":
            await self._send(chat_id, WELCOME, markdown=True)
        else:
            await self._send(chat_id, "Unknown command. Try /help")

    # ------------------------------------------------------------------
    async def _answer(self, chat_id: int, text: str) -> None:
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        if lock.locked():
            await self._send(chat_id, "⏳ Still working on your last message — one at a time.")
            return
        async with lock:
            typing = asyncio.create_task(self._typing_loop(chat_id))
            try:
                reply = await self._session(chat_id).send(text)
                await self._send_reply(chat_id, reply)
            except AgentError as exc:
                await self._send(chat_id, f"⚠️ {exc}")
            except Exception as exc:  # noqa: BLE001 — never crash the poller
                log.exception("chat run failed")
                await self._send(chat_id, f"⚠️ Something went wrong: {exc}")
            finally:
                typing.cancel()

    async def _send_reply(self, chat_id: int, reply: ChatReply) -> None:
        text = reply.render()
        for i, chunk in enumerate(split_message(text)):
            await self._send(chat_id, chunk, markdown=(i == 0))

    async def _typing_loop(self, chat_id: int) -> None:
        try:
            while True:
                await self._api("sendChatAction", chat_id=chat_id, action="typing")
                await asyncio.sleep(4.5)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    def _session(self, chat_id: int) -> ChatSession:
        if chat_id not in self._sessions:
            self._sessions[chat_id] = ChatSession(
                f"telegram:{chat_id}", self.settings
            )
        return self._sessions[chat_id]

    async def _send(self, chat_id: int, text: str, *, markdown: bool = False) -> None:
        for chunk in split_message(text):
            try:
                if markdown:
                    try:
                        await self._api(
                            "sendMessage", chat_id=chat_id, text=chunk, parse_mode="Markdown"
                        )
                        continue
                    except httpx.HTTPStatusError:
                        pass  # malformed markdown -> fall back to plain text
                await self._api("sendMessage", chat_id=chat_id, text=chunk)
            except Exception as exc:  # noqa: BLE001 — delivery must not kill polling
                log.error("sendMessage to %s failed: %s", chat_id, exc)

    async def _api(self, method: str, **params: object) -> dict:
        resp = await self._client.post(f"{_API}/bot{self.token}/{method}", json=params)
        if resp.status_code == 409:
            raise RuntimeError(
                "Telegram returned 409: another bot instance is already polling "
                "with this token. Only one instance may run (stop the other one)."
            )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API error on {method}: {data}")
        return data.get("result", {})

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._offset = int(data.get("offset", 0))
            owner = data.get("owner")
            self._owner = int(owner) if owner is not None else None
        except (OSError, ValueError, TypeError):
            pass

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({"offset": self._offset, "owner": self._owner}),
                encoding="utf-8",
            )
        except OSError:
            pass
