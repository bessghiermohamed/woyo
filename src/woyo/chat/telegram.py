"""Telegram frontend: chat with woyo from any phone (ADR-10).

Zero new dependencies — plain long-polling against the Bot API with the
httpx client woyo already ships. Safety properties:

- The bot token never leaves this process (server-side only).
- Access control: TELEGRAM_ALLOWED_CHAT_IDS (comma-separated) wins;
  when unset, the FIRST chat to send /start claims the bot — everyone
  else is refused. The claim persists in ~/.woyo/telegram_state.json.
- Replies are chunked to fit Telegram's 4096-char message limit and
  fall back from Markdown to plain text when parsing fails.
- One in-flight run per chat (lock); message handling runs as tasks so
  the poller keeps receiving — that is what makes inline Approve/Deny
  buttons possible while an agent run is waiting on an approval.
- Approval buttons carry an id bound to the chat that was asked; presses
  from any other chat are ignored. No press within the timeout = deny.
- A global 409 means another instance is polling with the same token
  (only one may run).
- Files (v0.6): inbound attachments are downloaded, saved to the durable
  store + workspace, and ingested into the agent's prompt (see
  chat/files.py); the send_file tool delivers workspace files back to
  the requesting chat via sendDocument — no other destination exists.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import mimetypes
import shutil
from pathlib import Path

import httpx

from woyo.chat.files import (
    DownloadError,
    FileTooBig,
    attachment_ref,
    ingest_attachment,
    unique_path,
)
from woyo.chat.session import ChatReply, ChatSession
from woyo.config import Settings, env_value
from woyo.errors import AgentError

log = logging.getLogger("woyo.telegram")

_API = "https://api.telegram.org"
_CHUNK = 4000  # Telegram hard limit is 4096; leave headroom
_POLL_TIMEOUT = 50  # seconds the server holds the connection
_HTTP_TIMEOUT = httpx.Timeout(65.0, connect=15.0)
_UPLOAD_TIMEOUT = httpx.Timeout(300.0, connect=15.0)  # big document uploads

WELCOME = (
    "👋 I'm *woyo* — an AI agent you can talk to.\n\n"
    "Send me anything: questions to research, things to calculate, code to "
    "run — and *files*: documents, code, PDFs, archives, photos (I can view "
    "them), voice notes. I'll read what you send, work on it, and cite "
    "verified sources.\n\n"
    "Ask me to *create* files too — reports, code, CSVs, exports — and I'll "
    "send them right here in the chat. I can also run Python, keep a "
    "workspace, spawn sub-agents, and (with your approval) run shell "
    "commands.\n\n"
    "Commands:\n"
    "/new — start a fresh conversation\n"
    "/status — usage so far\n"
    "/help — this message"
)

#: What the agent is told when a file arrives with no caption.
_NO_CAPTION_INSTRUCTION = (
    "(The user sent this file with no message. Look at it, then briefly "
    "summarize or analyze it and ask what they would like done with it.)"
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
        self._had_state = self.state_path.exists()
        self._sessions: dict[int, ChatSession] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        # approval id -> (future, chat_id that was asked)
        self._pending: dict[str, tuple[asyncio.Future[bool], int]] = {}
        self._approval_seq = 0
        self._load_state()

    # ------------------------------------------------------------------
    async def run_forever(self) -> None:
        me = await self._api("getMe")
        bot_name = me.get("username", "?")
        log.info("logged in as @%s — polling for messages…", bot_name)
        print(f"woyo telegram bot @{bot_name} is up — Ctrl-C to stop")
        if not self._had_state:
            await self._skip_backlog()
        while True:
            try:
                updates = await self._poll()
                for update in updates:
                    await self._dispatch(update)
            except asyncio.CancelledError:
                raise
            except RuntimeError as exc:
                if "409" in str(exc):
                    # another instance holds this token — stop loudly, not in a loop
                    print(f"stopping: {exc}")
                    raise
                log.error("poll cycle failed: %s", exc)
                await asyncio.sleep(3)
            except Exception as exc:  # noqa: BLE001 — the poller must survive
                log.error("poll cycle failed: %s", exc)
                await asyncio.sleep(3)

    async def _skip_backlog(self) -> None:
        """Fresh start with no persisted offset: confirm-and-drop pending updates.

        Messages that arrived while the bot was down are skipped rather than
        re-answered — for a personal assistant, duplicate replies are worse
        than a gap. Ephemeral hosts (GitHub Actions runners) rely on this.
        """
        result = await self._api("getUpdates", timeout=0, offset=-1)
        updates = result if isinstance(result, list) else []
        if updates:
            self._offset = int(updates[-1]["update_id"]) + 1
            log.info("skipped %s update(s) that predate this start", len(updates))
            self._save_state()

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
        callback = update.get("callback_query")
        if callback:
            await self._handle_callback(callback)
            return
        msg = update.get("message") or update.get("edited_message") or {}
        chat_id = int((msg.get("chat") or {}).get("id", 0))
        if not chat_id:
            return
        text = (msg.get("text") or msg.get("caption") or "").strip()
        attachment = attachment_ref(msg)
        if msg.get("sticker") and not text and not attachment:
            if not self._authorized(chat_id):
                return
            await self._send(
                chat_id, "😅 I can't do much with stickers — send text or a file."
            )
            return
        if not text and attachment is None:
            await self._send(
                chat_id,
                "🤔 I couldn't read that message type. Text and files work best.",
            )
            return
        if not self._authorized(chat_id):
            await self._send(chat_id, "🔒 This bot is private.")
            return
        if text.startswith("/") and attachment is None:
            await self._command(chat_id, text)
            return
        # Run as a task so the poller keeps receiving updates — this is
        # what lets a button press arrive while the run awaits an approval.
        if attachment is not None:
            task = asyncio.create_task(self._answer_attachment(chat_id, msg, text))
        else:
            task = asyncio.create_task(self._answer(chat_id, text))
        task.add_done_callback(self._log_task_crash)

    @staticmethod
    def _log_task_crash(task: asyncio.Task) -> None:  # pragma: no cover
        if not task.cancelled() and task.exception():
            log.error("answer task crashed: %s", task.exception())

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
                reply = await self._session(chat_id).send(
                    text,
                    approval_cb=lambda tool, args: self._request_approval(
                        chat_id, tool, args
                    ),
                )
                await self._send_reply(chat_id, reply)
            except AgentError as exc:
                await self._send(chat_id, f"⚠️ {exc}")
            except Exception as exc:  # noqa: BLE001 — never crash the poller
                log.exception("chat run failed")
                await self._send(chat_id, f"⚠️ Something went wrong: {exc}")
            finally:
                typing.cancel()

    async def _answer_attachment(self, chat_id: int, msg: dict, caption: str) -> None:
        """Ingest one attached file, then run the agent over it."""
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        if lock.locked():
            await self._send(chat_id, "⏳ Still working on your last message — one at a time.")
            return
        async with lock:
            typing = asyncio.create_task(self._typing_loop(chat_id))
            try:
                try:
                    ingested = await ingest_attachment(
                        self._client, self.token, chat_id, msg, self.settings
                    )
                except FileTooBig as exc:
                    await self._send(
                        chat_id,
                        f"📦 That file is too big ({exc}). I can handle up to "
                        f"{self.settings.chat_max_file_mb} MB — Telegram's bot "
                        "API won't let me download more.",
                    )
                    return
                except (DownloadError, httpx.HTTPError) as exc:
                    log.warning("attachment download failed: %s", exc)
                    await self._send(
                        chat_id,
                        "⚠️ I couldn't download that file from Telegram. "
                        "Try sending it again.",
                    )
                    return
                text = caption or _NO_CAPTION_INSTRUCTION
                reply = await self._session(chat_id).send(
                    f"{text}\n\n{ingested.note}",
                    images=ingested.images or None,
                    approval_cb=lambda tool, args: self._request_approval(
                        chat_id, tool, args
                    ),
                )
                await self._send_reply(chat_id, reply)
            except AgentError as exc:
                await self._send(chat_id, f"⚠️ {exc}")
            except Exception as exc:  # noqa: BLE001 — never crash the poller
                log.exception("attachment run failed")
                await self._send(chat_id, f"⚠️ Something went wrong: {exc}")
            finally:
                typing.cancel()

    # ------------------------------------------------------------------
    # Inline approval buttons (Phase 7 capability, pulled forward)
    # ------------------------------------------------------------------
    async def _request_approval(self, chat_id: int, tool: str, args_json: str) -> bool:
        """Ask `chat_id` to approve a tool call via inline buttons.

        Returns True only on an explicit Approve press from that same chat
        within chat_approval_timeout_s. Timeout, deny, or a press from any
        other chat => denied (fail safe).
        """
        self._approval_seq += 1
        ap_id = str(self._approval_seq)
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._pending[ap_id] = (fut, chat_id)

        try:
            pretty = json.dumps(json.loads(args_json or "{}"), indent=2, ensure_ascii=False)
        except ValueError:
            pretty = args_json or "{}"
        text = (
            "🔐 <b>Approval needed</b>\n\n"
            f"Tool: <code>{html.escape(tool)}</code>\n\n"
            f"<pre>{html.escape(pretty[:1200])}</pre>\n\n"
            "Approve this action?"
        )
        keyboard = {
            "inline_keyboard": [[
                {"text": "✅ Approve", "callback_data": f"woyo_apr:{ap_id}:1"},
                {"text": "❌ Deny", "callback_data": f"woyo_apr:{ap_id}:0"},
            ]]
        }
        sent = await self._api(
            "sendMessage", chat_id=chat_id, text=text,
            parse_mode="HTML", reply_markup=keyboard,
        )
        message_id = (sent or {}).get("message_id")

        approved = False
        try:
            approved = await asyncio.wait_for(
                fut, timeout=self.settings.chat_approval_timeout_s
            )
        except TimeoutError:
            approved = False
        finally:
            self._pending.pop(ap_id, None)

        verdict = "✅ approved" if approved else "❌ denied (timeout counts as deny)"
        if message_id is not None:
            try:
                await self._api(
                    "editMessageText", chat_id=chat_id, message_id=message_id,
                    text=f"🔐 {html.escape(tool)} — {verdict}",
                )
            except Exception:  # noqa: BLE001 — cosmetic edit only
                pass
        return approved

    async def _handle_callback(self, callback: dict) -> None:
        """Resolve a pending approval when its button is pressed."""
        cbq_id = callback.get("id")
        data = str(callback.get("data", ""))
        chat_id = int(((callback.get("message") or {}).get("chat") or {}).get("id", 0))
        try:
            await self._api("answerCallbackQuery", callback_query_id=cbq_id)
        except Exception:  # noqa: BLE001 — never crash the poller
            pass
        if not data.startswith("woyo_apr:"):
            return
        parts = data.split(":")
        if len(parts) != 3:
            return
        _, ap_id, verdict = parts
        entry = self._pending.get(ap_id)
        if entry is None:
            return
        fut, asked_chat = entry
        # only the chat that was asked (and is still authorized) may answer
        if chat_id != asked_chat or not self._authorized(chat_id):
            log.warning("approval press from wrong chat %s ignored", chat_id)
            return
        self._pending.pop(ap_id, None)
        if not fut.done():
            fut.set_result(verdict == "1")

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
                f"telegram:{chat_id}",
                self.settings,
                file_sender=self._make_file_sender(chat_id),
            )
        return self._sessions[chat_id]

    def _make_file_sender(self, chat_id: int):
        """Chat-scoped send_file transport: sendDocument + outbox archive."""

        async def sender(path: Path, caption: str) -> None:
            await self._send_document(chat_id, path, caption)
            # keep a durable copy so "send me that file again" survives rotation
            try:
                outbox = (
                    Path(self.settings.files_dir).expanduser() / str(chat_id) / "outbox"
                )
                outbox.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, unique_path(outbox, path.name))
            except OSError:
                pass  # archiving is best-effort

        return sender

    async def _send_document(self, chat_id: int, path: Path, caption: str) -> None:
        """Upload one file to the chat via multipart sendDocument."""
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        form: dict[str, str] = {"chat_id": str(chat_id)}
        if caption:
            form["caption"] = caption[:1024]
        with path.open("rb") as fh:
            resp = await self._client.post(
                f"{_API}/bot{self.token}/sendDocument",
                data=form,
                files={"document": (path.name, fh, mime)},
                timeout=_UPLOAD_TIMEOUT,
            )
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"sendDocument failed: {payload}")

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
