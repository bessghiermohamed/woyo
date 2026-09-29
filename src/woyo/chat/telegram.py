"""Telegram frontend: chat with woyo from any phone (ADR-10).

Zero new dependencies — plain long-polling against the Bot API with the
httpx client woyo already ships. Safety properties:

- The bot token never leaves this process (server-side only).
- Access control: TELEGRAM_ALLOWED_CHAT_IDS (comma-separated) wins;
  when unset, the FIRST chat to send /start claims the bot — everyone
  else is refused. The claim persists in ~/.woyo/telegram_state.json.
- Replies are chunked to fit Telegram's 4096-char message limit and
  fall back from Markdown to plain text when parsing fails.
- Messages QUEUE per chat (one in-flight run per chat; the next message
  waits and a short acknowledgement is sent — nothing is dropped).
- Message handling runs as tasks so the poller keeps receiving — that is
  what makes inline Approve/Deny buttons possible while an agent run is
  waiting on an approval.
- DURABILITY (v0.7): every fetched update is journaled (persisted in the
  state file) BEFORE processing and removed only after the reply is sent.
  If the host dies mid-turn, the next boot drains the journal and the
  request is answered instead of silently lost — the offset alone would
  have swallowed it.
- run_forever accepts a runtime budget / shutdown event so ephemeral
  hosts (GitHub Actions) can rotate gracefully: stop polling, wait for
  in-flight turns, notify affected chats, sync state, return.
- Approval buttons carry an id bound to the chat that was asked; presses
  from any other chat are ignored. No press within the timeout = deny.
- A global 409 means another instance is polling with the same token
  (only one may run).
- Files (v0.6): inbound attachments are downloaded, saved to the durable
  store + workspace, and ingested into the agent's prompt (see
  chat/files.py); the send_file tool delivers workspace files back to
  the requesting chat via sendDocument — no other destination exists.
- AWARENESS + REACH (v0.8): the bot keeps an index of the chats it is in
  (persisted in the state file) and injects it — with its @username and
  the current chat id — into every agent prompt, so the agent knows
  where it lives. A TelegramTransport exposes proactive, addressable
  sends (send_telegram_message / send_document to any KNOWN chat;
  unknown destinations are refused, cross-chat sends pass the approval
  gate).
- FOLLOW-THROUGH (v0.8): scheduled jobs (store/jobs.py) execute when due
  and report back in the owning chat — "I will" became a commitment.
  The job loop runs beside the poller; jobs interrupted by rotation are
  re-queued on the next host (bounded attempts).
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import mimetypes
import shutil
from datetime import UTC, datetime
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
from woyo.store.jobs import JobStore

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
    "Ask me to *create* files too — reports, code, CSVs, PDFs in Arabic or "
    "any script — and I'll send them right here. I can also send messages "
    "and files to *other chats I'm in* (I'll ask you to approve those), run "
    "Python, keep a workspace, spawn sub-agents, and (with your approval) "
    "run shell commands.\n\n"
    "🌐 I also drive a *real browser*: I can open pages, click, fill forms, "
    "extract what they say, and take screenshots I actually look at. Forms "
    "and irreversible actions always wait for your Approve button.\n\n"
    "⏰ *I keep my promises:* ask me to do something *later* — a reminder, a "
    "delayed report, a follow-up — and I'll schedule it so it actually runs "
    "at that time and reports back here, with the result or the reason it "
    "failed.\n\n"
    "Commands:\n"
    "/new — start a fresh conversation\n"
    "/status — usage so far\n"
    "/tasks — pending scheduled tasks\n"
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


class _BotTransport:
    """TelegramTransport implementation bound to one TelegramBot.

    Deliveries raise on failure (the tools turn that into observations);
    the allowlist is the set of chats the bot actually knows — the current
    conversation, the persisted chat index, or the deployment allowlist.
    """

    def __init__(self, bot: TelegramBot, current_chat_id: int):
        self._bot = bot
        self.current_chat_id = current_chat_id

    async def send_message(self, chat_id: int, text: str) -> None:
        for chunk in split_message(text):
            await self._bot._api("sendMessage", chat_id=chat_id, text=chunk)

    async def send_document(
        self, chat_id: int, path: Path, caption: str
    ) -> None:
        await self._bot._send_document(chat_id, path, caption)

    def known_chats(self) -> list[dict]:
        out: list[dict] = []
        for chat_id, entry in self._bot._chats.items():
            row = {"chat_id": chat_id, **entry}
            out.append(row)
        if self.current_chat_id not in self._bot._chats:
            out.append({
                "chat_id": self.current_chat_id,
                "title": "this conversation",
                "type": "current",
                "last_seen": "now",
            })
        return sorted(out, key=lambda c: c.get("chat_id", 0))

    async def chat_info(self, chat_id: int) -> dict:
        return await self._bot._api("getChat", chat_id=chat_id)

    def is_known(self, chat_id: int) -> bool:
        b = self._bot
        return (
            chat_id == self.current_chat_id
            or chat_id in b._chats
            or chat_id in b.allowed
            or chat_id == b._owner
        )


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
        # update_id -> raw update, fetched but not yet answered (the journal)
        self._journal: dict[int, dict] = {}
        # in-flight answer tasks (drained on graceful shutdown)
        self._inflight: set[asyncio.Task] = set()
        self._stopping = False
        # v0.8 awareness: @username + the chats this bot is actually in
        self._bot_username = "?"
        self._chats: dict[int, dict] = {}
        # v0.8 follow-through: scheduled jobs + a sync hook for the host
        self.jobs = JobStore(settings.db_file())
        self.jobs_changed_cb = None  # host wires this (e.g. run_bot _push_db)
        self._job_task: asyncio.Task | None = None
        self._load_state()

    # ------------------------------------------------------------------
    async def run_forever(
        self,
        *,
        max_runtime_s: float | None = None,
        shutdown_event: asyncio.Event | None = None,
        drain_grace_s: float = 90.0,
    ) -> None:
        """Poll until a runtime budget / shutdown signal, then leave gracefully.

        On the way out: stop fetching, wait (bounded) for in-flight turns,
        tell still-busy chats their request is saved, and return — the host
        wrapper syncs state and respawns. Journaled-but-unanswered updates
        stay in the state file and are drained on the next boot.
        """
        import time as _time

        started = _time.monotonic()
        me = await self._api("getMe")
        self._bot_username = str(me.get("username") or "?")
        bot_name = self._bot_username
        log.info("logged in as @%s — polling for messages…", bot_name)
        print(f"woyo telegram bot @{bot_name} is up — Ctrl-C to stop")
        if self._had_state:
            await self._drain_journal()
        else:
            await self._skip_backlog()
        # v0.8 follow-through: jobs left 'running' by a killed host go back
        # on the queue; then the scheduler ticks beside the poller
        recovered = self.jobs.recover(max_attempts=self.settings.job_max_attempts)
        if recovered:
            log.info("recovered %s interrupted job(s)", len(recovered))
            self._jobs_changed()
        self._job_task = asyncio.create_task(self._job_loop())
        try:
            while True:
                if shutdown_event is not None and shutdown_event.is_set():
                    log.info("shutdown signal — rotating out")
                    break
                if max_runtime_s is not None:
                    remaining = max_runtime_s - (_time.monotonic() - started)
                    # never start a long poll that would outlive the budget
                    if remaining < _POLL_TIMEOUT + 10:
                        log.info(
                            "runtime budget reached (%.0fs) — rotating out",
                            max_runtime_s,
                        )
                        break
                try:
                    poll_started = _time.monotonic()
                    updates = await self._poll()
                    for update in updates:
                        await self._dispatch(update)
                    if not updates and _time.monotonic() - poll_started < 5.0:
                        # an empty long-poll should hold ~50s server-side; one
                        # returning instantly means a broken/proxied API (or a
                        # test double) — never hot-loop the Bot API
                        await asyncio.sleep(1.0)
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
        finally:
            if self._job_task is not None:
                self._job_task.cancel()
                self._job_task = None
        await self._graceful_exit(drain_grace_s)

    async def _graceful_exit(self, grace_s: float) -> None:
        """Bounded drain of in-flight turns + rotation notice for stragglers."""
        self._stopping = True
        deadline = asyncio.get_running_loop().time() + grace_s
        while self._inflight and asyncio.get_running_loop().time() < deadline:
            await asyncio.wait(
                list(self._inflight), timeout=1.0,
            )
        stragglers = {task for task in self._inflight if not task.done()}
        if stragglers:
            # who is busy BEFORE cancelling (locks release on cancel)
            busy_chats = {
                chat_id
                for chat_id, lock in self._locks.items()
                if lock.locked()
            }
            for task in stragglers:
                task.cancel()
            # their journaled updates stay pending: the next host drains them
            for chat_id in busy_chats:
                await self._send(
                    chat_id,
                    "🔄 I'm switching host runners (routine rotation). Your "
                    "request is saved and will be picked up automatically in "
                    "a minute or two — no need to resend it.",
                )
        self._save_state()
        # release chromium before the runner goes away (browser sessions are
        # ephemeral by design; browser_navigate starts a fresh one on the
        # next host)
        try:
            from woyo.tools.builtin.browser import close_default_manager

            await close_default_manager()
        except Exception as exc:  # noqa: BLE001 — cleanup must never throw
            log.debug("browser cleanup skipped: %s", exc)

    async def _drain_journal(self) -> None:
        """Answer requests that were fetched but left unanswered by a
        previous (killed) host — the anti-silent-loss path (v0.7)."""
        if not self._journal:
            return
        entries = sorted(self._journal.items())
        log.info("draining %s journaled update(s) from previous host", len(entries))
        for _update_id, update in entries:
            try:
                await self._dispatch(update)
            except Exception as exc:  # noqa: BLE001 — one bad entry must not block
                log.error("journal drain failed for %s: %s", _update_id, exc)
        # give spawned tasks a moment so the journal empties naturally
        if self._inflight:
            await asyncio.wait(
                list(self._inflight), timeout=5.0
            )

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
            # journal BEFORE any processing: a host killed mid-turn must not
            # lose the request (the offset alone would swallow it)
            for update in updates:
                self._journal_update(update)
            self._save_state()
        return updates

    # ------------------------------------------------------------------
    def _journal_update(self, update: dict) -> None:
        """Record a fetched update as unanswered (durable via _save_state)."""
        update_id = update.get("update_id")
        if update_id is None:
            return
        self._journal[int(update_id)] = update
        # bounded: keep the newest 50
        if len(self._journal) > 50:
            for key in sorted(self._journal)[:-50]:
                self._journal.pop(key, None)

    def _journal_done(self, update_id) -> None:
        """Mark an update answered; persists immediately."""
        if update_id is None:
            return
        if self._journal.pop(int(update_id), None) is not None:
            self._save_state()

    async def _dispatch(self, update: dict) -> None:
        callback = update.get("callback_query")
        if callback:
            await self._handle_callback(callback)
            return
        msg = update.get("message") or update.get("edited_message") or {}
        chat_id = int((msg.get("chat") or {}).get("id", 0))
        if not chat_id:
            return
        self._note_chat(msg)  # awareness: remember every (authorized) chat
        text = (msg.get("text") or msg.get("caption") or "").strip()
        attachment = attachment_ref(msg)
        update_id = update.get("update_id")
        if msg.get("sticker") and not text and not attachment:
            if not self._authorized_msg(chat_id, msg):
                self._journal_done(update_id)
                return
            await self._send(
                chat_id, "😅 I can't do much with stickers — send text or a file."
            )
            self._journal_done(update_id)
            return
        if not text and attachment is None:
            await self._send(
                chat_id,
                "🤔 I couldn't read that message type. Text and files work best.",
            )
            self._journal_done(update_id)
            return
        if not self._authorized_msg(chat_id, msg):
            await self._send(chat_id, "🔒 This bot is private.")
            self._journal_done(update_id)
            return
        if text.startswith("/") and attachment is None:
            await self._command(chat_id, text)
            self._journal_done(update_id)
            return
        # Run as a task so the poller keeps receiving updates — this is
        # what lets a button press arrive while the run awaits an approval.
        # The wrapper un-journals the update once the reply is out.
        if attachment is not None:
            inner = self._answer_attachment(chat_id, msg, text)
        else:
            inner = self._answer(chat_id, text)

        async def _run_and_unjournal() -> None:
            # Un-journal ONLY on normal completion (a reply went out).
            # On cancellation (host rotation / kill) the entry STAYS so the
            # next boot re-processes the request — that is the durability
            # contract; the rotation notice already told the user.
            await inner
            self._journal_done(update_id)

        task = asyncio.create_task(_run_and_unjournal())
        self._inflight.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._inflight.discard(task)
        if not task.cancelled() and task.exception():
            log.error("answer task crashed: %s", task.exception())

    def _authorized(self, chat_id: int) -> bool:
        if self.allowed:
            return chat_id in self.allowed
        return self._owner in (None, chat_id)

    def _authorized_msg(self, chat_id: int, msg: dict) -> bool:
        """Access check with message context.

        Beyond the chat-level rule, the CLAIMING OWNER speaking in a group
        they added the bot to authorizes that chat: `from.id` comes from
        Telegram's servers, and the owner's account is the trust root of a
        claimed bot. Strangers in any chat stay refused.
        """
        if self._authorized(chat_id):
            return True
        if self._owner is None:
            return False
        from_id = (msg.get("from") or {}).get("id")
        return from_id == self._owner

    # ------------------------------------------------------------------
    # v0.8 — awareness: the chat index the agent is told about
    # ------------------------------------------------------------------
    def _note_chat(self, msg: dict) -> None:
        """Remember an (authorized) chat so the agent knows where it lives.

        Only chats that pass the access check are indexed — strangers must
        not become sendable destinations.
        """
        chat = msg.get("chat") or {}
        chat_id = chat.get("id")
        if not isinstance(chat_id, int) or not self._authorized_msg(chat_id, msg):
            return
        now = datetime.now(UTC).isoformat(timespec="seconds")
        entry = self._chats.get(chat_id, {})
        # never downgrade: a later sparse chat object (no title/username)
        # must not erase what an earlier richer one recorded
        title = (
            chat.get("title")
            or " ".join(
                p for p in (chat.get("first_name"), chat.get("last_name")) if p
            )
            or chat.get("username")
            or entry.get("title")
            or str(chat_id)
        )
        self._chats[chat_id] = {
            "title": str(title)[:120],
            "type": str(chat.get("type", entry.get("type", "?"))),
            "username": chat.get("username") or entry.get("username"),
            "first_seen": entry.get("first_seen", now),
            "last_seen": now,
        }
        # bounded index: forget the least recently seen beyond 100 entries
        if len(self._chats) > 100:
            by_age = sorted(
                self._chats.items(), key=lambda kv: kv[1].get("last_seen", "")
            )
            for stale_id, _ in by_age[: len(self._chats) - 100]:
                self._chats.pop(stale_id, None)
        self._save_state()

    def _environment_provider(self, chat_id: int):
        """Build the environment facts injected into this chat's prompts."""

        def env() -> str:
            others = {
                cid: c for cid, c in self._chats.items() if cid != chat_id
            }
            lines = [
                f"You are woyo, running as the Telegram bot @{self._bot_username} "
                "(verified via the Bot API).",
                "You are INSIDE Telegram right now: user messages arrive as "
                "Telegram updates and your replies are delivered to the user "
                "as Telegram messages. Saying 'I can't reach Telegram' is "
                "simply wrong — you live here.",
                "A server-side bot token is configured for Bot API calls "
                "(never reveal, paste or send it).",
                f"Current conversation: chat_id {chat_id} — your replies and "
                "send_file deliveries land here.",
            ]
            if others:
                listing = "; ".join(
                    f"{c.get('title', '?')} (id {cid}, {c.get('type', '?')})"
                    for cid, c in sorted(others.items())
                )
                lines.append(
                    "Other chats you are in (addressable with "
                    "send_telegram_message / send_document, which ask the "
                    f"user's approval first): {listing}."
                )
            else:
                lines.append(
                    "You are not in any other chat besides this one right now."
                )
            lines.append(
                "You CANNOT: make voice or video calls, send SMS or email, "
                "join chats by yourself, read messages from chats you are not "
                "in, access the user's Telegram account or contacts, or "
                "receive verification codes. If a task needs something your "
                "tool list does not offer, say plainly what is missing — "
                "never invent a capability."
            )
            return "\n".join(lines)

        return env

    # ------------------------------------------------------------------
    # v0.8 — follow-through: the scheduled-job loop
    # ------------------------------------------------------------------
    def _jobs_changed(self) -> None:
        if self.jobs_changed_cb is not None:
            try:
                self.jobs_changed_cb()
            except Exception as exc:  # noqa: BLE001 — sync must not kill jobs
                log.warning("jobs sync hook failed: %s", exc)

    async def _job_loop(self) -> None:
        """Tick job_poll_s: run what is due, report to the owning chat."""
        while not self._stopping:
            try:
                await self._run_due_jobs()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — the scheduler must survive
                log.error("job sweep failed: %s", exc)
            await asyncio.sleep(self.settings.job_poll_s)

    async def _run_due_jobs(self) -> None:
        for job in self.jobs.due(limit=5):
            log.info("job #%s '%s' due — running", job.id, job.title)
            task = asyncio.create_task(self._run_job(job.id))
            self._inflight.add(task)
            task.add_done_callback(self._task_done)

    async def _run_job(self, job_id: int) -> None:
        """Execute one scheduled job and report the outcome in its chat."""
        job = self.jobs.get(job_id)
        if job is None or job.status != "scheduled":
            return
        self.jobs.mark_running(job_id)
        self._jobs_changed()
        chat_id = job.chat_id
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        typing = asyncio.create_task(self._typing_loop(chat_id))
        try:
            async with lock:
                await self._send(
                    chat_id, f"⏰ Running your scheduled task: {job.title}"
                )
                prompt = (
                    "(This is a SCHEDULED task you committed to earlier in "
                    "this chat; its time has come — execute it now and report "
                    f"the result. Title: {job.title!r}. Your final reply is "
                    "delivered to this chat automatically — do NOT call "
                    "send_telegram_message for this chat; that tool is only "
                    "for reaching OTHER chats and nobody may be awake to "
                    "press its approval button.)\n\n" + job.prompt
                )
                reply = await self._session(chat_id).send(prompt)
                await self._send_reply(chat_id, reply)
                self.jobs.finish(
                    job_id, "done", result_preview=reply.answer
                )
                self._jobs_changed()
        except asyncio.CancelledError:
            # host rotation: put it back so the next host retries (bounded)
            cur = self.jobs.get(job_id)
            if cur is not None and cur.status == "running":
                if cur.attempts >= self.settings.job_max_attempts:
                    self.jobs.finish(
                        job_id, "failed",
                        error="interrupted by host rotation too many times",
                    )
                else:
                    self.jobs.reschedule(job_id)
                self._jobs_changed()
            raise
        except Exception as exc:  # noqa: BLE001 — failures must be reported
            log.exception("scheduled job #%s failed", job_id)
            self.jobs.finish(job_id, "failed", error=str(exc))
            self._jobs_changed()
            await self._send(
                chat_id,
                f"⚠️ Scheduled task '{job.title}' failed: {exc}\n"
                "Nothing is running now — tell me to retry it if you want.",
            )
        finally:
            typing.cancel()

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
        elif cmd == "/tasks":
            jobs = self.jobs.active_for_chat(chat_id)
            if not jobs:
                await self._send(
                    chat_id, "📭 No scheduled tasks pending in this chat."
                )
            else:
                lines = ["⏰ Pending scheduled tasks:"]
                for j in jobs:
                    lines.append(f"#{j.id} — {j.title} (runs at {j.run_at} UTC)")
                lines.append(
                    "\nAsk me to cancel any of them by id (cancel_scheduled_task)."
                )
                await self._send(chat_id, "\n".join(lines))
        elif cmd == "/help":
            await self._send(chat_id, WELCOME, markdown=True)
        else:
            await self._send(chat_id, "Unknown command. Try /help")

    # ------------------------------------------------------------------
    async def _answer(self, chat_id: int, text: str) -> None:
        lock = self._locks.setdefault(chat_id, asyncio.Lock())
        if lock.locked():
            # queue, never drop: the lock is released when the previous
            # turn finishes, then this one runs
            await self._send(
                chat_id,
                "⏳ Got it — I'm still working on your previous message. "
                "This one is queued and will run right after (nothing is "
                "dropped).",
            )
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
            await self._send(
                chat_id,
                "⏳ Got it — I'm still working on your previous message. "
                "This one is queued and will run right after (nothing is "
                "dropped).",
            )
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
        # only the chat that was asked (and is still known to us) may answer
        if chat_id != asked_chat or not (
            self._authorized(chat_id) or chat_id in self._chats
        ):
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
                environment=self._environment_provider(chat_id),
                telegram_transport=_BotTransport(self, chat_id),
                job_store=self.jobs,
                chat_id=chat_id,
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
            pending = data.get("pending")
            if isinstance(pending, dict):
                self._journal = {
                    int(k): v for k, v in pending.items() if isinstance(v, dict)
                }
            chats = data.get("chats")
            if isinstance(chats, dict):
                self._chats = {
                    int(cid): entry
                    for cid, entry in chats.items()
                    if isinstance(entry, dict)
                }
        except (OSError, ValueError, TypeError):
            pass

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps({
                    "offset": self._offset,
                    "owner": self._owner,
                    "pending": {str(k): v for k, v in self._journal.items()},
                    "chats": {str(k): v for k, v in self._chats.items()},
                }),
                encoding="utf-8",
            )
        except OSError:
            pass
