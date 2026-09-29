"""Telegram reach: proactive, addressable sends from inside a chat turn (v0.8).

The chat frontend injects a ``TelegramTransport`` at registry build time; the
tools below are the agent's hands on the Bot API beyond plain replies:

- ``list_chats`` / ``get_chat_info`` — see WHERE the bot can deliver (read-only)
- ``send_telegram_message`` — deliver text to any chat the bot is in
- ``send_document`` — deliver a workspace file to any chat the bot is in

Security posture (allowlists, approvals, limits):

- ALLOWLIST: destinations are restricted to chats the bot actually knows —
  the current conversation, chats recorded in its state index, or ids listed
  in the deployment's explicit allowlist. An arbitrary chat_id is refused
  with the known list attached, so the model self-corrects instead of
  spamming strangers.
- APPROVAL: both send tools are WRITES_EXTERNAL — the Telegram frontend
  shows an inline Approve/Deny button naming the destination before
  anything leaves. Same-chat file delivery keeps using the sandboxed
  ``send_file`` (no friction where the destination is the requester).
- LIMITS: message length and file size are capped in the args/tools; the
  underlying Bot API enforces its own quotas on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from woyo.config import Settings
from woyo.errors import ErrorKind
from woyo.tools.base import Permission, Tool, ToolResult
from woyo.tools.builtin.sandbox import resolve_ws_path, workspace_root

#: hard cap for one send_telegram_message call (Telegram allows 4096)
_MAX_MESSAGE_CHARS = 3500


class TelegramTransport(Protocol):
    """What a chat frontend exposes to the Telegram tools (v0.8)."""

    @property
    def current_chat_id(self) -> int: ...

    async def send_message(self, chat_id: int, text: str) -> None:
        """Deliver text; raises on any delivery failure."""

    async def send_document(
        self, chat_id: int, path: Path, caption: str
    ) -> None:
        """Upload a file; raises on any delivery failure."""

    def known_chats(self) -> list[dict]:
        """[{"chat_id", "title", "type", "username", "last_seen"}, ...]"""

    async def chat_info(self, chat_id: int) -> dict:
        """Fresh chat details from getChat; raises for unknown chats."""

    def is_known(self, chat_id: int) -> bool:
        """True if the destination is on the allowlist (known/allowed)."""


def _known_chats_help(transport: TelegramTransport) -> str:
    chats = transport.known_chats()
    if not chats:
        return "No known chats."
    listing = ", ".join(
        f"{c.get('title', '?')} (id {c.get('chat_id')})" for c in chats[:15]
    )
    return f"Known chats: {listing}."


class SendMessageArgs(BaseModel):
    chat_id: int = Field(description="Destination chat id (see list_chats)")
    text: str = Field(
        min_length=1, max_length=_MAX_MESSAGE_CHARS,
        description="Message text to deliver",
    )


class SendMessageTool(Tool):
    name = "send_telegram_message"
    description = (
        "Send a Telegram message to a chat this bot is in — proactively, at "
        "any point during your turn. Use it ONLY to reach a chat OTHER than "
        "the current one: in THIS conversation your ordinary reply is "
        "already delivered as a Telegram message, so sending here with a "
        "tool call adds nothing. Resolve the destination with list_chats "
        "when unsure; the user approves the send with a button. For files, "
        "use send_document."
    )
    permission = Permission.WRITES_EXTERNAL  # addressable destination
    timeout_s = 60.0
    Args = SendMessageArgs

    def __init__(self, transport: TelegramTransport):
        self._transport = transport

    async def run(self, args: SendMessageArgs) -> ToolResult:
        if not self._transport.is_known(args.chat_id):
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"Chat {args.chat_id} is not one of your chats — the bot has "
                "never seen a message from it, so sending there is refused. "
                + _known_chats_help(self._transport),
            )
        try:
            await self._transport.send_message(args.chat_id, args.text)
        except Exception as exc:  # noqa: BLE001 — delivery problems are observations
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Delivering the message to chat {args.chat_id} failed: {exc}",
            )
        return ToolResult.ok_result(
            f"Message delivered to chat {args.chat_id}.",
            data={"chat_id": args.chat_id, "chars": len(args.text)},
        )


class ListChatsArgs(BaseModel):
    pass


class ListChatsTool(Tool):
    name = "list_chats"
    description = (
        "List every chat this bot is in: chat id, type (private / group / "
        "supergroup / channel), title, and when a message was last seen "
        "there. Call this before sending anywhere so you address a real, "
        "known chat — never guess a chat id."
    )
    permission = Permission.READ_ONLY
    Args = ListChatsArgs

    def __init__(self, transport: TelegramTransport):
        self._transport = transport

    async def run(self, args: ListChatsArgs) -> ToolResult:
        chats = self._transport.known_chats()
        if not chats:
            return ToolResult.ok_result(
                "No chats known yet — this conversation is the only one."
            )
        lines = [f"Chats you are in ({len(chats)}):"]
        for c in chats:
            marker = (
                "  <- this conversation"
                if c.get("chat_id") == self._transport.current_chat_id
                else ""
            )
            username = f", @{c['username']}" if c.get("username") else ""
            lines.append(
                f"- {c.get('title', '?')} (id {c.get('chat_id')}, "
                f"{c.get('type', '?')}{username}, last seen "
                f"{c.get('last_seen', '?')}){marker}"
            )
        return ToolResult.ok_result("\n".join(lines[:30]))


class GetChatInfoArgs(BaseModel):
    chat_id: int = Field(description="Chat id to inspect (see list_chats)")


class GetChatInfoTool(Tool):
    name = "get_chat_info"
    description = (
        "Fresh details about one chat you are in (title, type, username, "
        "description, member count for groups), straight from Telegram. Use "
        "it to confirm which chat an id refers to before sending."
    )
    permission = Permission.READ_ONLY
    timeout_s = 30.0
    Args = GetChatInfoArgs

    def __init__(self, transport: TelegramTransport):
        self._transport = transport

    async def run(self, args: GetChatInfoArgs) -> ToolResult:
        if not self._transport.is_known(args.chat_id):
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"Chat {args.chat_id} is not one of your chats. "
                + _known_chats_help(self._transport),
            )
        try:
            info = await self._transport.chat_info(args.chat_id)
        except Exception as exc:  # noqa: BLE001 — lookups can fail
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"getChat failed for {args.chat_id}: {exc}",
            )
        keys = ("id", "type", "title", "username", "first_name", "last_name",
                "description", "member_count")
        pretty = "\n".join(
            f"{k}: {info[k]}" for k in keys if info.get(k) is not None
        )
        return ToolResult.ok_result(pretty or f"chat {args.chat_id} (no details)")


class SendDocumentArgs(BaseModel):
    chat_id: int = Field(description="Destination chat id (see list_chats)")
    path: str = Field(
        min_length=1, max_length=300,
        description="Workspace-relative path of the file to send",
    )
    caption: str = Field(
        default="", max_length=900,
        description="Optional caption shown with the file",
    )


class SendDocumentTool(Tool):
    name = "send_document"
    description = (
        "Send a workspace file to a specific chat this bot is in — the same "
        "delivery as send_file, but addressable to any known chat instead of "
        "only the current one (e.g. 'send this to the study group'). "
        "Resolve the destination with list_chats when unsure; the user "
        "approves the send with a button."
    )
    permission = Permission.WRITES_EXTERNAL  # addressable destination
    timeout_s = 300.0  # large uploads on slow links
    Args = SendDocumentArgs

    def __init__(self, settings: Settings, transport: TelegramTransport):
        self._settings = settings
        self._transport = transport

    async def run(self, args: SendDocumentArgs) -> ToolResult:
        root = workspace_root(self._settings)
        try:
            path = resolve_ws_path(root, args.path)
        except ValueError as exc:
            return ToolResult.error(ErrorKind.INVALID_INPUT, str(exc))
        if not path.is_file():
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"No such file in the workspace: '{args.path}'.",
            )
        size = path.stat().st_size
        if size == 0:
            return ToolResult.error(
                ErrorKind.INVALID_INPUT, f"'{args.path}' is empty (0 bytes)."
            )
        cap = self._settings.chat_max_send_file_mb * 1024 * 1024
        if size > cap:
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"'{args.path}' is {size / 1_048_576:.1f} MB — over the "
                f"{self._settings.chat_max_send_file_mb} MB sending cap.",
            )
        if not self._transport.is_known(args.chat_id):
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"Chat {args.chat_id} is not one of your chats — refused. "
                + _known_chats_help(self._transport),
            )
        try:
            await self._transport.send_document(args.chat_id, path, args.caption)
        except Exception as exc:  # noqa: BLE001 — delivery problems are observations
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Sending '{args.path}' to chat {args.chat_id} failed: {exc}",
            )
        return ToolResult.ok_result(
            f"Sent '{path.name}' ({size / 1024:.1f} KB) to chat {args.chat_id}.",
            data={"path": args.path, "chat_id": args.chat_id, "bytes": size},
        )


__all__ = [
    "TelegramTransport",
    "SendMessageTool",
    "ListChatsTool",
    "GetChatInfoTool",
    "SendDocumentTool",
]
