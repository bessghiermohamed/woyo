"""send_file: deliver a workspace file to the user's chat (v0.6, ADR-13).

The tool is transport-agnostic: a chat-scoped ``file_sender`` callable is
injected at registry build time (the Telegram frontend wires it to
sendDocument; CLI/task mode registers nothing). Only the requesting chat
can ever receive a file — there is no addressable "somewhere else".

Permission is SANDBOXED, not WRITES_EXTERNAL: the destination is the user
who is already talking to the agent, and the source is workspace-contained
(the same containment write_file already enforces). The filename and
caption are visible to the user, so nothing is covert.
"""

from __future__ import annotations

from collections.abc import Awaitable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from woyo.config import Settings
from woyo.errors import ErrorKind
from woyo.tools.base import Permission, Tool, ToolResult
from woyo.tools.builtin.sandbox import resolve_ws_path, workspace_root


class FileSender(Protocol):
    """Transport hook: send `path` (with `caption`) to the requesting chat."""

    def __call__(self, path: Path, caption: str) -> Awaitable[None]: ...


class SendFileArgs(BaseModel):
    path: str = Field(
        min_length=1, max_length=300,
        description="Workspace-relative path of the file to send",
    )
    caption: str = Field(
        default="", max_length=900,
        description="Optional one-line caption shown with the file",
    )


class SendFileTool(Tool):
    name = "send_file"
    description = (
        "Send a workspace file to the user in this chat as a downloadable "
        "document (optional caption). Use it whenever the user asks for a "
        "file, export, report, image or any artifact — e.g. after write_file "
        "or python_exec created it. Any workspace file works (text, code, "
        "CSV, PDF, images, archives). Tell the user what the file contains."
    )
    permission = Permission.SANDBOXED  # destination = the requesting chat only
    timeout_s = 300.0  # large uploads on slow links
    Args = SendFileArgs

    def __init__(self, settings: Settings, sender: FileSender):
        self._settings = settings
        self._sender = sender

    async def run(self, args: SendFileArgs) -> ToolResult:
        root = workspace_root(self._settings)
        try:
            path = resolve_ws_path(root, args.path)
        except ValueError as exc:
            return ToolResult.error(ErrorKind.INVALID_INPUT, str(exc))
        if not path.is_file():
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"No such file in the workspace: '{args.path}'. "
                "Create it first (write_file / python_exec), or check list_dir.",
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
                f"{self._settings.chat_max_send_file_mb} MB sending cap. "
                "Split it or export a smaller version.",
            )
        try:
            await self._sender(path, args.caption)
        except Exception as exc:  # noqa: BLE001 — delivery problems are observations
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Sending '{args.path}' failed: {exc}",
            )
        return ToolResult.ok_result(
            f"Sent '{path.name}' ({size / 1024:.1f} KB) to the user.",
            data={"path": args.path, "bytes": size},
        )


__all__ = ["FileSender", "SendFileTool"]
