"""create_document: PDF generation with real typography (v0.7, ADR-14).

The agent used to hand-write reportlab snippets in python_exec; with the
default Helvetica that produced PDFs whose Arabic text was a wall of black
squares (WinAnsi has no Arabic). This tool makes documents first-class:
markdown-ish content in, a polished, correctly-shaped PDF in the workspace
(Arabic/RTL, Latin, Greek, Cyrillic…), ready for send_file.

Characters no bundled font can draw (emoji…) are dropped and COUNTED — the
observation reports the number so the agent can tell the user honestly
instead of shipping tofu squares.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from woyo.config import Settings
from woyo.errors import ErrorKind
from woyo.tools.base import Permission, Tool, ToolResult
from woyo.tools.builtin.sandbox import resolve_ws_path, workspace_root


class CreateDocumentArgs(BaseModel):
    filename: str = Field(
        min_length=1, max_length=300,
        description="Workspace-relative output path ending in .pdf",
    )
    content: str = Field(
        min_length=1, max_length=120_000,
        description=(
            "Document body in simple markdown: #/##/### headings, bullets "
            "('- '), numbered items ('1. '), paragraphs, **bold**, '---' "
            "rules, fenced ```code blocks. Write content in the user's "
            "language — Arabic/RTL renders correctly."
        ),
    )
    title: str = Field(
        default="", max_length=300,
        description="Document title (PDF metadata; body should start with '# Title')",
    )
    subject: str = Field(default="", max_length=300, description="PDF subject metadata")


class CreateDocumentTool(Tool):
    name = "create_document"
    description = (
        "Create a polished PDF document in the workspace from markdown-like "
        "content (headings, bullets, bold, code blocks). Typography is "
        "handled for you: Arabic/RTL, Latin, Greek, Cyrillic all render "
        "correctly. ALWAYS use this tool for PDFs and reports — do NOT "
        "hand-write reportlab/PDF code in python_exec (that produces black "
        "squares for Arabic). Follow up with send_file to deliver it."
    )
    permission = Permission.SANDBOXED  # writes inside the workspace only
    timeout_s = 90.0
    Args = CreateDocumentArgs

    def __init__(self, settings: Settings):
        self._settings = settings

    async def run(self, args: CreateDocumentArgs) -> ToolResult:
        if not args.filename.lower().endswith(".pdf"):
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                "create_document makes PDFs — filename must end in .pdf. "
                "For plain text/code/CSV use write_file instead.",
            )
        root = workspace_root(self._settings)
        try:
            path = resolve_ws_path(root, args.filename)
        except ValueError as exc:
            return ToolResult.error(ErrorKind.INVALID_INPUT, str(exc))
        try:
            from woyo.docs import render_markdown_pdf
        except ImportError as exc:
            return ToolResult.error(
                ErrorKind.CONFIG,
                f"Document rendering is unavailable ({exc}). Ask the user to "
                "install the 'docs' extra (reportlab, arabic-reshaper, "
                "python-bidi).",
            )
        try:
            stats = await asyncio.to_thread(
                render_markdown_pdf,
                args.content,
                path,
                title=args.title,
                subject=args.subject,
            )
        except Exception as exc:  # noqa: BLE001 — render errors are observations
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE, f"PDF rendering failed: {exc}"
            )
        size = path.stat().st_size if path.exists() else 0
        if size == 0:
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE, "PDF rendering produced an empty file."
            )
        note = (
            f"Created '{args.filename}' ({size / 1024:.1f} KB, "
            f"{stats.pages} page(s), {stats.blocks} blocks"
        )
        if stats.rtl_blocks:
            note += f", {stats.rtl_blocks} RTL/Arabic block(s) shaped correctly"
        if stats.dropped_chars:
            note += (
                f". NOTE: {stats.dropped_chars} character(s) no bundled font "
                "covers (emoji etc.) were OMITTED — tell the user if that "
                "matters"
            )
        return ToolResult.ok_result(
            note + ". Deliver it with send_file.",
            # NB: "pages" is reserved for web-tool pagination data — use
            # page_count to avoid tripping the citation collector.
            data={"path": args.filename, "page_count": stats.pages, "bytes": size},
        )


__all__ = ["CreateDocumentTool"]
