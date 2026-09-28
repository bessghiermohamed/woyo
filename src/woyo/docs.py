"""Document rendering with real typography for any script (v0.7, ADR-14).

Why this module exists: an agent-generated PDF once shipped with every
Arabic glyph as a black square — hand-written reportlab code had used the
built-in Helvetica, whose WinAnsi encoding has no Arabic at all. Documents
are now a first-class capability instead of ad-hoc sandbox code:

- Fonts ship WITH the package (``woyo/resources/fonts``): Amiri (Arabic +
  Latin, proper Naskh) and DejaVu (Latin/Greek/Cyrillic + symbols). No
  dependence on whatever fonts a host happens to have.
- Arabic text is shaped (``arabic_reshaper``) and reordered
  (``python-bidi``) before layout; RTL paragraphs wrap right-to-left
  (reportlab ``wordWrap="RTL"``).
- Mixed-script lines are split into per-script runs, each drawn with a
  font that actually has the glyphs.
- Characters no bundled font covers (emoji, CJK…) are DROPPED and
  counted — never rendered as tofu squares — and the count is reported
  back to the agent so it can tell the user.

Import cost is deferred: reportlab & friends load lazily, so environments
without the ``docs`` extra simply get an honest error from the tool.

This module is also usable from sandboxed python_exec code
(``from woyo.docs import render_markdown_pdf``) for anything richer than
the tool's markdown dialect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_FONT_DIR = Path(__file__).resolve().parent / "resources" / "fonts"

#: Arabic + Arabic Supplement + Extended-A + presentation forms + digits.
_ARABIC_RE = re.compile(
    "[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def fonts_dir() -> Path:
    return _FONT_DIR


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


def is_rtl_block(text: str) -> bool:
    """Base direction from the first strong-directional character
    (the Unicode heuristic): Arabic anywhere early => the block reads RTL."""
    for ch in text:
        if _ARABIC_RE.match(ch):
            return True
        if ch.isalpha():  # first strong LTR letter wins
            return False
    return False


@dataclass(slots=True)
class RenderStats:
    """What a render produced (surfaced to the agent as an observation)."""

    pages: int = 0
    blocks: int = 0
    dropped_chars: int = 0
    rtl_blocks: int = 0


@dataclass(slots=True)
class _Block:
    kind: str  # title | h1 | h2 | p | bullet | item | hr | code
    text: str = ""
    lines: list[str] = field(default_factory=list)


# --- fonts ----------------------------------------------------------------------


def register_fonts() -> None:
    """Idempotently register the bundled font stack with reportlab."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if "WOYO-Amiri" in pdfmetrics.getRegisteredFontNames():
        return
    pdfmetrics.registerFont(TTFont("WOYO-Amiri", _FONT_DIR / "Amiri-Regular.ttf"))
    pdfmetrics.registerFont(TTFont("WOYO-Amiri-Bold", _FONT_DIR / "Amiri-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("WOYO-DejaVu", _FONT_DIR / "DejaVuSans.ttf"))
    pdfmetrics.registerFont(
        TTFont("WOYO-DejaVu-Bold", _FONT_DIR / "DejaVuSans-Bold.ttf")
    )
    pdfmetrics.registerFont(TTFont("WOYO-Mono", _FONT_DIR / "DejaVuSansMono.ttf"))


def _coverage(font_name: str) -> set[int]:
    """Codepoints the registered TTF face can actually draw."""
    from reportlab.pdfbase import pdfmetrics

    register_fonts()
    face = pdfmetrics.getFont(font_name).face
    return set(face.charToGlyph.keys())


# --- shaping / run splitting -------------------------------------------------------


def shape_text(text: str) -> str:
    """Reshape Arabic letters into contextual forms and apply the bidi
    algorithm, returning VISUAL order (what a renderer draws left-to-right)."""
    if not has_arabic(text):
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display

    return get_display(arabic_reshaper.reshape(text))


def _script_runs(visual: str, coverage: dict[str, set[int]]) -> tuple[list[tuple[str, str]], int]:
    """Split visual text into (font, chunk) runs.

    Arabic-range characters go to Amiri; everything covered by DejaVu goes
    to DejaVu; anything else is dropped (counted) — a dropped character
    beats a black square, and the caller reports the count honestly.
    """
    runs: list[tuple[str, str]] = []
    dropped = 0
    cur_font: str | None = None
    cur: list[str] = []
    for ch in visual:
        if ch in ("\n", "\r", "\t"):
            font = cur_font  # whitespace rides along with the current run
            if font is None:
                font = "WOYO-DejaVu"
        elif _ARABIC_RE.match(ch):
            font = "WOYO-Amiri"
        elif ord(ch) in coverage["WOYO-DejaVu"]:
            font = "WOYO-DejaVu"
        elif ord(ch) in coverage["WOYO-Amiri"]:
            font = "WOYO-Amiri"
        else:
            dropped += 1
            continue
        if font == cur_font:
            cur.append(ch)
        else:
            if cur:
                runs.append((cur_font or "WOYO-DejaVu", "".join(cur)))
            cur_font, cur = font, [ch]
    if cur:
        runs.append((cur_font or "WOYO-DejaVu", "".join(cur)))
    return runs, dropped


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def paragraph_markup(
    text: str,
    *,
    bold: bool = False,
    coverage: dict[str, set[int]] | None = None,
) -> tuple[str, int]:
    """Turn logical-order text into reportlab intra-paragraph markup.

    Handles **bold** spans, Arabic shaping/bidi and per-script font runs.
    Returns (markup, dropped_char_count).
    """
    if coverage is None:
        coverage = {
            name: _coverage(name)
            for name in ("WOYO-Amiri", "WOYO-DejaVu")
        }
    dropped_total = 0
    parts: list[str] = []
    pos = 0
    for match in _BOLD_RE.finditer(text):
        if match.start() > pos:
            markup, dropped = _runs_markup(text[pos:match.start()], False, coverage)
            parts.append(markup)
            dropped_total += dropped
        markup, dropped = _runs_markup(match.group(1), True, coverage)
        parts.append(f"<b>{markup}</b>")
        dropped_total += dropped
        pos = match.end()
    if pos < len(text):
        markup, dropped = _runs_markup(text[pos:], False, coverage)
        parts.append(markup)
        dropped_total += dropped
    return "".join(parts) or "&nbsp;", dropped_total


def _runs_markup(text: str, bold: bool, coverage: dict[str, set[int]]) -> tuple[str, int]:
    visual = shape_text(text)
    runs, dropped = _script_runs(visual, coverage)
    out: list[str] = []
    for font, chunk in runs:
        name = f"{font}-Bold" if bold and font != "WOYO-Mono" else font
        out.append(f'<font name="{name}">{_escape(chunk)}</font>')
    return "".join(out), dropped


# --- markdown parsing -----------------------------------------------------------------


def parse_markdown(content: str) -> list[_Block]:
    """Parse the small markdown dialect documents are written in.

    Supported: `# / ## / ###` headings, `- / * / •` bullets, `1.` numbered
    items, `---` rules, fenced or indented code blocks, blank-line-separated
    paragraphs, `**bold**` spans. Everything else is plain text.
    """
    blocks: list[_Block] = []
    lines = content.replace("\r\n", "\n").split("\n")
    i = 0
    para: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append(_Block("p", " ".join(s.strip() for s in para if s.strip())))
            para.clear()

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            flush_para()
            i += 1
            continue
        if line.startswith("```"):
            flush_para()
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # closing fence (or EOF)
            blocks.append(_Block("code", lines=code))
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            flush_para()
            level = len(m.group(1))
            blocks.append(
                _Block("title" if level == 1 and not any(
                    b.kind == "title" for b in blocks
                ) else f"h{level}", m.group(2))
            )
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", line):
            flush_para()
            blocks.append(_Block("hr"))
            i += 1
            continue
        m = re.match(r"^[-*•]\s+(.*)$", line)
        if m:
            flush_para()
            blocks.append(_Block("bullet", m.group(1)))
            i += 1
            continue
        m = re.match(r"^\d+[.)]\s+(.*)$", line)
        if m:
            flush_para()
            blocks.append(_Block("item", m.group(1)))
            i += 1
            continue
        para.append(raw)
        i += 1
    flush_para()
    return blocks


# --- rendering ---------------------------------------------------------------------


def _styles():
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle

    def style(name: str, **kw) -> ParagraphStyle:
        base = dict(fontName="WOYO-DejaVu", fontSize=10.5, leading=15.5,
                    spaceAfter=6, alignment=TA_LEFT)
        base.update(kw)
        return ParagraphStyle(name, **base)

    return {
        "title": style("d-title", fontName="WOYO-DejaVu-Bold", fontSize=20,
                       leading=26, spaceAfter=14),
        "title-rtl": style("d-title-rtl", fontName="WOYO-Amiri-Bold", fontSize=22,
                           leading=30, spaceAfter=14, alignment=TA_RIGHT,
                           wordWrap="RTL"),
        "h2": style("d-h2", fontName="WOYO-DejaVu-Bold", fontSize=15, leading=20,
                    spaceBefore=12, spaceAfter=6),
        "h2-rtl": style("d-h2-rtl", fontName="WOYO-Amiri-Bold", fontSize=16.5,
                        leading=24, spaceBefore=12, spaceAfter=6,
                        alignment=TA_RIGHT, wordWrap="RTL"),
        "h3": style("d-h3", fontName="WOYO-DejaVu-Bold", fontSize=12.5, leading=17,
                    spaceBefore=10, spaceAfter=4),
        "h3-rtl": style("d-h3-rtl", fontName="WOYO-Amiri-Bold", fontSize=14,
                        leading=20, spaceBefore=10, spaceAfter=4,
                        alignment=TA_RIGHT, wordWrap="RTL"),
        "body": style("d-body"),
        "body-rtl": style("d-body-rtl", fontName="WOYO-Amiri", fontSize=12.5,
                          leading=20, alignment=TA_RIGHT, wordWrap="RTL"),
        "bullet": style("d-bullet", leftIndent=14, bulletIndent=4),
        "bullet-rtl": style("d-bullet-rtl", fontName="WOYO-Amiri", fontSize=12.5,
                            leading=20, alignment=TA_RIGHT, wordWrap="RTL"),
    }


def _wrap_code(lines: list[str], width: int = 92) -> list[str]:
    out: list[str] = []
    for line in lines:
        while len(line) > width:
            out.append(line[:width])
            line = "    " + line[width:]
        out.append(line)
    return out or [""]


def _sanitize_code(
    code: str, coverage: dict[str, set[int]]
) -> tuple[str, int]:
    """Drop characters the mono font cannot draw (emoji…) — never tofu."""
    drawable = coverage["WOYO-DejaVu"] | coverage["WOYO-Amiri"]
    dropped = 0
    out: list[str] = []
    for ch in code:
        if ch in "\n\t" or ord(ch) in drawable or ch.isascii():
            out.append(ch)
        else:
            dropped += 1
    return "".join(out), dropped


def render_markdown_pdf(
    content: str,
    dest: str | Path,
    *,
    title: str = "",
    author: str = "woyo",
    subject: str = "",
) -> RenderStats:
    """Render markdown-ish `content` to a typographically correct PDF.

    Raises ImportError when the ``docs`` extra (reportlab, arabic-reshaper,
    python-bidi) is not installed — callers turn that into an honest error.
    """
    import asyncio  # noqa: F401 — documentation: safe in threads too

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.flowables import HRFlowable

    register_fonts()
    coverage = {
        name: _coverage(name) for name in ("WOYO-Amiri", "WOYO-DejaVu")
    }
    styles = _styles()
    stats = RenderStats()
    blocks = parse_markdown(content)
    stats.blocks = len(blocks)

    story: list = []
    for block in blocks:
        if block.kind == "hr":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color="#999999"))
            story.append(Spacer(1, 6))
            continue
        if block.kind == "code":
            code_text = "\n".join(_wrap_code(block.lines))
            code_text, dropped = _sanitize_code(code_text, coverage)
            stats.dropped_chars += dropped
            pre = Preformatted(code_text, styles["body"].clone(
                "d-code", fontName="WOYO-Mono", fontSize=8.5, leading=12
            ))
            box = Table([[pre]], colWidths=[17 * cm])
            box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), "#F4F4F4"),
                ("BOX", (0, 0), (-1, -1), 0.5, "#CCCCCC"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(Spacer(1, 2))
            story.append(box)
            story.append(Spacer(1, 6))
            continue

        rtl = is_rtl_block(block.text)
        if rtl:
            stats.rtl_blocks += 1
        if block.kind == "title":
            st = styles["title-rtl" if rtl else "title"]
            markup, dropped = paragraph_markup(
                block.text, bold=True, coverage=coverage
            )
            stats.dropped_chars += dropped
            story.append(Paragraph(markup, st))
        elif block.kind in ("h2", "h3"):
            key = f"{block.kind}-rtl" if rtl else block.kind
            markup, dropped = paragraph_markup(
                block.text, bold=True, coverage=coverage
            )
            stats.dropped_chars += dropped
            story.append(Paragraph(markup, styles[key]))
        elif block.kind in ("bullet", "item"):
            prefix = "• " if block.kind == "bullet" else ""
            st = styles["bullet-rtl" if rtl else "bullet"]
            markup, dropped = paragraph_markup(
                f"{prefix}{block.text}", coverage=coverage
            )
            stats.dropped_chars += dropped
            story.append(Paragraph(markup, st))
        else:
            st = styles["body-rtl" if rtl else "body"]
            markup, dropped = paragraph_markup(block.text, coverage=coverage)
            stats.dropped_chars += dropped
            story.append(Paragraph(markup, st))

    if not story:  # empty content still yields a valid single-page PDF
        story.append(Spacer(1, 1))

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(dest_path),
        pagesize=A4,
        leftMargin=2.1 * cm, rightMargin=2.1 * cm,
        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
        title=title or (blocks[0].text[:120] if blocks else "woyo document"),
        author=author or "woyo",
        subject=subject,
        creator="woyo",
    )

    def _footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont("WOYO-DejaVu", 8.5)
        canvas.setFillColorRGB(0.45, 0.45, 0.45)
        canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"{canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    stats.pages = doc.page
    return stats


__all__ = [
    "RenderStats",
    "fonts_dir",
    "has_arabic",
    "is_rtl_block",
    "paragraph_markup",
    "parse_markdown",
    "register_fonts",
    "render_markdown_pdf",
    "shape_text",
]
