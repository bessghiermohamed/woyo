"""v0.7 create_document + woyo.docs: typography that actually renders."""

from __future__ import annotations

from tests.conftest import make_settings
from woyo.docs import (
    is_rtl_block,
    paragraph_markup,
    parse_markdown,
    render_markdown_pdf,
    shape_text,
)
from woyo.tools.builtin.documents import CreateDocumentTool

AR_PARA = "فوائد النعناع عديدة وتشمل تحسين الهضم وتخفيف الصداع."
EN_PARA = "Mint helps with digestion and headaches."


# --- shaping helpers ---------------------------------------------------------------


def test_shape_text_returns_visual_order_for_arabic():
    logical = "النعناع"
    visual = shape_text(logical)
    assert visual != logical  # bidi reordered
    # presentation forms (contextually shaped glyphs) present
    assert any(0xFB50 <= ord(c) <= 0xFEFF for c in visual)


def test_shape_text_leaves_pure_latin_alone():
    assert shape_text("plain english 123") == "plain english 123"


def test_is_rtl_block():
    assert is_rtl_block(AR_PARA)
    assert not is_rtl_block(EN_PARA)
    # mixed but latin-dominant stays LTR
    assert not is_rtl_block(f"{EN_PARA} مع كلمة عربية")


def test_paragraph_markup_splits_scripts_and_counts_drops():
    markup, dropped = paragraph_markup(f"{AR_PARA} and {EN_PARA}")
    assert "WOYO-Amiri" in markup and "WOYO-DejaVu" in markup
    assert dropped == 0
    # emoji are not covered by any bundled font -> dropped, not tofu
    markup2, dropped2 = paragraph_markup("hello 🙂 world")
    assert dropped2 == 1
    assert "🙂" not in markup2


def test_paragraph_markup_bold_uses_bold_fonts():
    markup, _ = paragraph_markup(f"**{AR_PARA}** plain {EN_PARA}")
    assert "WOYO-Amiri-Bold" in markup
    assert markup.count("<b>") == 1


def test_parse_markdown_dialect():
    blocks = parse_markdown(
        "# Title\n\npara one\ncontinues here\n\n## Section\n\n- bullet a\n"
        "- bullet b\n\n1. first\n2. second\n\n---\n\n```py\nx = 1\n```\n"
    )
    kinds = [b.kind for b in blocks]
    assert kinds == [
        "title", "p", "h2", "bullet", "bullet",
        "item", "item", "hr", "code",
    ]
    assert blocks[1].text == "para one continues here"
    assert blocks[-1].lines == ["x = 1"]


# --- renderer ----------------------------------------------------------------------


def test_render_arabic_pdf(tmp_path):
    out = tmp_path / "mint.pdf"
    stats = render_markdown_pdf(
        f"# فوائد النعناع\n\n{AR_PARA}\n\n- يساعد على الهضم\n- يخفف الصداع",
        out,
        title="فوائد النعناع",
    )
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert stats.pages == 1
    assert stats.rtl_blocks >= 3
    # the Arabic-capable font is embedded (not Helvetica — no more squares)
    assert b"Amiri" in data

    import pypdf

    text = "".join(p.extract_text() for p in pypdf.PdfReader(str(out)).pages)
    # real shaped glyphs extract as presentation forms, proving text was drawn
    assert any(0xFB50 <= ord(c) <= 0xFEFF for c in text)
    assert len(text) > 30


def test_render_latin_pdf_uses_dejavu(tmp_path):
    out = tmp_path / "latin.pdf"
    stats = render_markdown_pdf(
        "# Report\n\nMint helps digestion. Ж Ω ünïcödé — dash",
        out,
    )
    assert stats.rtl_blocks == 0
    data = out.read_bytes()
    assert b"DejaVu" in data
    assert b"Amiri" not in data  # no Arabic -> Amiri not embedded


def test_render_drops_unsupported_chars_and_counts(tmp_path):
    out = tmp_path / "emoji.pdf"
    stats = render_markdown_pdf("# T\n\nhello 🙂🙂 world", out)
    assert stats.dropped_chars == 2


def test_render_mixed_code_block(tmp_path):
    out = tmp_path / "code.pdf"
    stats = render_markdown_pdf(
        "# Code\n\n```python\nprint('hi')  # 🙂\n```", out
    )
    assert stats.dropped_chars == 1  # emoji in code dropped, not tofu


def test_render_empty_content_still_makes_a_page(tmp_path):
    out = tmp_path / "empty.pdf"
    stats = render_markdown_pdf("", out)
    assert stats.pages == 1
    assert out.stat().st_size > 500


# --- the tool -------------------------------------------------------------------------


async def test_create_document_tool_writes_workspace_pdf(tmp_path):
    settings = make_settings(workspace_dir=str(tmp_path / "ws"))
    tool = CreateDocumentTool(settings)
    result = await tool.run(
        CreateDocumentTool.Args(
            filename="reports/mint.pdf",
            content=f"# فوائد النعناع\n\n{AR_PARA}",
            title="فوائد النعناع",
        )
    )
    assert result.ok
    pdf = tmp_path / "ws" / "reports" / "mint.pdf"
    assert pdf.exists()
    assert pdf.read_bytes().startswith(b"%PDF")
    assert b"Amiri" in pdf.read_bytes()
    assert "send_file" in result.content


async def test_create_document_rejects_non_pdf(tmp_path):
    settings = make_settings(workspace_dir=str(tmp_path / "ws"))
    tool = CreateDocumentTool(settings)
    result = await tool.run(
        CreateDocumentTool.Args(filename="data.csv", content="a,b\n1,2")
    )
    assert not result.ok
    assert "write_file" in result.content


async def test_create_document_rejects_traversal(tmp_path):
    settings = make_settings(workspace_dir=str(tmp_path / "ws"))
    tool = CreateDocumentTool(settings)
    result = await tool.run(
        CreateDocumentTool.Args(
            filename="../../etc/evil.pdf", content="# nope"
        )
    )
    assert not result.ok
    assert "workspace" in result.content


async def test_create_document_reports_dropped_emoji(tmp_path):
    settings = make_settings(workspace_dir=str(tmp_path / "ws"))
    tool = CreateDocumentTool(settings)
    result = await tool.run(
        CreateDocumentTool.Args(
            filename="x.pdf", content="# T\n\nhi 🙂 there"
        )
    )
    assert result.ok
    assert "1 character(s)" in result.content
    assert "OMITTED" in result.content


def test_registry_registers_create_document(tmp_path):
    from woyo.tools.builtin import build_default_registry

    settings = make_settings(workspace_dir=str(tmp_path / "ws"))
    registry = build_default_registry(settings)
    assert "create_document" in registry.names()
