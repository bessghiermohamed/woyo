"""v0.6 file transfer: ingestion, send_file, vision routing, durable sync."""

from __future__ import annotations

import asyncio
import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import httpx
import pytest

from tests.conftest import PLAN_JSON, make_settings
from woyo.chat.files import (
    FileTooBig,
    sanitize_filename,
    unique_path,
)
from woyo.chat.session import ChatReply, ChatSession
from woyo.chat.telegram import TelegramBot
from woyo.errors import ErrorKind
from woyo.models.base import Message, ModelResponse, Usage
from woyo.models.openai_compat import _to_openai_message
from woyo.models.router import ModelRouter
from woyo.tools.builtin import build_default_registry
from woyo.tools.builtin.file_transfer import SendFileTool

# --- naming -------------------------------------------------------------------


def test_sanitize_filename_neutralizes_traversal():
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\win.ini") == "win.ini"
    assert sanitize_filename("report\x00.pdf") == "report.pdf"
    assert sanitize_filename(".hidden") != ".hidden"  # never dot-leading
    # unicode survives (Arabic filenames are normal filenames)
    assert sanitize_filename("تقرير.txt") == "تقرير.txt"
    # empty / control-only falls back to a stamped name
    assert sanitize_filename("") .startswith("file_")
    assert sanitize_filename("///").startswith("file_")
    assert len(sanitize_filename("a" * 300 + ".txt")) <= 120


def test_unique_path_suffixes_collisions(tmp_path):
    first = unique_path(tmp_path, "notes.txt")
    first.write_text("one")
    second = unique_path(tmp_path, "notes.txt")
    assert second.name == "notes_2.txt"
    second.write_text("two")
    assert unique_path(tmp_path, "notes.txt").name == "notes_3.txt"


# --- send_file tool -----------------------------------------------------------------


class RecordingSender:
    def __init__(self):
        self.sent: list[tuple[Path, str]] = []

    async def __call__(self, path: Path, caption: str) -> None:
        self.sent.append((path, caption))


async def test_send_file_happy_path(tmp_path):
    settings = make_settings(workspace_dir=str(tmp_path / "ws"))
    sender = RecordingSender()
    tool = SendFileTool(settings, sender)
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "out.csv").write_text("a,b\n1,2\n")

    result = await tool.run(
        SendFileTool.Args(path="out.csv", caption="your export")
    )
    assert result.ok
    assert sender.sent == [(tmp_path / "ws" / "out.csv", "your export")]
    assert "out.csv" in result.content


async def test_send_file_rejects_traversal_and_missing(tmp_path):
    settings = make_settings(workspace_dir=str(tmp_path / "ws"))
    tool = SendFileTool(settings, RecordingSender())
    (tmp_path / "ws").mkdir()

    escape = await tool.run(SendFileTool.Args(path="../secrets.env"))
    assert not escape.ok and escape.error_kind == ErrorKind.INVALID_INPUT.value

    missing = await tool.run(SendFileTool.Args(path="nope.txt"))
    assert not missing.ok and "No such file" in missing.content


async def test_send_file_enforces_size_caps(tmp_path):
    settings = make_settings(
        workspace_dir=str(tmp_path / "ws"), chat_max_send_file_mb=1
    )
    tool = SendFileTool(settings, RecordingSender())
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "big.bin").write_bytes(b"x" * (2 * 1024 * 1024))
    (tmp_path / "ws" / "empty.txt").write_text("")

    big = await tool.run(SendFileTool.Args(path="big.bin"))
    assert not big.ok and "over the 1 MB sending cap" in big.content
    empty = await tool.run(SendFileTool.Args(path="empty.txt"))
    assert not empty.ok and "empty" in empty.content


async def test_send_file_surfaces_delivery_failures(tmp_path):
    settings = make_settings(workspace_dir=str(tmp_path / "ws"))

    async def boom(path: Path, caption: str) -> None:
        raise RuntimeError("telegram down")

    tool = SendFileTool(settings, boom)
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "f.txt").write_text("hi")
    result = await tool.run(SendFileTool.Args(path="f.txt"))
    assert not result.ok and "telegram down" in result.content


def test_registry_gates_send_file_on_sender():
    with_sender = build_default_registry(
        make_settings(), file_sender=RecordingSender()
    )
    without = build_default_registry(make_settings())
    assert with_sender.get("send_file") is not None
    assert without.get("send_file") is None


# --- vision plumbing -----------------------------------------------------------------


def test_openai_message_with_images_becomes_parts():
    m = Message(role="user", content="look", images=["data:image/png;base64,AB"])
    payload = _to_openai_message(m)
    assert isinstance(payload["content"], list)
    assert payload["content"][0] == {"type": "text", "text": "look"}
    assert payload["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AB"},
    }
    # plain messages stay plain strings
    plain = _to_openai_message(Message(role="user", content="hi"))
    assert plain["content"] == "hi"


def test_anthropic_image_blocks():
    pytest.importorskip("anthropic")
    from woyo.models.anthropic_provider import _convert_messages, _parse_data_uri

    assert _parse_data_uri("data:image/jpeg;base64,QUJD") == ("image/jpeg", "QUJD")
    system, converted = _convert_messages(
        [Message(role="user", content="hi", images=["data:image/png;base64,AA"])]
    )
    blocks = converted[0]["content"]
    assert blocks[0] == {"type": "text", "text": "hi"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AA"},
    }


class CapturingProvider:
    name = "capture"

    def __init__(self):
        self.calls: list[dict] = []

    async def complete(self, *, messages, tools, model, **kw):
        self.calls.append({"model": model, "messages": messages})
        return ModelResponse(content="ok", usage=Usage(input_tokens=1, output_tokens=1))


async def test_router_sends_image_calls_to_vision_model():
    vision = CapturingProvider()
    default = CapturingProvider()
    router = ModelRouter(
        make_settings(vision_model="openrouter:vl-1"),
        default_provider=default,
        extra_providers={"openrouter": vision},
    )
    msgs = [Message(role="user", content="hi", images=["data:image/jpeg;base64,AA"])]
    await router.complete("executor", messages=msgs)
    assert vision.calls and vision.calls[0]["model"] == "vl-1"
    assert not default.calls


async def test_router_strips_images_without_vision_model():
    default = CapturingProvider()
    router = ModelRouter(make_settings(), default_provider=default)
    msgs = [Message(role="user", content="hi", images=["data:image/jpeg;base64,AA"])]
    await router.complete("executor", messages=msgs)
    sent = default.calls[0]["messages"]
    assert sent[0].images is None and sent[0].content == "hi"
    # the caller's list is untouched (no mutation)
    assert msgs[0].images == ["data:image/jpeg;base64,AA"]


async def test_router_vision_failure_falls_back_honestly():
    from woyo.errors import AgentError

    class FlakyVision:
        """Fails the first call (rate limit), serves the second."""

        name = "flaky-vision"

        def __init__(self):
            self.calls = 0

        async def complete(self, **kw):
            self.calls += 1
            if self.calls == 1:
                raise AgentError("Provider failed after retries: 429")
            return ModelResponse(
                content="seen", usage=Usage(input_tokens=1, output_tokens=1)
            )

    default = CapturingProvider()
    flaky = FlakyVision()
    router = ModelRouter(
        # multiple vision refs: first fails -> second serves
        make_settings(vision_model="or:vl-1,or:vl-2"),
        default_provider=default,
        extra_providers={"or": flaky},
    )
    msgs = [Message(role="user", content="hi", images=["data:image/jpeg;base64,AA"])]
    resp = await router.complete("executor", messages=msgs)
    assert resp.content == "seen"
    assert flaky.calls == 2  # tried twice across the two refs

    # ALL vision refs down -> honest text fallback on the default model
    class FailingVision:
        name = "failing-vision"

        async def complete(self, **kw):
            raise AgentError("Provider failed after retries: 429")

    router2 = ModelRouter(
        make_settings(vision_model="or:vl-1"),
        default_provider=default,
        extra_providers={"or": FailingVision()},
    )
    resp2 = await router2.complete("executor", messages=msgs)
    assert resp2.content == "ok"  # served by the default model
    sent = default.calls[0]["messages"]
    assert sent[0].images is None
    # the honesty note rides along so the model cannot pretend it saw the image
    assert any(
        "did NOT see the image" in (m.content or "") for m in sent
    )


async def test_agent_run_carries_images_on_user_turn():
    from woyo.agent import Agent
    from woyo.events import EventBus
    from woyo.models.mock import MockProvider, text_response

    # vision_model routes image-bearing calls; point it at the same mock so
    # the scripted responses keep working and we can see what was sent
    settings = make_settings(
        direct_text_replies=True, vision_model="mock:mock/test"
    )
    mock = MockProvider(
        [text_response(PLAN_JSON), text_response("I see a red square.")]
    )
    router = ModelRouter(
        settings, default_provider=mock, extra_providers={"mock": mock}
    )
    from tests.conftest import default_test_tools, make_registry

    agent = Agent(settings, router, make_registry(default_test_tools()), bus=EventBus())
    uri = "data:image/jpeg;base64,AA"
    result = await agent.run("describe the image", images=[uri])
    assert result.outcome == "completed"
    assert any(
        m.images == [uri] for call in mock.calls for m in call["messages"]
    )


# --- telegram inbound ------------------------------------------------------------


def _attach_transport(
    *,
    file_bytes: bytes,
    file_size: int | None = None,
    get_file_ok: bool = True,
    download_status: int = 200,
):
    """MockTransport answering getFile + the file download endpoint."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/getFile"):
            if not get_file_ok:
                return httpx.Response(200, json={"ok": False, "description": "nope"})
            return httpx.Response(200, json={"ok": True, "result": {
                "file_id": "F1",
                "file_path": "documents/xyz",
                "file_size": file_size if file_size is not None else len(file_bytes),
            }})
        if "/file/bot" in path:
            if download_status != 200:
                return httpx.Response(download_status, json={"ok": False})
            return httpx.Response(200, content=file_bytes)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    return httpx.MockTransport(handler)


class FakeTGSession:
    def __init__(self):
        self.received: list[tuple[str, list[str] | None]] = []

    async def send(self, message, *, approval_cb=None, images=None) -> ChatReply:
        self.received.append((message, images))
        return ChatReply(answer="got it", verified=True)


async def _drain():
    for task in [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]:
        await task


def _doc_update(chat_id: int, *, name="notes.txt", mime="text/plain", caption=""):
    msg: dict = {
        "chat": {"id": chat_id},
        "document": {"file_id": "F1", "file_name": name, "mime_type": mime},
    }
    if caption:
        msg["caption"] = caption
    return {"update_id": 1, "message": msg}


def _make_bot(tmp_path, transport, **settings_kw) -> TelegramBot:
    client = httpx.AsyncClient(transport=transport)
    return TelegramBot(
        make_settings(
            sessions_dir=str(tmp_path / "sessions"),
            workspace_dir=str(tmp_path / "ws"),
            files_dir=str(tmp_path / "files"),
            **settings_kw,
        ),
        "T:x",
        set(),
        client=client,
        state_path=tmp_path / "state.json",
    )


async def test_telegram_document_is_read_into_the_prompt(tmp_path):
    bot = _make_bot(tmp_path, _attach_transport(file_bytes=b"hello file content"))
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    await bot._dispatch(
        _doc_update(111, caption="summarize this", name="notes.txt")
    )
    await _drain()

    assert len(fake.received) == 1
    text, images = fake.received[0]
    assert "summarize this" in text
    assert "[ATTACHMENT · document · notes.txt" in text
    assert "Saved to workspace: inbox/notes.txt" in text
    # the file content is inlined as untrusted data
    assert "<untrusted" in text and "hello file content" in text
    assert images is None
    # the file itself landed in the workspace + durable inbox
    assert (tmp_path / "ws" / "inbox" / "notes.txt").read_bytes() == b"hello file content"
    assert (tmp_path / "files" / "111" / "inbox" / "notes.txt").exists()
    await bot._client.aclose()


async def test_telegram_document_without_caption_gets_default_instruction(tmp_path):
    bot = _make_bot(tmp_path, _attach_transport(file_bytes=b"data"))
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    await bot._dispatch(_doc_update(111))
    await _drain()
    text, _ = fake.received[0]
    assert "sent this file with no message" in text


async def test_telegram_zip_attachment_is_listed_not_extracted(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner.txt", "x" * 10)
        zf.writestr("deep/nested.json", "{}")
    bot = _make_bot(tmp_path, _attach_transport(file_bytes=buf.getvalue()))
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    await bot._dispatch(
        _doc_update(111, name="bundle.zip", mime="application/zip")
    )
    await _drain()
    text, _ = fake.received[0]
    assert "Archive contents" in text
    assert "inner.txt" in text and "deep/nested.json" in text
    # listing only — the archive itself is never unpacked
    assert "NOT extracted" in text
    await bot._client.aclose()


async def test_telegram_binary_attachment_note(tmp_path):
    bot = _make_bot(tmp_path, _attach_transport(file_bytes=bytes(range(256)) * 40))
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    await bot._dispatch(_doc_update(111, name="blob.bin", mime="application/octet-stream"))
    await _drain()
    text, _ = fake.received[0]
    assert "Binary file saved" in text
    await bot._client.aclose()


async def test_telegram_photo_with_vision_becomes_image_input(tmp_path):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buf, format="JPEG")
    bot = _make_bot(
        tmp_path,
        _attach_transport(file_bytes=buf.getvalue()),
        vision_model="openrouter:vl-1",
    )
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    update = {
        "update_id": 2,
        "message": {
            "chat": {"id": 111},
            "photo": [{"file_id": "P1", "file_size": 10}, {"file_id": "P1", "file_size": 20}],
            "caption": "what is this?",
        },
    }
    await bot._dispatch(update)
    await _drain()

    text, images = fake.received[0]
    assert "what is this?" in text
    assert "image input" in text
    assert images and images[0].startswith("data:image/jpeg;base64,")
    # saved under a generated photo name
    assert any(
        p.suffix == ".jpg" for p in (tmp_path / "ws" / "inbox").iterdir()
    )
    await bot._client.aclose()


async def test_telegram_photo_without_vision_is_honest(tmp_path):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(buf, format="JPEG")
    bot = _make_bot(tmp_path, _attach_transport(file_bytes=buf.getvalue()))
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    update = {
        "update_id": 3,
        "message": {"chat": {"id": 111}, "photo": [{"file_id": "P1"}]},
    }
    await bot._dispatch(update)
    await _drain()
    text, images = fake.received[0]
    assert "Image viewing is NOT available" in text
    assert images is None
    await bot._client.aclose()


async def test_telegram_voice_transcript_inlined(tmp_path, monkeypatch):
    from woyo.chat import files as files_mod

    async def fake_transcribe(path, settings):
        return "hey check this number for me"

    monkeypatch.setattr(files_mod, "transcribe_audio", fake_transcribe)
    bot = _make_bot(
        tmp_path,
        _attach_transport(file_bytes=b"OGGDATA"),
        chat_asr_provider="groq",
    )
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    update = {
        "update_id": 4,
        "message": {
            "chat": {"id": 111},
            "voice": {"file_id": "V1", "duration": 7, "mime_type": "audio/ogg"},
        },
    }
    await bot._dispatch(update)
    await _drain()
    text, _ = fake.received[0]
    assert "Voice note · 7s" in text
    assert "hey check this number for me" in text
    assert "<untrusted" in text  # transcripts are external data too
    await bot._client.aclose()


async def test_telegram_voice_without_asr_is_honest(tmp_path):
    bot = _make_bot(tmp_path, _attach_transport(file_bytes=b"OGGDATA"))
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    update = {
        "update_id": 5,
        "message": {"chat": {"id": 111}, "voice": {"file_id": "V1"}},
    }
    await bot._dispatch(update)
    await _drain()
    text, _ = fake.received[0]
    assert "speech-to-text is not configured" in text
    await bot._client.aclose()


async def test_telegram_oversized_file_gets_honest_reply(tmp_path):
    sent: list[tuple[int, str]] = []

    async def recorder(chat_id, text, *, markdown=False):
        sent.append((chat_id, text))

    bot = _make_bot(
        tmp_path, _attach_transport(file_bytes=b"x", file_size=99 * 1024 * 1024)
    )
    bot._owner = 111
    bot._send = recorder  # type: ignore[method-assign]
    bot._sessions[111] = FakeTGSession()  # type: ignore[assignment]

    await bot._dispatch(_doc_update(111))
    await _drain()
    assert any("too big" in t for _, t in sent)
    # and the session was never invoked
    assert bot._sessions[111].received == []  # type: ignore[attr-defined]
    await bot._client.aclose()


async def test_telegram_attachment_from_stranger_is_refused(tmp_path):
    """Unauthorized chats must not trigger downloads (or answers)."""
    bot = _make_bot(tmp_path, _attach_transport(file_bytes=b"secret"))
    bot._owner = 111
    sent: list[tuple[int, str]] = []

    async def recorder(chat_id, text, *, markdown=False):
        sent.append((chat_id, text))

    bot._send = recorder  # type: ignore[method-assign]

    await bot._dispatch(_doc_update(222))
    await _drain()
    assert sent == [(222, "🔒 This bot is private.")]
    assert 222 not in bot._sessions
    await bot._client.aclose()


async def test_telegram_send_document_uploads_multipart(tmp_path):
    uploads: list[tuple[str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("sendDocument"):
            uploads.append((request.headers.get("content-type", ""), request.read()))
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 9}})
        return httpx.Response(200, json={"ok": True, "result": {}})

    bot = _make_bot(tmp_path, httpx.MockTransport(handler))
    doc = tmp_path / "export.csv"
    doc.write_text("a,b\n1,2\n")

    await bot._send_document(111, doc, "your export")
    assert uploads, "no upload happened"
    ctype, body = uploads[0]
    assert ctype.startswith("multipart/form-data")
    assert b'name="document"' in body and b"export.csv" in body
    assert b"your export" in body
    await bot._client.aclose()


# --- pdf -------------------------------------------------------------------------


def _mini_pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    ).encode()
    return out


async def test_telegram_pdf_text_is_extracted(tmp_path):
    pytest.importorskip("pypdf")
    pdf_bytes = _mini_pdf("Quarterly revenue is 42 units")
    bot = _make_bot(tmp_path, _attach_transport(file_bytes=pdf_bytes))
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    await bot._dispatch(
        _doc_update(111, name="report.pdf", mime="application/pdf", caption="read this")
    )
    await _drain()
    text, _ = fake.received[0]
    assert "report.pdf" in text
    assert "Quarterly revenue is 42 units" in text
    assert "<untrusted" in text
    await bot._client.aclose()


async def test_telegram_pdf_without_pypdf_is_honest(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)  # simulate a missing extra
    pdf_bytes = _mini_pdf("hidden text")
    bot = _make_bot(tmp_path, _attach_transport(file_bytes=pdf_bytes))
    bot._owner = 111
    fake = FakeTGSession()
    bot._sessions[111] = fake  # type: ignore[assignment]

    await bot._dispatch(_doc_update(111, name="report.pdf", mime="application/pdf"))
    await _drain()
    text, _ = fake.received[0]
    assert "No text could be extracted" in text
    await bot._client.aclose()


# --- session wiring ------------------------------------------------------------------


def test_chat_session_default_factory_registers_send_file(tmp_path):
    settings = make_settings(
        provider="mock",
        model="mock/test",
        sessions_dir=str(tmp_path / "sessions"),
        workspace_dir=str(tmp_path / "ws"),
    )
    session = ChatSession(
        "telegram:111", settings,
        chats_dir=tmp_path / "chats", file_sender=RecordingSender(),
    )
    agent = session._agent_factory(session.settings, __import__("woyo.events", fromlist=["EventBus"]).EventBus())
    assert agent.registry.get("send_file") is not None
    # the framing tells the model it can deliver files
    assert "send_file" in session._frame_task("make me a file")


def test_chat_session_without_sender_keeps_default_factory(tmp_path):
    settings = make_settings(
        provider="mock", model="mock/test", sessions_dir=str(tmp_path / "sessions")
    )
    session = ChatSession("telegram:111", settings, chats_dir=tmp_path / "chats")
    agent = session._agent_factory(session.settings, __import__("woyo.events", fromlist=["EventBus"]).EventBus())
    assert agent.registry.get("send_file") is None
    assert "send_file" not in session._frame_task("make me a file")


# --- durable sync (deploy/gha-telegram/run_bot.py) ------------------------------------


def _load_run_bot():
    path = Path(__file__).parent.parent / "deploy" / "gha-telegram" / "run_bot.py"
    spec = importlib.util.spec_from_file_location("woyo_run_bot_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mirror_with_caps_copies_and_skips_oversized(tmp_path):
    run_bot = _load_run_bot()
    src = tmp_path / "files" / "111" / "inbox"
    src.mkdir(parents=True)
    (src / "a.txt").write_text("hello")
    (src / "big.bin").write_bytes(b"x" * (9 * 1024 * 1024))  # over the 8 MB cap

    # sync operates on the tree ROOT (like _push_tree(FILES_DIR, "files"))
    root = tmp_path / "files"
    dst = tmp_path / "mirror"
    kept = run_bot.mirror_with_caps(
        root, dst, max_file_mb=8, total_mb=32
    )
    assert [p.name for p in kept] == ["a.txt"]
    assert (dst / "111" / "inbox" / "a.txt").read_text() == "hello"
    assert not (dst / "111" / "inbox" / "big.bin").exists()
    # the oversized original stays on disk (only the sync skips it)
    assert (src / "big.bin").exists()


def test_mirror_with_caps_prunes_oldest_over_budget(tmp_path):
    import os
    import time

    run_bot = _load_run_bot()
    src = tmp_path / "files"
    src.mkdir()
    now = time.time()
    # three 1 MB files: f0 oldest, f2 newest -> budget 2 MB must drop f0
    for i, age_s in enumerate((5000, 3000, 1000)):
        p = src / f"f{i}.bin"
        p.write_bytes(b"x" * (1024 * 1024))
        os.utime(p, (now - age_s, now - age_s))

    kept = run_bot.mirror_with_caps(
        src, tmp_path / "mirror", max_file_mb=8, total_mb=2
    )
    assert {p.name for p in kept} == {"f1.bin", "f2.bin"}
    assert not (src / "f0.bin").exists()  # oldest pruned from the source too
    assert (tmp_path / "mirror" / "f2.bin").exists()


def test_pull_state_restores_file_trees(tmp_path, monkeypatch):
    run_bot = _load_run_bot()

    def fake_git(*args: str, cwd=None) -> None:
        if args[0] == "clone":
            dest = Path(args[-1])
            (dest / "files" / "111" / "inbox").mkdir(parents=True)
            (dest / "files" / "111" / "inbox" / "a.txt").write_text("A")
            (dest / "files" / "111" / "outbox").mkdir(parents=True)
            (dest / "files" / "111" / "outbox" / "b.csv").write_text("B")
            (dest / "workspace-inbox").mkdir()
            (dest / "workspace-inbox" / "a.txt").write_text("A")
            (dest / "telegram_state.json").write_text('{"offset": 5, "owner": null}')

    monkeypatch.setattr(run_bot, "_git", fake_git)
    monkeypatch.setattr(run_bot, "STATE_REPO", "o/r")
    monkeypatch.setattr(run_bot, "STATE_TOKEN", "t")
    monkeypatch.setattr(run_bot, "STATE_FILE", tmp_path / "telegram_state.json")
    monkeypatch.setattr(run_bot, "FILES_DIR", tmp_path / "files")
    monkeypatch.setattr(run_bot, "WS_INBOX", tmp_path / "ws" / "inbox")
    monkeypatch.setattr(run_bot, "CHATS_DIR", tmp_path / "chats")
    monkeypatch.setattr(run_bot, "DB_FILE", tmp_path / "woyo.sqlite3")

    run_bot.pull_state()
    assert (tmp_path / "files" / "111" / "inbox" / "a.txt").read_text() == "A"
    assert (tmp_path / "files" / "111" / "outbox" / "b.csv").read_text() == "B"
    assert (tmp_path / "ws" / "inbox" / "a.txt").read_text() == "A"
    # a json file inside files/ must NOT be mistaken for a chat transcript
    assert not (tmp_path / "chats").exists() or not any(
        (tmp_path / "chats").iterdir()
    )


async def test_file_too_big_raised_from_download(tmp_path):
    # direct download cap check (transport-independent)
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/getFile"):
            return httpx.Response(200, json={"ok": True, "result": {
                "file_id": "F1", "file_path": "doc", "file_size": 50 * 1024 * 1024,
            }})
        return httpx.Response(200, content=b"x")

    from woyo.chat.files import download_telegram_file

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(FileTooBig):
            await download_telegram_file(client, "T:x", "F1", 15 * 1024 * 1024)
