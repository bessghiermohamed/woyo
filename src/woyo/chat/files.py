"""Inbound attachment ingestion for chat frontends (v0.6, ADR-13).

When a user sends a document / photo / voice note / video, the frontend:

1. downloads it through the Bot API (``getFile`` + the file endpoint),
2. saves it TWICE — into the durable ``~/.woyo/files/<chat>/inbox`` store
   (synced across ephemeral hosts) and into ``<workspace>/inbox`` so the
   agent's own tools (read_file, python_exec, shell_exec) can reach it,
3. extracts what it safely can (text, PDF text, zip listings, image data
   URIs for vision models, voice transcripts via optional ASR),
4. returns a framed note for the agent prompt.

Everything extracted from a user file is EXTERNAL DATA: it is wrapped with
``wrap_untrusted`` before it enters the prompt (prompt-injection defense,
same layer as web pages). Size and runtime caps everywhere — a hostile
PDF or zip must not hang or bloat the run:

- downloads capped at ``chat_max_file_mb`` (Bot API ceiling is 20 MB),
- extracted text capped at ``chat_extract_chars``,
- PDF parsing runs in a worker thread under a hard timeout, page-capped,
- zip handling LISTS entries only (never decompresses — zip bombs stay
  inert), entry count capped,
- images are downscaled + EXIF-stripped before they become model inputs.
"""

from __future__ import annotations

import asyncio
import base64
import io
import mimetypes
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

from woyo.config import Settings
from woyo.tools.base import wrap_untrusted

_API = "https://api.telegram.org"

#: Extensions treated as inline-able text (code, configs, data, markup).
_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log",
    ".py", ".pyw", ".js", ".mjs", ".ts", ".tsx", ".jsx",
    ".json", ".jsonl", ".csv", ".tsv", ".html", ".htm", ".xml",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".bat",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".hh", ".java", ".rs", ".go",
    ".rb", ".php", ".pl", ".lua", ".r", ".jl", ".swift", ".kt", ".kts",
    ".scala", ".css", ".scss", ".sass", ".vue", ".svelte", ".dart",
    ".cs", ".fs", ".vb", ".ex", ".exs", ".erl", ".hs", ".ml",
    ".env", ".gitignore", ".dockerignore", ".patch", ".diff",
}
_TEXT_MIMES = {
    "application/json", "application/xml", "application/javascript",
    "application/x-yaml", "application/yaml", "application/toml",
    "application/sql", "application/x-sh",
}
#: Zip containers get a listing (never extracted automatically).
_ZIPISH_SUFFIXES = {
    ".zip", ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp",
    ".jar", ".apk", ".epub", ".whl",
}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_AUDIO_SUFFIXES = {".ogg", ".oga", ".opus", ".mp3", ".m4a", ".wav", ".webm"}

_MAX_PDF_PAGES = 60
_MAX_ZIP_ENTRIES = 200
_PDF_TIMEOUT_S = 15.0
#: replacement-char ratio above which a "text" read is really binary
_BINARY_RATIO = 0.02


class FileTooBig(Exception):
    """The attachment exceeds the configured download cap."""


class DownloadError(Exception):
    """Telegram refused the download or transport failed."""


@dataclass(slots=True)
class Ingested:
    """What the frontend got out of one attachment."""

    note: str = ""  # framed block appended to the agent's task text
    images: list[str] = field(default_factory=list)  # data URIs for vision models
    saved: list[str] = field(default_factory=list)  # workspace-relative paths


@dataclass(slots=True)
class AttachmentRef:
    """The file pointer Telegram puts on a message (before download)."""

    kind: str  # document | photo | voice | audio | video | video_note | animation
    file_id: str
    file_name: str | None = None
    mime: str | None = None
    size: int | None = None
    duration_s: int | None = None


def attachment_ref(msg: dict) -> AttachmentRef | None:
    """Pull the first file-bearing field off a Telegram message."""
    doc = msg.get("document")
    if isinstance(doc, dict) and doc.get("file_id"):
        return AttachmentRef(
            "document", doc["file_id"], doc.get("file_name"),
            doc.get("mime_type"), doc.get("file_size"),
        )
    photos = msg.get("photo")
    if isinstance(photos, list) and photos:
        largest = photos[-1]  # Telegram sorts smallest -> largest
        return AttachmentRef(
            "photo", largest.get("file_id", ""), None, "image/jpeg",
            largest.get("file_size"),
        )
    for key, kind, default_mime in (
        ("voice", "voice", "audio/ogg"),
        ("audio", "audio", "audio/mpeg"),
        ("video", "video", "video/mp4"),
        ("video_note", "video_note", "video/mp4"),
        ("animation", "animation", "video/mp4"),
    ):
        item = msg.get(key)
        if isinstance(item, dict) and item.get("file_id"):
            return AttachmentRef(
                kind, item["file_id"], item.get("file_name"),
                item.get("mime_type") or default_mime, item.get("file_size"),
                item.get("duration"),
            )
    return None


# --- naming -------------------------------------------------------------------


def sanitize_filename(name: str | None, *, kind: str = "file") -> str:
    """Turn an attacker-controlled filename into something safe to store.

    Basename on both separators, strip control chars, keep Unicode letters
    (Arabic included), cap the length, never empty, never dot-leading.
    """
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)
    name = name.strip(" .")
    name = re.sub(r"[^\w.\- ()\[\]@+,;=&']", "_", name)
    name = re.sub(r"_{2,}", "_", name).strip("_")
    if len(name) > 120:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 16:
            name = stem[: 116 - len(ext)].rstrip("_") + "." + ext
        else:
            name = name[:120].rstrip("_")
    if not name or name.startswith("."):
        name = f"{kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
    return name


def _default_name(ref: AttachmentRef) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = {
        "photo": ".jpg", "voice": ".ogg", "audio": ".mp3",
        "video": ".mp4", "video_note": ".mp4", "animation": ".mp4",
    }.get(ref.kind, ".bin")
    return f"{ref.kind}_{stamp}{ext}"


def unique_path(directory: Path, name: str) -> Path:
    """`dir/name`, suffixed -2, -3… when it already exists."""
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    i = 2
    while True:
        candidate = directory / (f"{stem}_{i}" + (f".{ext}" if ext else ""))
        if not candidate.exists():
            return candidate
        i += 1


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n} B"


# --- download -------------------------------------------------------------------


async def download_telegram_file(
    client: httpx.AsyncClient, token: str, file_id: str, max_bytes: int
) -> bytes:
    """Fetch one file through the Bot API, enforcing the size cap."""
    resp = await client.post(f"{_API}/bot{token}/getFile", json={"file_id": file_id})
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("ok"):
        raise DownloadError(f"getFile failed: {payload}")
    info = payload.get("result") or {}
    file_path = info.get("file_path")
    if not file_path:
        raise DownloadError("Telegram returned no file_path (file expired?)")
    if (info.get("file_size") or 0) > max_bytes:
        raise FileTooBig(f"{_human_size(int(info['file_size']))} exceeds the cap")
    dresp = await client.get(f"{_API}/file/bot{token}/{file_path}")
    dresp.raise_for_status()
    if len(dresp.content) > max_bytes:
        raise FileTooBig(f"{_human_size(len(dresp.content))} exceeds the cap")
    return dresp.content


# --- extraction -------------------------------------------------------------------


def _decode_textish(data: bytes) -> str | None:
    """UTF-8 (tolerantly) decoded text, or None when it looks binary."""
    sample = data[:65_536]
    replaced = sample.decode("utf-8", errors="replace")
    if replaced.count("\ufffd") / max(len(replaced), 1) > _BINARY_RATIO:
        return None
    return data.decode("utf-8", errors="replace")


def _pdf_text(data: bytes, cap: int) -> str | None:
    """pypdf-based extraction (optional dep); returns None when unavailable."""
    try:
        import pypdf  # type: ignore[import-not-found]
    except ImportError:
        return None
    reader = pypdf.PdfReader(io.BytesIO(data))
    parts: list[str] = []
    taken = 0
    for i, page in enumerate(reader.pages):
        if i >= _MAX_PDF_PAGES or taken >= cap:
            break
        try:
            chunk = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — one bad page must not kill the file
            continue
        parts.append(chunk)
        taken += len(chunk)
    text = "\n".join(parts).strip()
    return text[:cap] if text else None


def _zip_listing(path: Path) -> str:
    """List zip entries WITHOUT decompressing (zip-bomb inert)."""
    lines: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()[:_MAX_ZIP_ENTRIES]
            total = len(zf.infolist())
            for info in infos:
                lines.append(f"- {info.filename} ({_human_size(info.file_size)})")
            if total > _MAX_ZIP_ENTRIES:
                lines.append(f"- … and {total - _MAX_ZIP_ENTRIES} more entries")
    except (zipfile.BadZipFile, OSError) as exc:
        return f"(could not read the archive: {exc})"
    return "\n".join(lines) or "(empty archive)"


def image_data_uri(
    path: Path, *, max_edge: int = 1568, max_b64: int = 6_000_000
) -> str | None:
    """A model-ready JPEG data URI, downscaled and EXIF-stripped.

    Returns None when the image cannot be prepared (too large, corrupt).
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        # No Pillow: fall back to the original bytes when small enough.
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if len(data) > 4_000_000:
            return None
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}" if len(b64) <= max_b64 else None
    try:
        with Image.open(path) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im.thumbnail((max_edge, max_edge))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
    except Exception:  # noqa: BLE001 — corrupt images are not fatal
        return None
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}" if len(b64) <= max_b64 else None


async def transcribe_audio(path: Path, settings: Settings) -> str | None:
    """Voice -> text via any OpenAI-compat /audio/transcriptions endpoint.

    Configured with WOYO_ASR_PROVIDER (e.g. "groq") + that provider's key.
    Any failure returns None — the caller degrades to an honest note.
    """
    provider = settings.chat_asr_provider
    if not provider:
        return None
    from woyo.config import resolve_api_key, resolve_base_url

    key = resolve_api_key(provider, settings)
    base = resolve_base_url(provider, settings)
    if not key or not base:
        return None
    mime = mimetypes.guess_type(path.name)[0] or "audio/ogg"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as ac:
            with path.open("rb") as fh:
                resp = await ac.post(
                    f"{base.rstrip('/')}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (path.name, fh, mime)},
                    data={"model": settings.chat_asr_model},
                )
        if resp.status_code != 200:
            return None
        text = (resp.json().get("text") or "").strip()
        return text or None
    except Exception:  # noqa: BLE001 — transcription is best-effort by design
        return None


# --- ingestion -------------------------------------------------------------------


async def ingest_attachment(
    client: httpx.AsyncClient,
    token: str,
    chat_id: int,
    msg: dict,
    settings: Settings,
) -> Ingested:
    """Download one attachment and build the agent-facing note for it."""
    from woyo.tools.builtin.sandbox import workspace_root

    ref = attachment_ref(msg)
    if ref is None:
        raise DownloadError("message carries no downloadable file")
    max_bytes = settings.chat_max_file_mb * 1024 * 1024
    data = await download_telegram_file(client, token, ref.file_id, max_bytes)

    name = sanitize_filename(ref.file_name or _default_name(ref), kind=ref.kind)
    ws_root = workspace_root(settings)
    ws_inbox = ws_root / "inbox"
    ws_inbox.mkdir(parents=True, exist_ok=True)
    ws_path = unique_path(ws_inbox, name)
    ws_path.write_bytes(data)
    rel = ws_path.relative_to(ws_root).as_posix()

    # durable copy (survives ephemeral hosts via the state-repo sync)
    try:
        durable = Path(settings.files_dir).expanduser() / str(chat_id) / "inbox"
        durable.mkdir(parents=True, exist_ok=True)
        unique_path(durable, name).write_bytes(data)
    except OSError:
        pass  # durability is best-effort; the workspace copy always exists

    result = Ingested(saved=[rel])
    suffix = ws_path.suffix.lower()
    size_line = _human_size(len(data))
    header = (
        f"[ATTACHMENT · {ref.kind} · {ws_path.name} · "
        f"{ref.mime or 'unknown type'} · {size_line}]\n"
        f"Saved to workspace: {rel} — your read_file / python_exec / "
        f"shell_exec tools can access it there."
    )

    async def body() -> str:
        # images: attach for vision when configured, regardless of kind
        if suffix in _IMAGE_SUFFIXES:
            if settings.vision_model:
                uri = await asyncio.to_thread(image_data_uri, ws_path)
                if uri:
                    result.images.append(uri)
                    return (
                        "The image is attached to this message as an image "
                        "input — actually look at it before answering."
                    )
                return (
                    "The image is saved but too large to pass to the vision "
                    "model. Tell the user honestly and offer python_exec-based "
                    "analysis instead."
                )
            return (
                "Image viewing is NOT available in this setup — you cannot see "
                "this photo's content. Say so honestly; offer what you CAN do "
                "(metadata, python-based analysis via python_exec)."
            )

        # voice / audio: optional transcription
        if ref.kind in ("voice", "audio") or suffix in _AUDIO_SUFFIXES:
            transcript = await transcribe_audio(ws_path, settings)
            if transcript:
                wrapped, _flagged = wrap_untrusted(
                    transcript, f"voice note transcript: {ws_path.name}"
                )
                dur = (
                    f" · {ref.duration_s}s" if ref.duration_s else ""
                )
                return (
                    f"Voice note{dur} — automatic transcript:\n{wrapped}"
                )
            if ref.kind == "voice":
                return (
                    "This is a voice note and speech-to-text is not "
                    "configured, so its content is unavailable. Tell the user "
                    "honestly (they can type or send a document instead)."
                )
            return "Audio file saved; audio content cannot be transcribed in this setup."

        # zip containers: list, never extract
        if suffix in _ZIPISH_SUFFIXES:
            listing = await asyncio.to_thread(_zip_listing, ws_path)
            wrapped, _flagged = wrap_untrusted(listing, f"archive listing: {ws_path.name}")
            hint = ""
            if suffix in (".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"):
                hint = (
                    "\nThis is a zip-based document format — you can extract "
                    "its text yourself with python_exec (zipfile + xml)."
                )
            return f"Archive contents (NOT extracted — listing only):\n{wrapped}{hint}"

        # PDF: pypdf extraction in a bounded worker thread
        if suffix == ".pdf":
            text = await asyncio.wait_for(
                asyncio.to_thread(_pdf_text, data, settings.chat_extract_chars),
                timeout=_PDF_TIMEOUT_S,
            )
            if text:
                wrapped, _flagged = wrap_untrusted(text, f"pdf text: {ws_path.name}")
                return f"Extracted text:\n{wrapped}"
            return (
                "No text could be extracted (pypdf missing or a scanned/image "
                "PDF). The file is saved in the workspace — python_exec or an "
                "approved shell_exec can work on it directly."
            )

        # plain text / code / data files
        text = await asyncio.to_thread(_decode_textish, data)
        if text is not None:
            if len(text) > settings.chat_extract_chars:
                text = text[: settings.chat_extract_chars] + "\n[… truncated …]"
            wrapped, _flagged = wrap_untrusted(text, f"telegram file: {ws_path.name}")
            return f"File contents:\n{wrapped}"
        return (
            "Binary file saved. Its contents were not inlined — analyze it "
            "with python_exec (or shell_exec with user approval) if needed."
        )

    result.note = f"{header}\n\n{await body()}"
    return result


__all__ = [
    "AttachmentRef",
    "FileTooBig",
    "DownloadError",
    "Ingested",
    "attachment_ref",
    "download_telegram_file",
    "image_data_uri",
    "ingest_attachment",
    "sanitize_filename",
    "transcribe_audio",
    "unique_path",
]
