"""Execution sandbox + workspace files (Phase 4, v0.5).

python_exec — run Python in an isolated subprocess: interpreter `-I`
(isolated mode), POSIX rlimits (CPU / memory / file size / fd count), a
SCRUBBED environment (secrets like TELEGRAM_BOT_TOKEN or provider keys
are never inherited — critical on shared hosts), and a persistent
workspace directory as cwd, so files survive across calls.

shell_exec — a real terminal in the same workspace. Approval-gated
(WRITES_EXTERNAL) AND behind WOYO_ENABLE_SHELL=true: default-deny twice.

read_file / write_file / list_dir — workspace-scoped file tools with
path-traversal protection.

Backends: "local" (subprocess + rlimits; always available; dev-grade)
and "docker" (one container per call, --network none; the real boundary;
used when WOYO_SANDBOX=docker and the docker binary exists). See
docs/SECURITY.md for the honest threat model of each.
"""

from __future__ import annotations

import asyncio
import os
import resource
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from woyo.config import Settings
from woyo.errors import ErrorKind
from woyo.tools.base import Permission, Tool, ToolResult

# --- workspace ---------------------------------------------------------------

def workspace_root(settings: Settings) -> Path:
    root = Path(settings.workspace_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_ws_path(root: Path, rel: str) -> Path:
    """Resolve `rel` inside `root`; refuse traversal or absolute escapes."""
    candidate = (root / rel).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(
            f"path '{rel}' escapes the workspace; only relative paths inside "
            "the workspace are allowed"
        )
    return candidate


# --- subprocess plumbing -----------------------------------------------------

_SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL")


def scrubbed_env(root: Path) -> dict[str, str]:
    """Minimal environment for child processes — secrets never inherit.

    On a GitHub Actions runner the process env carries TELEGRAM_BOT_TOKEN,
    provider keys and the state-repo token; on a laptop it may carry far
    more. Children get PATH/locale plus a sandbox HOME and TMPDIR, nothing
    else.
    """
    env: dict[str, str] = {
        "HOME": str(root),
        "TMPDIR": str(root / ".tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    for key in _SAFE_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            env[key] = value
    (root / ".tmp").mkdir(parents=True, exist_ok=True)
    return env


def _python_limits() -> None:  # runs inside the child process
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_AS, (512_000_000, 512_000_000))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8_000_000, 8_000_000))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _shell_limits() -> None:  # runs inside the child process
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16_000_000, 16_000_000))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _format_output(out: bytes, returncode: int | None, *, backend: str) -> str:
    text = out.decode("utf-8", errors="replace")[:20_000]
    status = "exit 0" if returncode == 0 else f"exit {returncode}"
    return f"[{backend} · {status}]\n{text or '(no output)'}"


async def _run_local(
    argv: list[str], *, cwd: Path, env: dict[str, str],
    timeout_s: int, preexec,
) -> ToolResult:
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            preexec_fn=preexec,  # noqa: PLW1509 — POSIX-only by design
        )
    except OSError as exc:
        return ToolResult.error(ErrorKind.TOOL_FAILURE, f"Failed to start: {exc}")
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        return ToolResult.error(
            ErrorKind.TRANSIENT, f"Exceeded {timeout_s}s and was killed."
        )
    return ToolResult.ok_result(_format_output(out, proc.returncode, backend="local"))


async def _run_docker(
    argv: list[str], *, root: Path, timeout_s: int,
) -> ToolResult:
    """Run argv inside a throwaway container: no network, capped resources."""
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "--memory", "512m", "--cpus", "1",
        "-v", f"{root.resolve()}:/work", "-w", "/work",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONIOENCODING=utf-8",
        "python:3.12-slim", *argv,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        return ToolResult.error(ErrorKind.TOOL_FAILURE, f"Failed to start docker: {exc}")
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s + 30)
    except TimeoutError:
        proc.kill()
        return ToolResult.error(
            ErrorKind.TRANSIENT, f"Exceeded {timeout_s}s and was killed."
        )
    return ToolResult.ok_result(_format_output(out, proc.returncode, backend="docker"))


def _backend(settings: Settings) -> str:
    if settings.sandbox_backend == "docker" and _docker_available():
        return "docker"
    return "local"


# --- python_exec ---------------------------------------------------------------

class CodeArgs(BaseModel):
    code: str = Field(min_length=1, max_length=20_000)
    timeout_s: int = Field(default=20, ge=1, le=60)


class PythonExecTool(Tool):
    name = "python_exec"
    description = (
        "Run a Python snippet and get stdout/stderr. Runs isolated "
        "(-I, CPU/memory limits, no secrets in env) with the workspace "
        "directory as cwd — files you write persist across calls; use "
        "read_file/write_file/list_dir to manage them. print() your results. "
        "For PDFs/documents use the create_document tool instead of writing "
        "reportlab code here — it shapes Arabic and other scripts correctly."
    )
    permission = Permission.SANDBOXED
    timeout_s = 70.0
    Args = CodeArgs

    def __init__(self, settings: Settings):
        self._settings = settings

    async def run(self, args: CodeArgs) -> ToolResult:
        if not self._settings.enable_code_exec:
            return ToolResult.error(
                ErrorKind.CONFIG,
                "Code execution is disabled. Ask the user to set "
                "WOYO_ENABLE_CODE_EXEC=true if they want it "
                "(see docs/SECURITY.md).",
            )
        root = workspace_root(self._settings)
        backend = _backend(self._settings)
        script = root / ".woyo_snippet.py"
        script.write_text(args.code, encoding="utf-8")
        try:
            if backend == "docker":
                return await _run_docker(
                    ["python", "-I", "/work/.woyo_snippet.py"],
                    root=root, timeout_s=args.timeout_s,
                )
            return await _run_local(
                [sys.executable, "-I", str(script)],
                cwd=root, env=scrubbed_env(root),
                timeout_s=args.timeout_s, preexec=_python_limits,
            )
        finally:
            script.unlink(missing_ok=True)


# --- shell_exec -----------------------------------------------------------------

class ShellArgs(BaseModel):
    command: str = Field(min_length=1, max_length=2_000)
    timeout_s: int = Field(default=30, ge=1, le=120)
    why: str = Field(
        default="", max_length=300,
        description="One line telling the user why this command is needed",
    )


class ShellExecTool(Tool):
    name = "shell_exec"
    description = (
        "Run a shell command in the workspace (a real terminal: ls, grep, "
        "pip install --user, curl is blocked offline in docker mode…). "
        "Each call needs explicit user approval. Keep commands "
        "single-purpose; the workspace persists between calls."
    )
    permission = Permission.WRITES_EXTERNAL  # approval gate applies
    timeout_s = 130.0
    Args = ShellArgs

    def __init__(self, settings: Settings):
        self._settings = settings

    async def run(self, args: ShellArgs) -> ToolResult:
        if not self._settings.enable_shell:
            return ToolResult.error(
                ErrorKind.CONFIG,
                "Shell execution is disabled. Ask the user to set "
                "WOYO_ENABLE_SHELL=true (approval is still required for "
                "every call; see docs/SECURITY.md).",
            )
        root = workspace_root(self._settings)
        backend = _backend(self._settings)
        if backend == "docker":
            return await _run_docker(
                ["sh", "-c", args.command], root=root, timeout_s=args.timeout_s
            )
        return await _run_local(
            ["/bin/sh", "-c", args.command],
            cwd=root, env=scrubbed_env(root),
            timeout_s=args.timeout_s, preexec=_shell_limits,
        )


# --- workspace files --------------------------------------------------------------

class ReadFileArgs(BaseModel):
    path: str = Field(min_length=1, max_length=300,
                      description="Relative path inside the workspace")
    max_chars: int = Field(default=12_000, ge=100, le=50_000)


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a text file from the workspace (created earlier by you, "
        "python_exec or shell_exec). Paths are relative to the workspace."
    )
    permission = Permission.READ_ONLY
    timeout_s = 10.0
    Args = ReadFileArgs

    def __init__(self, settings: Settings):
        self._settings = settings

    async def run(self, args: ReadFileArgs) -> ToolResult:
        root = workspace_root(self._settings)
        try:
            path = resolve_ws_path(root, args.path)
        except ValueError as exc:
            return ToolResult.error(ErrorKind.INVALID_INPUT, str(exc))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult.error(
                ErrorKind.INVALID_INPUT, f"Cannot read '{args.path}': {exc}"
            )
        if len(text) > args.max_chars:
            text = text[: args.max_chars] + "\n[... truncated ...]"
        return ToolResult.ok_result(
            text or "(empty file)",
            data={"path": args.path},
        )


class WriteFileArgs(BaseModel):
    path: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=50_000)


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write (or overwrite) a text file in the workspace. Use for "
        "artifacts the user asked for (notes, code, CSV) or intermediate "
        "data other tools should reuse. Paths are workspace-relative."
    )
    permission = Permission.SANDBOXED
    timeout_s = 10.0
    Args = WriteFileArgs

    def __init__(self, settings: Settings):
        self._settings = settings

    async def run(self, args: WriteFileArgs) -> ToolResult:
        root = workspace_root(self._settings)
        try:
            path = resolve_ws_path(root, args.path)
        except ValueError as exc:
            return ToolResult.error(ErrorKind.INVALID_INPUT, str(exc))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.content, encoding="utf-8")
        except OSError as exc:
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE, f"Cannot write '{args.path}': {exc}"
            )
        return ToolResult.ok_result(
            f"wrote {len(args.content)} chars to {args.path}"
        )


class ListDirArgs(BaseModel):
    path: str = Field(default=".", max_length=300)


class ListDirTool(Tool):
    name = "list_dir"
    description = "List workspace directory entries (name, size, dir flag)."
    permission = Permission.READ_ONLY
    timeout_s = 10.0
    Args = ListDirArgs

    def __init__(self, settings: Settings):
        self._settings = settings

    async def run(self, args: ListDirArgs) -> ToolResult:
        root = workspace_root(self._settings)
        try:
            path = resolve_ws_path(root, args.path or ".")
        except ValueError as exc:
            return ToolResult.error(ErrorKind.INVALID_INPUT, str(exc))
        try:
            entries: list[str] = []
            for child in sorted(path.iterdir()):
                if child.name.startswith("."):
                    continue
                size = child.stat().st_size if child.is_file() else 0
                entries.append(f"{'d' if child.is_dir() else '-'} {size:>9} {child.name}")
        except OSError as exc:
            return ToolResult.error(
                ErrorKind.INVALID_INPUT, f"Cannot list '{args.path}': {exc}"
            )
        return ToolResult.ok_result(
            "\n".join(entries) or "(empty directory)",
            data={"path": args.path},
        )


__all__ = [
    "PythonExecTool",
    "ShellExecTool",
    "ReadFileTool",
    "WriteFileTool",
    "ListDirTool",
    "workspace_root",
    "resolve_ws_path",
    "scrubbed_env",
]
