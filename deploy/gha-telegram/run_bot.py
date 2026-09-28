"""Run the woyo Telegram bot on a GitHub Actions runner (ephemeral host).

The runner vanishes after ~6h, so ALL durable state is synced to a PRIVATE
git repo (BOT_STATE_REPO + BOT_STATE_TOKEN env) after every message:
  * telegram_state.json  — update offset + owner claim
  * chats/*.json         — conversation transcripts
  * woyo.sqlite3         — tasks + long-term memory (online-backup snapshot)
  * files/**             — durable inbox (user attachments) + outbox
                           (bot deliverables), size-capped
  * workspace-inbox/**   — the agent-visible copy of user attachments
On boot the repo is cloned into a temp dir and restored into ~/.woyo, so
the next runner resumes exactly where this one stopped — including memory
and files.

Secrets this script expects (repo → Settings → Secrets and variables):
  TELEGRAM_BOT_TOKEN   the bot token from @BotFather
  COHERE_API_KEY       (or whichever provider WOYO_PROVIDER points to)
  BOT_STATE_REPO       owner/name of the private state repo
  BOT_STATE_TOKEN      a GitHub token with repo scope (sync only)
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

HOME = Path.home() / ".woyo"
STATE_FILE = HOME / "telegram_state.json"
CHATS_DIR = HOME / "chats"
DB_FILE = HOME / "woyo.sqlite3"
FILES_DIR = HOME / "files"
WS_INBOX = HOME / "workspace" / "inbox"

#: budgets that keep the state repo sane (git is not an artifact store)
_MAX_SYNC_FILE_MB = 8
_MAX_SYNC_TOTAL_MB = 32

STATE_REPO = os.environ.get("BOT_STATE_REPO", "")
STATE_TOKEN = os.environ.get("BOT_STATE_TOKEN", "")
CLONE_DIR: Path | None = None


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )


def pull_state() -> None:
    """Clone the state repo and restore its files into ~/.woyo."""
    global CLONE_DIR
    if not STATE_REPO:
        print("no BOT_STATE_REPO — starting with empty state")
        return
    CLONE_DIR = Path(tempfile.mkdtemp(prefix="woyo-state-"))
    url = f"https://x-access-token:{STATE_TOKEN}@github.com/{STATE_REPO}.git"
    _git("clone", "--depth", "5", url, str(CLONE_DIR))
    _git("config", "user.name", "woyo-bot", cwd=CLONE_DIR)
    _git("config", "user.email", "woyo-bot@users.noreply.github.com", cwd=CLONE_DIR)
    restored = 0
    for path in CLONE_DIR.rglob("*"):
        if not path.is_file() or path.name == ".gitignore":
            continue
        rel = path.relative_to(CLONE_DIR)
        top = rel.parts[0] if len(rel.parts) > 1 else ""
        if top == "files":  # durable inbox/outbox tree
            dest = FILES_DIR.joinpath(*rel.parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
            restored += 1
        elif top == "workspace-inbox":  # agent-visible attachment copies
            dest = WS_INBOX.joinpath(*rel.parts[1:])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
            restored += 1
        elif path.name == "telegram_state.json":
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_bytes(path.read_bytes())
            restored += 1
        elif path.name == "woyo.sqlite3":
            # drop stale WAL sidecars so sqlite opens the snapshot clean
            for suffix in ("-wal", "-shm"):
                (DB_FILE.parent / f"woyo.sqlite3{suffix}").unlink(missing_ok=True)
            DB_FILE.parent.mkdir(parents=True, exist_ok=True)
            DB_FILE.write_bytes(path.read_bytes())
            restored += 1
        elif path.suffix == ".json":
            CHATS_DIR.mkdir(parents=True, exist_ok=True)
            (CHATS_DIR / path.name).write_bytes(path.read_bytes())
            restored += 1
    print(f"state restored: {restored} file(s) from {STATE_REPO}")


def snapshot_db(dest: Path) -> None:
    """Online-backup the live db to dest (safe while idle; no WAL sidecars)."""
    if not DB_FILE.exists():
        return
    src = sqlite3.connect(str(DB_FILE))
    dst = sqlite3.connect(str(dest))
    try:
        with dst:
            src.backup(dst)
    finally:
        src.close()
        dst.close()


def _push_db() -> None:
    """Snapshot tasks+memory db into the state clone and push if it changed."""
    if CLONE_DIR is None:
        return
    try:
        snapshot_db(CLONE_DIR / DB_FILE.name)
        _git("add", DB_FILE.name, cwd=CLONE_DIR)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=CLONE_DIR,
            capture_output=True,
        )
        if staged.returncode == 0:
            return  # db unchanged since the last sync
        _git("commit", "-m", "sync woyo.sqlite3", cwd=CLONE_DIR)
        _git("push", cwd=CLONE_DIR)
    except (subprocess.CalledProcessError, sqlite3.Error) as exc:
        print(f"warn: state sync of {DB_FILE.name} failed: {exc}")


def _push_file(name: str, path: Path) -> None:
    if CLONE_DIR is None or not path.exists():
        return
    target = CLONE_DIR / name
    try:
        target.write_bytes(path.read_bytes())
        _git("add", name, cwd=CLONE_DIR)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=CLONE_DIR,
            capture_output=True,
        )
        if staged.returncode == 0:
            return  # nothing changed since the last sync
        _git("commit", "-m", f"sync {name}", cwd=CLONE_DIR)
        _git("push", cwd=CLONE_DIR)
    except subprocess.CalledProcessError as exc:
        print(f"warn: state sync of {name} failed: {exc.stderr.strip()[:200]}")


def mirror_with_caps(
    src: Path, dst: Path, *, max_file_mb: int, total_mb: int
) -> list[Path]:
    """Mirror `src` into `dst` under size budgets; returns the kept files.

    Files larger than `max_file_mb` are skipped; when the tree exceeds
    `total_mb`, the oldest files (by mtime) are pruned from BOTH sides —
    a safety valve so a git-backed state repo never becomes an artifact
    store. Normal usage stays far below the caps.
    """
    if not src.is_dir():
        return []
    max_file = max_file_mb * 1024 * 1024
    kept: list[Path] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > max_file:
                print(f"sync: skipping oversized file {path.name}")
                continue
            kept.append(path)
        except OSError:
            continue
    # oldest-first prune when the total budget is exceeded
    budget = total_mb * 1024 * 1024
    total = sum(p.stat().st_size for p in kept)
    if total > budget:
        kept.sort(key=lambda p: p.stat().st_mtime)
        dropped: list[Path] = []
        while total > budget and kept:
            victim = kept.pop(0)
            total -= victim.stat().st_size
            dropped.append(victim)
        for victim in dropped:
            print(f"sync: pruning old file {victim.name} (budget)")
            victim.unlink(missing_ok=True)
    # mirror what's left
    if dst.exists():
        shutil.rmtree(dst)
    for path in kept:
        dest = dst.joinpath(*path.relative_to(src).parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    return kept


def _push_tree(src: Path, clone_rel: str) -> None:
    """Mirror a durable tree into the state clone and push if it changed."""
    if CLONE_DIR is None:
        return
    try:
        mirror_with_caps(
            src, CLONE_DIR / clone_rel,
            max_file_mb=_MAX_SYNC_FILE_MB, total_mb=_MAX_SYNC_TOTAL_MB,
        )
        _git("add", "-A", clone_rel, cwd=CLONE_DIR)
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", clone_rel], cwd=CLONE_DIR,
            capture_output=True,
        )
        if staged.returncode == 0:
            return  # nothing changed since the last sync
        _git("commit", "-m", f"sync {clone_rel}", cwd=CLONE_DIR)
        _git("push", cwd=CLONE_DIR)
    except subprocess.CalledProcessError as exc:
        print(f"warn: state sync of {clone_rel} failed: {exc.stderr.strip()[:200]}")


def main() -> None:
    pull_state()

    from woyo.chat.telegram import TelegramBot, telegram_credentials
    from woyo.config import Settings

    settings = Settings()
    token, allowed = telegram_credentials()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    bot = TelegramBot(settings, token, allowed)

    if CLONE_DIR is not None:
        # offset/owner state: sync after every update batch
        orig_save_state = bot._save_state

        def save_state() -> None:
            orig_save_state()
            _push_file(STATE_FILE.name, STATE_FILE)

        bot._save_state = save_state  # type: ignore[method-assign]

        # chat transcripts: sync after every message
        orig_session = bot._session

        def session_with_sync(chat_id: int):
            sess = orig_session(chat_id)
            if not getattr(sess, "_state_synced", False):
                sess._state_synced = True  # type: ignore[attr-defined]
                orig_save = sess._save

                def save() -> None:
                    orig_save()
                    _push_file(sess._state_path().name, sess._state_path())
                    _push_db()  # tasks + long-term memory ride along
                    _push_tree(FILES_DIR, "files")  # durable inbox/outbox
                    _push_tree(WS_INBOX, "workspace-inbox")

                sess._save = save  # type: ignore[method-assign]
            return sess

        bot._session = session_with_sync  # type: ignore[method-assign]

    print("starting woyo telegram bot (GitHub Actions runner)…", flush=True)
    asyncio.run(bot.run_forever())


if __name__ == "__main__":
    main()
