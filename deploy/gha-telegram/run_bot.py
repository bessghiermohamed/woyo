"""Run the woyo Telegram bot on a GitHub Actions runner (ephemeral host).

The runner vanishes after ~6h, so conversation + offset state is synced to
a PRIVATE git repo (BOT_STATE_REPO + BOT_STATE_TOKEN env) after every
message. On boot the repo is cloned into a temp dir and its *.json files
are restored into ~/.woyo, so the next runner resumes exactly where this
one stopped.

Secrets this script expects (repo → Settings → Secrets and variables):
  TELEGRAM_BOT_TOKEN   the bot token from @BotFather
  COHERE_API_KEY       (or whichever provider WOYO_PROVIDER points to)
  BOT_STATE_REPO       owner/name of the private state repo
  BOT_STATE_TOKEN      a GitHub token with repo scope (sync only)
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

HOME = Path.home() / ".woyo"
STATE_FILE = HOME / "telegram_state.json"
CHATS_DIR = HOME / "chats"

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
    for path in CLONE_DIR.rglob("*.json"):
        if path.name == "telegram_state.json":
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_bytes(path.read_bytes())
            restored += 1
        else:
            CHATS_DIR.mkdir(parents=True, exist_ok=True)
            (CHATS_DIR / path.name).write_bytes(path.read_bytes())
            restored += 1
    print(f"state restored: {restored} file(s) from {STATE_REPO}")


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

                sess._save = save  # type: ignore[method-assign]
            return sess

        bot._session = session_with_sync  # type: ignore[method-assign]

    print("starting woyo telegram bot (GitHub Actions runner)…", flush=True)
    asyncio.run(bot.run_forever())


if __name__ == "__main__":
    main()
