"""Run the woyo Telegram bot on a GitHub Actions runner (ephemeral host).

The runner vanishes after ~6h, so conversation + offset state is synced to
a private GitHub gist after every event (BOT_STATE_GIST_ID +
BOT_STATE_GIST_TOKEN env). On boot the gist is pulled back into ~/.woyo,
so the next runner resumes exactly where this one stopped.

Secrets this script expects (repo → Settings → Secrets and variables):
  TELEGRAM_BOT_TOKEN   the bot token from @BotFather
  COHERE_API_KEY       (or whichever provider WOYO_PROVIDER points to)
  BOT_STATE_GIST_ID    id of the private gist used as state store
  BOT_STATE_GIST_TOKEN a GitHub token with gist scope (sync only)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx

GIST_API = "https://api.github.com/gists"
HOME = Path.home() / ".woyo"
STATE_FILE = HOME / "telegram_state.json"
CHATS_DIR = HOME / "chats"

GIST_ID = os.environ.get("BOT_STATE_GIST_ID", "")
GIST_TOKEN = os.environ.get("BOT_STATE_GIST_TOKEN", "")


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {GIST_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def pull_state() -> None:
    """Restore ~/.woyo from the gist before the bot starts."""
    if not GIST_ID:
        print("no BOT_STATE_GIST_ID — starting with empty state")
        return
    with httpx.Client(timeout=30, headers=_headers()) as client:
        gist = client.get(f"{GIST_API}/{GIST_ID}").raise_for_status().json()
    for name, info in gist.get("files", {}).items():
        content = info.get("content") or ""
        if not content:
            continue
        if name == STATE_FILE.name:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(content, encoding="utf-8")
        elif name.endswith(".json"):
            CHATS_DIR.mkdir(parents=True, exist_ok=True)
            (CHATS_DIR / name).write_text(content, encoding="utf-8")
    print(f"state restored from gist: {len(gist.get('files', {}))} file(s)")


def _push_file(name: str, path: Path) -> None:
    if not GIST_ID or not path.exists():
        return
    try:
        with httpx.Client(timeout=30, headers=_headers()) as client:
            client.patch(
                f"{GIST_API}/{GIST_ID}",
                json={"files": {name: {"content": path.read_text(encoding="utf-8")}}},
            ).raise_for_status()
    except Exception as exc:  # noqa: BLE001 — sync must never kill the bot
        print(f"warn: gist sync of {name} failed: {exc}")


def main() -> None:
    pull_state()

    from woyo.chat.telegram import TelegramBot, telegram_credentials
    from woyo.config import Settings

    settings = Settings()
    token, allowed = telegram_credentials()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set")
    bot = TelegramBot(settings, token, allowed)

    if GIST_ID:
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
            if not getattr(sess, "_gist_synced", False):
                sess._gist_synced = True  # type: ignore[attr-defined]
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
