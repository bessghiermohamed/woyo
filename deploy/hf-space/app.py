"""Hugging Face Space entrypoint: web chat + optional Telegram poller.

Secrets to set in Space settings (Settings → Variables and secrets):
  WOYO_CHAT_PASSWORD   required — the passcode visitors must type
  COHERE_API_KEY       (or any other provider key; see woyo .env.example)
  WOYO_PROVIDER        e.g. cohere
  WOYO_MODEL           e.g. command-a-03-2025
  TELEGRAM_BOT_TOKEN   optional — add it to also chat via Telegram
"""

from __future__ import annotations

import asyncio
import os
import threading


def main() -> None:
    from woyo.config import Settings

    settings = Settings()

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if token:
        from woyo.chat.telegram import TelegramBot

        bot = TelegramBot(settings, token)
        threading.Thread(
            target=lambda: asyncio.run(bot.run_forever()),
            daemon=True,
            name="woyo-telegram",
        ).start()
        print("telegram poller started", flush=True)
    else:
        print(
            "TELEGRAM_BOT_TOKEN not set — web chat only "
            "(add the token in Settings → Secrets to enable the bot)",
            flush=True,
        )

    from woyo.chat.web import serve

    port = int(os.environ.get("PORT", "7860"))
    serve(settings, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
