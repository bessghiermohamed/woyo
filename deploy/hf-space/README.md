---
title: woyo
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# woyo — agent chat

A general-purpose AI agent you can chat with from any phone browser:
it plans, searches the web, reads pages, verifies its sources and
replies with citations.

This Space runs the `woyo` package from
[github.com/bessghiermohamed/woyo](https://github.com/bessghiermohamed/woyo).

## Setup (Settings → Variables and secrets)

| Secret | Why |
|---|---|
| `WOYO_CHAT_PASSWORD` | **required** — passcode visitors must enter |
| `COHERE_API_KEY` | model provider key (or any provider woyo supports) |
| `WOYO_PROVIDER` / `WOYO_MODEL` | e.g. `cohere` / `command-a-03-2025` |
| `TELEGRAM_BOT_TOKEN` | optional — enables the Telegram bot alongside the web chat |

## Notes

- Free CPU Spaces sleep after ~48h without HTTP traffic. Ping
  `https://<this-space>.hf.space/api/health` every few minutes with a
  free uptime monitor (e.g. cron-job.org) to keep the agent online.
- While asleep, the Telegram poller is down too; Telegram queues
  messages server-side and they are processed once the Space wakes.
