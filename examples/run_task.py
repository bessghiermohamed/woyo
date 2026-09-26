"""Example: use woyo as a library.

Run with a configured provider (see .env.example):
    python examples/run_task.py "What time is it in UTC right now?"
"""

from __future__ import annotations

import asyncio

from woyo import Agent, Settings
from woyo.events import EventBus
from woyo.models.router import ModelRouter
from woyo.tools.builtin import build_default_registry


async def main(task: str) -> None:
    settings = Settings()
    bus = EventBus()
    registry = build_default_registry(settings, bus=bus)
    router = ModelRouter(settings, bus=bus)
    agent = Agent(settings, router, registry, bus=bus)

    result = await agent.run(
        task,
        approval_cb=lambda tool, args: input(f"Allow {tool}? [y/N] ").lower() == "y",
    )
    print(result.final_answer)
    print("---")
    print(result.summary())


if __name__ == "__main__":
    import sys

    asyncio.run(main(" ".join(sys.argv[1:]) or "What is 2^40 divided by 1024? Use the calculate tool."))
