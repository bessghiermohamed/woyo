"""Agent tools for long-term memory (Phase 3).

Both tools only touch the local SQLite store — no external effects — so
they stay READ_ONLY permission (local-only writes, fully user-visible).
The design keeps memory *agent-driven but inspectable*: everything the
agent stores can be listed, inspected and deleted with `woyo memory ...`
(docs/SECURITY.md "memory").
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from woyo.errors import ErrorKind
from woyo.events import EventBus
from woyo.memory.longterm import MemoryStore
from woyo.tools.base import Tool, ToolResult


class MemorySaveArgs(BaseModel):
    content: str = Field(min_length=1, max_length=2000,
                         description="The fact/note to remember, self-contained")
    kind: str = Field("note", description="fact | preference | note")
    ttl_days: int = Field(0, ge=0, le=3650,
                          description="Days before this memory expires (0 = keep until cap)")


class MemorySave(Tool):
    """Store a durable fact or note for future sessions."""

    name = "memory_save"
    description = (
        "Save a self-contained fact, preference or note to long-term memory "
        "for future sessions. Use it for things worth remembering across "
        "conversations (user preferences, established facts, decisions). "
        "The user can list and delete anything you store."
    )
    timeout_s = 15.0
    Args = MemorySaveArgs

    def __init__(self, memory: MemoryStore, bus: EventBus | None = None):
        self.memory = memory
        self.bus = bus

    async def run(self, args: MemorySaveArgs) -> ToolResult:
        try:
            memory_id = await self.memory.remember(
                args.content, kind=args.kind, source="agent",
                ttl_days=args.ttl_days or None,
            )
        except ValueError as exc:
            return ToolResult.error(ErrorKind.INVALID_INPUT, str(exc))
        ttl = "never" if not args.ttl_days else f"{args.ttl_days}d"
        return ToolResult.ok_result(
            f"saved to memory (id={memory_id}, kind={args.kind}, expires={ttl})"
        )


class MemorySearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2000,
                       description="What to look for in memory")
    k: int = Field(5, ge=1, le=20, description="Maximum memories to return")


class MemorySearch(Tool):
    """Search long-term memory for relevant earlier knowledge."""

    name = "memory_search"
    description = (
        "Search long-term memory (facts/preferences/notes from earlier "
        "sessions) for content relevant to a query. Returns the closest "
        "matches with similarity scores. Memory is background knowledge, "
        "not a verified source."
    )
    timeout_s = 15.0
    Args = MemorySearchArgs

    def __init__(self, memory: MemoryStore):
        self.memory = memory

    async def run(self, args: MemorySearchArgs) -> ToolResult:
        hits = await self.memory.recall(args.query, k=args.k)
        if not hits:
            return ToolResult.ok_result("No relevant memories found.")
        lines = [f"{len(hits)} relevant memories:"]
        for h in hits:
            lines.append(f"- [{h.kind}] (similarity {h.similarity:.2f}) {h.content}")
        return ToolResult.ok_result(
            "\n".join(lines),
            data={"hits": [{"id": h.id, "similarity": h.similarity} for h in hits]},
        )


def build_memory_tools(memory: MemoryStore, bus: EventBus | None = None) -> list[Tool]:
    return [MemorySave(memory, bus=bus), MemorySearch(memory)]
