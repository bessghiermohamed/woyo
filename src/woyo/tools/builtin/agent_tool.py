"""spawn_agent: delegate a subtask to a fresh sub-agent (Phase "agents").

Why a tool and not a framework: the parent's executor decides when
delegation helps (parallel research threads, context isolation for messy
subtasks). The child is a full Agent — own plan, own budgets — but:

- ONE level of nesting: the child's registry has no spawn_agent, so
  recursion is structurally impossible (no depth counter to go wrong).
- The child SHARES the parent's ModelRouter: token/cost usage accumulates
  in one place, so the parent's budget gate covers sub-agent spend.
- The child gets read/compute tools only — no approval-gated tools
  (there is no human attached to a sub-agent) and no ask_user.
- The child's cited sources flow back as observed URLs, so the parent's
  citation verification still applies to everything it reports.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from woyo.config import Settings
from woyo.errors import ErrorKind
from woyo.events import EventBus
from woyo.tools.base import Permission, Tool, ToolResult

#: What a sub-agent may use. Deliberately excludes spawn_agent (depth),
#: shell_exec / ask_user (no human attached), write_file (read-only role).
_CHILD_TOOLS = [
    "web_search",
    "fetch_url",
    "crawl_site",
    "calculate",
    "now",
    "python_exec",
    "memory_search",
    "finish",
]


class SpawnAgentArgs(BaseModel):
    task: str = Field(
        min_length=10, max_length=2_000,
        description="A self-contained subtask: goal + success criteria + "
                    "any context the sub-agent needs (it sees nothing else)",
    )


class SpawnAgentTool(Tool):
    name = "spawn_agent"
    description = (
        "Delegate a self-contained subtask to a fresh sub-agent with its own "
        "plan and budgets (web search, page fetch, crawl, calculate, python). "
        "It returns a final answer with sources. Use it to split a research "
        "question into threads, or to keep messy exploration out of your own "
        "context. Give a complete task statement — the sub-agent sees ONLY "
        "what you pass. One level of nesting only."
    )
    permission = Permission.READ_ONLY  # bounded: own budgets + shared router
    timeout_s = 150.0
    Args = SpawnAgentArgs

    def __init__(
        self,
        settings: Settings,
        router,  # ModelRouter (typed loosely to avoid an import cycle)
        *,
        bus: EventBus | None = None,
        memory=None,  # MemoryStore
    ):
        self._settings = settings
        self._router = router
        self._bus = bus
        self._memory = memory

    async def run(self, args: SpawnAgentArgs) -> ToolResult:
        from woyo.agent import Agent
        from woyo.tools.builtin import build_default_registry

        s = self._settings
        child_settings = s.model_copy(
            update={
                "max_steps": s.subagent_max_steps,
                "max_tool_calls": 16,
                "max_time_s": s.subagent_max_time_s,
                "max_tokens": 40_000,
                "max_cost_usd": s.subagent_max_cost_usd,
                "max_search_calls": s.subagent_max_search_calls,
                "direct_text_replies": True,
                "task_checkpoint": False,  # no store wiring for children
                "require_approval": False,  # registry has no approval tools
            }
        )
        child_bus = EventBus()
        child_registry = build_default_registry(
            child_settings, bus=child_bus, memory=self._memory,
            include=_CHILD_TOOLS,
        )
        child = Agent(child_settings, self._router, child_registry, bus=child_bus)

        if self._bus:
            self._bus.emit("subagent_started", task=args.task[:200])
        try:
            result = await child.run(args.task)
        except Exception as exc:  # noqa: BLE001 — delegation must not kill the parent
            if self._bus:
                self._bus.emit("subagent_finished", outcome="failed")
            return ToolResult.error(
                ErrorKind.TOOL_FAILURE,
                f"Sub-agent run failed: {type(exc).__name__}: {exc}",
            )
        if self._bus:
            self._bus.emit(
                "subagent_finished", outcome=result.outcome, steps=result.steps
            )

        sources = [s_.get("url", "") for s_ in result.sources if s_.get("url")]
        lines = [
            f"Sub-agent finished ({result.outcome}, {result.steps} steps, "
            f"{result.duration_s:.0f}s). Final answer:",
            "",
            result.final_answer,
        ]
        if result.open_questions:
            lines += ["", "Open questions: " + "; ".join(result.open_questions[:3])]
        if sources:
            lines += ["", "Sources:", *[f"- {u}" for u in sources[:8]]]
        # child-observed URLs ride along so the parent's citation
        # verification covers claims sourced from the sub-agent
        data: dict[str, object] = {"subagent": True}
        if sources:
            data["results"] = [{"url": u} for u in sources]
        return ToolResult.ok_result("\n".join(lines), data=data)
