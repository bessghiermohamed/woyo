"""The woyo agent loop: plan -> act -> observe -> adapt -> verify -> finish.

Design notes (see ARCHITECTURE.md §3):
- Native tool-calling only; arguments validated by the ToolRegistry.
- Budgets enforced every iteration; exhaustion yields an honest partial result.
- Loop detection on identical repeated calls; replan pressure on failures.
- Approval gates for writes_external/destructive tools — denied when no
  approval channel exists (fail safe).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from woyo.agent.planner import create_plan
from woyo.agent.prompts import executor_system_prompt
from woyo.agent.results import RunResult, build_usage_report
from woyo.config import Settings
from woyo.errors import AgentError, ErrorKind
from woyo.events import EventBus
from woyo.memory.working import compact_context
from woyo.models.base import Message, ToolCall, ToolSpec
from woyo.models.router import ModelRouter
from woyo.tools.base import Permission, ToolRegistry, ToolResult

ApprovalCallback = Callable[[str, str], bool]  # (tool_name, args_json) -> approved


class Agent:
    """Runs one task at a time through the supervised loop."""

    def __init__(
        self,
        settings: Settings,
        router: ModelRouter,
        registry: ToolRegistry,
        bus: EventBus | None = None,
    ):
        self.settings = settings
        self.router = router
        self.registry = registry
        self.bus = bus or EventBus()

    # ------------------------------------------------------------------
    async def run(
        self,
        task: str,
        *,
        approval_cb: ApprovalCallback | None = None,
    ) -> RunResult:
        task = task.strip()
        if not task:
            raise AgentError("empty task")
        started = time.monotonic()
        self.bus.emit("run_started", task=task[:300])

        # 1) PLAN ------------------------------------------------------
        catalog = "\n".join(
            f"- {item['name']} ({item['permission']}): {item['description']}"
            for item in self.registry.safety_overview()
        )
        plan = await create_plan(self.router, task, catalog, self.bus)
        budget_text = (
            f"max {self.settings.max_steps} steps, {self.settings.max_tool_calls} tool "
            f"calls, {self.settings.max_time_s:.0f}s, ~${self.settings.max_cost_usd:.2f}"
        )
        system = executor_system_prompt(
            today=datetime.now(UTC).strftime("%Y-%m-%d"),
            timezone_name=self.settings.timezone,
            task=task,
            plan_text=plan.render(),
            budget_text=budget_text,
        )
        messages: list[Message] = [
            Message(role="system", content=system),
            Message(role="user", content=f"Task: {task}"),
        ]

        # 2) EXECUTE LOOP ----------------------------------------------
        steps = 0
        tool_calls = 0
        consecutive_failures = 0
        text_only_strikes = 0
        repeat_counter: Counter[tuple[str, str]] = Counter()
        outcome = "failed"
        outcome_detail = ""
        final: dict[str, Any] | None = None

        while final is None:
            # --- budget gate ------------------------------------------
            budget_reason = self._check_budgets(steps, tool_calls, started)
            if budget_reason is not None:
                outcome, outcome_detail = "budget_exhausted", budget_reason
                self.bus.emit("limit_hit", limit=budget_reason)
                break

            # --- executor call ----------------------------------------
            compact_context(messages, self.settings.context_soft_limit_tokens)
            steps += 1
            resp = await self.router.complete(
                "executor",
                messages=messages,
                tools=self._tool_specs(),
            )
            messages.append(
                Message(
                    role="assistant",
                    content=resp.content,
                    tool_calls=resp.tool_calls or None,
                )
            )

            if not resp.tool_calls:
                # text-only reply: nudge once, accept the second as final
                text_only_strikes += 1
                if text_only_strikes >= 2:
                    final = {
                        "summary": resp.content or "(no content)",
                        "verified": False,
                        "open_questions": [],
                        "sources": [],
                    }
                    outcome = "completed"
                    outcome_detail = "text-only final answer"
                    break
                messages.append(
                    Message(
                        role="user",
                        content=(
                            "Continue with your tools, or call `finish` with your "
                            "final summary when the task is done."
                        ),
                    )
                )
                continue

            text_only_strikes = 0
            stop_reason: str | None = None
            for tc in resp.tool_calls:
                result, finish_payload, blocked_reason = await self._execute_one(
                    tc,
                    approval_cb=approval_cb,
                    repeat_counter=repeat_counter,
                )
                observation = self._observation_text(result)
                messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=observation,
                    )
                )
                tool_calls += 1
                if finish_payload is not None:
                    final = finish_payload
                    outcome = "completed"
                    break
                if blocked_reason:
                    stop_reason = blocked_reason
                    break
                if result.ok:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 4:
                        stop_reason = "too many consecutive tool failures"
                        break
            if final is not None:
                break
            if stop_reason:
                outcome, outcome_detail = "budget_exhausted", stop_reason
                self.bus.emit("limit_hit", limit=stop_reason)
                break

        # 3) RESULT -----------------------------------------------------
        if final is None:
            final = {
                "summary": (
                    "The run stopped before completing "
                    f"({outcome_detail or outcome}). No final answer was produced; "
                    "the transcript above shows how far it got."
                ),
                "verified": False,
                "open_questions": [task],
                "sources": [],
            }
        result = RunResult(
            task=task,
            outcome=outcome,
            outcome_detail=outcome_detail,
            final_answer=final["summary"],
            plan=plan,
            verified=bool(final.get("verified")),
            sources=final.get("sources") or [],
            open_questions=final.get("open_questions") or [],
            usage=build_usage_report(self.router.usage_summary(), tool_calls),
            duration_s=time.monotonic() - started,
            steps=steps,
            events_log=[e.to_dict() for e in self.bus.events],
        )
        self.bus.emit(
            "run_finished",
            outcome=outcome,
            steps=steps,
            tool_calls=tool_calls,
            cost_usd=result.usage.cost_usd_est,
        )
        return result

    # ------------------------------------------------------------------
    def _tool_specs(self) -> list[ToolSpec]:
        specs = []
        for spec in self.registry.specs():
            fn = spec["function"]
            specs.append(
                ToolSpec(
                    name=fn["name"],
                    description=fn["description"],
                    parameters=fn.get("parameters", {}),
                )
            )
        return specs

    def _check_budgets(self, steps: int, tool_calls: int, started: float) -> str | None:
        s = self.settings
        if steps >= s.max_steps:
            return f"step limit ({s.max_steps}) reached"
        if tool_calls >= s.max_tool_calls:
            return f"tool-call limit ({s.max_tool_calls}) reached"
        if time.monotonic() - started > s.max_time_s:
            return f"time limit ({s.max_time_s:.0f}s) reached"
        tokens = self.router.total_usage.input_tokens + self.router.total_usage.output_tokens
        if tokens >= s.max_tokens:
            return f"token limit ({s.max_tokens}) reached"
        if self.router.cost_usd_est >= s.max_cost_usd:
            return f"cost limit (${s.max_cost_usd:.2f}) reached"
        return None

    @staticmethod
    def _observation_text(result: ToolResult) -> str:
        if result.ok:
            return result.content
        kind = result.error_kind or "tool_failure"
        return f"[ERROR: {kind}] {result.content}"

    # ------------------------------------------------------------------
    async def _execute_one(
        self,
        tc: ToolCall,
        *,
        approval_cb: ApprovalCallback | None,
        repeat_counter: Counter[tuple[str, str]],
    ) -> tuple[ToolResult, dict[str, Any] | None, str | None]:
        """Run one tool call through the gates.

        Returns (result, finish_payload, blocked_reason):
        - finish_payload is set only when the `finish` tool was called
        - blocked_reason is set when the run must stop (loop detected)
        """
        tool = self.registry.get(tc.name)
        if tool is None:
            return (
                ToolResult.error(
                    ErrorKind.INVALID_INPUT,
                    f"Unknown tool '{tc.name}'. Available: "
                    f"{', '.join(self.registry.names())}",
                ),
                None,
                None,
            )

        # approval gate (fail-safe: no channel -> deny)
        needs_approval = tool.permission in (
            Permission.WRITES_EXTERNAL,
            Permission.DESTRUCTIVE,
        )
        if needs_approval and self.settings.require_approval:
            self.bus.emit(
                "approval_requested", tool=tc.name, args_digest=tc.arguments[:120]
            )
            approved = bool(approval_cb and approval_cb(tc.name, tc.arguments))
            self.bus.emit("approval_result", tool=tc.name, approved=approved)
            if not approved:
                return (
                    ToolResult.error(
                        ErrorKind.NEEDS_APPROVAL,
                        f"The user did not approve '{tc.name}'. Do not retry it; "
                        "adapt: skip the action, or finish with an honest note.",
                    ),
                    None,
                    None,
                )

        # loop detection on identical repeated calls
        key = (tc.name, tc.arguments)
        repeat_counter[key] += 1
        if repeat_counter[key] == 3:
            self.bus.emit("notice", tool=tc.name, reason="repeated identical call x3")
            return (
                ToolResult.error(
                    ErrorKind.INVALID_INPUT,
                    "You already called this tool with identical arguments twice and "
                    "it did not help. Change your approach or call `finish`.",
                ),
                None,
                None,
            )
        if repeat_counter[key] >= 4:
            return (
                ToolResult.error(
                    ErrorKind.IMPOSSIBLE, "Repeated identical call blocked."
                ),
                None,
                f"loop detected: repeated identical calls to '{tc.name}'",
            )

        result = await self.registry.execute(tc.name, tc.arguments)
        if result.control == "finish":
            data = result.data or {}
            return (
                result,
                {
                    "summary": result.content,
                    "verified": data.get("verified", False),
                    "open_questions": data.get("open_questions", []),
                    "sources": data.get("sources", []),
                },
                None,
            )
        return result, None, None
