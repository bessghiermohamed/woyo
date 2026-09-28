"""The woyo agent loop: plan -> act -> observe -> adapt -> verify -> finish.

Design notes (see ARCHITECTURE.md §3):
- Native tool-calling only; arguments validated by the ToolRegistry.
- Budgets enforced every iteration; exhaustion yields an honest partial result.
- Loop detection on identical repeated calls; replan pressure on failures.
- Approval gates for writes_external/destructive tools — denied when no
  approval channel exists (fail safe).
- Phase 3: the loop's whole mutable state lives in RunState and can be
  checkpointed after every step (resume after restart) and steered between
  steps by an external control channel (pause / cancel).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from woyo.agent.citations import collect_observed_urls, normalize_url, verify_sources
from woyo.agent.planner import create_plan
from woyo.agent.prompts import executor_system_prompt, render_memory_section
from woyo.agent.results import RunResult, build_usage_report
from woyo.agent.state import RunState
from woyo.config import Settings
from woyo.errors import AgentError, ErrorKind
from woyo.events import EventBus
from woyo.memory.working import compact_context
from woyo.models.base import Message, ToolCall, ToolSpec
from woyo.models.router import ModelRouter
from woyo.tools.base import Permission, ToolRegistry, ToolResult

if TYPE_CHECKING:
    from woyo.memory.longterm import MemoryStore

ApprovalCallback = Callable[[str, str], bool]  # (tool_name, args_json) -> approved
CheckpointCallback = Callable[[dict[str, Any]], None]  # RunState.to_dict() -> stored
ControlPoll = Callable[[], str | None]  # -> "pause" | "cancel" | None

_SEARCH_TOOLS = {"web_search", "crawl_site"}


class Agent:
    """Runs one task at a time through the supervised loop."""

    def __init__(
        self,
        settings: Settings,
        router: ModelRouter,
        registry: ToolRegistry,
        bus: EventBus | None = None,
        *,
        memory: MemoryStore | None = None,
    ):
        self.settings = settings
        self.router = router
        self.registry = registry
        self.bus = bus or EventBus()
        self.memory = memory

    # ------------------------------------------------------------------
    async def run(
        self,
        task: str,
        *,
        approval_cb: ApprovalCallback | None = None,
        resume_state: dict[str, Any] | None = None,
        checkpoint_cb: CheckpointCallback | None = None,
        control_poll: ControlPoll | None = None,
    ) -> RunResult:
        task = task.strip()
        if not task:
            raise AgentError("empty task")
        started = time.monotonic()
        self.bus.emit("run_started", task=task[:300])

        if resume_state is not None:
            state = RunState.from_dict(resume_state)
            if state.task.strip() != task:
                raise AgentError(
                    "resume state belongs to a different task; retry from scratch"
                )
            plan = None
            self.bus.emit("notice", reason="resuming from checkpoint", steps_done=state.steps)
        else:
            # 1) PLAN --------------------------------------------------
            catalog = "\n".join(
                f"- {item['name']} ({item['permission']}): {item['description']}"
                for item in self.registry.safety_overview()
            )
            plan = await create_plan(self.router, task, catalog, self.bus)
            budget_text = (
                f"max {self.settings.max_steps} steps, {self.settings.max_tool_calls} tool "
                f"calls, {self.settings.max_time_s:.0f}s, ~${self.settings.max_cost_usd:.2f}"
            )
            memory_text = ""
            if self.memory is not None:
                try:
                    hits = await self.memory.recall(task, k=self.settings.memory_recall_k)
                    memory_text = render_memory_section(hits) if hits else ""
                except Exception:  # noqa: BLE001 — recall must never block a run
                    memory_text = ""
            system = executor_system_prompt(
                today=datetime.now(UTC).strftime("%Y-%m-%d"),
                timezone_name=self.settings.timezone,
                task=task,
                plan_text=plan.render(),
                budget_text=budget_text,
                memory_text=memory_text,
            )
            state = RunState(
                task=task,
                system_prompt=system,
                plan_text=plan.render(),
                messages=[
                    Message(role="system", content=system),
                    Message(role="user", content=f"Task: {task}"),
                ],
            )

        prior_elapsed = state.elapsed_s

        def elapsed_now() -> float:
            return prior_elapsed + (time.monotonic() - started)

        # 2) EXECUTE LOOP ----------------------------------------------
        outcome = "failed"
        outcome_detail = ""
        final: dict[str, Any] | None = None

        while final is None:
            # --- external control (pause/cancel between steps) --------
            action = control_poll() if control_poll is not None else None
            if action == "cancel":
                outcome, outcome_detail = "cancelled", "cancelled by user"
                self.bus.emit("limit_hit", limit="cancelled by user")
                break
            if action == "pause":
                outcome, outcome_detail = "paused", "paused by user"
                self.bus.emit("limit_hit", limit="paused by user")
                break

            # --- budget gate ------------------------------------------
            budget_reason = self._check_budgets(state, elapsed_now())
            if budget_reason is not None:
                outcome, outcome_detail = "budget_exhausted", budget_reason
                self.bus.emit("limit_hit", limit=budget_reason)
                break

            # --- executor call ----------------------------------------
            compact_context(state.messages, self.settings.context_soft_limit_tokens)
            state.steps += 1
            resp = await self.router.complete(
                "executor",
                messages=state.messages,
                tools=self._tool_specs(),
            )
            state.messages.append(
                Message(
                    role="assistant",
                    content=resp.content,
                    tool_calls=resp.tool_calls or None,
                )
            )

            if not resp.tool_calls:
                # text-only reply: task mode nudges once and accepts the second
                # as final; chat mode (direct_text_replies) accepts prose directly
                state.text_only_strikes += 1
                if state.text_only_strikes >= 2 or self.settings.direct_text_replies:
                    final = {
                        "summary": resp.content or "(no content)",
                        "verified": False,
                        "open_questions": [],
                        "sources": [],
                    }
                    outcome = "completed"
                    outcome_detail = (
                        "direct reply" if self.settings.direct_text_replies else
                        "text-only final answer"
                    )
                    break
                state.messages.append(
                    Message(
                        role="user",
                        content=(
                            "Respond with TOOL CALLS, not prose. If the task is "
                            "done, call the `finish` tool NOW with your final "
                            "summary, `verified`, and `sources`. Do not describe "
                            "what you will do — do it."
                        ),
                    )
                )
                self._checkpoint(state, elapsed_now(), checkpoint_cb)
                continue

            state.text_only_strikes = 0
            stop_reason: str | None = None
            for tc in resp.tool_calls:
                # per-task search budget (Phase 2)
                if (tc.name in _SEARCH_TOOLS
                        and state.search_calls >= self.settings.max_search_calls):
                    self.bus.emit(
                        "limit_hit",
                        limit=f"search budget ({self.settings.max_search_calls} calls)",
                    )
                    result = ToolResult.error(
                        ErrorKind.INVALID_INPUT,
                        "Search budget for this task is exhausted. Do NOT search "
                        "or crawl again — work with the material already in "
                        "your context and finish.",
                    )
                    state.messages.append(
                        Message(
                            role="tool",
                            tool_call_id=tc.id,
                            name=tc.name,
                            content=self._observation_text(result),
                        )
                    )
                    state.tool_calls += 1
                    continue

                result, finish_payload, blocked_reason = await self._execute_one(
                    tc,
                    approval_cb=approval_cb,
                    repeat_counter=state.repeat_counter,
                )
                if tc.name in _SEARCH_TOOLS:
                    state.search_calls += 1
                state.observed_urls.update(
                    normalize_url(u)
                    for u in collect_observed_urls(result)
                    if normalize_url(u)
                )
                observation = self._observation_text(result)
                state.messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=observation,
                    )
                )
                state.tool_calls += 1
                if finish_payload is not None:
                    final = finish_payload
                    outcome = "completed"
                    break
                if blocked_reason:
                    stop_reason = blocked_reason
                    break
                if result.ok:
                    state.consecutive_failures = 0
                else:
                    state.consecutive_failures += 1
                    if state.consecutive_failures >= 4:
                        stop_reason = "too many consecutive tool failures"
                        break

            # --- checkpoint after every executor step -----------------
            self._checkpoint(state, elapsed_now(), checkpoint_cb)

            if final is not None:
                break
            if stop_reason:
                outcome, outcome_detail = "budget_exhausted", stop_reason
                self.bus.emit("limit_hit", limit=stop_reason)
                break

        # 3) RESULT -----------------------------------------------------
        if final is None:
            if outcome == "paused":
                final = {
                    "summary": (
                        "The run was paused by the user before completing. "
                        "Progress so far is checkpointed; resume to continue."
                    ),
                    "verified": False,
                    "open_questions": [task],
                    "sources": [],
                }
            elif outcome == "cancelled":
                final = {
                    "summary": (
                        "The run was cancelled by the user. No final answer was "
                        "produced; the transcript shows how far it got."
                    ),
                    "verified": False,
                    "open_questions": [],
                    "sources": [],
                }
            else:
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

        # citation verification: claimed sources must have been observed
        verified_sources, dropped_sources = verify_sources(
            final.get("sources"), state.observed_urls
        )
        if dropped_sources:
            self.bus.emit(
                "citation_flagged",
                dropped=[s.get("url", "") for s in dropped_sources][:5],
            )
            final["sources"] = verified_sources
            final.setdefault("open_questions", []).append(
                "Some cited sources could not be verified against the pages "
                "actually seen this run and were removed."
            )

        result = RunResult(
            task=task,
            outcome=outcome,
            outcome_detail=outcome_detail,
            final_answer=final["summary"],
            plan=plan,
            verified=bool(final.get("verified")),
            sources=final.get("sources") or [],
            sources_dropped=[
                s.get("url", "") for s in dropped_sources if isinstance(s, dict)
            ],
            open_questions=final.get("open_questions") or [],
            usage=build_usage_report(self.router.usage_summary(), state.tool_calls),
            duration_s=elapsed_now(),
            steps=state.steps,
            events_log=[e.to_dict() for e in self.bus.events],
        )
        self.bus.emit(
            "run_finished",
            outcome=outcome,
            steps=state.steps,
            tool_calls=state.tool_calls,
            cost_usd=result.usage.cost_usd_est,
        )
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _checkpoint(
        state: RunState, elapsed: float, cb: CheckpointCallback | None
    ) -> None:
        if cb is None:
            return
        state.elapsed_s = elapsed
        try:
            cb(state.to_dict())
        except Exception:  # noqa: BLE001 — checkpointing must never kill a run
            pass

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

    def _check_budgets(self, state: RunState, elapsed_s: float) -> str | None:
        s = self.settings
        if state.steps >= s.max_steps:
            return f"step limit ({s.max_steps}) reached"
        if state.tool_calls >= s.max_tool_calls:
            return f"tool-call limit ({s.max_tool_calls}) reached"
        if elapsed_s > s.max_time_s:
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
        repeat_counter,
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
