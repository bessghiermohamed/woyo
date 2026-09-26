"""RunResult: what a completed (or stopped) run reports back."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from woyo.agent.planner import TaskPlan


@dataclass(slots=True)
class UsageReport:
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    cost_usd_est: float = 0.0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    task: str
    outcome: str  # completed | budget_exhausted | failed
    final_answer: str
    plan: TaskPlan | None = None
    verified: bool = False
    sources: list[dict[str, str]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    usage: UsageReport = field(default_factory=UsageReport)
    duration_s: float = 0.0
    steps: int = 0
    outcome_detail: str = ""
    events_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome == "completed"

    def summary(self) -> str:
        lines = [
            f"outcome     : {self.outcome}"
            + (f" ({self.outcome_detail})" if self.outcome_detail else ""),
            f"steps       : {self.steps}",
            f"llm calls   : {self.usage.llm_calls}",
            f"tool calls  : {self.usage.tool_calls}",
            f"tokens      : {self.usage.input_tokens} in / {self.usage.output_tokens} out",
            f"cost (est)  : ${self.usage.cost_usd_est:.4f}",
            f"duration    : {self.duration_s:.1f}s",
            f"verified    : {self.verified}",
        ]
        if self.sources:
            lines.append("sources     :")
            lines += [f"  - {s.get('url', '')} {s.get('title', '')}" for s in self.sources]
        if self.open_questions:
            lines.append("open        :")
            lines += [f"  ? {q}" for q in self.open_questions]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "outcome": self.outcome,
            "outcome_detail": self.outcome_detail,
            "final_answer": self.final_answer,
            "verified": self.verified,
            "sources": self.sources,
            "open_questions": self.open_questions,
            "steps": self.steps,
            "duration_s": round(self.duration_s, 2),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "llm_calls": self.usage.llm_calls,
                "tool_calls": self.usage.tool_calls,
                "cost_usd_est": self.usage.cost_usd_est,
                "by_model": self.usage.by_model,
            },
        }


def build_usage_report(router_summary: dict[str, Any], tool_calls: int) -> UsageReport:
    total = router_summary.get("total", {})
    return UsageReport(
        input_tokens=total.get("input_tokens", 0),
        output_tokens=total.get("output_tokens", 0),
        llm_calls=total.get("calls", 0),
        tool_calls=tool_calls,
        cost_usd_est=router_summary.get("cost_usd_est", 0.0),
        by_model=router_summary.get("by_model", {}),
    )
