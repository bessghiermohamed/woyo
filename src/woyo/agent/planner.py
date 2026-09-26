"""Planner: task + tool catalog -> TaskPlan (with graceful fallback)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from woyo.agent.prompts import PLANNER_SYSTEM, PLANNER_USER
from woyo.events import EventBus
from woyo.models.base import Message
from woyo.models.router import ModelRouter


class PlanStep(BaseModel):
    id: str = "s1"
    description: str
    status: str = "pending"  # pending | in_progress | done | failed


class TaskPlan(BaseModel):
    goal: str
    success_criteria: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)

    def render(self) -> str:
        lines = [f"Goal: {self.goal}"]
        if self.success_criteria:
            lines.append("Success criteria:")
            lines += [f"  - {c}" for c in self.success_criteria]
        lines.append("Steps:")
        lines += [f"  [{s.id}] {s.description} ({s.status})" for s in self.steps]
        return "\n".join(lines)

    @classmethod
    def fallback(cls, task: str) -> TaskPlan:
        return cls(
            goal=task,
            success_criteria=["The user's question is answered or the blockage is explained"],
            steps=[PlanStep(id="s1", description=f"Accomplish: {task}")],
        )


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of model output (fences or bare)."""
    if not text:
        return None
    fenced = None
    if "```" in text:
        for part in text.split("```"):
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                fenced = stripped
                break
    candidates = [fenced] if fenced else []
    depth, start, in_string, escape = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(text[start : i + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


async def create_plan(
    router: ModelRouter, task: str, tool_catalog: str, bus: EventBus | None = None
) -> TaskPlan:
    messages_payload = PLANNER_USER.format(tools=tool_catalog, task=task)

    last_text = ""
    for _attempt in range(2):
        resp = await router.complete(
            "planner",
            messages=[
                Message(role="system", content=PLANNER_SYSTEM),
                Message(role="user", content=messages_payload),
            ],
            tools=[],
            temperature=0.0,
            max_tokens=800,
        )
        last_text = resp.content or ""
        obj = extract_json_object(last_text)
        if obj is not None:
            try:
                plan = TaskPlan.model_validate(obj)
                if plan.steps:
                    if bus:
                        bus.emit(
                            "plan_created",
                            goal=plan.goal,
                            steps=[s.description for s in plan.steps],
                        )
                    return plan
            except ValidationError:
                pass
        # retry once with the error visible
        messages_payload = (
            f"{PLANNER_USER.format(tools=tool_catalog, task=task)}\n\n"
            f"Your previous output was not valid plan JSON:\n{last_text[:500]}\n"
            "Output ONLY the JSON object now."
        )
    plan = TaskPlan.fallback(task)
    if bus:
        bus.emit("plan_fallback", reason="planner output unparseable", task=task[:200])
    return plan
