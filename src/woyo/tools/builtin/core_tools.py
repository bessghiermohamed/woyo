"""Small built-in tools: calculate, now, ask_user, finish."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from woyo.errors import ErrorKind
from woyo.tools.base import Tool, ToolResult, UserInteraction

# python_exec lives in sandbox.py since v0.5; re-exported here for compat
from woyo.tools.builtin.sandbox import CodeArgs, PythonExecTool  # noqa: F401

# --- calculate ---------------------------------------------------------------

class CalculateArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=500,
                            description="Math expression, e.g. '3.5*12 + 2^10'")


class CalculateTool(Tool):
    name = "calculate"
    description = (
        "Evaluate a math expression exactly (arithmetic, powers, sqrt, "
        "percent). Use instead of doing arithmetic in your head."
    )
    timeout_s = 5.0
    Args = CalculateArgs

    async def run(self, args: CalculateArgs) -> ToolResult:
        from simpleeval import simple_eval

        try:
            value = simple_eval(args.expression, names={}, functions={})
        except Exception as exc:  # noqa: BLE001 — user-expression errors
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"Could not evaluate expression: {exc}",
            )
        pretty = f"{value:.10g}" if isinstance(value, float) else str(value)
        return ToolResult.ok_result(f"{args.expression} = {pretty}")


# --- now ---------------------------------------------------------------------

class NowArgs(BaseModel):
    pass


class NowTool(Tool):
    name = "now"
    description = (
        "Current date and time (UTC and the user's timezone). Use it whenever "
        "a task depends on 'today' or recency — never guess the date."
    )
    timeout_s = 3.0
    Args = NowArgs

    def __init__(self, tz: str = "UTC"):
        self._tz = tz

    async def run(self, args: NowArgs) -> ToolResult:
        utc = datetime.now(UTC)
        try:
            local = utc.astimezone(ZoneInfo(self._tz))
            local_str = local.strftime("%Y-%m-%d %H:%M (%Z)")
        except Exception:  # noqa: BLE001 — bad tz config shouldn't break the tool
            local_str = f"invalid timezone '{self._tz}'"
        return ToolResult.ok_result(
            f"UTC: {utc.strftime('%Y-%m-%d %H:%M')}\nUser timezone ({self._tz}): {local_str}"
        )


# --- ask_user ----------------------------------------------------------------

class AskUserArgs(BaseModel):
    question: str = Field(min_length=4, max_length=2000)
    options: list[str] | None = Field(
        default=None, max_length=6,
        description="Optional short list of choices to present",
    )


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "Ask the human a question when you are blocked on information or a "
        "decision only they can make. Do not use it for things you can find "
        "out yourself with tools."
    )
    timeout_s = 3600.0  # humans are slow
    Args = AskUserArgs

    def __init__(self, interaction: UserInteraction | None = None):
        self._interaction = interaction

    async def run(self, args: AskUserArgs) -> ToolResult:
        if self._interaction is None:
            return ToolResult.error(
                ErrorKind.MISSING_INFO,
                "No interactive user is available in this mode. Proceed with "
                "what you know and note the question in your final summary.",
            )
        if args.options:
            question = args.question + " Options: " + " / ".join(args.options)
        else:
            question = args.question
        answer = self._interaction.ask(question)
        return ToolResult.ok_result(f"User answered: {answer}")


# --- finish ------------------------------------------------------------------
class Source(BaseModel):
    title: str = ""
    url: str = ""


class FinishArgs(BaseModel):
    summary: str = Field(min_length=10, description="Final answer for the user")
    verified: bool = Field(
        default=False,
        description="True ONLY if you checked the task's success criteria",
    )
    open_questions: list[str] = Field(
        default_factory=list, description="What remains unknown or uncertain"
    )
    sources: list[Source] = Field(
        default_factory=list, description="URLs actually used for the answer"
    )


class FinishTool(Tool):
    name = "finish"
    description = (
        "End the task. Provide: a clear final summary answering the user's "
        "goal; `verified` (true only if you actually checked the success "
        "criteria); `sources` (URLs you actually used — never invent them); "
        "`open_questions` (honest unknowns)."
    )
    timeout_s = 5.0
    Args = FinishArgs

    async def run(self, args: FinishArgs) -> ToolResult:
        return ToolResult.ok_result(
            args.summary,
            control="finish",
            data={
                "verified": args.verified,
                "open_questions": args.open_questions,
                "sources": [s.model_dump() for s in args.sources],
            },
        )
