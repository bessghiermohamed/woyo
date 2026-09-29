"""schedule_task / list_scheduled_tasks / cancel_scheduled_task (v0.8).

The follow-through tools: an agent that says "I will do X later" must commit
it here, or the promise is empty words. A scheduled job is a row in
woyo.sqlite3 that the chat frontend's job loop executes when due and then
reports back in the chat that created it — success or failure, the user
always hears the outcome.

Permission notes (ADR-15):
- Creating/cancelling a job is SANDBOXED, not WRITES_EXTERNAL: the effect is
  contained to the bot's own database and the owning chat (the job runs as
  an ordinary turn of that chat with the same approval gates as a typed
  message). The tool's result is visible in the transcript, and the tool
  description instructs the agent to state the job id and time to the user.
- Ownership is structural: list/cancel only ever see the calling chat's
  jobs; a chat cannot touch another chat's commitments.
- Limits: active jobs per chat (max_scheduled_jobs), scheduling horizon
  (max_schedule_horizon_days), and no timestamps in the past.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, model_validator

from woyo.config import Settings
from woyo.errors import ErrorKind
from woyo.store.jobs import JobStore, describe_run_at
from woyo.tools.base import Permission, Tool, ToolResult

#: grace window: a run_at slightly in the past still fires immediately
_PAST_GRACE_S = 300.0


def _parse_run_at(
    raw: str, tz_name: str
) -> tuple[datetime | None, str | None]:
    """Parse an absolute local timestamp -> (aware UTC datetime, error)."""
    try:
        dt = datetime.fromisoformat(raw.strip())
    except ValueError:
        return None, (
            f"Could not parse run_at {raw!r} — use ISO format like "
            "'2026-09-30 21:00' or '2026-09-30T21:00:30'."
        )
    if dt.tzinfo is None:
        try:
            dt = dt.replace(tzinfo=ZoneInfo(tz_name))
        except Exception:  # noqa: BLE001 — bad tz config falls back to UTC
            dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC), None


class ScheduleTaskArgs(BaseModel):
    title: str = Field(
        min_length=1, max_length=120,
        description="Short name for the commitment, shown to the user",
    )
    prompt: str = Field(
        min_length=1, max_length=4000,
        description="The task to execute at the scheduled time, as a "
        "self-contained instruction (it runs as a fresh agent turn)",
    )
    delay_minutes: float | None = Field(
        default=None, gt=0, le=60 * 24 * 366,
        description="Relative delay in minutes (e.g. 90 = in 1.5 hours)",
    )
    run_at: str | None = Field(
        default=None, max_length=40,
        description="Absolute time in the user's timezone, ISO format "
        "(e.g. '2026-09-30 21:00')",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> ScheduleTaskArgs:
        if (self.delay_minutes is None) == (self.run_at is None):
            raise ValueError(
                "Give exactly one of delay_minutes or run_at."
            )
        return self


class ScheduleTaskTool(Tool):
    name = "schedule_task"
    description = (
        "Commit to a future action — the ONLY real form of 'I will do it "
        "later'. The task runs by itself at the scheduled time and the "
        "result (or failure) is reported back in this chat automatically. "
        "Give delay_minutes (e.g. 90 for 'in 1.5 hours') or an absolute "
        "run_at in the user's timezone. After scheduling, ALWAYS tell the "
        "user the job id and when it will run — that confirmation is the "
        "promise."
    )
    permission = Permission.SANDBOXED  # own db + owning chat (see ADR-14)
    Args = ScheduleTaskArgs

    def __init__(self, settings: Settings, store: JobStore, chat_id: int):
        self._settings = settings
        self._store = store
        self._chat_id = chat_id

    async def run(self, args: ScheduleTaskArgs) -> ToolResult:
        now = datetime.now(UTC)
        if args.delay_minutes is not None:
            run_at_dt = now + timedelta(minutes=args.delay_minutes)
        else:
            assert args.run_at is not None  # validator guarantees it
            run_at_dt, err = _parse_run_at(args.run_at, self._settings.timezone)
            if run_at_dt is None:
                return ToolResult.error(ErrorKind.INVALID_INPUT, err or "bad run_at")
        if run_at_dt < now - timedelta(seconds=_PAST_GRACE_S):
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"That time ({run_at_dt.isoformat(timespec='minutes')}) is "
                "already in the past. Current UTC time: "
                f"{now.isoformat(timespec='minutes')}. Use a future time or "
                "delay_minutes.",
            )
        horizon = timedelta(days=self._settings.max_schedule_horizon_days)
        if run_at_dt > now + horizon:
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"That is further ahead than the "
                f"{self._settings.max_schedule_horizon_days}-day scheduling "
                "horizon. Split the work or schedule closer.",
            )
        if self._store.count_active(self._chat_id) >= self._settings.max_scheduled_jobs:
            return ToolResult.error(
                ErrorKind.INVALID_INPUT,
                f"This chat already has {self._settings.max_scheduled_jobs} "
                "active scheduled tasks (the limit). Cancel one with "
                "cancel_scheduled_task before scheduling more.",
            )
        job = self._store.create(
            self._chat_id, args.title, args.prompt,
            run_at_dt.isoformat(timespec="seconds"),
        )
        when = describe_run_at(job.run_at, tz=self._settings.timezone)
        return ToolResult.ok_result(
            f"Scheduled (job #{job.id}): '{job.title}' will run {when}. "
            "The result — or a failure notice — will be posted in this chat "
            "automatically. Tell the user the job id and the time now.",
            data={"job_id": job.id, "run_at": job.run_at},
        )


class ListScheduledTasksArgs(BaseModel):
    pass


class ListScheduledTasksTool(Tool):
    name = "list_scheduled_tasks"
    description = (
        "Show this chat's scheduled tasks: pending ones with their due time "
        "and id, plus recently finished ones with their outcome. Use it when "
        "the user asks what is pending or what happened to a commitment."
    )
    permission = Permission.READ_ONLY
    Args = ListScheduledTasksArgs

    def __init__(self, settings: Settings, store: JobStore, chat_id: int):
        self._settings = settings
        self._store = store
        self._chat_id = chat_id

    async def run(self, args: ListScheduledTasksArgs) -> ToolResult:
        jobs = self._store.recent_for_chat(self._chat_id, limit=15)
        if not jobs:
            return ToolResult.ok_result(
                "No scheduled tasks in this chat. Create one with "
                "schedule_task when you commit to a future action."
            )
        lines = []
        active = [j for j in jobs if j.status in ("scheduled", "running")]
        finished = [j for j in jobs if j.status not in ("scheduled", "running")]
        if active:
            lines.append(f"Pending ({len(active)}):")
            for j in active:
                when = describe_run_at(j.run_at, tz=self._settings.timezone)
                lines.append(
                    f"- #{j.id} '{j.title}' — {when}"
                    + (" [running now]" if j.status == "running" else "")
                )
        if finished:
            lines.append(f"Recent results ({len(finished)}):")
            for j in finished[:5]:
                if j.status == "done":
                    preview = (j.result_preview or "").replace("\n", " ")[:100]
                    lines.append(f"- #{j.id} '{j.title}' — done: {preview}")
                elif j.status == "failed":
                    lines.append(
                        f"- #{j.id} '{j.title}' — FAILED: "
                        f"{(j.error or 'unknown error')[:120]}"
                    )
                else:
                    lines.append(f"- #{j.id} '{j.title}' — {j.status}")
        return ToolResult.ok_result("\n".join(lines))


class CancelScheduledTaskArgs(BaseModel):
    job_id: int = Field(description="Id of the scheduled task to cancel")


class CancelScheduledTaskTool(Tool):
    name = "cancel_scheduled_task"
    description = (
        "Cancel one of this chat's scheduled tasks by its job id (find ids "
        "with list_scheduled_tasks). Only tasks from this chat can be "
        "cancelled."
    )
    permission = Permission.SANDBOXED
    Args = CancelScheduledTaskArgs

    def __init__(self, store: JobStore, chat_id: int):
        self._store = store
        self._chat_id = chat_id

    async def run(self, args: CancelScheduledTaskArgs) -> ToolResult:
        ok, message = self._store.cancel(args.job_id, self._chat_id)
        if not ok:
            return ToolResult.error(ErrorKind.INVALID_INPUT, message)
        return ToolResult.ok_result(message, data={"job_id": args.job_id})


__all__ = [
    "ScheduleTaskTool",
    "ListScheduledTasksTool",
    "CancelScheduledTaskTool",
]
