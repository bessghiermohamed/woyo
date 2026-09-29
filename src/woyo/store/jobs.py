"""Scheduled jobs: the follow-through layer behind "I will do it later" (v0.8).

A job is a commitment recorded in the database instead of words in a reply:
`schedule_task` inserts a row, and the chat frontend's job loop executes it
when due and reports the outcome back to the chat that created it. Jobs live
in woyo.sqlite3 next to tasks and memories, so they already survive host
rotation through the existing state-sync machinery.

Ownership model (allowlist philosophy):
- A job belongs to the chat that created it; `cancel` and `list` are scoped
  to that chat. One chat can never touch another chat's jobs.
- The prompt stored in a job is agent input at execution time — it runs with
  the exact same budget, sandbox and approval gates as a typed message, in
  the owning chat's session (approval buttons included; timeout = deny).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from woyo.store.db import connect, migrate

JOB_STATUSES = ("scheduled", "running", "done", "failed", "cancelled")
#: statuses that still owe the chat an execution
ACTIVE = ("scheduled", "running")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _parse(ts: str) -> datetime:
    """Parse an ISO timestamp (naive -> UTC)."""
    dt = datetime.fromisoformat(ts)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


@dataclass(slots=True)
class Job:
    id: int
    chat_id: int
    title: str
    prompt: str
    run_at: str
    status: str
    attempts: int
    created_at: str
    updated_at: str | None
    finished_at: str | None
    error: str | None
    result_preview: str | None


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        chat_id=row["chat_id"],
        title=row["title"],
        prompt=row["prompt"],
        run_at=row["run_at"],
        status=row["status"],
        attempts=row["attempts"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        finished_at=row["finished_at"],
        error=row["error"],
        result_preview=row["result_preview"],
    )


class JobStore:
    """CRUD + lifecycle for scheduled jobs (single writer, same file as tasks)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = migrate(connect(self.path))

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    def create(
        self, chat_id: int, title: str, prompt: str, run_at: str
    ) -> Job:
        now = now_iso()
        title = title.strip()[:120] or "(untitled)"
        cur = self.conn.execute(
            "INSERT INTO jobs (chat_id, title, prompt, run_at, status,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
            (chat_id, title, prompt.strip(), run_at, "scheduled", now, now),
        )
        self.conn.commit()
        return self.get(cur.lastrowid)  # type: ignore[return-value]

    def get(self, job_id: int) -> Job | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        return _row_to_job(row) if row else None

    def due(self, now: str | None = None, *, limit: int = 10) -> list[Job]:
        """Scheduled jobs whose time has come, earliest first."""
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE status='scheduled' AND run_at<=?"
            " ORDER BY run_at, id LIMIT ?",
            (now or now_iso(), limit),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def active_for_chat(self, chat_id: int) -> list[Job]:
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE chat_id=? AND status IN ('scheduled',"
            "'running') ORDER BY run_at",
            (chat_id,),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def recent_for_chat(self, chat_id: int, *, limit: int = 10) -> list[Job]:
        """Newest jobs (any status) — the 'what did I commit to' view."""
        rows = self.conn.execute(
            "SELECT * FROM jobs WHERE chat_id=? ORDER BY id DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
        return [_row_to_job(r) for r in rows]

    def count_active(self, chat_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE chat_id=? AND status IN"
            " ('scheduled','running')",
            (chat_id,),
        ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    def mark_running(self, job_id: int) -> None:
        self.conn.execute(
            "UPDATE jobs SET status='running', updated_at=?,"
            " attempts=attempts+1 WHERE id=?",
            (now_iso(), job_id),
        )
        self.conn.commit()

    def finish(
        self, job_id: int, status: str, *,
        result_preview: str | None = None, error: str | None = None,
    ) -> None:
        if status not in ("done", "failed", "cancelled"):
            raise ValueError(f"unknown finish status {status!r}")
        self.conn.execute(
            "UPDATE jobs SET status=?, result_preview=?, error=?,"
            " finished_at=?, updated_at=? WHERE id=?",
            (status, (result_preview or "")[:500] or None,
             (error or "")[:500] or None, now_iso(), now_iso(), job_id),
        )
        self.conn.commit()

    def reschedule(self, job_id: int, run_at: str | None = None) -> None:
        """Put a job back on the queue (rotation retry / boot recovery)."""
        self.conn.execute(
            "UPDATE jobs SET status='scheduled', run_at=?, finished_at=NULL,"
            " updated_at=? WHERE id=?",
            (run_at or now_iso(), now_iso(), job_id),
        )
        self.conn.commit()

    def cancel(self, job_id: int, chat_id: int) -> tuple[bool, str]:
        """Cancel a job — only from the chat that owns it."""
        job = self.get(job_id)
        if job is None or job.chat_id != chat_id:
            return False, f"No scheduled task #{job_id} in this chat."
        if job.status not in ACTIVE:
            return False, f"Task #{job_id} already finished ({job.status})."
        self.finish(job_id, "cancelled")
        return True, f"Cancelled: {job.title}"

    # ------------------------------------------------------------------
    def recover(
        self, *, max_attempts: int = 3
    ) -> list[int]:
        """Boot-time sweep: `running` rows from a killed host go back to
        the queue (or fail out after too many rotations)."""
        rows = self.conn.execute(
            "SELECT id, attempts FROM jobs WHERE status='running'"
        ).fetchall()
        ids: list[int] = []
        for row in rows:
            ids.append(row["id"])
            if row["attempts"] >= max_attempts:
                self.finish(
                    row["id"], "failed",
                    error="interrupted by host rotation too many times",
                )
            else:
                self.reschedule(row["id"])
        return ids

    def prune(self, *, days: int) -> int:
        """Delete finished jobs older than `days`."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        cur = self.conn.execute(
            "DELETE FROM jobs WHERE status IN ('done','cancelled','failed')"
            " AND COALESCE(finished_at, updated_at, created_at) < ?",
            (cutoff,),
        )
        self.conn.commit()
        return cur.rowcount


def describe_run_at(run_at: str, *, tz: str = "UTC") -> str:
    """Human-readable 'when' for prompts and confirmations."""
    try:
        from zoneinfo import ZoneInfo

        local = _parse(run_at).astimezone(ZoneInfo(tz))
        local_txt = local.strftime("%Y-%m-%d %H:%M (%Z)")
    except Exception:  # noqa: BLE001 — formatting must never raise
        local_txt = run_at
    delta = _parse(run_at) - datetime.now(UTC)
    if delta.total_seconds() >= 0:
        mins = int(delta.total_seconds() // 60)
        rel = f"in {mins // 60}h {mins % 60}m" if mins >= 60 else f"in {mins}m"
    else:
        mins = int(-delta.total_seconds() // 60)
        rel = f"{mins}m ago"
    return f"{local_txt}, {rel}"


__all__ = ["Job", "JobStore", "JOB_STATUSES", "describe_run_at", "now_iso"]
