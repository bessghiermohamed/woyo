"""Task queue + state machine persistence (Phase 3).

Statuses:
    pending -> running -> (waiting_approval -> running)*
    -> completed | failed | cancelled | paused
    paused/running rows carry a checkpoint and can be resumed;
    retry re-queues a finished task (attempts += 1, fresh checkpoint).

Control crosses process boundaries through the `control` column:
another process (or the CLI) writes 'pause' / 'cancel' and the running
TaskRunner polls it between steps and acts.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from woyo.store.db import connect, migrate

STATUSES = (
    "pending",
    "running",
    "waiting_approval",
    "paused",
    "completed",
    "failed",
    "cancelled",
)
#: statuses a `run` may pick up (interrupted `running` needs staleness check)
RESUMABLE = ("pending", "paused", "waiting_approval")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(slots=True)
class Task:
    id: int
    title: str
    prompt: str
    status: str
    source: str
    created_at: str
    updated_at: str
    finished_at: str | None
    heartbeat_at: str | None
    attempts: int
    error: str | None
    result: dict[str, Any] | None
    checkpoint: dict[str, Any] | None
    meta: dict[str, Any] | None

    @property
    def resumable(self) -> bool:
        return self.status in RESUMABLE


def _row_to_task(row: sqlite3.Row) -> Task:
    def loads(raw: str | None) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    return Task(
        id=row["id"],
        title=row["title"],
        prompt=row["prompt"],
        status=row["status"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        finished_at=row["finished_at"],
        heartbeat_at=row["heartbeat_at"],
        attempts=row["attempts"],
        error=row["error"],
        result=loads(row["result_json"]),
        checkpoint=loads(row["checkpoint_json"]),
        meta=loads(row["meta_json"]),
    )


class TaskStore:
    """CRUD + state transitions for tasks, with an event sink per task."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = migrate(connect(self.path))

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    def create(self, prompt: str, *, title: str | None = None,
               source: str = "cli", meta: dict | None = None) -> Task:
        now = _now()
        title = (title or prompt).strip()[:120] or "(untitled)"
        cur = self.conn.execute(
            "INSERT INTO tasks (title, prompt, status, source, created_at,"
            " updated_at, attempts, meta_json) VALUES (?,?,?,?,?,?,0,?)",
            (title, prompt.strip(), "pending", source, now, now,
             json.dumps(meta) if meta else None),
        )
        self.conn.commit()
        return self.get(cur.lastrowid)  # type: ignore[return-value]

    def get(self, task_id: int) -> Task | None:
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        return _row_to_task(row) if row else None

    def list(self, *, status: str | None = None, limit: int = 30) -> list[Task]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY id DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM tasks ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_task(r) for r in rows]

    # ------------------------------------------------------------------
    def set_status(self, task_id: int, status: str, *,
                   error: str | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        finished = _now() if status in ("completed", "failed", "cancelled") else None
        self.conn.execute(
            "UPDATE tasks SET status=?, error=?, updated_at=?, finished_at=COALESCE(?, finished_at)"
            " WHERE id=?",
            (status, error, _now(), finished, task_id),
        )
        self.conn.commit()

    def mark_running(self, task_id: int) -> None:
        self.conn.execute(
            "UPDATE tasks SET status='running', updated_at=?, heartbeat_at=?,"
            " attempts=attempts+1 WHERE id=?",
            (_now(), _now(), task_id),
        )
        self.conn.commit()

    def heartbeat(self, task_id: int) -> None:
        self.conn.execute(
            "UPDATE tasks SET heartbeat_at=? WHERE id=?", (_now(), task_id)
        )
        self.conn.commit()

    def save_result(self, task_id: int, result: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE tasks SET result_json=?, updated_at=? WHERE id=?",
            (json.dumps(result, default=str), _now(), task_id),
        )
        self.conn.commit()

    def save_checkpoint(self, task_id: int, checkpoint: dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE tasks SET checkpoint_json=?, heartbeat_at=?, updated_at=? WHERE id=?",
            (json.dumps(checkpoint, default=str), _now(), _now(), task_id),
        )
        self.conn.commit()

    def clear_checkpoint(self, task_id: int) -> None:
        self.conn.execute(
            "UPDATE tasks SET checkpoint_json=NULL WHERE id=?", (task_id,)
        )
        self.conn.commit()

    def retry(self, task_id: int) -> Task | None:
        """Re-queue a finished/failed/cancelled task from scratch."""
        task = self.get(task_id)
        if task is None or task.status not in ("completed", "failed", "cancelled"):
            return task
        self.conn.execute(
            "UPDATE tasks SET status='pending', error=NULL, result_json=NULL,"
            " checkpoint_json=NULL, finished_at=NULL, updated_at=? WHERE id=?",
            (_now(), task_id),
        )
        self.conn.commit()
        return self.get(task_id)

    # ------------------------------------------------------------------
    # cross-process control channel
    def request_control(self, task_id: int, action: str) -> None:
        if action not in ("pause", "cancel", "resume"):
            raise ValueError(f"unknown control action {action!r}")
        self.conn.execute(
            "UPDATE tasks SET control=?, updated_at=? WHERE id=?",
            (action, _now(), task_id),
        )
        self.conn.commit()

    def poll_control(self, task_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT control FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        return row["control"] if row else None

    def clear_control(self, task_id: int) -> None:
        self.conn.execute(
            "UPDATE tasks SET control=NULL WHERE id=?", (task_id,)
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    def recover_stale(self, stale_minutes: int) -> list[int]:
        """Flip interrupted `running` rows to paused so they can be resumed."""
        cutoff = (datetime.now(UTC) - timedelta(minutes=stale_minutes)).isoformat(
            timespec="seconds"
        )
        rows = self.conn.execute(
            "SELECT id FROM tasks WHERE status='running' AND"
            " COALESCE(heartbeat_at, updated_at, created_at) < ?",
            (cutoff,),
        ).fetchall()
        ids = [r["id"] for r in rows]
        for task_id in ids:
            self.set_status(task_id, "paused", error="interrupted (stale heartbeat)")
        return ids

    def append_event(self, task_id: int, seq: int, kind: str, data: dict) -> None:
        self.conn.execute(
            "INSERT INTO task_events (task_id, seq, kind, data_json, created_at)"
            " VALUES (?,?,?,?,?)",
            (task_id, seq, kind, json.dumps(data, default=str), _now()),
        )
        self.conn.commit()

    def events(self, task_id: int, limit: int = 500) -> list[dict]:
        rows = self.conn.execute(
            "SELECT seq, kind, data_json, created_at FROM task_events"
            " WHERE task_id=? ORDER BY seq LIMIT ?",
            (task_id, limit),
        ).fetchall()
        out = []
        for r in rows:
            try:
                data = json.loads(r["data_json"]) if r["data_json"] else {}
            except ValueError:
                data = {}
            out.append({"seq": r["seq"], "kind": r["kind"], "data": data,
                        "at": r["created_at"]})
        return out

    def prune(self, *, days: int, keep_failed: bool = True) -> int:
        """Delete finished tasks older than `days`; keeps history lean."""
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
        statuses = ("completed", "cancelled") + (("failed",) if keep_failed else ())
        marks = ",".join("?" for _ in statuses)
        cur = self.conn.execute(
            f"DELETE FROM tasks WHERE status IN ({marks}) AND"
            " COALESCE(finished_at, updated_at, created_at) < ?",
            (*statuses, cutoff),
        )
        self.conn.commit()
        return cur.rowcount
