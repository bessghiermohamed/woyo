"""TaskStore: state machine, control channel, events, recovery (Phase 3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from woyo.store.tasks import TaskStore


@pytest.fixture
def store(tmp_path):
    s = TaskStore(tmp_path / "tasks.sqlite3")
    yield s
    s.close()


def _age_heartbeat(store: TaskStore, task_id: int, minutes: int) -> None:
    past = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat(timespec="seconds")
    store.conn.execute(
        "UPDATE tasks SET heartbeat_at=? WHERE id=?", (past, task_id)
    )
    store.conn.commit()


class TestCrud:
    def test_create_get_roundtrip(self, store):
        t = store.create("Research the best ORM for SQLite", title="orm research")
        assert t.status == "pending"
        assert t.source == "cli"
        assert t.attempts == 0
        fetched = store.get(t.id)
        assert fetched.prompt == "Research the best ORM for SQLite"
        assert fetched.title == "orm research"

    def test_create_generates_title_from_prompt(self, store):
        t = store.create("a very long prompt " * 20)
        assert t.title.startswith("a very long")
        assert len(t.title) <= 120

    def test_list_newest_first(self, store):
        first = store.create("first task")
        second = store.create("second task")
        ids = [t.id for t in store.list()]
        assert ids[0] == second.id
        assert ids[1] == first.id

    def test_get_missing_returns_none(self, store):
        assert store.get(999) is None


class TestStateMachine:
    def test_set_status_terminal_sets_finished(self, store):
        t = store.create("t")
        store.set_status(t.id, "completed")
        row = store.get(t.id)
        assert row.status == "completed"
        assert row.finished_at is not None

    def test_set_status_nonterminal_keeps_finished_null(self, store):
        t = store.create("t")
        store.mark_running(t.id)
        row = store.get(t.id)
        assert row.status == "running"
        assert row.finished_at is None
        assert row.attempts == 1

    def test_unknown_status_rejected(self, store):
        t = store.create("t")
        with pytest.raises(ValueError):
            store.set_status(t.id, "exploded")

    def test_retry_requeues_finished(self, store):
        t = store.create("t")
        store.mark_running(t.id)
        store.save_result(t.id, {"outcome": "completed"})
        store.set_status(t.id, "completed")
        retried = store.retry(t.id)
        assert retried.status == "pending"
        assert retried.result is None
        assert retried.checkpoint is None
        assert retried.attempts == 1  # from mark_running; retry keeps history

    def test_retry_leaves_pending_alone(self, store):
        t = store.create("t")
        assert store.retry(t.id).status == "pending"


class TestControlChannel:
    def test_request_and_poll(self, store):
        t = store.create("t")
        assert store.poll_control(t.id) is None
        store.request_control(t.id, "pause")
        assert store.poll_control(t.id) == "pause"
        store.clear_control(t.id)
        assert store.poll_control(t.id) is None

    def test_unknown_action_rejected(self, store):
        t = store.create("t")
        with pytest.raises(ValueError):
            store.request_control(t.id, "explode")


class TestCheckpointsAndResults:
    def test_save_checkpoint(self, store):
        t = store.create("t")
        store.save_checkpoint(t.id, {"steps": 3, "messages": []})
        assert store.get(t.id).checkpoint["steps"] == 3

    def test_save_result(self, store):
        t = store.create("t")
        store.save_result(t.id, {"outcome": "completed", "final": "ok"})
        assert store.get(t.id).result["final"] == "ok"


class TestRecovery:
    def test_recover_stale_flips_running_to_paused(self, store):
        t = store.create("t")
        store.mark_running(t.id)
        _age_heartbeat(store, t.id, minutes=30)
        recovered = store.recover_stale(stale_minutes=10)
        assert recovered == [t.id]
        assert store.get(t.id).status == "paused"

    def test_fresh_running_row_not_recovered(self, store):
        t = store.create("t")
        store.mark_running(t.id)
        assert store.recover_stale(stale_minutes=10) == []

    def test_resumable_property(self, store):
        t = store.create("t")
        assert t.resumable
        store.set_status(t.id, "completed")
        assert not store.get(t.id).resumable


class TestEvents:
    def test_append_and_fetch_in_order(self, store):
        t = store.create("t")
        store.append_event(t.id, 1, "run_started", {"task": "x"})
        store.append_event(t.id, 2, "tool_call", {"tool": "echo"})
        events = store.events(t.id)
        assert [e["seq"] for e in events] == [1, 2]
        assert events[0]["kind"] == "run_started"
        assert events[1]["data"]["tool"] == "echo"


class TestPrune:
    def test_prune_deletes_old_finished(self, store):
        t = store.create("t")
        store.set_status(t.id, "completed")
        past = (datetime.now(UTC) - timedelta(days=60)).isoformat(timespec="seconds")
        store.conn.execute(
            "UPDATE tasks SET finished_at=?, updated_at=? WHERE id=?",
            (past, past, t.id),
        )
        store.conn.commit()
        kept = store.create("recent")
        store.set_status(kept.id, "completed")
        assert store.prune(days=30) == 1
        assert store.get(t.id) is None
        assert store.get(kept.id) is not None

    def test_prune_never_touches_pending(self, store):
        t = store.create("old pending")
        past = (datetime.now(UTC) - timedelta(days=90)).isoformat(timespec="seconds")
        store.conn.execute("UPDATE tasks SET created_at=? WHERE id=?", (past, t.id))
        store.conn.commit()
        store.prune(days=30)
        assert store.get(t.id) is not None


class TestSchema:
    def test_reopen_is_idempotent(self, tmp_path):
        path = tmp_path / "tasks.sqlite3"
        s1 = TaskStore(path)
        t = s1.create("persisted")
        s1.close()
        s2 = TaskStore(path)
        assert s2.get(t.id).prompt == "persisted"
        s2.close()

    def test_wal_mode(self, store):
        mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
