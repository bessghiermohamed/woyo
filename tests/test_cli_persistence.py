"""CLI surface for Phase 3: `woyo memory ...` and `woyo tasks ...`."""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from woyo.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every CLI test gets its own store; the real ~/.woyo stays untouched."""
    db = tmp_path / "cli.sqlite3"
    monkeypatch.setenv("WOYO_DB_PATH", str(db))
    monkeypatch.setenv("WOYO_MEMORY_ENABLED", "true")
    monkeypatch.setenv("WOYO_PROVIDER", "mock")
    monkeypatch.setenv("WOYO_MODEL", "mock/test")
    monkeypatch.delenv("WOYO_CACHE_ENABLED", raising=False)
    monkeypatch.setenv("WOYO_CACHE_ENABLED", "false")
    yield
    if os.environ.get("WOYO_MEMORY_EMBEDDER"):
        os.environ.pop("WOYO_MEMORY_EMBEDDER", None)


class TestMemoryCli:
    def test_add_list_show_delete(self):
        r = runner.invoke(app, ["memory", "add", "user likes rust", "--kind", "preference"])
        assert r.exit_code == 0, r.output
        assert "saved" in r.output

        r = runner.invoke(app, ["memory", "list"])
        assert r.exit_code == 0, r.output
        assert "user likes rust" in r.output

        r = runner.invoke(app, ["memory", "show", "1"])
        assert r.exit_code == 0, r.output
        assert "preference" in r.output

        r = runner.invoke(app, ["memory", "delete", "1", "--yes"])
        assert r.exit_code == 0, r.output
        assert "deleted" in r.output

        r = runner.invoke(app, ["memory", "list"])
        assert "no memories stored yet" in r.output

    def test_search_finds_related(self):
        runner.invoke(app, ["memory", "add", "kubernetes 1.34 shipped in 2026"])
        r = runner.invoke(app, ["memory", "search", "when did kubernetes 1.34 ship"])
        assert r.exit_code == 0, r.output
        assert "kubernetes 1.34" in r.output

    def test_search_empty_store(self):
        r = runner.invoke(app, ["memory", "search", "nothing at all"])
        assert r.exit_code == 0, r.output
        assert "no relevant memories" in r.output

    def test_stats(self):
        runner.invoke(app, ["memory", "add", "one"])
        r = runner.invoke(app, ["memory", "stats"])
        assert r.exit_code == 0, r.output
        assert '"total": 1' in r.output

    def test_prune_dry_run(self):
        r = runner.invoke(app, ["memory", "prune", "--dry-run"])
        assert r.exit_code == 0, r.output
        assert "would be deleted" in r.output

    def test_show_missing_id_fails(self):
        r = runner.invoke(app, ["memory", "show", "42"])
        assert r.exit_code == 1
        assert "no memory" in r.output


class TestTasksCli:
    def test_add_list_show(self):
        r = runner.invoke(app, ["tasks", "add", "check the weather in algiers"])
        assert r.exit_code == 0, r.output
        assert "queued" in r.output

        r = runner.invoke(app, ["tasks", "list"])
        assert r.exit_code == 0, r.output
        assert "check the weather" in r.output
        assert "pending" in r.output

        r = runner.invoke(app, ["tasks", "show", "1"])
        assert r.exit_code == 0, r.output
        assert "pending" in r.output

    def test_cancel_and_retry(self):
        runner.invoke(app, ["tasks", "add", "a task to cancel"])
        r = runner.invoke(app, ["tasks", "cancel", "1"])
        assert r.exit_code == 0, r.output
        assert "cancel requested" in r.output

        # cancel only requests; flip status manually as the runner would
        from woyo.config import Settings
        from woyo.store.db import db_path_from_settings
        from woyo.store.tasks import TaskStore

        store = TaskStore(db_path_from_settings(Settings()))
        store.set_status(1, "cancelled")
        store.close()

        r = runner.invoke(app, ["tasks", "retry", "1"])
        assert r.exit_code == 0, r.output
        assert "re-queued" in r.output

    def test_show_missing_fails(self):
        r = runner.invoke(app, ["tasks", "show", "99"])
        assert r.exit_code == 1

    def test_run_no_pending_is_graceful(self):
        r = runner.invoke(app, ["tasks", "run"])
        assert r.exit_code == 0, r.output
        assert "no pending tasks" in r.output


class TestDoctor:
    def test_doctor_shows_new_rows(self):
        r = runner.invoke(app, ["doctor"])
        assert r.exit_code == 0, r.output
        assert "memory" in r.output
        assert "tasks" in r.output
