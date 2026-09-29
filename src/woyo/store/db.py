"""Database connection + schema migrations for the woyo store.

The schema is versioned with PRAGMA user_version; migrate() is idempotent
and only ever moves forward. Tasks, task events and memories live here;
the Phase 2 research cache keeps its own file (cache.sqlite3) untouched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from woyo.config import Settings

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT NOT NULL,
    prompt           TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
        -- pending | running | waiting_approval | paused
        -- | completed | failed | cancelled
    source           TEXT NOT NULL DEFAULT 'cli',   -- cli | chat | api
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    finished_at      TEXT,
    heartbeat_at     TEXT,
    attempts         INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    result_json      TEXT,
    checkpoint_json  TEXT,
    control          TEXT,          -- cross-process: pause | cancel | resume
    meta_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_updated ON tasks(updated_at);

CREATE TABLE IF NOT EXISTS task_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    data_json   TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_events_task ON task_events(task_id);

CREATE TABLE IF NOT EXISTS memories (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    kind             TEXT NOT NULL DEFAULT 'note', -- fact | preference | note
    content          TEXT NOT NULL,
    content_hash     TEXT NOT NULL UNIQUE,
    source           TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT,
    last_accessed_at TEXT,
    expires_at       TEXT,                         -- data minimization
    access_count     INTEGER NOT NULL DEFAULT 0,
    embedding        BLOB,                         -- float32 little-endian
    embed_dim        INTEGER,
    embed_model      TEXT,
    meta_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);

-- v0.8: scheduled jobs (the follow-through layer — "I will" becomes a row)
CREATE TABLE IF NOT EXISTS jobs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id        INTEGER NOT NULL,             -- telegram chat to run + report in
    title          TEXT NOT NULL,
    prompt         TEXT NOT NULL,                -- agent task executed when due
    run_at         TEXT NOT NULL,                -- ISO UTC timestamp
    status         TEXT NOT NULL DEFAULT 'scheduled',
        -- scheduled | running | done | failed | cancelled
    attempts       INTEGER NOT NULL DEFAULT 0,   -- host rotations survived
    created_at     TEXT NOT NULL,
    updated_at     TEXT,
    finished_at    TEXT,
    error          TEXT,
    result_preview TEXT,
    meta_json      TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_chat ON jobs(chat_id);
"""


def db_path_from_settings(settings: Settings) -> Path:
    path = settings.db_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the store with pragmas suited for a single-writer agent."""
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def migrate(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create/upgrade the schema (idempotent, forward-only).

    Every object is CREATE ... IF NOT EXISTS, so a v1 database upgraded to
    v2 simply grows the `jobs` table — existing rows are never touched.
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.commit()
    return conn
