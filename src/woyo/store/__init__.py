"""SQLite persistence for tasks and long-term memory (Phase 3, ADR-11).

One database file (~/.woyo/woyo.sqlite3, WAL mode) holds the task queue
(with per-step checkpoints for resume-after-restart) and the memory store.
Connections are per-TaskStore/MemoryStore instance; WAL keeps concurrent
readers safe while a run is checkpointing.
"""

from woyo.store.db import connect, db_path_from_settings, migrate
from woyo.store.tasks import TaskStore

__all__ = ["connect", "migrate", "db_path_from_settings", "TaskStore"]
