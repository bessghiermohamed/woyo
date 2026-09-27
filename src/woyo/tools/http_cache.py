"""SQLite-backed TTL cache for tool responses (Phase 2).

Repeated fetches of the same URL — within a run or across runs — hit the
cache instead of the network. That saves free-tier search quota, speeds up
re-runs, and keeps fetches polite. Stdlib only (sqlite3); the ops are
sub-millisecond on the small payloads tools produce, so the sync calls do
not starve the event loop in practice.

Cache lives at <cache_dir>/cache.sqlite3 (default ~/.woyo/). Set
WOYO_CACHE_ENABLED=false to disable.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key     TEXT PRIMARY KEY,
    kind    TEXT NOT NULL,
    value   TEXT NOT NULL,
    created REAL NOT NULL,
    size    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created);
"""

_MAX_VALUE_BYTES = 512 * 1024  # never cache a single payload larger than this


class HttpCache:
    """Tiny TTL cache with entry- and byte-budget eviction (oldest first)."""

    def __init__(
        self,
        path: str | Path,
        *,
        ttl_s: int = 86_400,
        max_entries: int = 2_000,
        max_bytes: int = 64 * 1024 * 1024,
    ):
        self.path = Path(path)
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    # -- lifecycle ----------------------------------------------------------
    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # -- core -----------------------------------------------------------------
    @staticmethod
    def _hash_key(kind: str, key: str) -> str:
        return hashlib.sha256(f"{kind}:{key}".encode()).hexdigest()

    def get(self, kind: str, key: str, *, ttl_s: int | None = None) -> Any | None:
        """Return the cached JSON value, or None on miss/expiry/corruption."""
        h = self._hash_key(kind, key)
        ttl = ttl_s if ttl_s is not None else self.ttl_s
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT value, created FROM cache WHERE key = ?", (h,)
            ).fetchone()
        if row is None:
            return None
        value_blob, created = row
        if time.time() - created > ttl:
            self._delete(h)
            return None
        try:
            return json.loads(value_blob)
        except (json.JSONDecodeError, TypeError):
            self._delete(h)
            return None

    def put(self, kind: str, key: str, value: Any) -> None:
        blob = json.dumps(value, ensure_ascii=False)
        if len(blob) > _MAX_VALUE_BYTES:
            return  # too big to be worth caching
        h = self._hash_key(kind, key)
        now = time.time()
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                "INSERT INTO cache (key, kind, value, created, size) VALUES (?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "created=excluded.created, size=excluded.size",
                (h, kind, blob, now, len(blob)),
            )
            self._conn.commit()
        self._evict_if_needed()

    def _delete(self, h: str) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (h,))
                self._conn.commit()

    def _evict_if_needed(self) -> None:
        with self._lock:
            if self._conn is None:
                return
            count, total_bytes = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size),0) FROM cache"
            ).fetchone()
            if count <= self.max_entries and total_bytes <= self.max_bytes:
                return
            # evict oldest rows until both budgets are satisfied
            while True:
                count, total_bytes = self._conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size),0) FROM cache"
                ).fetchone()
                if (count <= self.max_entries and total_bytes <= self.max_bytes) or count == 0:
                    break
                self._conn.execute(
                    "DELETE FROM cache WHERE key = (SELECT key FROM cache "
                    "ORDER BY created ASC LIMIT 1)"
                )
            self._conn.commit()

    # -- utilities ------------------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.execute("DELETE FROM cache")
                self._conn.commit()

    def stats(self) -> dict[str, int]:
        with self._lock:
            if self._conn is None:
                return {"entries": 0, "bytes": 0}
            count, total = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size),0) FROM cache"
            ).fetchone()
            return {"entries": int(count), "bytes": int(total)}


def build_cache_from_settings(settings) -> HttpCache | None:
    """Cache from Settings, or None when disabled. Import-lazy to avoid cycles."""
    if not getattr(settings, "cache_enabled", True):
        return None
    from pathlib import Path

    base = Path(getattr(settings, "cache_dir", "~/.woyo")).expanduser()
    return HttpCache(
        base / "cache.sqlite3",
        ttl_s=getattr(settings, "cache_ttl_s", 86_400),
    )
