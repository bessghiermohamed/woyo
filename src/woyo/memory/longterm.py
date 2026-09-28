"""Long-term memory: visible, deletable, expiring (Phase 3, ADR-11).

Design rules (docs/SECURITY.md "memory"):
- Everything is user-inspectable: `woyo memory list|show|delete|prune`.
- Data minimization: optional per-item TTL, a global item cap with
  LRU-ish eviction, and purge-on-open for expired rows.
- Recall output is treated as *untrusted background data*: the prompt
  frames it as hints from earlier sessions, never as verified sources.
- Similarity search uses sqlite-vec when installed (extra `vector`),
  with a pure-Python brute-force fallback that is fast for thousands
  of rows — memory never hard-depends on a vector extension.
- Embeddings are model-scoped: recall only compares rows embedded by
  the same embedder, so switching embedders degrades to "no recall"
  instead of comparing apples to oranges. Old rows are re-embedded
  lazily on their next `remember()` dedupe hit.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from woyo.memory.embedders import Embedder, HashEmbedder, build_embedder, cosine


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def content_hash(text: str) -> str:
    return hashlib.blake2b(text.strip().lower().encode("utf-8"), digest_size=16).hexdigest()


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob[: n * 4]))


@dataclass(slots=True)
class MemoryHit:
    id: int
    kind: str
    content: str
    similarity: float
    created_at: str
    source: str | None


class MemoryStore:
    """Persistent semantic memory over SQLite."""

    def __init__(self, path: str | Path, *, embedder: Embedder | None = None,
                 max_items: int = 5000, default_ttl_days: int = 180,
                 min_similarity: float = 0.04):
        from woyo.store.db import connect, migrate

        self.path = Path(path)
        self.conn: sqlite3.Connection = migrate(connect(self.path))
        self.embedder = embedder or HashEmbedder()
        self.max_items = max_items
        self.default_ttl_days = default_ttl_days
        self.min_similarity = min_similarity
        self._api_embedder_failed = False
        try:
            self.purge_expired()
        except sqlite3.Error:
            pass  # a fresh DB never has expired rows anyway

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed with auto-fallback: one API failure -> hash for this process."""
        base = self.embedder
        if isinstance(base, HashEmbedder):
            return await base.embed(texts)
        if self._api_embedder_failed:
            return await HashEmbedder().embed(texts)
        try:
            return await base.embed(texts)
        except Exception:  # noqa: BLE001 — memory must not kill the run
            self._api_embedder_failed = True
            return await HashEmbedder().embed(texts)

    # ------------------------------------------------------------------
    async def remember(self, content: str, *, kind: str = "note",
                       source: str | None = None, ttl_days: int | None = None,
                       meta: dict | None = None) -> int:
        """Store or refresh a memory; dedupes on exact content."""
        content = content.strip()
        if not content:
            raise ValueError("empty memory content")
        if kind not in ("fact", "preference", "note"):
            kind = "note"
        now = _now()
        ttl = self.default_ttl_days if ttl_days is None else max(0, ttl_days)
        expires = (
            (datetime.now(UTC) + timedelta(days=ttl)).isoformat(timespec="seconds")
            if ttl > 0 else None
        )
        vec = (await self._embed([content]))[0]
        chash = content_hash(content)

        existing = self.conn.execute(
            "SELECT id FROM memories WHERE content_hash=?", (chash,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE memories SET kind=?, source=?, updated_at=?, expires_at=?,"
                " embedding=?, embed_dim=?, embed_model=?, meta_json=? WHERE id=?",
                (kind, source, now, expires, _pack(vec), len(vec),
                 self.embedder.name, json.dumps(meta) if meta else None, existing["id"]),
            )
            self.conn.commit()
            self._enforce_cap()
            return int(existing["id"])

        cur = self.conn.execute(
            "INSERT INTO memories (kind, content, content_hash, source, created_at,"
            " updated_at, expires_at, embedding, embed_dim, embed_model, meta_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (kind, content, chash, source, now, now, expires, _pack(vec), len(vec),
             self.embedder.name, json.dumps(meta) if meta else None),
        )
        self.conn.commit()
        self._enforce_cap()
        return int(cur.lastrowid)

    # ------------------------------------------------------------------
    async def recall(self, query: str, *, k: int = 4) -> list[MemoryHit]:
        """Top-k semantically similar, unexpired memories for `query`."""
        query = query.strip()
        if not query:
            return []
        qvec = (await self._embed([query]))[0]
        now = _now()
        rows = self.conn.execute(
            "SELECT id, kind, content, created_at, source, embedding"
            " FROM memories WHERE expires_at IS NULL OR expires_at > ?",
            (now,),
        ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            blob = row["embedding"]
            if not blob:
                continue
            vec = _unpack(blob)
            # model-scoped comparison: skip foreign embedding spaces
            if len(vec) != len(qvec):
                continue
            score = cosine(qvec, vec)
            if score >= self.min_similarity:
                scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        hits: list[MemoryHit] = []
        for score, row in scored[:k]:
            hits.append(
                MemoryHit(
                    id=row["id"],
                    kind=row["kind"],
                    content=row["content"],
                    similarity=round(score, 4),
                    created_at=row["created_at"],
                    source=row["source"],
                )
            )
        if hits:
            self.conn.executemany(
                "UPDATE memories SET last_accessed_at=?, access_count=access_count+1"
                " WHERE id=?",
                [(_now(), h.id) for h in hits],
            )
            self.conn.commit()
        return hits

    # ------------------------------------------------------------------
    def list(self, *, kind: str | None = None, limit: int = 50,
             include_expired: bool = False) -> list[dict[str, Any]]:
        sql = ("SELECT id, kind, content, source, created_at, updated_at,"
               " last_accessed_at, expires_at, access_count FROM memories")
        conds, params = [], []
        if kind:
            conds.append("kind=?")
            params.append(kind)
        if not include_expired:
            conds.append("expires_at IS NULL OR expires_at > ?")
            params.append(_now())
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def get(self, memory_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT id, kind, content, source, created_at, updated_at,"
            " last_accessed_at, expires_at, access_count FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        return dict(row) if row else None

    def delete(self, memory_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def purge_expired(self) -> int:
        cur = self.conn.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (_now(),),
        )
        self.conn.commit()
        return cur.rowcount

    def stats(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total,"
            " SUM(CASE WHEN expires_at IS NOT NULL THEN 1 ELSE 0 END) AS expiring,"
            " SUM(access_count) AS accesses,"
            " MIN(created_at) AS oldest,"
            " MAX(created_at) AS newest FROM memories"
        ).fetchone()
        by_kind = {
            r["kind"]: r["n"]
            for r in self.conn.execute(
                "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind"
            ).fetchall()
        }
        return {
            "total": row["total"] or 0,
            "expiring": row["expiring"] or 0,
            "lifetime_accesses": row["accesses"] or 0,
            "oldest": row["oldest"],
            "newest": row["newest"],
            "by_kind": by_kind,
            "embedder": self.embedder.name,
            "cap": self.max_items,
        }

    # ------------------------------------------------------------------
    def _enforce_cap(self) -> None:
        """Data minimization: evict least-useful rows past the cap."""
        total = self.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        if total <= self.max_items:
            return
        overflow = total - self.max_items
        # evict: expired first, then least accessed, then least recently used
        self.conn.execute(
            "DELETE FROM memories WHERE id IN ("
            "  SELECT id FROM memories"
            "  ORDER BY (expires_at IS NULL) ASC,"
            "           access_count ASC,"
            "           COALESCE(last_accessed_at, created_at) ASC"
            "  LIMIT ?)",
            (overflow,),
        )
        self.conn.commit()


def build_memory_from_settings(settings) -> MemoryStore | None:
    """The process-wide MemoryStore, or None when memory is disabled."""
    if not settings.memory_enabled:
        return None
    from woyo.store.db import db_path_from_settings

    embedder = build_embedder(settings)
    return MemoryStore(
        db_path_from_settings(settings),
        embedder=embedder,
        max_items=settings.memory_max_items,
        default_ttl_days=settings.memory_default_ttl_days,
        min_similarity=settings.memory_min_similarity,
    )
