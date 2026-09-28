"""MemoryStore: remember/recall, dedupe, expiry, cap, visibility (Phase 3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from woyo.memory.embedders import HashEmbedder, cosine
from woyo.memory.longterm import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "mem.sqlite3", default_ttl_days=0)
    yield s
    s.close()


class TestRemember:
    async def test_remember_and_get(self, store):
        mid = await store.remember("User prefers concise answers", kind="preference")
        row = store.get(mid)
        assert row["content"] == "User prefers concise answers"
        assert row["kind"] == "preference"
        assert row["expires_at"] is None  # ttl 0 = never

    async def test_dedupe_same_content_updates(self, store):
        a = await store.remember("Python 3.14 released October 2025")
        b = await store.remember("Python 3.14 released October 2025", kind="fact")
        assert a == b
        assert store.get(a)["kind"] == "fact"

    async def test_empty_content_rejected(self, store):
        with pytest.raises(ValueError):
            await store.remember("   ")

    async def test_ttl_sets_expiry(self, tmp_path):
        s = MemoryStore(tmp_path / "m.sqlite3", default_ttl_days=7)
        mid = await s.remember("temporary fact")
        assert s.get(mid)["expires_at"] is not None
        s.close()


class TestRecall:
    async def test_recall_ranks_relevant_first(self, store):
        await store.remember("Python 3.14 was released in October 2025", kind="fact")
        await store.remember("The Eiffel tower is in Paris", kind="fact")
        await store.remember("Best pizza dough needs 24h cold fermentation", kind="note")
        hits = await store.recall("when was python 3.14 released?", k=3)
        assert hits, "expected at least one hit"
        assert "Python 3.14" in hits[0].content
        assert hits[0].similarity > 0.05

    async def test_recall_bumps_access_count(self, store):
        mid = await store.remember("User's timezone is Africa/Algiers")
        before = store.get(mid)["access_count"]
        await store.recall("what timezone is the user in?")
        assert store.get(mid)["access_count"] == before + 1

    async def test_recall_skips_expired(self, store):
        mid = await store.remember("expired fact about kotlin")
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")
        store.conn.execute(
            "UPDATE memories SET expires_at=? WHERE id=?", (past, mid)
        )
        store.conn.commit()
        assert await store.recall("fact about kotlin") == []

    async def test_recall_min_similarity_threshold(self, tmp_path):
        s = MemoryStore(tmp_path / "m.sqlite3", min_similarity=0.99)
        await s.remember("completely unrelated quantum entanglement note")
        assert await s.recall("pizza toppings for a party") == []
        s.close()

    async def test_recall_empty_query(self, store):
        assert await store.recall("   ") == []


class TestVisibility:
    async def test_list_filter_kind(self, store):
        await store.remember("a fact", kind="fact")
        await store.remember("a note", kind="note")
        kinds = {r["kind"] for r in store.list(kind="fact")}
        assert kinds == {"fact"}

    async def test_delete(self, store):
        mid = await store.remember("delete me")
        assert store.delete(mid)
        assert store.get(mid) is None
        assert store.delete(mid) is False

    async def test_stats(self, store):
        await store.remember("fact one", kind="fact")
        await store.remember("note one", kind="note")
        stats = store.stats()
        assert stats["total"] == 2
        assert stats["by_kind"] == {"fact": 1, "note": 1}
        assert stats["embedder"] == "hash-512"

    async def test_purge_expired(self, store):
        keep = await store.remember("kept")
        gone = await store.remember("expired")
        past = (datetime.now(UTC) - timedelta(days=2)).isoformat(timespec="seconds")
        store.conn.execute(
            "UPDATE memories SET expires_at=? WHERE id=?", (past, gone)
        )
        store.conn.commit()
        n = store.purge_expired()
        assert n == 1
        assert store.get(keep) is not None
        assert store.get(gone) is None


class TestDataMinimization:
    async def test_cap_evicts_least_useful(self, tmp_path):
        s = MemoryStore(tmp_path / "m.sqlite3", max_items=3, default_ttl_days=0)
        for i in range(5):
            await s.remember(f"memory number {i}")
        rows = s.list(limit=100)
        assert len(rows) == 3
        # newest survive: numbers 2,3,4 (insertion order, none accessed yet)
        contents = {r["content"] for r in rows}
        assert "memory number 4" in contents
        assert "memory number 0" not in contents
        s.close()

    async def test_cap_prefers_evicting_expired(self, tmp_path):
        s = MemoryStore(tmp_path / "m.sqlite3", max_items=2, default_ttl_days=0)
        await s.remember("fresh important fact")
        await s.remember("fresh other fact")
        old = await s.remember("stale thing")
        past = (datetime.now(UTC) - timedelta(days=1)).isoformat(timespec="seconds")
        s.conn.execute(
            "UPDATE memories SET expires_at=? WHERE id=?", (past, old)
        )
        s.conn.commit()
        await s.remember("newest fact forces eviction")
        rows = s.list(limit=100, include_expired=True)
        contents = {r["content"] for r in rows}
        assert "stale thing" not in contents  # expired evicted first
        assert "newest fact forces eviction" in contents
        s.close()


class TestEmbedders:
    def test_hash_embedder_deterministic_and_normalized(self):
        import asyncio

        emb = HashEmbedder()
        a = asyncio.run(emb.embed(["hello world python"]))[0]
        b = asyncio.run(emb.embed(["hello world python"]))[0]
        assert a == b
        assert abs(sum(x * x for x in a) - 1.0) < 1e-6
        assert len(a) == 512

    def test_hash_similarity_intuition(self):
        import asyncio

        emb = HashEmbedder()
        near = asyncio.run(emb.embed(["python release date"]))[0]
        related = asyncio.run(emb.embed(["when was python released?"]))[0]
        far = asyncio.run(emb.embed(["train schedules in tokyo"]))[0]
        assert cosine(near, related) > cosine(near, far)

    async def test_foreign_embedding_space_skipped(self, store):
        """Rows embedded with a different dim/model are ignored, not compared."""
        await store.remember("a dim-3 memory")
        store.conn.execute(
            "UPDATE memories SET embedding=?, embed_dim=3 WHERE content LIKE 'a dim-3%'",
            (__import__("struct").pack("<3f", 1.0, 0.0, 0.0),),
        )
        store.conn.commit()
        assert await store.recall("dim memory") == []
