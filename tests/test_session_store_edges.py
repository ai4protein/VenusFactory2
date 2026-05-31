"""Corner-case tests for src/agent/session_store.py.

Covers datetime serialization, missing fields, int keys, large history,
concurrent saves/creates, TTL cleanup, delete consistency, runtime restore,
and SQL-special-character safety.

These tests use a lightweight ``ensure_runtime_fn`` stub so we don't pull in
the full chat_agent stack — what we test is purely the storage layer.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# conftest adds src/ to sys.path for pytest; do the same for unittest direct.
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from agent.session_store import (  # noqa: E402
    SessionStore,
    SqliteBackend,
    persistable_state,
    _PERSISTABLE_KEYS,
)


def _stub_ensure_runtime(state: dict[str, Any]) -> dict[str, Any]:
    """Lightweight ensure_runtime mimicking the real one's *behavior* on
    persistable-field preservation. Fills in runtime-only defaults without
    importing the heavy chat_agent module."""
    state.setdefault("llm", object())  # placeholder runtime field
    state.setdefault("all_tools", [])
    state.setdefault("memory", object())
    state.setdefault("tool_cache", {})
    state.setdefault("temp_files", [])
    state.setdefault("dialogue_memory", [])
    state.setdefault("history", [])
    state.setdefault("conversation_log", [])
    state.setdefault("tool_executions", [])
    state.setdefault("execution_failed", False)
    return state


def _make_store(tmpdir: str) -> tuple[SessionStore, SqliteBackend]:
    db_path = os.path.join(tmpdir, "sessions.db")
    backend = SqliteBackend(db_path)
    store = SessionStore(backend, _stub_ensure_runtime)
    return store, backend


def _async(fn):
    """Decorator: run an async test method via asyncio.run."""
    def wrapper(self):
        return asyncio.run(fn(self))
    return wrapper


class TestSessionStoreEdges(unittest.TestCase):

    # ------------------------------------------------------------------ T1
    @_async
    async def test_T1_datetime_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            now = datetime.now()
            nested_ts = datetime.now() - timedelta(minutes=5)
            sid = "t1"
            state = {
                "session_id": sid,
                "owner_key": "owner1",
                "created_at": now,
                "tool_executions": [
                    {"timestamp": nested_ts, "tool": "x", "args": {}}
                ],
                "history": [
                    {"role": "user", "content": "hi", "ts": nested_ts}
                ],
            }
            await store.create(state)
            await store.save(sid)
            # drop mem
            store._mem.pop(sid, None)
            restored = await store.get(sid)
            self.assertIsNotNone(restored)
            # top-level datetime -> string ISO is acceptable per persistable_state
            self.assertIn("created_at", restored)
            ca = restored["created_at"]
            self.assertTrue(isinstance(ca, str), f"created_at type={type(ca)}")
            self.assertTrue(ca.startswith(str(now.year)))
            # nested datetime -> stringified via json default=str
            self.assertEqual(len(restored["tool_executions"]), 1)
            te = restored["tool_executions"][0]
            self.assertEqual(te["tool"], "x")
            # nested datetime gets str()-ified by json default=str (not lost)
            self.assertIsInstance(te["timestamp"], str)
            self.assertTrue(len(te["timestamp"]) > 0)
            # history nested ts preserved as string too
            self.assertEqual(restored["history"][0]["role"], "user")
            self.assertIsInstance(restored["history"][0]["ts"], str)

    # ------------------------------------------------------------------ T2
    @_async
    async def test_T2_minimum_state_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            sid = "t2"
            state = {"session_id": sid, "owner_key": "owner1"}
            await store.create(state)
            await store.save(sid)
            store._mem.pop(sid, None)
            restored = await store.get(sid)
            self.assertIsNotNone(restored)
            # stub ensure_runtime sets sane defaults
            self.assertEqual(restored.get("history"), [])
            self.assertEqual(restored.get("tool_executions"), [])
            self.assertEqual(restored.get("dialogue_memory"), [])
            self.assertEqual(restored.get("execution_failed"), False)
            # session_id intact
            self.assertEqual(restored["session_id"], sid)

    # ------------------------------------------------------------------ T3
    @_async
    async def test_T3_int_keys_in_step_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            sid = "t3"
            state = {
                "session_id": sid,
                "owner_key": "owner1",
                "step_results": {1: {"raw_output": "one"}, 2: {"raw_output": "two"}},
            }
            await store.create(state)
            await store.save(sid)
            store._mem.pop(sid, None)
            restored = await store.get(sid)
            sr = restored["step_results"]
            keys = list(sr.keys())
            # FB fix restores int keys after JSON round-trip (SessionStore.get
            # coerces numeric-string keys back to int for step_results).
            int_keys = [k for k in keys if isinstance(k, int)]
            self.assertEqual(len(int_keys), 2, f"expected 2 int keys after restore, got {keys!r}")
            self.assertEqual(sr[1]["raw_output"], "one")
            self.assertEqual(sr[2]["raw_output"], "two")

    # ------------------------------------------------------------------ T4
    @_async
    async def test_T4_large_history_10mb(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            sid = "t4"
            big_history = [{"role": "user", "content": "x" * 100} for _ in range(100_000)]
            state = {
                "session_id": sid,
                "owner_key": "owner1",
                "history": big_history,
            }
            t0 = time.time()
            await store.create(state)
            t_create = time.time() - t0
            t0 = time.time()
            await store.save(sid)
            t_save = time.time() - t0
            store._mem.pop(sid, None)
            t0 = time.time()
            restored = await store.get(sid)
            t_load = time.time() - t0
            print(f"\n[T4 PERF] create={t_create:.2f}s save={t_save:.2f}s load={t_load:.2f}s")
            self.assertEqual(len(restored["history"]), 100_000)
            # spot-check first / last entries -- no truncation
            self.assertEqual(restored["history"][0], {"role": "user", "content": "x" * 100})
            self.assertEqual(restored["history"][-1], {"role": "user", "content": "x" * 100})

    # ------------------------------------------------------------------ T5
    @_async
    async def test_T5_concurrent_save_same_sid(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            sid = "t5"
            state = {"session_id": sid, "owner_key": "owner1", "history": []}
            await store.create(state)

            async def writer(tag: int):
                # mutate then save -- last writer wins
                async with store._lock:
                    store._mem[sid]["history"] = [{"role": "user", "content": f"msg-{tag}"}]
                    store._mem[sid]["_last_writer"] = tag  # type: ignore[index]
                await store.save(sid)

            tasks = [writer(i) for i in range(10)]
            await asyncio.gather(*tasks)
            # whatever the last writer was, db should reflect SOME complete payload (no corruption)
            store._mem.pop(sid, None)
            restored = await store.get(sid)
            self.assertIsNotNone(restored)
            self.assertEqual(len(restored["history"]), 1)
            self.assertTrue(restored["history"][0]["content"].startswith("msg-"))
            # db row count remains 1 (no dup, no broken row)
            ids = backend.list_ids("owner1")
            self.assertEqual(ids.count(sid), 1)

    # ------------------------------------------------------------------ T6
    @_async
    async def test_T6_concurrent_create_distinct_sids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            n = 50

            async def maker(i: int):
                sid = f"t6-{i:03d}"
                await store.create({"session_id": sid, "owner_key": "shared"})

            await asyncio.gather(*[maker(i) for i in range(n)])
            ids = await store.list_ids("shared")
            self.assertEqual(len(ids), n)
            self.assertEqual(len(set(ids)), n)

    # ------------------------------------------------------------------ T7
    @_async
    async def test_T7_ttl_cleanup_mem_vs_db_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            sid = "t7"
            await store.create({"session_id": sid, "owner_key": "owner1"})
            # manually backdate the db row's last_accessed to 100 days ago
            very_old = time.time() - 100 * 86400
            with backend._lock:
                backend._conn.execute(
                    "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                    (very_old, sid),
                )
                backend._conn.commit()
            removed = backend.cleanup_older_than(idle_seconds=1)
            self.assertEqual(removed, 1)
            # db is gone
            self.assertIsNone(backend.load(sid))
            self.assertNotIn(sid, backend.list_ids())
            # mem still has the entry (cleanup_older_than is db-only)
            self.assertIn(sid, store._mem)
            # store.get(sid) returns the in-memory copy (mem-first lookup)
            got = await store.get(sid)
            self.assertIsNotNone(got)
            self.assertEqual(got["session_id"], sid)
            print(
                f"\n[T7 NOTE] After backend.cleanup_older_than removed the db row, "
                f"SessionStore.get() still returns the mem-cached state. "
                f"list_ids reflects db-only sessions PLUS mem-only sessions matching owner."
            )
            # list_ids should still see it (mem fallback in SessionStore.list_ids)
            ids = await store.list_ids("owner1")
            self.assertIn(sid, ids)

    # ------------------------------------------------------------------ T8
    @_async
    async def test_T8_delete_filters_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            sid = "t8"
            await store.create({"session_id": sid, "owner_key": "owner1"})
            await store.save(sid)
            ids_before = await store.list_ids("owner1")
            self.assertIn(sid, ids_before)
            await store.delete(sid)
            ids_after = await store.list_ids("owner1")
            self.assertNotIn(sid, ids_after)
            self.assertNotIn(sid, store._mem)
            self.assertIsNone(backend.load(sid))

    # ------------------------------------------------------------------ T9
    @_async
    async def test_T9_ensure_runtime_does_not_clobber_persistable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            sid = "t9"
            state = {
                "session_id": sid,
                "owner_key": "owner1",
                "pi_report": "TEST_REPORT",
                "plan": [{"step": 1, "action": "noop"}],
                "current_step_index": 3,
            }
            await store.create(state)
            await store.save(sid)
            store._mem.pop(sid, None)
            restored = await store.get(sid)
            self.assertEqual(restored["pi_report"], "TEST_REPORT")
            self.assertEqual(restored["plan"], [{"step": 1, "action": "noop"}])
            self.assertEqual(restored["current_step_index"], 3)

    # ------------------------------------------------------------------ T10
    @_async
    async def test_T10_sql_special_chars_in_owner_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, backend = _make_store(tmp)
            evil_owner = "';DROP TABLE sessions;--"
            sid = "t10"
            await store.create({"session_id": sid, "owner_key": evil_owner, "history": [{"role": "user", "content": "hello"}]})
            await store.save(sid)
            # If injection had succeeded, the table would be gone -> upsert/load would fail.
            store._mem.pop(sid, None)
            restored = await store.get(sid)
            self.assertIsNotNone(restored)
            self.assertEqual(restored["owner_key"], evil_owner)
            # list_ids filtered by evil owner
            ids = await store.list_ids(evil_owner)
            self.assertIn(sid, ids)
            # also escape-y session_id
            evil_sid = "x' OR '1'='1"
            await store.create({"session_id": evil_sid, "owner_key": "o"})
            await store.save(evil_sid)
            store._mem.pop(evil_sid, None)
            r2 = await store.get(evil_sid)
            self.assertIsNotNone(r2)
            self.assertEqual(r2["session_id"], evil_sid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
