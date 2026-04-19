"""Tests for session persistence."""
import sys
import os
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from agent.session import (
    FileSession,
    InMemorySession,
    Session,
    SessionABC,
    SessionManager,
    SessionManagerConfig,
)


class TestSessionProtocol:
    def test_in_memory_satisfies_protocol(self):
        s = InMemorySession("test-1")
        assert isinstance(s, Session)

    def test_file_session_satisfies_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = FileSession("test-1", tmpdir)
            assert isinstance(s, Session)


class TestInMemorySession:
    def test_basic_operations(self):
        s = InMemorySession("s1")
        assert s.session_id == "s1"
        assert s.get("key") is None
        assert s.get("key", "default") == "default"

        s.set("key", "value")
        assert s.get("key") == "value"

        s.delete("key")
        assert s.get("key") is None

    def test_get_all(self):
        s = InMemorySession("s1")
        s.set("a", 1)
        s.set("b", 2)
        data = s.get_all()
        assert data == {"a": 1, "b": 2}

    def test_clear(self):
        s = InMemorySession("s1")
        s.set("a", 1)
        s.clear()
        assert s.get_all() == {}

    def test_touch_updates_last_accessed(self):
        s = InMemorySession("s1")
        t1 = s.last_accessed
        time.sleep(0.01)
        s.touch()
        assert s.last_accessed > t1

    def test_age_and_idle(self):
        s = InMemorySession("s1")
        assert s.age_seconds >= 0
        assert s.idle_seconds >= 0


class TestFileSession:
    def test_basic_operations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = FileSession("s1", tmpdir)
            s.set("key", "value")
            assert s.get("key") == "value"

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = FileSession("s1", tmpdir)
            s1.set("data", {"nested": [1, 2, 3]})

            s2 = FileSession("s1", tmpdir)
            assert s2.get("data") == {"nested": [1, 2, 3]}

    def test_delete_and_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = FileSession("s1", tmpdir)
            s1.set("a", 1)
            s1.set("b", 2)
            s1.delete("a")

            s2 = FileSession("s1", tmpdir)
            assert s2.get("a") is None
            assert s2.get("b") == 2

    def test_clear_and_persist(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = FileSession("s1", tmpdir)
            s1.set("key", "value")
            s1.clear()

            s2 = FileSession("s1", tmpdir)
            assert s2.get_all() == {}

    def test_destroy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = FileSession("s1", tmpdir)
            s.set("key", "value")
            file_path = s._file_path
            assert file_path.exists()

            s.destroy()
            assert not file_path.exists()


class TestSessionManager:
    def test_create_and_get(self):
        mgr = SessionManager()
        s = mgr.create("s1")
        assert s.session_id == "s1"

        retrieved = mgr.get("s1")
        assert retrieved is s

    def test_get_nonexistent(self):
        mgr = SessionManager()
        assert mgr.get("nonexistent") is None

    def test_delete(self):
        mgr = SessionManager()
        mgr.create("s1")
        mgr.delete("s1")
        assert mgr.get("s1") is None

    def test_list_sessions(self):
        mgr = SessionManager()
        mgr.create("s1")
        mgr.create("s2")
        ids = mgr.list_sessions()
        assert set(ids) == {"s1", "s2"}

    def test_max_sessions_eviction(self):
        cfg = SessionManagerConfig(max_sessions=2)
        mgr = SessionManager(cfg)
        mgr.create("s1")
        time.sleep(0.01)
        mgr.create("s2")
        time.sleep(0.01)
        mgr.create("s3")
        assert mgr.count == 2
        assert mgr.get("s1") is None

    def test_ttl_expiry(self):
        cfg = SessionManagerConfig(ttl_seconds=0.01)
        mgr = SessionManager(cfg)
        mgr.create("s1")
        time.sleep(0.02)
        assert mgr.get("s1") is None

    def test_idle_expiry(self):
        cfg = SessionManagerConfig(idle_ttl_seconds=0.01, ttl_seconds=100)
        mgr = SessionManager(cfg)
        mgr.create("s1")
        time.sleep(0.02)
        assert mgr.get("s1") is None

    def test_cleanup_expired(self):
        cfg = SessionManagerConfig(ttl_seconds=0.01)
        mgr = SessionManager(cfg)
        mgr.create("s1")
        mgr.create("s2")
        time.sleep(0.02)
        removed = mgr.cleanup_expired()
        assert removed == 2
        assert mgr.count == 0

    def test_file_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = SessionManagerConfig(use_file_sessions=True, storage_dir=tmpdir)
            mgr = SessionManager(cfg)
            s = mgr.create("s1")
            assert isinstance(s, FileSession)
            s.set("key", "value")

            retrieved = mgr.get("s1")
            assert retrieved.get("key") == "value"
