"""Metadata timing/concurrency checks; not scientific model evidence."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from mcp_tools.models.evolved import catalog, service


@pytest.fixture
def library(tmp_path):
    (tmp_path / "index").mkdir()
    value = {"models": [{"entry_id": "a" * 64,
                        "key": {"plugin_id": "fixture.v1", "scope": {"protein": "fixture"}},
                        "origin": {}, "plugin_sha256": "current"}],
             "indexed_entries": 1, "counting_rule": "fixture"}
    return SimpleNamespace(root=tmp_path, discovery_metadata=lambda: deepcopy(value))


def test_slow_reads_get_a_full_cache_lifetime_after_completion(library, monkeypatch):
    now = [100.0]
    calls = []
    original = library.discovery_metadata

    def slow():
        calls.append(1)
        now[0] += 30
        return original()

    library.discovery_metadata = slow
    monkeypatch.setattr(catalog.time, "monotonic", lambda: now[0])
    first = catalog.recorded_metadata(library)
    first["models"][0]["key"]["scope"]["protein"] = "changed"
    assert catalog.recorded_metadata(library)["models"][0]["key"]["scope"]["protein"] == "fixture"
    assert len(calls) == 1
    now[0] = 146.0
    catalog.recorded_metadata(library)
    assert len(calls) == 2


def test_concurrent_searches_share_one_read_and_receive_separate_values(library):
    gate, entered, release = Barrier(4), Event(), Event()
    original = library.discovery_metadata
    calls = []

    def slow():
        calls.append(1)
        entered.set()
        assert release.wait(10)
        return original()

    def read():
        gate.wait(timeout=10)
        return catalog.recorded_metadata(library)

    library.discovery_metadata = slow
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(read) for _ in range(4)]
        assert entered.wait(10)
        release.set()
        values = [f.result(timeout=10) for f in futures]
    assert len(calls) == 1
    values[0]["models"].clear()
    assert all(len(v["models"]) == 1 for v in values[1:])


def test_failed_read_is_not_kept_or_reused(library):
    original = library.discovery_metadata
    calls = []

    def fail_once():
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("changed evidence")
        return original()

    library.discovery_metadata = fail_once
    with pytest.raises(ValueError, match="changed evidence"):
        catalog.recorded_metadata(library)
    assert catalog.recorded_metadata(library)["indexed_entries"] == 1
    assert len(calls) == 2


def test_registry_change_during_a_read_rejects_that_snapshot(library):
    original = library.discovery_metadata

    def changed():
        index = library.root / "index"
        prior = index.stat().st_mtime_ns
        (index / "new.json").write_text("{}")
        os.utime(index, ns=(prior + 1000000, prior + 1000000))
        return original()

    library.discovery_metadata = changed
    with pytest.raises(ValueError, match="changed during discovery"):
        catalog.recorded_metadata(library)
    library.discovery_metadata = original
    assert catalog.recorded_metadata(library)["indexed_entries"] == 1


def test_runtime_compatibility_is_rechecked_for_cached_metadata(library, monkeypatch):
    plugin = SimpleNamespace(source_sha256="current")
    monkeypatch.setattr("agent.rsi.engine.model_plugins.get_model_plugin", lambda _: plugin)
    assert service.catalog_cards(library)[0][0]["runtime_compatible"]

    def unexpected():
        raise AssertionError("Fresh metadata should be reused")

    library.discovery_metadata = unexpected
    plugin.source_sha256 = "changed"
    assert not service.catalog_cards(library)[0][0]["runtime_compatible"]


def test_symlinked_index_is_rejected_even_with_a_cached_result(library):
    catalog.recorded_metadata(library)
    index = library.root / "index"
    index.rename(library.root / "saved-index")
    index.symlink_to(library.root / "saved-index", target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        catalog.recorded_metadata(library)


def test_model_switches_use_one_catalog_snapshot_but_invocation_rechecks(tmp_path, monkeypatch):
    from mcp_tools.models.catalog_snapshot import catalog_snapshot
    from mcp_tools.tools.capability_policy import CapabilityStore, CapabilityDisabled

    store = CapabilityStore(tmp_path / "capabilities.sqlite3")
    reads = []
    original = store.rows

    def read():
        reads.append(1)
        return original()

    monkeypatch.setattr(store, "rows", read)
    key = "tool:predict_with_evolved_model"

    @catalog_snapshot
    def listing():
        assert all(store.enabled(key) for _ in range(100))
        assert len(reads) == 1
        store.set(key, False, 0, {}, "fixture-admin")
        with pytest.raises(CapabilityDisabled):
            store.assert_call("predict_with_evolved_model")

    listing()
    assert len(reads) == 2
    assert not store.enabled(key) and len(reads) == 3
