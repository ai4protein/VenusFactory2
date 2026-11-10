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
    assert len(calls) == 1  # The authenticated disk metadata survives the memory TTL.


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


def test_worker_restart_reuses_private_metadata_without_loading_models(library):
    expected = catalog.recorded_metadata(library)
    catalog._CACHE.clear()
    def forbidden():
        raise AssertionError("Published model packets must not be loaded again")
    library.discovery_metadata = forbidden
    actual = catalog.recorded_metadata(library)
    assert actual == expected
    actual["models"].clear()
    assert catalog.recorded_metadata(library) == expected


def test_usage_and_inference_receipts_do_not_rebuild_discovery(library):
    expected = catalog.recorded_metadata(library)
    for area in ("usage", ".inference", "publication_attempts"):
        (library.root / area).mkdir()
        (library.root / area / "new.json").write_text("{}")
    catalog._CACHE.clear()
    library.discovery_metadata = lambda: pytest.fail("Unrelated receipts changed discovery")
    assert catalog.recorded_metadata(library) == expected


@pytest.mark.parametrize("area", ["index", "entries", "origins", "packets"])
def test_registry_change_invalidates_hot_and_persistent_metadata(library, area):
    target = library.root / area / "record.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text("original")
    expected = catalog.recorded_metadata(library)
    original = library.discovery_metadata
    reads = []
    library.discovery_metadata = lambda: (reads.append(1) or original())
    info = target.stat()
    target.write_text("modified")
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert catalog.recorded_metadata(library) == expected
    assert len(reads) == 1
    catalog._CACHE.clear()
    assert catalog.recorded_metadata(library) == expected
    assert len(reads) == 1


@pytest.mark.parametrize("attack", ["bad_json", "edited_metadata", "cross_library"])
def test_untrusted_disk_record_is_rebuilt_from_original_objects(library, attack):
    import json
    expected = catalog.recorded_metadata(library)
    target = library.root / ".discovery/catalog.json"
    if attack == "bad_json":
        target.write_text("not JSON")
    else:
        value = json.loads(target.read_text())
        if attack == "edited_metadata":
            value["record"]["metadata"]["models"][0]["key"]["scope"]["protein"] = "forged"
        else:
            value["record"]["library"] = str(library.root / "another-owner")
        target.write_text(json.dumps(value))
    catalog._CACHE.clear()
    original = library.discovery_metadata
    reads = []
    library.discovery_metadata = lambda: (reads.append(1) or original())
    assert catalog.recorded_metadata(library) == expected
    assert len(reads) == 1


@pytest.mark.parametrize("name", [".discovery", ".discovery/catalog.json", ".discovery/signing-key"])
def test_discovery_cache_permissions_are_host_only(library, name):
    catalog.recorded_metadata(library)
    catalog._CACHE.clear()
    path = library.root / name
    path.chmod(0o755 if path.is_dir() else 0o644)
    with pytest.raises(ValueError, match="private"):
        catalog.recorded_metadata(library)


@pytest.mark.parametrize("area", ["index", "entries", "origins", "packets"])
def test_registry_symlink_is_rejected_with_an_existing_cache(library, area):
    target = library.root / area
    target.mkdir(exist_ok=True)
    catalog.recorded_metadata(library)
    saved = library.root / (area + "-saved")
    target.rename(saved)
    target.symlink_to(saved, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        catalog.recorded_metadata(library)


def test_non_posix_hosts_keep_original_discovery(library, monkeypatch):
    monkeypatch.setattr(catalog, "os", SimpleNamespace())
    expected = library.discovery_metadata()
    assert catalog.recorded_metadata(library) == expected
    assert not (library.root / ".discovery").exists()


def test_first_discovery_does_not_require_a_writable_library(library):
    if os.getuid() == 0:
        pytest.skip("Root can write despite directory permissions")
    library.root.chmod(0o500)
    try:
        assert catalog.recorded_metadata(library) == library.discovery_metadata()
        assert not (library.root / ".discovery").exists()
    finally:
        library.root.chmod(0o700)


def test_registry_file_budget_is_enforced(library, monkeypatch):
    monkeypatch.setattr(catalog, "_MAX_REGISTRY_PATHS", 2)
    with pytest.raises(ValueError, match="path budget"):
        catalog.recorded_metadata(library)


def test_separate_workers_share_one_persistent_build(library):
    import multiprocessing
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("The worker concurrency check needs POSIX fork")
    context = multiprocessing.get_context("fork")
    ready = context.Barrier(3)
    results = context.Queue()
    counter = library.root / "build-count.txt"
    expected = library.discovery_metadata()
    def worker():
        catalog._CACHE.clear()
        catalog._PENDING.clear()
        def build():
            with counter.open("a") as stream:
                stream.write("built\n")
            return expected
        library.discovery_metadata = build
        ready.wait(timeout=10)
        results.put(catalog.recorded_metadata(library))
    processes = [context.Process(target=worker) for _ in range(3)]
    try:
        for process in processes:
            process.start()
        actual = [results.get(timeout=20) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert actual == [expected] * 3
        assert counter.read_text().splitlines() == ["built"]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()


@pytest.mark.parametrize("error", [13, 30])
def test_unwritable_existing_cache_keeps_original_reader(library, monkeypatch, error):
    from contextlib import contextmanager
    expected = catalog.recorded_metadata(library)
    catalog._CACHE.clear()
    (library.root / ".discovery/catalog.json").unlink()
    @contextmanager
    def unavailable(path):
        raise OSError(error, "Synthetic read-only cache")
        yield
    monkeypatch.setattr(catalog, "file_lock", unavailable)
    assert catalog.recorded_metadata(library) == expected
