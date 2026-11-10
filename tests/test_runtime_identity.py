import os
import shutil

from backend import runtime_identity


def test_revision_tracks_source_contents_and_inventory_without_git(tmp_path):
    root = tmp_path / "release"
    code = root / "backend/app.py"
    code.parent.mkdir(parents=True)
    code.write_text("value = 1")
    original = runtime_identity.source_revision(root)
    stat = code.stat()
    code.write_text("value = 2")
    os.utime(code, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert runtime_identity.source_revision(root) != original
    code.write_text("value = 1")
    extra = root / "backend/added.py"
    extra.write_text("value = 1")
    assert runtime_identity.source_revision(root) != original
    extra.unlink()
    assert runtime_identity.source_revision(root) == original
    clone = tmp_path / "copy"
    shutil.copytree(root, clone)
    assert runtime_identity.source_revision(clone) == original


def test_revision_excludes_private_runtime_and_credentials(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/config.py").write_text("mode = 'local'")
    before = runtime_identity.source_revision(tmp_path)
    for name in (".env", "runtime/local/queue/bootstrap.py", "runtime/models/weights.pt",
                 "src/__pycache__/ignored.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private fixture content")
    assert runtime_identity.source_revision(tmp_path) == before
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/catalog.json").write_text('{"model":"example"}')
    assert runtime_identity.source_revision(tmp_path) != before


def test_loaded_revision_does_not_change_for_running_process(monkeypatch):
    runtime_identity.loaded_source_revision.cache_clear()
    version = ["first"]
    monkeypatch.setattr(runtime_identity, "source_revision", lambda: version[0])
    try:
        assert runtime_identity.loaded_source_revision() == "first"
        version[0] = "second"
        assert runtime_identity.loaded_source_revision() == "first"
    finally:
        runtime_identity.loaded_source_revision.cache_clear()


def test_worker_generation_cannot_be_overridden_by_submitted_environment(monkeypatch):
    from backend.web_v2 import worker_pool
    version = ["old"]
    monkeypatch.setattr(worker_pool, "loaded_source_revision", lambda: version[0])
    env = {"WEBUI_V2_MODE": "local", "__venus_source_revision__": "user-value"}
    old = worker_pool.namespace_for(env)
    version[0] = "new"
    assert worker_pool.namespace_for(env) != old
    env["__venus_source_revision__"] = "another-user-value"
    new = worker_pool.namespace_for(env)
    env["__venus_source_revision__"] = "user-value"
    assert worker_pool.namespace_for(env) == new
