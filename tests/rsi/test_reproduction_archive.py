"""The long-job host must start from an immutable source snapshot."""

import importlib.util
from pathlib import Path


def test_readonly_overlapping_source_groups_are_archived_once(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts/rsi"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("reproduction_archive_test", scripts / "run_harness_reproduction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "source"
    path = root / "src/agent/rsi/engine/runtime.py"
    path.parent.mkdir(parents=True)
    path.write_text("frozen = True\n")
    path.chmod(0o444)
    monkeypatch.setattr(module, "ROOT", root)
    real_copy = module.shutil.copy2
    copied = []

    def record_copy(src, dst):
        copied.append(src)
        return real_copy(src, dst)

    monkeypatch.setattr(module.shutil, "copy2", record_copy)
    hashes = module.archive_implementation([path, path], tmp_path / "run")
    archived = tmp_path / "run/implementation/src/agent/rsi/engine/runtime.py"
    assert copied == [path]
    assert archived.read_bytes() == path.read_bytes() == b"frozen = True\n"
    assert hashes == {"src/agent/rsi/engine/runtime.py": module.file_hash(archived)}
    assert path.stat().st_mode & 0o222 == 0
