import hashlib

import pytest

from agent.rsi.context_runtime_compat import FILES, metadata_extension, verify_sources

OLD = "GROUP_LEVELS = {'biological_replicate'}\ndef score(x):\n    return x + 1\n"
NEW = "GROUP_LEVELS = {'biological_replicate', 'genetic_population'}\ndef score(x):\n    return x + 1\n"


def test_only_additive_metadata_changes_are_compatible():
    assert metadata_extension(OLD, NEW) == {"biological_replicate"}
    with pytest.raises(ValueError, match="numerical code"):
        metadata_extension(OLD, NEW.replace("x + 1", "x + 2"))
    with pytest.raises(ValueError, match="numerical code"):
        metadata_extension(NEW, OLD)
    with pytest.raises(ValueError, match="numerical code"):
        metadata_extension(OLD, OLD)
    with pytest.raises(ValueError, match="Ambiguous"):
        metadata_extension(OLD + "GROUP_LEVELS={'second'}\n", NEW)
    assert metadata_extension(OLD, NEW + "# formatting only\n") == {"biological_replicate"}


@pytest.fixture
def archived(tmp_path):
    root, current = tmp_path / "goal", tmp_path / "current"
    current.mkdir()
    protocol = {"implementation": {}}
    blobs = []
    for name in FILES:
        old = OLD.encode() if name == "generated_context.py" else b"# identical dependency\n"
        new = NEW.encode() if name == "generated_context.py" else old
        relative = "src/agent/rsi/" + name
        path = root / "implementation" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(old)
        (current / name).write_bytes(new)
        protocol["implementation"][relative] = hashlib.sha256(old).hexdigest()
        blobs.append(old)
    policy = {
        "plugin_sha256": hashlib.sha256(b"".join(blobs)).hexdigest(),
        "scope": {"grouping": {"level": "biological_replicate"}},
    }
    return root, current, policy, protocol


def test_original_hash_and_scope_are_required(archived):
    root, current, policy, protocol = archived
    verify_sources(root, policy, protocol, current_root=current)
    wrong = {**policy, "plugin_sha256": "0" * 64}
    with pytest.raises(ValueError, match="policy"):
        verify_sources(root, wrong, protocol, current_root=current)
    wrong = {**policy, "scope": {"grouping": {"level": "genetic_population"}}}
    with pytest.raises(ValueError, match="Historical scope"):
        verify_sources(root, wrong, protocol, current_root=current)


def test_dependency_and_archive_mutations_are_rejected(archived):
    root, current, policy, protocol = archived
    dep = current / "code_worker.py"
    dep.write_text("# changed dependency\n")
    with pytest.raises(ValueError, match="dependency"):
        verify_sources(root, policy, protocol, current_root=current)
    dep.write_bytes((root / "implementation/src/agent/rsi/code_worker.py").read_bytes())
    original = root / "implementation/src/agent/rsi/generated_context.py"
    original.write_text(OLD + "# tampered archive\n")
    with pytest.raises(ValueError, match="Archived"):
        verify_sources(root, policy, protocol, current_root=current)
