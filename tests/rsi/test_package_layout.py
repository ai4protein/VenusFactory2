import importlib
import subprocess
import sys
from pathlib import Path


def test_legacy_imports_share_canonical_identity():
    for old,new in [('model_gateway','engine.model_gateway'),('protein_references','engine.protein_references'),
                    ('case_plugins','campaigns.case_plugins'),('rfantibody_targets','cases.rfantibody_targets')]:
        assert importlib.import_module('agent.rsi.'+old) is importlib.import_module('agent.rsi.'+new)


def test_importing_rsi_does_not_eagerly_load_model_implementations():
    result=subprocess.run([sys.executable,'-c','import sys; import agent.rsi; assert "agent.rsi.engine.generated_model" not in sys.modules; assert "torch" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_source_snapshot_contains_subpackage_and_classified_scripts():
    from agent.rsi.engine.source_snapshot import research_implementation_paths
    root=Path(__file__).resolve().parents[2]
    names={p.relative_to(root).as_posix() for p in research_implementation_paths(root)}
    assert {'src/agent/rsi/_compat.py','src/agent/rsi/artifacts/package.py',
            'scripts/rsi/cases/run_clean_annotation.py','scripts/rsi/data_preparation/prepare_poisk.py'}<=names


def test_relocated_source_keeps_complete_hash_binding_and_existing_path_precedence(tmp_path):
    import hashlib
    import pytest
    from agent.rsi.datasets.model_tasks import bound_file
    old = tmp_path / "checkout/src/agent/rsi/sabio_assay_encoder.py"
    current = old.parent / "engine/sabio_assay_encoder.py"
    current.parent.mkdir(parents=True)
    original = b"original encoder source\n"
    current.write_bytes(original)
    reference = {"path": str(old), "sha256": hashlib.sha256(original).hexdigest()}
    assert bound_file(reference) == current
    current.write_bytes(b"modified encoder source\n")
    with pytest.raises(ValueError, match="changed"):
        bound_file(reference)
    current.write_bytes(original)
    old.write_bytes(b"a different existing source")
    with pytest.raises(ValueError, match="changed"):
        bound_file(reference)
    old.write_bytes(original)
    assert bound_file(reference) == old


def test_relocated_source_does_not_search_other_roots_or_guess_unknown_names(tmp_path):
    import pytest
    from agent.rsi.datasets.model_tasks import bound_file
    from agent.rsi._compat import relocated_source
    for old in [tmp_path / "unknown.py", tmp_path / "src/agent/rsi/unregistered.py",
                tmp_path / "arbitrary/rsi/sabio_assay_encoder.py"]:
        assert relocated_source(old) == old
        with pytest.raises(ValueError, match="Missing"):
            bound_file({"path": str(old), "sha256": "0" * 64})
    relative = Path("src/agent/rsi/sabio_assay_encoder.py")
    with pytest.raises(ValueError, match="absolute"):
        bound_file({"path": str(relative), "sha256": "0" * 64})


def test_relocated_sources_reject_symlinks_and_special_files(tmp_path):
    import hashlib
    import os
    import pytest
    from agent.rsi.datasets.model_tasks import bound_file
    old = tmp_path / "checkout/src/agent/rsi/sabio_assay_encoder.py"
    current = old.parent / "engine/sabio_assay_encoder.py"
    current.parent.mkdir(parents=True)
    original = b"original source"
    current.write_bytes(original)
    reference = {"path": str(old), "sha256": hashlib.sha256(original).hexdigest()}
    old.symlink_to(current)
    with pytest.raises(ValueError, match="non-symlinked"):
        bound_file(reference)
    old.unlink()
    saved = current.with_suffix(".saved")
    current.rename(saved)
    current.symlink_to(saved)
    with pytest.raises(ValueError, match="non-symlinked"):
        bound_file(reference)
    current.unlink()
    os.mkfifo(current)
    with pytest.raises(ValueError, match="Missing"):
        bound_file(reference)
