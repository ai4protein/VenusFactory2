"""Installer boundaries: portable inputs, private environments and honest readiness."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.bootstrap import cli, structure_runtimes as structure
from scripts.bootstrap.common import atomic_json
from scripts.bootstrap.runtimes import Runtimes

ROOT = Path(__file__).resolve().parents[1]


def test_aliases_and_database_download_are_explicit():
    manifest = json.loads((ROOT / "configs/bootstrap.json").read_text())
    args = cli.parser().parse_args(["install", "--tools", "AlphaFold2,Proteinix,Mol-1"])
    assert cli.selection(args, manifest) == (["tools"], ["alphafold2", "protenix", "mol1"])
    assert args.alphafold_databases == "none"
    args = cli.parser().parse_args(["install", "--tools", "mol1", "--alphafold-databases", "full"])
    with pytest.raises(ValueError, match="requires"):
        cli.selection(args, manifest)


def test_database_none_never_starts_an_external_process(tmp_path):
    structure.alphafold_databases(SimpleNamespace(), tmp_path, "none")


def test_database_disk_check_happens_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(structure.shutil, "disk_usage", lambda _: SimpleNamespace(free=10 * 2**30))
    with pytest.raises(ValueError, match="650 GiB"):
        structure.alphafold_databases(SimpleNamespace(), tmp_path, "reduced")


def test_sources_are_pinned_and_restricted_weights_are_not_selected():
    manifest = json.loads((ROOT / "configs/bootstrap.json").read_text())["tools"]
    for name in ("alphafold2", "protenix", "mol1"):
        spec = manifest[name]
        assert len(spec["revision"]) == 40
        assert spec["installer"] == "structure"
    assert manifest["protenix"]["checkpoint"] == "protenix_base_default_v1.0.0.pt"
    for name in ("prosst", "esm1v", "esmif1", "protssn"):
        assert manifest[name]["python"]
        assert not manifest[name].get("platform_environment")


def test_unmanaged_source_is_not_overwritten(tmp_path):
    target = tmp_path / "runtime"
    (target / "source").mkdir(parents=True)
    (target / "source/user.py").write_text("preserve")
    with pytest.raises(ValueError, match="Existing source"):
        structure.source_snapshot(None, {"repository": "https://github.com/example/tool.git", "revision": "a" * 40}, target)
    assert (target / "source/user.py").read_text() == "preserve"


def test_existing_administrator_runtime_is_not_changed(tmp_path):
    path = tmp_path / "configs/model_runtimes.json"
    original = {"mol1": {"enabled": False, "python": "/private/python"}}
    atomic_json(path, original)
    atomic_json(path.with_name("model_runtimes.example.json"), {})
    runtime = SimpleNamespace(root=tmp_path, config=path, base=tmp_path / "runtime", manifest={"tools": {"mol1": {}}})
    assert structure.install(runtime, "mol1")["status"] == "existing"
    assert json.loads(path.read_text()) == original


def test_native_tools_stay_in_private_environment(tmp_path, monkeypatch):
    calls = []
    class FakeEnvironment:
        def __init__(self, root, run, **kwargs):
            assert kwargs["manager"] == "conda"
        def create(self, folder, version):
            folder.mkdir(parents=True, exist_ok=True)
            assert folder == tmp_path / "mol1/bioenv"
        def ensure_manager(self):
            return ["/conda"]
    monkeypatch.setattr(structure, "Environment", FakeEnvironment)
    runtime = SimpleNamespace(root=tmp_path, run=lambda args, **kw: calls.append(args))
    structure.bio_dependencies(runtime, "mol1", tmp_path / "mol1")
    args = calls[0]
    assert "--prefix" in args and "--name" not in args
    assert args[args.index("--prefix") + 1] == tmp_path / "mol1/bioenv"
    assert "kalign3=3.3.5" in args
    structure.bio_dependencies(runtime, "mol1", tmp_path / "mol1")
    assert len(calls) == 1


def test_runtime_cache_and_alignment_paths_are_private(tmp_path):
    target = tmp_path / "mol1"
    target.mkdir()
    python = target / "env/bin/python"
    env = structure.runtime_environment("mol1", target, python)
    assert env["MOL_ROOT_DIR"] == str(target / "data")
    assert env["TORCH_EXTENSIONS_DIR"] == str(target / "cache/extensions")
    assert env["PATH"].split(":")[:2] == [str(python.parent), str(target / "bioenv/bin")]


def test_foundation_download_uses_own_python(tmp_path, monkeypatch):
    from scripts.bootstrap import runtimes
    called = []
    def snapshot(run, python, model, folder, *args):
        called.append(python)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.safetensors").write_bytes(b"fixture")
    monkeypatch.setattr(runtimes, "snapshot", snapshot)
    runtime = Runtimes(tmp_path, SimpleNamespace(run=None), Path("/platform/python"), {})
    runtime.foundation("esm1v", {"python": "runtime/esm1v/env/bin/python"}, tmp_path / "source", tmp_path / "esm1v")
    assert called == [tmp_path / "runtime/esm1v/env/bin/python"]


def test_native_cli_uses_the_installed_interpreter_and_no_model_api_keys(tmp_path, monkeypatch):
    from scripts.run_structure import prepare
    target = tmp_path / "runtime/local/model-runtimes/protenix"
    python = target / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture")
    receipt = target / "installation.json"
    atomic_json(receipt, {"runtime": "protenix"})
    atomic_json(tmp_path / "configs/model_runtimes.json", {"protenix": {
        "installation": str(receipt.relative_to(tmp_path)),
        "python": str(python.relative_to(tmp_path))
    }})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-fixture")
    args, env = prepare("protenix", ["pred", "-i", "input with spaces.json"], root=tmp_path)
    assert args == [str(python.parent / "protenix"), "pred", "-i", "input with spaces.json"]
    assert "OPENROUTER_API_KEY" not in env
    assert env["PROTENIX_ROOT_DIR"] == str(target / "data")


def test_reference_data_symlink_cannot_escape_runtime(tmp_path):
    target = tmp_path / "runtime"
    target.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (target / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="leaves"):
        structure.data_archive(target, "https://example.org/common.tar.gz", ["common/components.cif"])


@pytest.mark.parametrize("name", ["alphafold2", "protenix", "mol1"])
def test_new_runtime_cannot_execute_without_inference_evidence(tmp_path, name):
    import sys
    from mcp_tools.models.execution import profile
    receipt = tmp_path / "installation.json"
    atomic_json(receipt, {"runtime": name, "verified": False})
    with pytest.raises(ValueError, match="verified inference"):
        profile(name, settings={name: {"enabled": True, "python": sys.executable, "installation": str(receipt)}})
