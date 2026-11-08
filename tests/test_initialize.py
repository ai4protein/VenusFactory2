"""Provisioning invariants: isolated fixtures, no network/model calls."""
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.bootstrap import assets, cli
from scripts.bootstrap.checks import factory_health, failed
from scripts.bootstrap.common import InstallLock, Runner, add_missing, atomic_json, inside
from scripts.bootstrap.runtimes import Runtimes
from scripts.bootstrap.system import Environment, probe


def test_entrypoint_resolves_repository():
    assert cli.ROOT == Path(__file__).resolve().parents[1]


def test_plan_is_non_mutating(tmp_path, monkeypatch, capsys):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/bootstrap.json").write_text(json.dumps({"profiles": {"minimal": []}, "tools": {}}))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    assert cli.main(["plan", "--profile", "minimal", "--json"]) == 0
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    assert json.loads(capsys.readouterr().out)["factory_bytes"] == 0


def test_selection_validates_before_installing():
    args = cli.parser().parse_args(["install", "--stages", "tools,environment", "--tools", "x,x"])
    assert cli.selection(args, {"tools": {"x": {}}}) == (["environment", "tools"], ["x"])
    with pytest.raises(ValueError, match="Unknown tool"):
        cli.selection(args, {"tools": {}})


def test_path_escape_and_symlink_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        inside(tmp_path, "../elsewhere")
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        inside(tmp_path, "link/outside")
    assert inside(tmp_path, "allowed/a") == tmp_path / "allowed/a"


def test_preserve_administrator_profile_and_nested_variants():
    old = {"enabled": False, "python": "/own/env/python", "variants": {"20": {"weights": "own"}}}
    new = {"enabled": True, "python": "new", "cache": "cache", "variants": {"20": {"weights": "replace"}, "128": {"weights": "128"}}}
    result = add_missing(old, new)
    assert result["enabled"] is False and result["python"] == "/own/env/python"
    assert result["variants"]["20"]["weights"] == "own"
    assert result["variants"]["128"]["weights"] == "128"


def test_atomic_json_leaves_valid_file(tmp_path):
    path = tmp_path / "a/config.json"
    atomic_json(path, {"one": 1})
    atomic_json(path, {"two": 2})
    assert json.loads(path.read_text()) == {"two": 2}
    assert not list(path.parent.glob(".init-*"))


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_archive_traversal_is_rejected_before_extracting(tmp_path, kind):
    path = tmp_path / ("bad." + kind)
    if kind == "zip":
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("../outside", "bad")
    else:
        with tarfile.open(path, "w") as z:
            member = tarfile.TarInfo("../outside")
            member.size = 3
            z.addfile(member, io.BytesIO(b"bad"))
    with pytest.raises(ValueError):
        assets.extract(path, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


def test_archive_links_and_bombs_are_rejected(tmp_path):
    path = tmp_path / "link.tar"
    with tarfile.open(path, "w") as z:
        entry = tarfile.TarInfo("link")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "/etc"
        z.addfile(entry)
    with pytest.raises(ValueError):
        assets.extract(path, tmp_path / "out")
    path = tmp_path / "large.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", "12345")
    with pytest.raises(ValueError):
        assets.extract(path, tmp_path / "out", max_bytes=4)


def test_download_checksum_and_retry_cleanup(tmp_path):
    path = tmp_path / "weights.pt"
    with patch.object(assets.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong")):
        with pytest.raises(ValueError, match="Checksum"):
            assets.download("https://example.org/weights", path, expected="0" * 64)
    assert not path.exists() and not list(tmp_path.glob(".download-*"))
    expected = hashlib.sha256(b"right").hexdigest()
    with patch.object(assets.urllib.request, "urlopen", return_value=io.BytesIO(b"right")):
        assets.download("https://example.org/weights", path, expected=expected)
    with patch.object(assets.urllib.request, "urlopen", side_effect=AssertionError("No network on repeat")):
        assert assets.download("https://example.org/weights", path, expected=expected) == path
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Existing asset"):
        assets.download("https://example.org/weights", path, expected=expected)
    assert path.read_bytes() == b"corrupt"


def test_unknown_environment_is_never_modified(tmp_path):
    folder = tmp_path / "old"
    (folder / "bin").mkdir(parents=True)
    (folder / "bin/python").write_text("preserve")
    def no_run(*args, **kwargs):
        raise AssertionError("must not invoke a package manager")
    env = Environment(tmp_path, no_run)
    with pytest.raises(ValueError, match="not initializer-owned"):
        env.create(folder, "3.12")
    assert env.platform(folder, reuse=True) == folder / "bin/python"


def test_different_installer_cannot_claim_active_lock(tmp_path):
    path = tmp_path / "lock"
    with InstallLock(path):
        with pytest.raises(RuntimeError, match="Another initializer"):
            with InstallLock(path):
                pass
    with InstallLock(path):
        pass


def test_factory_manifest_checks_hash_not_only_size(tmp_path):
    folder = tmp_path / "ckpt"
    folder.mkdir()
    entry = {"path": "demo/model.pt", "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    atomic_json(folder / "manifest.json", {"presets": {"demo": ["demo/**"]}, "files": [entry]})
    assert factory_health(tmp_path, "demo")["status"] == "failed"
    (folder / "demo").mkdir()
    (folder / "demo/model.pt").write_bytes(b"xyz")
    assert factory_health(tmp_path, "demo")["status"] == "ready"
    assert factory_health(tmp_path, "demo", hashes=True)["status"] == "failed"
    (folder / "demo/model.pt").write_bytes(b"abc")
    assert factory_health(tmp_path, "demo", hashes=True)["status"] == "ready"


def test_manual_or_failed_stage_makes_exit_unsuccessful():
    assert failed({"steps": {"tool": {"status": "manual"}}})
    assert failed({"steps": {"tool": {"status": "needs_validation"}}})
    assert not failed({"sandbox": {"status": "missing"}, "steps": {"status": "ready"}})


def test_command_failure_is_reported_without_environment(tmp_path):
    run = Runner(tmp_path, tmp_path / "log")
    with pytest.raises(RuntimeError, match="exited 7"):
        run([sys.executable, "-c", "raise SystemExit(7)"])


def test_cpu_is_selected_for_old_or_unknown_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.bootstrap.system.executable", lambda name: "/fake/smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr("scripts.bootstrap.system.capture", lambda args, **kw: "GPU 0" if "-L" in args else "CUDA Version: 12.4")
    assert probe(tmp_path)["torch_type"] == "cpu"


def test_new_runtime_profile_merging_keeps_disabled_admin_settings(tmp_path):
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs/model_runtimes.json"
    atomic_json(path, {"s4pred": {"enabled": False, "timeout_seconds": 90}})
    env = Environment(tmp_path, None)
    runtimes = Runtimes(tmp_path, env, Path(sys.executable), {})
    runtimes.merge("s4pred", {"enabled": True, "python": "env/python", "timeout_seconds": 300})
    result = json.loads(path.read_text())["s4pred"]
    assert result["enabled"] is False and result["timeout_seconds"] == 90
    assert result["python"] == "env/python"
    assert len(list((tmp_path / "runtime/local/bootstrap/backups").glob("model_runtimes-*.json"))) == 1


def test_uv_and_conda_create_only_requested_environment(tmp_path):
    for manager in ["uv", "conda"]:
        calls = []
        env = Environment(tmp_path, lambda args, **kw: calls.append(args), manager=manager)
        env.command = [manager]
        target = tmp_path / manager / "environment"
        python = env.create(target, "3.12")
        assert python == target / "bin/python"
        assert str(target) in list(map(str, calls[0]))
        assert not any("--user" in args or "sudo" in args for args in calls)
        assert "--prefix" in calls[0] if manager == "conda" else "--seed" in calls[0]


def test_failed_dependency_install_is_not_marked_ready(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='sample'\n")
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements/bootstrap-constraints.txt").write_text("torch==2.8.0\n")
    calls = []
    def fail(args, **kw):
        calls.append(args)
        if "pip" in args:
            raise RuntimeError("failed download")
    env = Environment(tmp_path, fail)
    env.command = ["uv"]
    with pytest.raises(RuntimeError):
        env.platform(tmp_path / "env")
    assert not (tmp_path / "env/.venus-dependencies.json").exists()


def test_git_fetch_avoids_lazy_blob_network_requests(tmp_path):
    calls = []
    def run(args, **kw):
        calls.append(args)
    with patch.object(assets, "capture", return_value="a" * 40):
        assets.clone(run, "https://github.com/example/model.git", "a" * 40, tmp_path / "source")
    joined = repr(calls)
    assert "--filter=blob:none" not in joined
    assert "fetch" in joined and "--depth" in joined and "FETCH_HEAD" in joined


def test_hf_endpoint_never_accepts_embedded_credentials(monkeypatch, tmp_path):
    (tmp_path / "configs").mkdir()
    atomic_json(tmp_path / "configs/bootstrap.json", {"profiles": {"minimal": []}, "tools": {}})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    assert cli.main(["plan", "--profile", "minimal", "--hf-endpoint", "https://user:secret@mirror.example"]) == 2


def test_frontend_prebuild_has_python_without_environment_activation(tmp_path, monkeypatch):
    from scripts.bootstrap import system
    binaries = tmp_path / "node/bin"
    binaries.mkdir(parents=True)
    for name in ("node", "npm"):
        (binaries / name).write_text("")
    monkeypatch.setattr(system, "executable", lambda name: str(binaries / name))
    monkeypatch.setattr(system, "capture", lambda *a, **kw: "v24.15.0")
    calls = []
    system.frontend(tmp_path, lambda args, **kw: calls.append((args, kw)), Path(sys.executable))
    launcher = tmp_path / "runtime/local/bootstrap/bin/python"
    assert launcher.is_file() and 'exec ' in launcher.read_text()
    assert str(launcher.parent) in calls[-1][1]["extra_env"]["PATH"]
    assert calls[0][0][1] == "ci" and calls[1][0][1:] == ["run", "build"]


def test_tar_with_npz_members_is_not_misclassified_as_zip(tmp_path):
    # Real parameter bundles contain ZIP-formatted NPZ files inside a TAR.
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as handle:
        handle.writestr("alphafold/weight.npy", b"tensor-fixture")
    payload = nested.getvalue()
    archive = tmp_path / "params.tar"
    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo("params_model_1.npz")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    assert zipfile.is_zipfile(archive)  # This was the misleading old detection.
    assets.extract(archive, tmp_path / "params")
    assert (tmp_path / "params/params_model_1.npz").read_bytes() == payload
    assert not (tmp_path / "params/alphafold").exists()
