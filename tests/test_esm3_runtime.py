"""ESM3 must stay outside the platform interpreter and fail closed."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mcp_tools.models.factory.training.data import esm3_runtime as bridge
from mcp_tools.models.factory.training.data import get_esm3_structure_seq as encoder
from mcp_tools.models import execution
from scripts.bootstrap import esm3_runtime as installer
from scripts.bootstrap.runtimes import Runtimes


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    pdb = tmp_path / "protein with spaces.pdb"
    pdb.write_text("HEADER fixture")
    weights = tmp_path / "encoder.pth"
    weights.write_bytes(b"unit-test-only")
    receipt = tmp_path / "installation.json"
    receipt.write_text(json.dumps({"runtime": "esm3", "verified": True}))
    cfg = {"esm3": {"enabled": True, "python": sys.executable, "weights": str(weights),
                    "installation": str(receipt), "timeout_seconds": 19}}
    monkeypatch.setattr(bridge, "load_runtime_settings", lambda: cfg)
    return pdb, cfg


def output_for(spec, rows):
    assert spec.argv[1:3] == ["-m", "mcp_tools.models.factory.training.data.get_esm3_structure_seq"]
    inputs = json.loads(Path(spec.argv[spec.argv.index("--pdb_manifest") + 1]).read_text())
    Path(spec.argv[spec.argv.index("--out_file") + 1]).write_text(json.dumps(rows))
    return inputs


def valid(pdb):
    return {"name": pdb.name.split(".")[0], "esm3_structure_seq": [0, 1, 4095], "plddt": 80.0}


def test_dispatch_is_explicit_bounded_and_excludes_credentials(fixture, monkeypatch):
    pdb, cfg = fixture
    monkeypatch.setenv("DMX_API_KEY", "unit-test-secret")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/imports")
    def run(spec):
        assert spec.timeout == 19
        assert "DMX_API_KEY" not in spec.env
        assert "/untrusted/imports" not in spec.env["PYTHONPATH"]
        assert spec.env["PYTHON_DOTENV_DISABLED"] == "1"
        assert output_for(spec, [valid(pdb)]) == [str(pdb)]
    monkeypatch.setattr(execution, "execute", run)
    assert bridge.encode_structures([pdb]) == [valid(pdb)]


@pytest.mark.parametrize("change,message", [
    ("missing", "Install"), ("python", "Install"), ("disabled", "disabled"),
    ("weights", "checkpoint"), ("receipt", "verified inference"),
])
def test_no_host_fallback_or_unverified_execution(fixture, monkeypatch, change, message):
    pdb, cfg = fixture
    if change == "missing":
        cfg.clear()
    elif change == "python":
        cfg["esm3"].pop("python")
    elif change == "disabled":
        cfg["esm3"]["enabled"] = False
    elif change == "weights":
        cfg["esm3"]["weights"] = "/no/checkpoint"
    else:
        cfg["esm3"].pop("installation")
    monkeypatch.setattr(execution, "execute", lambda *_: pytest.fail("must not run"))
    with pytest.raises(ValueError, match=message):
        bridge.encode_structures([pdb])


@pytest.mark.parametrize("case", ["count", "name", "negative", "large", "bool", "nan", "empty", "missing"])
def test_invalid_worker_results_cannot_become_training_features(fixture, monkeypatch, case):
    pdb, _ = fixture
    row = valid(pdb)
    rows = [row]
    if case == "count": rows = []
    elif case == "name": row["name"] = "another sample"
    elif case == "nan": row["plddt"] = float("nan")
    elif case == "missing": row.pop("esm3_structure_seq")
    else: row["esm3_structure_seq"] = {"negative": [-1], "large": [4096], "bool": [True], "empty": []}[case]
    monkeypatch.setattr(execution, "execute", lambda spec: output_for(spec, rows))
    with pytest.raises(ValueError):
        bridge.encode_structures([pdb])


def test_worker_timeout_propagates_without_partial_output(fixture, monkeypatch):
    pdb, _ = fixture
    def expired(spec):
        raise subprocess.TimeoutExpired(spec.argv, spec.timeout)
    monkeypatch.setattr(execution, "execute", expired)
    with pytest.raises(subprocess.TimeoutExpired):
        bridge.encode_structures([pdb])


def test_old_csv_and_json_cli_use_the_managed_runtime(fixture, tmp_path, monkeypatch):
    pdb, _ = fixture
    monkeypatch.setattr(bridge, "encode_structures", lambda paths, device: [valid(pdb)])
    out = tmp_path / "output.json"
    encoder.main(["--pdb_file", str(pdb), "--out_format", "json", "--out_file", str(out)])
    assert json.loads(out.read_text()) == valid(pdb)
    out = tmp_path / "output.csv"
    encoder.main(["--pdb_file", str(pdb), "--out_file", str(out)])
    import csv
    with out.open() as stream:
        rows = list(csv.DictReader(stream))
    assert json.loads(rows[0]["esm3_structure_seq"]) == [0, 1, 4095]


def test_platform_imports_do_not_require_esm():
    result = subprocess.run([sys.executable, "-c",
        "import sys; from mcp_tools.models.factory.training.data import esm3_runtime,get_esm3_structure_seq; "
        "assert not any(n == 'esm' or n.startswith('esm.') for n in sys.modules)"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_installer_without_weights_stays_unverified(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "requirements/optional").mkdir(parents=True)
    for rel in ["configs/model_runtimes.example.json", "requirements/optional/esm3.txt"]:
        (tmp_path / rel).write_text((root / rel).read_text())
    calls = []
    class Env:
        torch_type = "cpu"
        def create(self, folder, version):
            python = folder / "bin/python"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"test interpreter")
            return python
        def pip(self, *args, **kwargs):
            calls.append((args, kwargs))
        def run(self, *args, **kwargs):
            pass
    env = Env()
    manifest = json.loads((root / "configs/bootstrap.json").read_text())
    runtime = Runtimes(tmp_path, env, Path(sys.executable), manifest)
    result = installer.install(runtime, "esm3")
    assert result["status"] == "needs_validation"
    config = json.loads((tmp_path / "configs/model_runtimes.json").read_text())["esm3"]
    assert config["enabled"] is False
    receipt = json.loads((tmp_path / config["installation"]).read_text())
    assert receipt["verified"] is False
    assert receipt["verification"] == "dependencies_only"
    assert all("bootstrap-constraints" not in str(call) for call in calls)


def test_esm3_advisories_do_not_exempt_platform_or_sandbox_profiles():
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("esm_audit", root / "scripts/audit_dependencies.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    from datetime import date
    entry = {"name": "transformers", "version": "4.57.6", "profiles": ["esm3"],
             "ids": ["GHSA-29pf-2h5f-8g72"], "review_by": "2026-12-08"}
    findings = {("transformers", "4.57.6"): set(entry["ids"])}
    assert audit.classify(findings, [entry], "esm3", date(2026, 11, 9))["status"] == "known_open"
    for profile in ["linux", "macos", "cube", "e2b"]:
        assert audit.classify(findings, [entry], profile, date(2026, 11, 9))["status"] == "action_required"


def test_doctor_cannot_advertise_an_unverified_encoder(tmp_path, monkeypatch):
    from scripts.bootstrap import checks
    (tmp_path / "configs").mkdir()
    python = tmp_path / "python"
    python.write_text("fixture")
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fixture")
    (tmp_path / "configs/model_runtimes.json").write_text(json.dumps({
        "esm3": {"enabled": True, "python": str(python), "weights": str(weights)},
    }))
    monkeypatch.setattr(checks, "capture", lambda *a, **k: pytest.fail("unverified runtime must not run"))
    health = checks.tool_health(tmp_path, "esm3", sys.executable)
    assert health["status"] == "failed"
    assert {"installation", "inference_verification"} <= set(health["issues"])
