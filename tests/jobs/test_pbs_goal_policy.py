"""Administrator policy, path and secret boundaries for standalone PBS tools."""
import json
import os
from pathlib import Path

import pytest

from agent.harness import pbs_goal_policy as policy
from agent.rsi.engine.storage import atomic_json
from mcp_tools.tools.capability_policy import CapabilityDisabled, CapabilityStore


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    monkeypatch.delenv("VENUS_CAPABILITY_POLICY_SNAPSHOT", raising=False)
    install = tmp_path / "installation"
    install.mkdir()
    database = install / "runtime/local/state/capabilities.sqlite3"
    store = CapabilityStore(database)
    store.set("tool:calculate_physchem_from_fasta", False, 0, {}, "fixture@example.invalid")
    store.set("model:private-fixture", False, 0, {"adapter": "capability", "id": "private-fixture", "tools": ["run_generated_capability"]}, "fixture")
    config = install / "configs"
    config.mkdir()
    (config / "model_runtimes.json").write_text(json.dumps({"fixture": {"enabled": True, "python": "env/bin/python", "weights": "weights/model.pt"}}))
    catalog = config / "free.json"
    catalog.write_text('{"models": []}')
    output = tmp_path / "attempt"
    output.mkdir(mode=0o700)
    settings = {"runtime_root": str(install), "repository": str(install), "environment": {}, "execution_catalog": str(catalog)}
    record = {"record_path": str(output / "submission.json"), "settings": settings, "execution": {}}
    record["tool_settings"] = policy.freeze_settings(settings, output)
    return store, record


def test_frozen_disabled_switches_remain_disabled_without_admin_database_mutation(prepared, monkeypatch):
    store, record = prepared
    before = store.path.read_bytes()
    env = policy.verify_settings(record)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    frozen = CapabilityStore()
    assert not frozen.tool_enabled("calculate_physchem_from_sequence")
    with pytest.raises(CapabilityDisabled):
        frozen.assert_call("calculate_physchem_from_fasta", {})
    with pytest.raises(CapabilityDisabled, match="read-only"):
        frozen.set("tool:calculate_physchem_from_fasta", True, 1, {}, "fixture")
    assert store.path.read_bytes() == before
    raw = Path(env["VENUS_CAPABILITY_POLICY_SNAPSHOT"]).read_text()
    assert "fixture@example.invalid" not in raw


def test_snapshot_is_frozen_and_runtime_paths_use_installation_not_snapshot(prepared, monkeypatch):
    store, record = prepared
    store.set("tool:calculate_physchem_from_fasta", True, 1, {}, "fixture")
    env = policy.verify_settings(record)
    monkeypatch.setenv("VENUS_CAPABILITY_POLICY_SNAPSHOT", env["VENUS_CAPABILITY_POLICY_SNAPSHOT"])
    assert not CapabilityStore().tool_enabled("calculate_physchem_from_fasta")
    values = json.loads(Path(env["VENUS_MODEL_RUNTIMES_CONFIG"]).read_text())
    assert values["fixture"]["python"] == record["settings"]["runtime_root"] + "/env/bin/python"
    assert values["fixture"]["weights"] == record["settings"]["runtime_root"] + "/weights/model.pt"


@pytest.mark.parametrize("fault", ["modified", "symlink", "outside", "missing"])
def test_bound_configuration_cannot_be_replaced(prepared, tmp_path, fault):
    _, record = prepared
    reference = record["tool_settings"]["catalog"]
    target = Path(reference["path"])
    if fault == "modified":
        target.write_text('{"models": [], "changed": true}')
    elif fault == "symlink":
        other = tmp_path / "same-bytes.json"
        other.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(other)
    elif fault == "outside":
        other = tmp_path / "same-bytes.json"
        other.write_bytes(target.read_bytes())
        reference["path"] = str(other)
    else:
        target.unlink()
    with pytest.raises((ValueError, OSError)):
        policy.verify_settings(record)


def test_snapshot_mode_hash_and_private_permissions_are_checked(prepared, monkeypatch):
    _, record = prepared
    path = Path(record["tool_settings"]["capabilities"]["path"])
    monkeypatch.setenv("VENUS_CAPABILITY_POLICY_SNAPSHOT", str(path))
    monkeypatch.setenv("WEBUI_V2_MODE", "online")
    with pytest.raises(ValueError):
        CapabilityStore().rows()
    monkeypatch.setenv("WEBUI_V2_MODE", "local")
    path.chmod(0o644)
    with pytest.raises(ValueError):
        CapabilityStore().rows()
    path.chmod(0o600)
    value = json.loads(path.read_text())
    value["switches"]["tool:calculate_physchem_from_fasta"]["enabled"] = True
    atomic_json(path, value)
    with pytest.raises(ValueError):
        CapabilityStore().rows()


def test_only_named_tool_keys_are_returned_and_never_exported(prepared, tmp_path):
    _, record = prepared
    source = tmp_path / "provider.env"
    source.write_text('TAVILY_API_KEY=fixture-search\nDMXAPI_API_KEY=fixture-llm\nSERPAPI_API_KEY=fixture-serp\n')
    source.chmod(0o600)
    record["execution"]["env_file"] = str(source)
    record["settings"]["tool_credential_refs"] = ["TAVILY_API_KEY"]
    before = dict(os.environ)
    assert policy.service_environment(record) == {"TAVILY_API_KEY": "fixture-search"}
    assert dict(os.environ) == before
    record["settings"]["tool_credential_refs"] = ["DMXAPI_API_KEY"]
    with pytest.raises(ValueError):
        policy.service_environment(record)
    record["settings"]["tool_credential_refs"] = ["NCBI_API_KEY"]
    with pytest.raises(ValueError):
        policy.service_environment(record)


def test_explicit_missing_admin_configuration_fails_closed(prepared, tmp_path):
    _, record = prepared
    settings = record["settings"].copy()
    settings["capability_db"] = str(tmp_path / "missing.sqlite")
    with pytest.raises(ValueError):
        policy.freeze_settings(settings, tmp_path)
    settings.pop("capability_db")
    settings["model_runtimes"] = str(tmp_path / "missing.json")
    with pytest.raises(ValueError):
        policy.freeze_settings(settings, tmp_path)


def test_variant_weights_resolve_against_installation_and_preserve_interpreter_links(prepared, tmp_path, monkeypatch):
    from mcp_tools.models.profiles import load_runtime_settings

    _, record = prepared
    install = Path(record["settings"]["runtime_root"])
    (install / "configs/model_runtimes.json").write_text(json.dumps({"prosst": {
        "python": "env/bin/python", "weights": "runtime/models/prosst-2048",
        "variants": {"128": {"weights": "runtime/models/prosst-128", "label": "tiny variant"},
                     "512": {"weights": "/explicit/shared/512"}},
    }}))
    record["tool_settings"] = policy.freeze_settings(record["settings"], Path(record["record_path"]).parent)
    env = policy.verify_settings(record)
    frozen = tmp_path / "unrelated-source"
    frozen.mkdir()
    values = load_runtime_settings(root=frozen, env=env)
    assert values["prosst"]["python"] == str(install / "env/bin/python")
    assert values["prosst"]["variants"]["128"]["weights"] == str(install / "runtime/models/prosst-128")
    assert values["prosst"]["variants"]["512"]["weights"] == "/explicit/shared/512"
    assert values["prosst"]["variants"]["128"]["label"] == "tiny variant"


@pytest.mark.parametrize("variants", [{"128": {"api_key": "synthetic-forbidden"}}, {"128": "invalid"}, []])
def test_nested_runtime_credentials_and_malformed_variants_are_rejected(prepared, variants):
    _, record = prepared
    install = Path(record["settings"]["runtime_root"])
    (install / "configs/model_runtimes.json").write_text(json.dumps({"prosst": {"variants": variants}}))
    with pytest.raises(ValueError):
        policy.freeze_settings(record["settings"], Path(record["record_path"]).parent)



def test_asset_settings_resolve_relative_to_operator_configuration(tmp_path):
    import sys

    from agent.harness.pbs_goal_config import configuration

    raw = {"schema": "venusfactory.pbs-config.v1", "repository": "source", "control_root": "control",
           "runtime_root": "installation", "python": sys.executable, "pbs_bin": "pbs",
           "goal_roots": ["goals"], "checkpoint_root": "shared/weights", "huggingface_cache": "cache/hub",
           "profiles": {"gpu": {"queue": "ai4090d", "ncpus": 2, "ngpus": 1,
                                "memory_gb": 16, "wall_seconds": 7200}}}
    configured = configuration(raw, base=tmp_path)
    assert configured["checkpoint_root"] == str(tmp_path / "shared/weights")
    assert configured["huggingface_cache"] == str(tmp_path / "cache/hub")
    raw["environment"] = {"HOME": "/operator/home"}
    with pytest.raises(ValueError, match="environment"):
        configuration(raw, base=tmp_path)


def test_default_asset_root_does_not_share_home_or_require_models(prepared, monkeypatch):
    _, record = prepared
    monkeypatch.setenv("HF_HUB_CACHE", "/unselected/operator/cache")
    env = policy.verify_settings(record)
    assert env["VENUS_CKPT_DIR"] == record["settings"]["runtime_root"] + "/ckpt"
    assert "HF_HUB_CACHE" not in env and "HOME" not in env and "HF_HOME" not in env
    assert not Path(env["VENUS_CKPT_DIR"]).exists(), "Discovery must not install/create weights"


@pytest.mark.parametrize("field", ["checkpoint_root", "huggingface_cache"])
def test_explicit_missing_asset_directory_is_rejected(prepared, tmp_path, field):
    _, record = prepared
    record["settings"][field] = str(tmp_path / "absent")
    with pytest.raises(ValueError, match="asset directory is unavailable"):
        policy.freeze_settings(record["settings"], Path(record["record_path"]).parent)


@pytest.mark.parametrize("fault", ["changed_manifest", "symlink_manifest", "redirected_root", "missing_root"])
def test_asset_binding_rejects_tampering_and_missing_selected_cache(prepared, tmp_path, fault):
    _, record = prepared
    cache = tmp_path / "selected-cache"
    cache.mkdir()
    record["settings"]["huggingface_cache"] = str(cache)
    record["tool_settings"] = policy.freeze_settings(record["settings"], Path(record["record_path"]).parent)
    manifest = Path(record["tool_settings"]["assets"]["path"])
    if fault == "changed_manifest":
        value = json.loads(manifest.read_text())
        value["huggingface_cache"] = str(tmp_path)
        atomic_json(manifest, value)
    elif fault == "symlink_manifest":
        replacement = tmp_path / "manifest.json"
        replacement.write_bytes(manifest.read_bytes())
        manifest.unlink()
        manifest.symlink_to(replacement)
    else:
        cache.rmdir()
        if fault == "redirected_root":
            replacement = tmp_path / "other-cache"
            replacement.mkdir()
            cache.symlink_to(replacement, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        policy.verify_settings(record)


def test_readonly_assets_work_from_frozen_source_and_private_home(prepared, tmp_path):
    """Real offline subprocess imports copied sources and reads tiny asset fixtures.

    This proves path selection, not inference/weight integrity or OS read-only mounts.
    """
    import hashlib
    import shutil
    import subprocess
    import sys

    from agent.harness.pbs_goal_worker import _environment

    _, record = prepared
    install = Path(record["settings"]["runtime_root"])
    adapter = install / "ckpt/DeepSol/fixture"
    adapter.mkdir(parents=True)
    (adapter / "weights.pt").write_bytes(b"synthetic fixture, not trained weights")
    (adapter / "config.json").write_text("{}")
    mpnn = install / "ckpt/ProteinMPNN/vanilla_model_weights"
    mpnn.mkdir(parents=True)
    (mpnn / "v_48_020.pt").write_bytes(b"fixture")
    cache = tmp_path / "operator-home/.cache/huggingface/hub"
    model = cache / "models--fixture--backbone"
    snapshot = model / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (model / "refs").mkdir()
    (model / "refs/main").write_text("a" * 40)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"fixture")
    assets = [p for base in (install / "ckpt", cache) for p in base.rglob("*") if p.is_file()]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
    for p in assets:
        p.chmod(0o444)
    record["settings"]["huggingface_cache"] = str(cache)
    record["tool_settings"] = policy.freeze_settings(record["settings"], Path(record["record_path"]).parent)
    source = tmp_path / "frozen-source"
    repository = Path(__file__).resolve().parents[2]
    for name in ("src/ckpt_hub.py", "src/hub_progress.py", "mcp_tools/models/profiles.py",
                 "mcp_tools/models/catalog_snapshot.py", "mcp_tools/models/checkpoint_cache.py",
                 "mcp_tools/models/execution.py", "mcp_tools/models/pretrained/runners/proteinmpnn.py"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / name, target)
    record.update(source={"root": str(source)}, resources={"ncpus": 2, "ngpus": 1, "memory_gb": 16},
                  execution={"python": sys.executable, "environment": {}})
    env = _environment(record["record_path"], record)
    assert env["HOME"] != str(cache.parents[2])
    assert Path(env["HOME"]).is_relative_to(Path(record["record_path"]).parent)
    script = """
import json, os
from src.ckpt_hub import ensure_ckpt_path
from mcp_tools.models.pretrained.runners.proteinmpnn import weight_root
from mcp_tools.models.checkpoint_cache import cached_model
from mcp_tools.models.execution import environment
import mcp_tools.models.profiles as profiles
child = environment({})
print(json.dumps({'source': str(profiles.PROJECT),
 'adapter': str(ensure_ckpt_path('ckpt/DeepSol/fixture', download=False)),
 'mpnn': str(weight_root({})), 'backbone': cached_model('fixture/backbone'),
 'child_ckpt': child.get('VENUS_CKPT_DIR'), 'child_cache': child.get('HF_HUB_CACHE'),
 'home': child['HOME']}))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=env["HOME"], env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr[-4000:]
    values = json.loads(result.stdout)
    assert values == {"source": str(source), "adapter": str(adapter), "mpnn": str(mpnn.parent),
                      "backbone": str(snapshot), "child_ckpt": str(install / "ckpt"),
                      "child_cache": str(cache), "home": env["HOME"]}
    assert before == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
    assert not (source / "ckpt").exists()
