import json
from types import SimpleNamespace
import pytest
from scripts.bootstrap.generative_runtimes import checkpoint_files, install
from scripts.bootstrap.checks import tool_health

def test_sharded_checkpoint_requires_every_shard(tmp_path):
    (tmp_path/"config.json").write_text("{}")
    (tmp_path/"model-1.safetensors").write_bytes(b"fixture")
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"a":"model-1.safetensors","b":"model-2.safetensors"}}))
    with pytest.raises(ValueError,match="missing"):
        checkpoint_files(tmp_path)
    (tmp_path/"model-2.safetensors").write_bytes(b"fixture")
    assert tmp_path/"model-2.safetensors" in checkpoint_files(tmp_path)

def test_original_esmfold_can_use_official_bin_checkpoint(tmp_path):
    (tmp_path/"config.json").write_text("{}")
    (tmp_path/"pytorch_model.bin").write_bytes(b"fixture")
    with pytest.raises(ValueError,match="incomplete"):
        checkpoint_files(tmp_path)
    assert tmp_path/"pytorch_model.bin" in checkpoint_files(tmp_path,allow_legacy_bin=True)

@pytest.mark.parametrize("name",["../outside.safetensors","/etc/passwd", "."])
def test_checkpoint_index_cannot_escape_its_directory(tmp_path,name):
    (tmp_path/"config.json").write_text("{}")
    (tmp_path/"model.safetensors").write_bytes(b"fixture")
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"a":name}}))
    with pytest.raises(ValueError,match="shard path"):
        checkpoint_files(tmp_path)

def test_existing_administrator_runtime_is_preserved(tmp_path):
    config=tmp_path/"configs";config.mkdir()
    profile={"esmfold2":{"enabled":False,"python":"/custom/python","weights":"/custom/weights"}}
    path=config/"model_runtimes.json";path.write_text(json.dumps(profile))
    (config/"model_runtimes.example.json").write_text("{}")
    runner=SimpleNamespace(root=tmp_path,base=tmp_path/"runtime",config=path,manifest={"tools":{"esmfold2":{}}})
    assert install(runner,"esmfold2")["status"]=="existing"
    assert json.loads(path.read_text())==profile

def test_doctor_detects_missing_ccd_without_importing_model(tmp_path,monkeypatch):
    config=tmp_path/"configs";config.mkdir()
    (config/"model_runtimes.json").write_text(json.dumps({"esmfold2":{"ccd":"missing.pkl"}}))
    result=tool_health(tmp_path,"esmfold2","unused")
    assert result["status"]=="failed" and "ccd" in result["issues"]

def test_doctor_detects_missing_receipt_assets_without_hash_mode(tmp_path):
    config=tmp_path/"configs";config.mkdir()
    receipt=tmp_path/"receipt.json";receipt.write_text(json.dumps({"verified":True,"required_files":["missing.weight"]}))
    (config/"model_runtimes.json").write_text(json.dumps({"custom":{"installation":"receipt.json"}}))
    result=tool_health(tmp_path,"custom","unused",hashes=False)
    assert result["status"]=="failed"
    assert "required_asset:missing.weight" in result["issues"]


def test_bindcraft_animation_requires_encoder_in_runtime_environment(tmp_path, monkeypatch):
    config = tmp_path / "configs"
    config.mkdir()
    (tmp_path / "advanced.json").write_text('{"save_design_animations": true}')
    (config / "model_runtimes.json").write_text(json.dumps({
        "bindcraft": {"advanced": "advanced.json"}
    }))
    monkeypatch.setattr("scripts.bootstrap.checks.shutil.which", lambda *a, **k: None)
    result = tool_health(tmp_path, "bindcraft", "/runtime/bin/python")
    assert result["status"] == "failed" and "ffmpeg" in result["issues"]
