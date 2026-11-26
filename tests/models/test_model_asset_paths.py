"""Checkpoint selection and child environment boundaries without loading a model."""
import argparse

from mcp_tools.models.execution import environment
from mcp_tools.models.pretrained.runners.proteinmpnn import prepare, weight_root


def test_mpnn_uses_shared_checkpoint_root_but_explicit_runtime_has_priority(tmp_path, monkeypatch):
    checkpoint = tmp_path / "installed"
    default = checkpoint / "ProteinMPNN/vanilla_model_weights"
    default.mkdir(parents=True)
    (default / "v_48_020.pt").write_bytes(b"fixture")
    explicit = tmp_path / "profile"
    (explicit / "vanilla_model_weights").mkdir(parents=True)
    (explicit / "vanilla_model_weights/v_48_020.pt").write_bytes(b"other fixture")
    monkeypatch.setenv("VENUS_CKPT_DIR", str(checkpoint))
    assert weight_root({}) == default.parent
    assert weight_root({"proteinmpnn": {"weights": str(explicit)}}) == explicit
    args = argparse.Namespace(model_name="v_48_020", out_folder=str(tmp_path / "out"))
    assert prepare(args, {})["path_to_model_weights"] == str(default)
    assert prepare(args, {"proteinmpnn": {"weights": str(explicit)}})["path_to_model_weights"] == str(explicit / "vanilla_model_weights")


def test_runtime_forwards_only_asset_paths_not_provider_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("VENUS_CKPT_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HOME", str(tmp_path / "private-home"))
    monkeypatch.setenv("DMXAPI_API_KEY", "synthetic-not-for-child")
    monkeypatch.setenv("TAVILY_API_KEY", "synthetic-not-for-child")
    env = environment({})
    assert env["VENUS_CKPT_DIR"] == str(tmp_path / "weights")
    assert env["HF_HUB_CACHE"] == str(tmp_path / "hub")
    assert env["HOME"] == str(tmp_path / "private-home")
    assert "DMXAPI_API_KEY" not in env and "TAVILY_API_KEY" not in env
