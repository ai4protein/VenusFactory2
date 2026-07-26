"""Unit tests for Hugging Face checkpoint hub helpers."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ckpt_hub
from config import CkptHubConfig


@pytest.fixture
def ckpt_env(tmp_path, monkeypatch):
    root = tmp_path / "ckpt"
    root.mkdir()
    monkeypatch.setenv("VENUS_CKPT_DIR", str(root))
    monkeypatch.setenv("VENUS_CKPT_REPO_ID", "AI4Protein/VenusFactory2-ckpts")
    monkeypatch.setenv("VENUS_CKPT_REVISION", "main")
    monkeypatch.setenv("VENUS_CKPT_AUTO_DOWNLOAD", "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    return root


class TestNormalizeAndPaths:
    def test_normalize_strips_ckpt_prefix(self, ckpt_env):
        assert ckpt_hub.normalize_rel_path("ckpt/DeepSol/ankh-large") == "DeepSol/ankh-large"
        assert ckpt_hub.normalize_rel_path("DeepSol/ankh-large") == "DeepSol/ankh-large"
        assert ckpt_hub.normalize_rel_path(ckpt_env / "demo" / "x.pt") == "demo/x.pt"

    def test_normalize_rejects_outside_root(self, ckpt_env, tmp_path):
        outside = tmp_path / "other" / "model.pt"
        outside.parent.mkdir()
        outside.write_bytes(b"x")
        with pytest.raises(ValueError, match="outside ckpt root"):
            ckpt_hub.normalize_rel_path(outside)

    def test_local_path(self, ckpt_env):
        assert ckpt_hub.local_path("demo/a.pt") == (ckpt_env / "demo" / "a.pt").resolve()


class TestPresets:
    def test_list_presets_contains_core(self):
        names = ckpt_hub.list_presets()
        assert "demo" in names
        assert "predict-core" in names
        assert "proteinmpnn" in names
        assert "all" in names

    def test_preset_patterns_predict_core(self):
        patterns = ckpt_hub.preset_patterns("predict-core")
        assert "demo/**" in patterns
        assert "DeepSol/ankh-large/**" in patterns
        assert "ProteinMPNN/**" not in patterns

    def test_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown ckpt preset"):
            ckpt_hub.preset_patterns("not-a-real-preset")


class TestEnsureLocalHits:
    def test_ensure_file_uses_local_without_download(self, ckpt_env, monkeypatch):
        target = ckpt_env / "demo" / "demo_solubility.pt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"weights")

        def boom(*_a, **_k):
            raise AssertionError("should not download")

        monkeypatch.setattr(ckpt_hub, "download_file", boom)
        got = ckpt_hub.ensure_ckpt_file("demo/demo_solubility.pt")
        assert got == target.resolve()

    def test_ensure_dir_uses_local_adapter(self, ckpt_env, monkeypatch):
        adapter = ckpt_env / "DeepSol" / "ankh-large"
        adapter.mkdir(parents=True)
        (adapter / "lr5e-4_bt12k_ga8.pt").write_bytes(b"pt")
        (adapter / "lr5e-4_bt12k_ga8.json").write_text("{}", encoding="utf-8")

        def boom(*_a, **_k):
            raise AssertionError("should not download")

        monkeypatch.setattr(ckpt_hub, "download_patterns", boom)
        got = ckpt_hub.ensure_ckpt_dir("DeepSol/ankh-large", require_json=True)
        assert got == adapter.resolve()

    def test_ensure_path_custom_outside_hub(self, ckpt_env, tmp_path):
        custom = tmp_path / "my_model"
        custom.mkdir()
        (custom / "model.pt").write_bytes(b"x")
        got = ckpt_hub.ensure_ckpt_path(custom)
        assert got == custom.resolve()


class TestAutoDownload:
    def test_ensure_file_downloads_when_missing(self, ckpt_env, monkeypatch):
        calls = []

        def fake_download(rel, *, force=False):
            calls.append(rel)
            dest = ckpt_env / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"downloaded")
            return dest.resolve()

        monkeypatch.setattr(ckpt_hub, "download_file", fake_download)
        got = ckpt_hub.ensure_ckpt_file("demo/demo_solubility.pt")
        assert calls == ["demo/demo_solubility.pt"]
        assert got.is_file()
        assert got.read_bytes() == b"downloaded"

    def test_ensure_dir_downloads_prefix(self, ckpt_env, monkeypatch):
        calls = []

        def fake_patterns(patterns, *, force=False):
            calls.append(list(patterns))
            adapter = ckpt_env / "DeepSol" / "ankh-large"
            adapter.mkdir(parents=True, exist_ok=True)
            (adapter / "a.pt").write_bytes(b"pt")
            (adapter / "a.json").write_text("{}", encoding="utf-8")
            return ckpt_env

        monkeypatch.setattr(ckpt_hub, "download_patterns", fake_patterns)
        got = ckpt_hub.ensure_ckpt_dir("DeepSol/ankh-large", require_json=True)
        assert calls == [["DeepSol/ankh-large/**"]]
        assert got.is_dir()

    def test_auto_download_disabled_raises(self, ckpt_env, monkeypatch):
        monkeypatch.setenv("VENUS_CKPT_AUTO_DOWNLOAD", "0")
        with pytest.raises(FileNotFoundError, match="download_ckpts.py"):
            ckpt_hub.ensure_ckpt_file("demo/missing.pt", download=None)

    def test_offline_env_disables_auto_download(self, monkeypatch):
        monkeypatch.setenv("VENUS_CKPT_AUTO_DOWNLOAD", "1")
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert ckpt_hub.auto_download_enabled() is False

    def test_download_file_calls_hf_hub(self, ckpt_env, monkeypatch):
        recorded = {}

        def fake_hf_hub_download(**kwargs):
            recorded.update(kwargs)
            dest = Path(kwargs["local_dir"]) / kwargs["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"hub")
            return str(dest)

        def fake_snapshot_download(**kwargs):
            raise AssertionError("snapshot should not be used for single file")

        monkeypatch.setattr(
            ckpt_hub,
            "_import_hub",
            lambda: (fake_hf_hub_download, fake_snapshot_download),
        )
        got = ckpt_hub.download_file("demo/demo_solubility.pt")
        assert recorded["repo_id"] == "AI4Protein/VenusFactory2-ckpts"
        assert recorded["filename"] == "demo/demo_solubility.pt"
        assert recorded["local_dir"] == str(ckpt_env.resolve())
        assert got.read_bytes() == b"hub"

    def test_download_patterns_calls_snapshot(self, ckpt_env, monkeypatch):
        recorded = {}

        def fake_hf_hub_download(**kwargs):
            raise AssertionError("single-file download unexpected")

        def fake_snapshot_download(**kwargs):
            recorded.update(kwargs)
            return str(ckpt_env)

        monkeypatch.setattr(
            ckpt_hub,
            "_import_hub",
            lambda: (fake_hf_hub_download, fake_snapshot_download),
        )
        ckpt_hub.download_patterns(["demo/**", "DeepSol/ankh-large/**"])
        assert recorded["allow_patterns"] == ["demo/**", "DeepSol/ankh-large/**"]
        assert recorded["repo_id"] == "AI4Protein/VenusFactory2-ckpts"


class TestProteinMPNN:
    def test_ensure_proteinmpnn_weights(self, ckpt_env, monkeypatch):
        def fake_ensure(rel, *, download=None):
            dest = ckpt_env / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"mpnn")
            return dest.resolve()

        monkeypatch.setattr(ckpt_hub, "ensure_ckpt_file", fake_ensure)
        got = ckpt_hub.ensure_proteinmpnn_weights(model_name="v_48_020", variant="vanilla")
        assert got.name == "v_48_020.pt"
        assert "vanilla_model_weights" in got.as_posix()


class TestConfig:
    def test_ckpt_hub_config_from_env(self, monkeypatch):
        monkeypatch.setenv("VENUS_CKPT_REPO_ID", "org/other")
        monkeypatch.setenv("VENUS_CKPT_DIR", "/tmp/venus-ckpt")
        monkeypatch.setenv("VENUS_CKPT_REVISION", "v1")
        monkeypatch.setenv("VENUS_CKPT_AUTO_DOWNLOAD", "0")
        cfg = CkptHubConfig.from_env()
        assert cfg.repo_id == "org/other"
        assert cfg.local_dir == "/tmp/venus-ckpt"
        assert cfg.revision == "v1"
        assert cfg.auto_download is False


class TestManifestAndCLI:
    def test_bundled_manifest_exists_and_has_files(self):
        path = Path(__file__).resolve().parent.parent / "ckpt" / "manifest.json"
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["repo_id"] == "AI4Protein/VenusFactory2-ckpts"
        assert "predict-core" in data["presets"]
        assert isinstance(data["files"], list)
        assert len(data["files"]) > 0
        assert {"path", "size", "sha256"} <= set(data["files"][0])

    def test_download_script_list_presets(self):
        import importlib.util

        repo = Path(__file__).resolve().parent.parent
        script = repo / "scripts" / "download_ckpts.py"
        spec = importlib.util.spec_from_file_location("download_ckpts_cli", script)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.main(["--list-presets"]) == 0
