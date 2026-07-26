"""Unit tests for frpc download helpers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import frpc_hub


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    # Point repo_root / downloads into a temp tree.
    monkeypatch.setattr(frpc_hub, "_REPO_ROOT", tmp_path)
    (tmp_path / "install_config.json").write_text(
        """
{
  "frpc_versions": {
    "v0.3": {
      "linux_amd64": "https://example.invalid/frpc_linux_amd64"
    }
  }
}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("VENUS_FRPC_VERSION", "v0.3")
    monkeypatch.setenv("VENUS_FRPC_REPO_ID", "AI4Protein/VenusFactory2-ckpts")
    monkeypatch.setenv("VENUS_FRPC_HF_PREFIX", "assets/frpc")
    return tmp_path


class TestNaming:
    def test_filename_linux(self, monkeypatch):
        monkeypatch.setattr(frpc_hub, "system_name", lambda: "linux")
        monkeypatch.setattr(frpc_hub, "arch_str", lambda: "amd64")
        assert frpc_hub.frpc_filename() == "frpc_linux_amd64_v0.3"

    def test_filename_windows(self, monkeypatch):
        monkeypatch.setattr(frpc_hub, "system_name", lambda: "windows")
        monkeypatch.setattr(frpc_hub, "arch_str", lambda: "amd64")
        assert frpc_hub.frpc_filename().endswith(".exe")


class TestDownload:
    def test_uses_local_without_network(self, isolated_repo, monkeypatch):
        name = "frpc_linux_amd64_v0.3"
        local = isolated_repo / name
        local.write_bytes(b"binary")
        monkeypatch.setattr(frpc_hub, "system_name", lambda: "linux")
        monkeypatch.setattr(frpc_hub, "arch_str", lambda: "amd64")

        def boom(*_a, **_k):
            raise AssertionError("should not download")

        monkeypatch.setattr(frpc_hub, "_download_cdn", boom)
        monkeypatch.setattr(frpc_hub, "_download_hf", boom)
        got = frpc_hub.download_frpc()
        assert got == local.resolve() or got == local

    def test_falls_back_to_hf_when_cdn_fails(self, isolated_repo, monkeypatch):
        monkeypatch.setattr(frpc_hub, "system_name", lambda: "linux")
        monkeypatch.setattr(frpc_hub, "arch_str", lambda: "amd64")

        def fail_cdn(url, dest):
            raise OSError("cdn down")

        def ok_hf(filename, dest):
            dest.write_bytes(b"from-hf")
            dest.chmod(0o755)
            return dest

        monkeypatch.setattr(frpc_hub, "_download_cdn", fail_cdn)
        monkeypatch.setattr(frpc_hub, "_download_hf", ok_hf)
        got = frpc_hub.download_frpc(force=True)
        assert got.read_bytes() == b"from-hf"

    def test_install_to_gradio_cache(self, isolated_repo, tmp_path, monkeypatch):
        monkeypatch.setattr(frpc_hub, "system_name", lambda: "linux")
        monkeypatch.setattr(frpc_hub, "arch_str", lambda: "amd64")
        cache = tmp_path / "gradio-frpc"
        monkeypatch.setattr(frpc_hub, "gradio_cache_dir", lambda: cache)

        local = isolated_repo / "frpc_linux_amd64_v0.3"
        local.write_bytes(b"binary-data")

        dest = frpc_hub.install_to_gradio_cache()
        assert dest == cache / "frpc_linux_amd64_v0.3"
        assert dest.read_bytes() == b"binary-data"


class TestCLI:
    def test_download_script_help(self):
        import importlib.util

        script = Path(__file__).resolve().parent.parent / "scripts" / "download_frpc.py"
        spec = importlib.util.spec_from_file_location("download_frpc_cli", script)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with pytest.raises(SystemExit) as exc:
            mod.main(["--help"])
        assert exc.value.code == 0
