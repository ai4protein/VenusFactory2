"""Tests for shared download notice helpers."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hub_progress
import ckpt_hub


def test_format_bytes():
    assert hub_progress.format_bytes(512) == "512 B"
    assert "KB" in hub_progress.format_bytes(2048)
    assert "MB" in hub_progress.format_bytes(5 * 1024 * 1024)


def test_announce_capture_and_quiet(monkeypatch):
    monkeypatch.delenv("VENUS_DOWNLOAD_QUIET", raising=False)
    with hub_progress.capture_notices() as notes:
        hub_progress.announce("hello-download")
    assert notes == ["hello-download"]

    monkeypatch.setenv("VENUS_DOWNLOAD_QUIET", "1")
    assert hub_progress.quiet() is True


def test_estimate_download_bytes_from_manifest():
    size = ckpt_hub.estimate_download_bytes(["demo/**"])
    assert size is not None
    assert size > 0


def test_ensure_announces_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("VENUS_CKPT_DIR", str(tmp_path))
    monkeypatch.setenv("VENUS_CKPT_AUTO_DOWNLOAD", "1")
    monkeypatch.delenv("VENUS_DOWNLOAD_QUIET", raising=False)

    def fake_download(rel, *, force=False):
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest

    monkeypatch.setattr(ckpt_hub, "download_file", fake_download)
    with hub_progress.capture_notices() as notes:
        ckpt_hub.ensure_ckpt_file("demo/demo_solubility.pt")
    assert any("auto-downloading" in n for n in notes)
