"""Tests for auto torch profile detection."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".install"))

import torch_detect


def test_resolve_override_cpu():
    kind, reason = torch_detect.resolve_install_type("cpu")
    assert kind == "cpu"
    assert "override" in reason


def test_resolve_override_cu128():
    kind, reason = torch_detect.resolve_install_type("cu128")
    assert kind == "cu128"
    assert "override" in reason


def test_detect_cpu_without_gpu(monkeypatch):
    from pathlib import Path as P

    monkeypatch.setattr(torch_detect.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch_detect.shutil, "which", lambda _n: None)
    monkeypatch.setattr(torch_detect, "_nvidia_cuda_version", lambda: None)
    real_exists = P.exists

    def fake_exists(self):
        if str(self) == "/dev/nvidia0":
            return False
        return real_exists(self)

    monkeypatch.setattr(P, "exists", fake_exists)
    kind, reason = torch_detect.detect_install_type()
    assert kind == "cpu"
    assert "CPU" in reason


def test_detect_old_cuda_prefers_cpu(monkeypatch):
    class Proc:
        returncode = 0
        stdout = "GPU 0: NVIDIA GeForce GTX 1080\n"

    monkeypatch.setattr(torch_detect.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch_detect.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(torch_detect, "_nvidia_cuda_version", lambda: 11.8)
    monkeypatch.setattr(torch_detect.subprocess, "run", lambda *a, **k: Proc())
    kind, reason = torch_detect.detect_install_type()
    assert kind == "cpu"
    assert "11.8" in reason or "12.0" in reason


def test_detect_modern_nvidia(monkeypatch):
    class Proc:
        returncode = 0
        stdout = "GPU 0: NVIDIA GeForce RTX 4090\n"

    monkeypatch.setattr(torch_detect.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch_detect.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(torch_detect, "_nvidia_cuda_version", lambda: 12.4)
    monkeypatch.setattr(torch_detect.subprocess, "run", lambda *a, **k: Proc())
    kind, reason = torch_detect.detect_install_type()
    assert kind == "cu128"
    assert "NVIDIA" in reason
