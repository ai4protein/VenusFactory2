import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("deployment_services", Path(__file__).resolve().parents[1] / "scripts/deploy_services.py")
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)


def test_both_modes_override_inherited_paths_and_use_distinct_ports(tmp_path):
    args = deployment.arguments(["--runtime-root", str(tmp_path)])
    plans = deployment.service_plan(args, {"VENUS_JOB_QUEUE_ROOT": "/old/shared", "VENUS_USAGE_DB": "/old/shared.db"})
    assert [p["mode"] for p in plans] == ["online", "local"]
    assert len({p[k] for p in plans for k in ["port", "mcp_port"]}) == 4
    for plan in plans:
        assert Path(plan["env"]["VENUS_JOB_QUEUE_ROOT"]) == tmp_path / plan["mode"] / "queue"
        assert Path(plan["env"]["VENUS_USAGE_DB"]).is_relative_to(tmp_path / plan["mode"])
        assert "--mode" in plan["command"] and plan["mode"] in plan["command"]
        assert Path(plan["env"]["KIMI_CODE_HOME"]) == tmp_path / plan["mode"] / "control/kimi"


def test_harness_settings_are_private_and_do_not_copy_global_sessions_or_mcp(tmp_path):
    source = tmp_path / "personal"
    source.mkdir()
    for name in ["config.toml", "mcp.json", "sessions.json"]:
        (source / name).write_text("fixture")
    plans = deployment.service_plan(deployment.arguments(["--runtime-root", str(tmp_path / "runtime")]), {})
    for plan in plans:
        deployment.prepare_harness_home(plan, source)
        home = Path(plan["env"]["KIMI_CODE_HOME"])
        assert (home / "config.toml").read_text() == "fixture"
        assert not (home / "mcp.json").exists()
        assert not (home / "sessions.json").exists()
        (home / "config.toml").write_text("mode-specific")
        deployment.prepare_harness_home(plan, source)
        assert (home / "config.toml").read_text() == "mode-specific"


def test_port_collision_and_partial_tunnel_configuration_fail_before_launch():
    with pytest.raises(ValueError, match="distinct"):
        deployment.service_plan(deployment.arguments(["--online-port", "7862"]))
    with pytest.raises(SystemExit):
        deployment.arguments(["--frp-config", "private.ini"])


def test_stopping_bundle_leaves_unrelated_process_alive():
    import subprocess
    import sys
    managed = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        deployment.stop_children([managed])
        assert managed.poll() is not None
        assert unrelated.poll() is None
    finally:
        for child in [managed, unrelated]:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
