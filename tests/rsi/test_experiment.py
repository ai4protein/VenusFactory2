import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/rsi/run_experiment.py"
spec = importlib.util.spec_from_file_location("rsi_engineering_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.asyncio
async def test_provider_auth_error_stops_queued_tasks(tmp_path, monkeypatch):
    attempts = []

    async def failed_execute(task, *, run_dir, **kwargs):
        run_dir.mkdir(parents=True)
        attempts.append(run_dir)
        return (
            {"status": "runtime_error", "error_code": "provider_authentication_failed"},
            {"task_id": task["task_id"], "passed": False, "reason": "runtime_not_completed"},
        )

    monkeypatch.setattr(runner, "execute_task", failed_execute)
    config = {
        "variants": 1,
        "concurrency": 1,
        "kimi_bin": "/unused/kimi",
        "kimi_config": "/unused/config",
        "python_root": "/unused/python",
        "task_timeout_s": 10,
        "max_tool_calls": 12,
        "suite": "fixture",
        "claim_scope": "test",
    }
    status = await runner.experiment(
        config,
        tmp_path,
        {"model": {"configured_id": "fixture"}, "kimi_version": "fixture"},
        tmp_path,
        {"version_id": "fixture"},
        None,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert status == 1
    assert len(attempts) == summary["attempted"] == 1
    assert len(summary["skipped"]) == 5
    assert summary["state"] == "blocked"
    assert summary["failed"] == 1
    assert len(list((tmp_path / "runs").iterdir())) == 1
