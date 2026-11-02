"""Shared version-bound execution and external scoring for engineering fixtures.

This is not a general protein evaluator or an RSI selection controller.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .diagnostic_audit import audit
from .model_policy import read_model_configuration
from agent.rsi.campaigns.pilot import evaluate
from .runtime import run_task
from .sandbox import SandboxKimi
from .storage import atomic_json, safe_child, sha256_file
from .versions import VersionStore


def load_binding(capability_dir: Path, version_id: str, config: dict, baseline: dict) -> tuple[str, dict]:
    capability_dir = capability_dir.resolve()
    store = VersionStore(capability_dir.parent.parent)
    manifest = store.verify(version_id)
    if capability_dir != (store.root / "versions" / version_id).resolve():
        raise ValueError("Capability directory does not match the requested version")
    prompt_path = capability_dir / "executor_prompt.md"
    prompt_bytes = prompt_path.read_bytes()
    prompt_hash = hashlib.sha256(prompt_bytes).hexdigest()
    if manifest["files"].get("executor_prompt.md") != prompt_hash:
        raise ValueError("Executor prompt does not match the version manifest")
    _, model = read_model_configuration(
        Path(config["kimi_config"]), expected=baseline["model"], required=config.get("required_model")
    )
    binding = {
        "schema_version": 2,
        "version_id": version_id,
        "parent_version_id": manifest["parent"],
        "prompt_sha256": prompt_hash,
        "capability_files": manifest["files"],
        "model": model["configured_id"],
        "model_identity": model,
        "runtime_overrides": {"max_output_size": 4096, "thinking": "medium"},
        "budget": {"timeout_s": config["task_timeout_s"], "max_tool_calls": config["max_tool_calls"]},
        "evaluator_sha256": sha256_file(Path(__file__).with_name("diagnostic_audit.py")),
    }
    return prompt_bytes.decode("utf-8"), binding


async def execute_task(
    task: dict,
    *,
    config: dict,
    baseline: dict,
    capability_dir: Path,
    version_id: str,
    run_dir: Path,
    metadata: dict | None = None,
) -> tuple[dict, dict]:
    # Reject changed bundles/configuration before creating a runtime or making API calls.
    prompt, binding = load_binding(capability_dir, version_id, config, baseline)
    work = run_dir / "workspace"
    work.mkdir(parents=True, exist_ok=False)
    for name, content in task["files"].items():
        path = safe_child(work, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (work / "skills").mkdir(exist_ok=True)
    binding["task_prompt_sha256"] = hashlib.sha256(task["prompt"].encode()).hexdigest()
    binding["input_sha256"] = {name: sha256_file(safe_child(work, name)) for name in task["files"]}
    binding["metadata"] = metadata or {}
    atomic_json(run_dir / "policy_binding.json", binding)
    sandbox = SandboxKimi(
        workspace=work,
        control_dir=run_dir / "control",
        capability_dir=capability_dir,
        kimi_bin=Path(config["kimi_bin"]),
        config_path=Path(config["kimi_config"]),
        python_root=Path(config["python_root"]),
        expected_model=binding["model_identity"],
    )
    try:
        async with sandbox:
            run = await run_task(
                sandbox.client,
                prompt=task["prompt"],
                system_prompt=prompt,
                model=binding["model"],
                output_dir=run_dir,
                timeout_s=config["task_timeout_s"],
                max_tool_calls=config["max_tool_calls"],
            )
    except Exception as exc:
        run = {"status": "runtime_error", "error": f"{type(exc).__name__}: {exc}"}
        atomic_json(run_dir / "run.json", run)
    VersionStore(capability_dir.parent.parent).verify(version_id)
    events = run_dir / "events.jsonl"
    try:
        legacy = evaluate(task, work, run, events)
    except (ValueError, TypeError, KeyError, OSError):
        legacy = {"task_id": task["task_id"], "passed": False, "reason": "legacy_evaluation_error"}
    score = audit(task, work, run, events)
    try:
        inputs_unchanged = all(
            sha256_file(safe_child(work, name)) == digest for name, digest in binding["input_sha256"].items()
        )
    except (OSError, ValueError):
        inputs_unchanged = False
    score["inputs_unchanged"] = inputs_unchanged
    if not inputs_unchanged:
        score.update(passed=False, reason="task_inputs_modified")
    score.update(
        task_id=task["task_id"],
        kind=task["kind"],
        variant=task["variant"],
        split=task.get("split"),
        version_id=version_id,
        prompt_sha256=binding["prompt_sha256"],
        runtime_status=run["status"],
        error_code=run.get("error_code"),
        sandbox_stopped=sandbox.stopped,
        wall_seconds=run.get("wall_seconds"),
        tool_calls=run.get("tool_calls"),
        usage_status=run.get("usage_status", "not_reported"),
        legacy_passed=legacy["passed"],
        legacy_reason=legacy["reason"],
    )
    atomic_json(run_dir / "legacy_evaluation.json", legacy)
    atomic_json(run_dir / "evaluation.json", score)
    return run, score
