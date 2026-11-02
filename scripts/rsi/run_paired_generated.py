"""Predeclared, counterbalanced dual-mode source-development/reuse experiments.

These are retrospective system comparisons, not tests of general RSI efficacy.
Both arms use the same upstream GLM alias, sampling, scientific capabilities,
training data, actual task text and request/tool/wall-clock ceilings. Native
system prompts and role-output schemas remain architecture-specific.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tomllib
from pathlib import Path

from run_generated_lifecycle import FILES, SYSTEM, policy_for, spec
from run_protein_lifecycle import ROOT, audit_events, prepare_case

from agent.rsi.engine.live_model_gateway import ModelGateway
from agent.rsi.engine.model_gateway import GatewayPolicy
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.model_policy import read_model_configuration
from agent.rsi.engine.protein_model import digest, evaluate, fit_candidate
from agent.rsi.engine.runtime import build_task_message, run_task
from agent.rsi.engine.sandbox import SandboxKimi
from agent.rsi.engine.scientist_runtime import run_scientist
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.rsi.engine.versions import VersionStore

REQUEST_TOTAL_DEADLINE_S = 300
CLIENT_READ_TIMEOUT_S = REQUEST_TOTAL_DEADLINE_S + 30


from agent.rsi.layout import source_relative

def make_bundle(
    folder,
    phase,
    inputs,
    scope,
    identity,
    lifecycle=None,
    model_id=None,
    *,
    experience=None,
    library_experience=None,
    plugin_id="generated_python.v1",
    training_baseline_feedback="none",
):
    if plugin_id not in {
        "generated_python.v1",
        "generated_sequence.v1",
        "generated_classifier.v1",
        "generated_classifier.large.v1",
        "generated_features.v1",
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        raise ValueError("Unsupported generated model plugin")
    from agent.rsi.engine.context_feedback_dispatch import FIELD, policy_for
    from agent.rsi.engine.development_handoff import FIELD as HANDOFF_FIELD
    from agent.rsi.engine.development_handoff import POLICY as HANDOFF_POLICY

    feedback = policy_for({FIELD: training_baseline_feedback, "scope": scope}, plugin_id=plugin_id)
    if feedback != "none" and phase != "develop":
        raise ValueError("Baseline feedback is available only during development")
    source = folder / "bundle_source"
    source.mkdir(parents=True)
    files = [*FILES, "development_handoff.py"]
    if plugin_id == "generated_context.v3":
        files.append("generated_context_v3.py")
    if plugin_id == "generated_assay.v1":
        files.append("generated_assay.py")
    if plugin_id == "generated_antibody.v1":
        files += ["generated_antibody.py", "generated_assay.py", "poisk_pair_encoder.py"]
    if feedback != "none":
        files += [
            "context_controls.py",
            "context_feedback.py",
            "reference_thermal_baselines.py",
            "context_feedback_profiles.py",
            "expression_baselines.py",
            "context_feedback_dispatch.py",
            "context_v2_controls.py",
            "context_v2_feedback.py",
        ]
    if feedback == "group_baselines.v3":
        files.extend(["context_v3_controls.py", "context_v3_feedback.py"])
    for name in dict.fromkeys(files):
        shutil.copy2(ROOT / source_relative(name), source / name)
    task = {"phase": phase, "scope": scope, "inputs": {name: digest(rows) for name, rows in inputs.items()}}
    if phase == "develop":
        task[HANDOFF_FIELD] = HANDOFF_POLICY
    if plugin_id != "generated_python.v1":
        task["plugin_id"] = plugin_id
    if feedback != "none":
        task[FIELD] = feedback
    if experience is not None:
        from agent.rsi.engine.experience import validate_projection

        validate_projection(experience)
        atomic_json(source / "experience.json", experience)
        task["experience_sha256"] = digest(experience)
    if library_experience is not None:
        from agent.rsi.engine.experience import validate_projection

        if phase != "develop":
            raise ValueError("Library experience is available only during development")
        validate_projection(library_experience)
        atomic_json(source / "library_experience.json", library_experience)
        task["library_experience_sha256"] = digest(library_experience)
    if lifecycle is not None:
        task["capability_sha256"] = digest(lifecycle.export(model_id, source / "registered"))
    atomic_json(source / "task.json", task)
    versions = VersionStore(folder / "bundles")
    manifest = versions.snapshot(source, metadata={"phase": phase, "model": identity})
    return versions, manifest["version_id"], folder / "bundles/versions" / manifest["version_id"]


def request_audit(path, mode, allowed):
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    requests = [row for row in records if row["event"] == "request"]
    responses = [row for row in records if row["event"] == "response"]
    expected = {"mcp__venusfactory__" + name for name in allowed}
    exposed = all(set(row["tools_sent"]) <= expected for row in requests) if mode == "science_agent" else None
    request_ids = [row["call"] for row in requests]
    response_ids = [row["call"] for row in responses]
    errors = [row for row in records if row["event"] in {"transport_error", "policy_rejected", "budget_rejected"}]
    passed = (
        bool(requests)
        and len(request_ids) == len(set(request_ids))
        and sorted(request_ids) == sorted(response_ids)
        and not errors
        and all(row.get("status") == 200 and row.get("completed") is True for row in responses)
    )
    if mode == "science_agent":
        passed = passed and exposed
    return {
        "passed": passed,
        "model_calls": len(requests),
        "responses": len(responses),
        "transport_or_policy_errors": errors,
        "completion_required": True,
        "scientific_api_tool_allowlist_enforced": exposed,
        "usage": [row["usage"] for row in responses if row.get("usage")],
        "architectural_role_output_schemas_retained": mode == "science_expert",
        "request_trace_sha256": sha256_file(path) if path.exists() else None,
    }


async def run_phase(
    folder,
    mode,
    phase,
    inputs,
    capability,
    version_id,
    prompt,
    config,
    identity,
    provider,
    timeout,
    *,
    use_experience=False,
):
    folder.mkdir(parents=True)
    work = folder / "workspace"
    work.mkdir()
    for name, value in inputs.items():
        atomic_json(work / name, value)
    phase_spec = spec(phase)
    allowed = phase_spec["tools"]
    required = phase_spec["required_tools"]
    if use_experience:
        if phase != "develop":
            raise ValueError("experience applies to model development only")
        allowed = allowed | {"read_model_experience"}
        required = required | {"read_model_experience"}
    submitted = build_task_message(SYSTEM, prompt)
    (folder / "common_task.txt").write_text(submitted)
    atomic_json(
        folder / "binding.json",
        {
            "mode": mode,
            "phase": phase,
            "model": identity,
            "capability_version": version_id,
            "task_sha256": digest(submitted),
            "allowed_scientific_tools": sorted(allowed),
            "temperature": 0.1,
            "max_output_tokens": 12288,
            "max_model_calls": 24,
            "max_tool_calls": 12,
            "timeout_s": timeout,
            "gateway_enforced": True,
            "request_total_deadline_s": REQUEST_TOTAL_DEADLINE_S,
            "scientist_client_read_timeout_s": CLIENT_READ_TIMEOUT_S,
        },
    )
    policy = GatewayPolicy(
        model=identity["model"], mode=mode, scientific_tools=allowed, max_tokens=12288, thinking="enabled"
    )
    async with ModelGateway(
        policy=policy,
        upstream_url=provider["base_url"],
        api_key=provider["api_key"],
        output=folder / "model_requests.jsonl",
        timeout_s=REQUEST_TOTAL_DEADLINE_S,
    ) as relay:
        if mode == "science_agent":
            sandbox = SandboxKimi(
                workspace=work,
                control_dir=folder / "control",
                capability_dir=capability,
                kimi_bin=Path(config["kimi_bin"]),
                config_path=Path(config["kimi_config"]),
                python_root=Path(config["python_root"]),
                expected_model=identity,
                mcp_entry="generated_tools.py",
                relay=relay,
            )
            async with sandbox:
                result = await run_task(
                    sandbox.client,
                    prompt=prompt,
                    system_prompt=SYSTEM,
                    model=identity["configured_id"],
                    output_dir=folder,
                    timeout_s=timeout,
                    max_tool_calls=12,
                    allowed_mcp_tools=frozenset(allowed),
                    request_gateway=True,
                )
        else:
            result = await run_scientist(
                workspace=work,
                capability=capability,
                python_root=Path(config["python_root"]),
                output=folder,
                prompt=submitted,
                model=identity["model"],
                base_url=relay.base_url,
                api_key=relay.token,
                allowed_tools=allowed,
                timeout_s=timeout,
                max_tool_calls=12,
                research_policy="task_context_only",
                phase=phase,
                request_timeout_s=CLIENT_READ_TIMEOUT_S,
                max_output_tokens=12288,
            )
    audit = audit_events(folder, phase, result, required, phase_spec["recoverable_tools"])
    audit["inputs_unchanged"] = all(
        digest(json.loads((work / name).read_text())) == digest(value) for name, value in inputs.items()
    )
    audit["request_gateway"] = request_audit(folder / "model_requests.jsonl", mode, allowed)
    audit["passed"] = audit["passed"] and audit["inputs_unchanged"] and audit["request_gateway"]["passed"]
    atomic_json(folder / "audit.json", audit)
    return result, audit, work


async def main(args):
    if not 1 <= args.pairs <= 6:
        raise ValueError("campaign must have 1..6 predeclared pairs")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    _, identity = read_model_configuration(Path(config["kimi_config"]), required=config["required_model"])
    private = tomllib.loads(Path(config["kimi_config"]).read_text())
    provider = private["providers"][identity["provider"]]
    scope, train, val, future, source, split = prepare_case(args.case, args.source_cache)
    development_prompt = (
        "Build a numerical predictor of " + scope["target"] + " for " + scope["protein"] + ". "
        "Assay: " + scope["assay"] + "; units: " + scope["unit"] + ". "
        "Each short amino-acid string encodes aligned mutable positions in a full protein; it is not a standalone short protein. "
        "Inspect the task, then author complete fit(sequences, targets, seed) and predict(model, sequences) Python. "
        "Implement a real supervised predictor with JSON model state and no placeholders. Evaluate source through the tool. "
        "Use training-only feedback to improve or repair within two attempts, and submit one actual successfully tested hash. "
        "Independent acceptance labels are unavailable. Finish with an evidence summary of at most 350 words."
    )
    modes = ["science_agent", "science_expert"]
    protocol = {
        "schema": 1,
        "case": args.case,
        "model": identity,
        "source": source,
        "split": split,
        "pairs": args.pairs,
        "orders": [modes if i % 2 == 0 else modes[::-1] for i in range(args.pairs)],
        "request_total_deadline_s": REQUEST_TOTAL_DEADLINE_S,
        "scientist_client_read_timeout_s": CLIENT_READ_TIMEOUT_S,
        "development_timeout_s": 900,
        "reuse_timeout_s": 420,
        "max_model_calls_per_phase": 24,
        "max_scientific_tool_calls_per_phase": 12,
        "code_feedback_attempts": 2,
        "candidates_per_arm": 1,
        "sampling": {"temperature": 0.1, "max_tokens": 12288, "thinking": {"type": "enabled", "clear_thinking": True}},
        "data_hashes": {"training": digest(train), "validation": digest(val), "future": digest(future)},
        "common_task_sha256": digest(build_task_message(SYSTEM, development_prompt)),
        "reuse_enabled": not args.skip_reuse,
        "comparison_scope": "retrospective architecture-level system comparison; no recursive improvement or superiority claim from small samples",
        "architecture_differences": [
            "native system prompts",
            "internal role-output schemas",
            "role coordination and planning",
        ],
        "limitations": [
            "provider alias is not an immutable snapshot",
            "historical dataset partitions used in earlier integration pilots",
            "failed earlier pilots are retained separately, not pooled with this frozen protocol",
            "provider thinking disable probe was ignored; enabled thinking requested equally; reasoning accounting unverified",
        ],
    }
    atomic_json(args.output / "protocol.json", protocol)
    paths = [
        Path(__file__),
        *list((ROOT / "src/agent/rsi").rglob("*.py")),
        ROOT / "src/agent/graph/finalize.py",
        ROOT / "src/agent/graph/state.py",
        ROOT / "src/agent/graph/planning/plan.py",
        ROOT / "src/agent/graph/execution/prepare.py",
        ROOT / "src/agent/graph/execution/execute.py",
        ROOT / "src/agent/graph/research/search.py",
    ]
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        dest = args.output / "implementation" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[str(relative)] = sha256_file(path)
    atomic_json(args.output / "implementation/manifest.json", hashes)
    versions, version, capability = make_bundle(
        args.output / "shared_development", "develop", {"training.json": train}, scope, identity
    )
    report = {"status": "running", "protocol_sha256": digest(protocol), "arms": []}
    atomic_json(args.output / "report.json", report)
    for pair, order in enumerate(protocol["orders"]):
        for mode in order:
            if any(sha256_file(ROOT / rel) != sha for rel, sha in hashes.items()):
                raise ValueError("implementation changed during campaign")
            folder = args.output / ("pair_" + str(pair)) / mode
            entry = {"pair": pair, "mode": mode, "status": "running"}
            report["arms"].append(entry)
            atomic_json(args.output / "report.json", report)
            print(json.dumps({"pair": pair, "mode": mode, "phase": "develop"}), flush=True)
            try:
                run, audit, work = await run_phase(
                    folder / "develop",
                    mode,
                    "develop",
                    {"training.json": train},
                    capability,
                    version,
                    development_prompt,
                    config,
                    identity,
                    provider,
                    protocol["development_timeout_s"],
                )
                versions.verify(version)
                entry["development"] = {"run": run, "audit": audit}
                entry["status"] = "development_failed"
                if audit["passed"] and (work / "candidate.json").exists():
                    lc = ModelLifecycle(folder / "controller", policy_for(scope, train, val))
                    mid = lc.submit(work / "candidate.json", session_id=run["session_id"], mode=mode)
                    entry["assessment"] = lc.assess(mid, training=train, validation=val)
                    entry["source_sha256"] = lc._model(mid)["source_sha256"]
                    entry["status"] = "accepted" if entry["assessment"]["passed"] else "candidate_rejected"
                    if entry["assessment"]["passed"] and not args.skip_reuse:
                        inputs = [{"id": row["id"], "sequence": row["sequence"]} for row in future]
                        rv, rid, rc = make_bundle(
                            folder / "reuse_capability", "reuse", {"inputs.json": inputs}, scope, identity, lc, mid
                        )
                        reuse_prompt = spec("reuse")["prompt"] + " Report only observed evidence in at most 350 words."
                        reused, ra, rw = await run_phase(
                            folder / "reuse",
                            mode,
                            "reuse",
                            {"inputs.json": inputs},
                            rc,
                            rid,
                            reuse_prompt,
                            config,
                            identity,
                            provider,
                            protocol["reuse_timeout_s"],
                        )
                        rv.verify(rid)
                        entry["reuse"] = {"run": reused, "audit": ra}
                        entry["status"] = "reuse_failed"
                        if ra["passed"] and (rw / "predictions.json").exists():
                            predictions = json.loads((rw / "predictions.json").read_text())
                            entry["reuse"]["event"] = lc.record_reuse(
                                mid,
                                session_id=reused["session_id"],
                                mode=mode,
                                inputs=inputs,
                                predictions=predictions,
                                scope=scope,
                            )
                            entry["future_metrics"] = get_model_plugin("generated_python.v1").evaluate(
                                lc._model(mid), future, scope
                            )
                            entry["status"] = "completed"
            except Exception as exc:
                # No automatic rerun: every allocated arm stays in the denominator.
                entry.update(status="controller_error", error_type=type(exc).__name__)
            atomic_json(folder / "report.json", entry)
            atomic_json(args.output / "report.json", report)
            print(json.dumps({"pair": pair, "mode": mode, "status": entry["status"]}), flush=True)
    fixed = fit_candidate(train, scope, {"encoding": "onehot", "alphas": [0.1, 1.0, 10.0, 100.0]})
    report["fixed_algorithm_comparator"] = {
        "validation": evaluate(fixed, val, scope),
        "future": evaluate(fixed, future, scope),
    }
    report["status"] = "campaign_completed"
    atomic_json(args.output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["alde_parpgb", "mcba_itopride"], default="alde_parpgb")
    p.add_argument("--pairs", type=int, default=1)
    p.add_argument("--skip-reuse", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rsi/generated_models_glm51.json")
    p.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    asyncio.run(main(p.parse_args()))
