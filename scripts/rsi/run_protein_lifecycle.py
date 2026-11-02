"""GLM-driven configured protein model development, host validation and fresh-session reuse.

Public retrospective data; not blind discovery, wet-lab replication or recursive
self-improvement. One candidate per case; no retries after seeing validation.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import shutil
from pathlib import Path

from agent.research_modes import get_research_mode
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.model_policy import read_model_configuration
from agent.rsi.protein_model import PLUGIN_ID, digest, evaluate, fit_candidate, validate_rows
from agent.rsi.runtime import run_task
from agent.rsi.sandbox import SandboxKimi
from agent.rsi.storage import atomic_json, sha256_file
from agent.rsi.versions import VersionStore

ROOT = Path(__file__).resolve().parents[2]
TOOL_SETS = {
    "develop": frozenset({"inspect_protein_task", "discover_protein_models", "develop_protein_model"}),
    "reuse": frozenset({"inspect_protein_task", "discover_protein_models", "predict_with_protein_model"}),
}
SYSTEM = """You are the adaptive scientific execution agent in VenusFactory.
Use the declared protein-model MCP tools to complete the task autonomously.
Do not ask for human input. No shell, network search or external files are available.
Use measured training data only. Internal CV is development feedback, not final validation.
MCP tools write the required machine-readable artifacts. Do not alter their outputs.
Describe this as a configured task-specific model, not a new modelling architecture.
"""
PROMPTS = {
    "develop": "Inspect this protein task. Select a defensible supported residue encoding and bounded ridge alpha grid for its measured training data, then develop exactly one candidate using the MCP tool. Explain the choice and report its artifact path. Do not claim independent validation or registration: a separate controller will assess it after this session ends.",
    "reuse": "This is a fresh protein prediction task. Inspect its scope, discover any compatible validated protein models, and reuse a suitable discovered model to predict every input. Do not train a new model. Use the MCP tool to save predictions.json and report its path and row count.",
}


def source_asset(cache: Path, repository: str, name: str):
    root = cache / repository
    record = json.loads((root / "source.json").read_text())
    entry = next(e for e in record["files"] if e["path"] == name)
    path = root / name
    if sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
        raise ValueError("source data integrity mismatch")
    return path, {"repository": record["repository"], "commit": record["commit"], **entry}


def prepare_case(case: str, cache: Path):
    import pandas as pd

    if case == "alde_parpgb":
        path, provenance = source_asset(cache, "ALDE", "analysis/ParPgb_fitness_all.csv")
        df = pd.read_csv(path)
        scope = {
            "protein": "ParPgb",
            "assay": "ALDE published carbene-transfer yield1 assay",
            "target": "yield1",
            "unit": "author-reported yield1",
            "sequence_length": 5,
        }

        def rows(part):
            return [
                {"id": str(r.Combo), "sequence": str(r.Combo), "target": float(r.yield1)} for r in part.itertuples()
            ]

        train = rows(df[df["round"].isin(["initial", "parent"])])
        val = rows(df[df["round"] == "round1"])
        future = rows(df[df["round"] == "round2"])
        split = "Author rounds: initial+parent training; round1 promotion; round2 fresh-session reuse assessment. Retrospective selection-biased cohorts."
    elif case == "mcba_itopride":
        path, provenance = source_asset(cache, "accelerated-enzyme-engineering", "data/HSS/itopride_train.xlsx")
        df = pd.read_excel(path)
        scope = {
            "protein": "McbA",
            "assay": "published itopride activity assay",
            "target": "Activity",
            "unit": "author-reported relative activity",
            "sequence_length": 4,
        }
        rows = [
            {"id": str(r.AminoAcid), "sequence": str(r.AminoAcid), "target": float(r.Activity)} for r in df.itertuples()
        ]
        # Fixed sequence-hash allocation, independent of labels and no split search.
        rows.sort(key=lambda r: digest({"split_seed": 20261011, "sequence": r["sequence"]}))
        train, val, future = rows[:49], rows[49:63], rows[63:]
        split = "Sequence-hash split seed 20261011: 49 training, 14 promotion, 14 later reuse; one substrate; no family transfer."
    else:
        raise ValueError("unsupported case")
    for group in (train, val, future):
        validate_rows(group, scope, labels=True)
    groups = [{r["sequence"] for r in group} for group in (train, val, future)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("case split overlaps")
    return scope, train, val, future, provenance, split


def audit_events(
    folder: Path,
    phase: str,
    result: dict,
    required_tools: set[str] | None = None,
    recoverable_tools: set[str] | None = None,
    retryable_read_tools: set[str] | None = None,
) -> dict:
    """Permit declared read-only corrections only after a later successful retry.

    This execution gate never replaces model acceptance or prediction replay.
    Existing recoverable development tools retain their separate semantics.
    """
    starts, ends, problems = {}, {}, []
    start_positions, end_positions = {}, {}
    events = folder / "events.jsonl"
    if not events.exists():
        return {"passed": False, "problems": ["missing event trace"]}
    for position, line in enumerate(events.read_text().splitlines()):
        event = json.loads(line)
        kind, cid = event.get("kind"), event.get("tool_call_id")
        if kind in {"tool_call_start", "tool_result"}:
            dest = starts if kind == "tool_call_start" else ends
            if not cid or cid in dest:
                problems.append("missing or duplicate tool call identity")
            dest[cid] = event
            positions = start_positions if kind == "tool_call_start" else end_positions
            positions[cid] = position
    if set(starts) != set(ends):
        problems.append("unpaired tool calls")
    names = {str(e.get("tool_name", "")).split("__")[-1] for e in starts.values()}
    required = (
        {"develop_protein_model"} if phase == "develop" else {"discover_protein_models", "predict_with_protein_model"}
    )
    if required_tools is not None:
        required = required_tools
    successful = {
        str(event.get("tool_name", "")).split("__")[-1]
        for cid, event in starts.items()
        if cid in ends and not ends[cid].get("is_error")
    }
    if not required <= successful:
        problems.append("required successful protein tool chain missing")
    recoverable = recoverable_tools or set()
    failed_tools, unrecovered, retries = [], [], []
    retryable = retryable_read_tools or set()
    for cid, event in ends.items():
        if not event.get("is_error"):
            continue
        name = str(starts.get(cid, {}).get("tool_name", "")).split("__")[-1]
        failed_tools.append(name)
        if name in recoverable:
            continue
        later_success = next(
            (
                retry_id
                for retry_id, start in starts.items()
                if name in retryable
                and str(start.get("tool_name", "")).split("__")[-1] == name
                and start_positions[retry_id] > end_positions[cid]
                and retry_id in ends
                and not ends[retry_id].get("is_error")
                and end_positions[retry_id] > start_positions[retry_id]
            ),
            None,
        )
        if later_success is None:
            unrecovered.append(name)
        else:
            retries.append({"tool": name, "failed_call_id": cid, "successful_retry_call_id": later_success})
    if unrecovered or result["status"] != "completed":
        problems.append("runtime or nonrecoverable tool failure")
    return {
        "passed": not problems,
        "problems": problems,
        "tool_names": sorted(names),
        "paired_calls": len(ends),
        "recovered_tool_failures": failed_tools,
        "read_tool_retries": retries,
        "unrecovered_tool_failures": unrecovered,
    }


async def phase_run(root, phase, inputs, scope, config, identity, lifecycle=None, model_id=None, plugin_spec=None):
    spec = plugin_spec or {}
    tools = frozenset(spec.get("tools", TOOL_SETS[phase]))
    system_prompt = spec.get("system_prompt", SYSTEM)
    prompt = spec.get("prompt", PROMPTS[phase])
    run = root / phase
    run.mkdir()
    workspace = run / "workspace"
    workspace.mkdir()
    for name, value in inputs.items():
        atomic_json(workspace / name, value)
    task = {"phase": phase, "scope": scope, "inputs": {name: digest(value) for name, value in inputs.items()}}
    source = run / "bundle_source"
    source.mkdir()
    for name in spec.get("files", ("protein_tools.py", "protein_model.py")):
        shutil.copy2(ROOT / "src/agent/rsi" / name, source / name)
    if phase == "reuse":
        packet = lifecycle.export(model_id, source / "registered")
        task["capability_sha256"] = digest(packet)
    atomic_json(source / "task.json", task)
    (source / "system_prompt.txt").write_text(system_prompt)
    (source / "task_prompt.txt").write_text(prompt)
    versions = VersionStore(run / "bundles")
    manifest = versions.snapshot(source, metadata={"phase": phase, "scope": scope, "model": identity})
    capability = run / "bundles/versions" / manifest["version_id"]
    binding = {
        "mode": "science_agent",
        "engine": get_research_mode("science_agent").engine,
        "model": identity,
        "capability_version": manifest["version_id"],
        "prompt_sha256": digest(prompt),
        "system_sha256": digest(system_prompt),
        "timeout_s": spec.get("timeout_s", 240),
        "max_tool_calls": spec.get("max_tool_calls", 8),
        "allowed_mcp_tools": sorted(tools),
        "runtime_overrides": {"max_output_size": 4096, "thinking": "medium"},
        "instruction_binding": "task_message_prelude; native system prompt retained",
        "tool_visibility_enforced": False,
    }
    atomic_json(run / "binding.json", binding)
    sandbox = SandboxKimi(
        workspace=workspace,
        control_dir=run / "control",
        capability_dir=capability,
        kimi_bin=Path(config["kimi_bin"]),
        config_path=Path(config["kimi_config"]),
        python_root=Path(config["python_root"]),
        expected_model=identity,
        mcp_entry=spec.get("mcp_entry", "protein_tools.py"),
    )
    async with sandbox:
        result = await run_task(
            sandbox.client,
            prompt=prompt,
            system_prompt=system_prompt,
            model=identity["configured_id"],
            output_dir=run,
            timeout_s=spec.get("timeout_s", 240),
            max_tool_calls=spec.get("max_tool_calls", 8),
            allowed_mcp_tools=tools,
        )
    versions.verify(manifest["version_id"])
    audit = audit_events(
        run,
        phase,
        result,
        spec.get("required_tools"),
        spec.get("recoverable_tools"),
        spec.get("retryable_read_tools"),
    )
    audit["inputs_unchanged"] = all(
        digest(json.loads((workspace / name).read_text())) == digest(value) for name, value in inputs.items()
    )
    audit["sandbox_stopped"] = sandbox.stopped
    audit["credential_copy_removed"] = not (run / "control/private_home/.kimi-code/config.toml").exists()
    audit["passed"] = audit["passed"] and all(
        audit[k] for k in ("inputs_unchanged", "sandbox_stopped", "credential_copy_removed")
    )
    atomic_json(run / "audit.json", audit)
    return result, audit, workspace


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    _, identity = read_model_configuration(Path(config["kimi_config"]), required=config["required_model"])
    scope, train, val, future, provenance, split = prepare_case(args.case, args.source_cache)
    plugin = get_model_plugin(PLUGIN_ID)
    policy = {
        "schema": 1,
        "scope": scope,
        "plugin_id": plugin.id,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(train),
        "validation_sha256": digest(val),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }
    lifecycle = ModelLifecycle(args.output / "controller", policy)
    for name, rows in (("training", train), ("validation", val), ("future", future)):
        atomic_json(args.output / "controller" / f"{name}.json", rows)
    report = {
        "schema": 1,
        "case": args.case,
        "status": "running",
        "scope": scope,
        "model": identity,
        "source": provenance,
        "split": split,
        "rows": {"training": len(train), "validation": len(val), "future": len(future)},
        "future_sha256": digest(future),
        "policy_sha256": digest(policy),
        "development_type": "configured_task_specific_model",
        "claim_scope": "retrospective configured-model creation, independent acceptance and fresh-session reuse; no algorithm invention, wet-lab replication, blind discovery, cross-family gain or RSI efficacy claim",
        "network_isolated": False,
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn", "fastmcp"]},
        "implementation_hashes": {
            str(f.relative_to(ROOT)): sha256_file(f)
            for f in [
                Path(__file__),
                ROOT / "src/agent/rsi/model_lifecycle.py",
                ROOT / "src/agent/rsi/runtime.py",
                ROOT / "src/agent/rsi/sandbox.py",
                ROOT / "src/agent/rsi/protein_tools.py",
            ]
        },
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps({"case": args.case, "phase": "develop", "rows": report["rows"]}), flush=True)
    dev, dev_audit, workspace = await phase_run(
        args.output, "develop", {"training.json": train}, scope, config, identity
    )
    report["development"] = {
        "status": dev["status"],
        "audit": dev_audit,
        "session_id": dev["session_id"],
        "tool_calls": dev["tool_calls"],
    }
    if dev_audit["passed"] and (workspace / "candidate.json").is_file():
        model_id = lifecycle.submit(workspace / "candidate.json", session_id=dev["session_id"], mode="science_agent")
        assessment = lifecycle.assess(model_id, training=train, validation=val)
        report["assessment"] = assessment
        report["model_id"] = model_id
        # A predefined fixed recipe is a modelling baseline, not an LLM/RSI control arm.
        fixed = fit_candidate(train, scope, {"encoding": "onehot", "alphas": [0.1, 1.0, 10.0, 100.0]})
        report["fixed_recipe_baseline"] = {
            "recipe": fixed["recipe"],
            "validation": evaluate(fixed, val, scope),
            "future": evaluate(fixed, future, scope),
            "scope": "algorithmic comparator only; not matched agent budget",
        }
        if assessment["passed"]:
            print(json.dumps({"case": args.case, "phase": "reuse", "validation": assessment["metrics"]}), flush=True)
            inputs = [{"id": r["id"], "sequence": r["sequence"]} for r in future]
            reuse, reuse_audit, work = await phase_run(
                args.output, "reuse", {"inputs.json": inputs}, scope, config, identity, lifecycle, model_id
            )
            report["reuse"] = {
                "status": reuse["status"],
                "audit": reuse_audit,
                "session_id": reuse["session_id"],
                "tool_calls": reuse["tool_calls"],
            }
            if reuse_audit["passed"] and (work / "predictions.json").is_file():
                predictions = json.loads((work / "predictions.json").read_text())
                report["reuse"]["event"] = lifecycle.record_reuse(
                    model_id,
                    session_id=reuse["session_id"],
                    mode="science_agent",
                    inputs=inputs,
                    predictions=predictions,
                    scope=scope,
                )
                report["reuse"]["future_metrics"] = evaluate(lifecycle._model(model_id), future, scope)
                report["status"] = "completed"
            else:
                report["status"] = "reuse_failed"
        else:
            report["status"] = "candidate_rejected"
    else:
        report["status"] = "development_failed"
    atomic_json(args.output / "report.json", report)
    print(
        json.dumps({"case": args.case, "status": report["status"], "report": str(args.output / "report.json")}),
        flush=True,
    )
    return 0 if report["status"] in {"completed", "candidate_rejected"} else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["alde_parpgb", "mcba_itopride"], required=True)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rsi/engineering_smoke_glm.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(main(args))
    except Exception as exc:
        report_path = args.output / "report.json"
        if report_path.is_file():
            failure_report = json.loads(report_path.read_text())
            failure_report.update(status="controller_error", error_type=type(exc).__name__)
            atomic_json(report_path, failure_report)
        raise
    raise SystemExit(exit_code)
