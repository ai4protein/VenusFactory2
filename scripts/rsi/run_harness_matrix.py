"""Predeclared backend comparison using the same generated-model lifecycle.

Workflow/backend/model are separate factors. No silent substitution and no
outcome-dependent reruns. All allocations are saved before any model request.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

from run_generated_lifecycle import SYSTEM, policy_for, spec
from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_paired_generated import make_bundle
from run_protein_lifecycle import audit_events, prepare_case

from agent.harness import Profile, Task, builtin_registry
from agent.harness.contracts import digest, file_hash
from agent.harness.host import execute, write_json
from agent.rsi.engine.experience import LIMITATION
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin


async def phase(folder, profile, phase_name, inputs, capability, prompt, *, cold_start=False):
    work = folder / "workspace"
    work.mkdir(parents=True)
    for name, value in inputs.items():
        write_json(work / name, value)
    declaration = spec(phase_name)
    if cold_start and phase_name == "develop":
        declaration["tools"] = declaration["tools"] | {"read_model_experience"}
        declaration["required_tools"] = declaration["required_tools"] | {"read_model_experience"}
    task = Task(
        work, capability, folder / "execution", prompt, SYSTEM, frozenset(declaration["tools"]), phase=phase_name
    )
    return task, declaration


async def main(args):
    matrix = json.loads(args.matrix.read_text())
    if type(matrix.get("cold_start", False)) is not bool:
        raise ValueError("cold_start must be an explicit boolean")
    profiles = [Profile.from_dict(x) for x in matrix["profiles"]]
    if not 1 <= matrix["replicates"] <= 20 or not profiles:
        raise ValueError("Invalid matrix allocation")
    args.output.mkdir(parents=True, exist_ok=False)
    runtimes = default_runtimes()
    if args.runtimes:
        runtimes.update(json.loads(args.runtimes.read_text()))
    registry = builtin_registry()
    checks = [registry.preflight(p, runtimes) for p in profiles]
    cases = {name: prepare_case(name, args.source_cache) for name in matrix["cases"]}
    allocations = []
    for ci, case in enumerate(cases):
        for repetition in range(matrix["replicates"]):
            order = list(range(len(profiles)))
            if (ci + repetition) % 2:
                order.reverse()
            for index in order:
                allocations.append(
                    {
                        "id": len(allocations),
                        "case": case,
                        "replicate": repetition,
                        "profile_index": index,
                        "mode": profiles[index].mode,
                        "harness": profiles[index].harness,
                        "status": "allocated" if checks[index]["ready"] else "unsupported",
                    }
                )
    paths = [
        Path(__file__),
        ROOT / "scripts/rsi/run_harness_probe.py",
        ROOT / "scripts/rsi/run_generated_lifecycle.py",
        ROOT / "scripts/rsi/run_protein_lifecycle.py",
        ROOT / "scripts/rsi/run_paired_generated.py",
        *sorted((ROOT / "src/agent/harness").glob("*.py")),
        *sorted((ROOT / "src/agent/rsi").rglob("*.py")),
        *sorted((ROOT / "src/agent/graph").rglob("*.py")),
    ]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for path in paths:
        target = args.output / "implementation" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    protocol = {
        "schema": 1,
        "matrix": matrix,
        "preflight": checks,
        "implementation": hashes,
        "allocations": allocations,
        "data": {
            name: {
                "scope": c[0],
                "source": c[4],
                "split": c[5],
                "training_sha256": digest(c[1]),
                "acceptance_sha256": digest(c[2]),
                "future_sha256": digest(c[3]),
            }
            for name, c in cases.items()
        },
        "transport": "buffered_complete_response_v1",
        "claim_scope": "integration pilot, no superiority inference",
        "factor_note": "Only profiles differing in harness alone support a harness contrast; model or mode changes are joint contrasts.",
        "future_policy": "Score after all development and reuse allocations; never feed future metrics to agents.",
        "negative_runs": "All launched arms retained; unsupported cells are compatibility exclusions, not biological failures.",
    }
    write_json(args.output / "protocol.json", protocol)
    report = {"status": "preflight_complete", "protocol_sha256": digest(protocol), "arms": allocations}
    write_json(args.output / "report.json", report)
    if not args.execute:
        return
    report["status"] = "running"
    future_queue = []
    for entry in report["arms"]:
        if entry["status"] == "unsupported":
            continue
        if any(file_hash(ROOT / name) != value for name, value in hashes.items()):
            report["status"] = "implementation_drift"
            write_json(args.output / "report.json", report)
            raise ValueError("Frozen implementation changed during campaign")
        profile = profiles[entry["profile_index"]]
        key = (
            credential_from_toml(args.credential_toml, profile.model)
            if args.credential_toml
            else os.environ.get(profile.model.credential_env)
        )
        scope, train, val, future, _, _ = cases[entry["case"]]
        folder = args.output / ("arm_{:03d}".format(entry["id"]))
        entry["status"] = "running"
        write_json(args.output / "report.json", report)
        print(
            json.dumps({"id": entry["id"], "case": entry["case"], "harness": profile.harness, "phase": "develop"}),
            flush=True,
        )
        try:
            versions, version, capability = make_bundle(
                folder / "development_bundle",
                "develop",
                {"training.json": train},
                scope,
                profile.public()["model"],
                experience={"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION}
                if matrix.get("cold_start")
                else None,
            )
            prompt = (
                "Build a numerical predictor of "
                + scope["target"]
                + " for "
                + scope["protein"]
                + ". Assay: "
                + scope["assay"]
                + "; units: "
                + scope["unit"]
                + ". "
                "Short amino-acid strings encode aligned mutable positions in full proteins. "
                "Inspect the task and discover validated models. Author complete deterministic fit/predict Python, "
                "with encoding, supervised fitting and JSON model state. Evaluate with training-only feedback; "
                "repair within two attempts, then submit one actually tested source hash. "
                "Independent acceptance labels are unavailable. Report observed evidence in at most 350 words."
            )
            if matrix.get("cold_start"):
                prompt += " Read the permitted model experience before developing code; an empty result is valid and requires developing from the supplied training data."
            task, declaration = await phase(
                folder / "develop",
                profile,
                "develop",
                {"training.json": train},
                capability,
                prompt,
                cold_start=matrix.get("cold_start", False),
            )
            run = await execute(profile, task, runtimes, credential=key)
            audit = audit_events(
                task.output, "develop", run, declaration["required_tools"], declaration["recoverable_tools"]
            )
            versions.verify(version)
            entry["development"] = {"run": run, "scientific_audit": audit}
            entry["status"] = "development_failed"
            if run["status"] == "completed" and audit["passed"] and (task.workspace / "candidate.json").exists():
                lifecycle = ModelLifecycle(folder / "controller", policy_for(scope, train, val))
                mid = lifecycle.submit(
                    task.workspace / "candidate.json", session_id=run["session_id"], mode=profile.mode
                )
                entry["assessment"] = lifecycle.assess(mid, training=train, validation=val)
                entry["source_sha256"] = lifecycle._model(mid)["source_sha256"]
                entry["status"] = "candidate_rejected"
                if entry["assessment"]["passed"]:
                    inputs = [{"id": row["id"], "sequence": row["sequence"]} for row in future]
                    rv, rid, rc = make_bundle(
                        folder / "reuse_bundle",
                        "reuse",
                        {"inputs.json": inputs},
                        scope,
                        profile.public()["model"],
                        lifecycle,
                        mid,
                    )
                    reuse_profile = replace(profile, budget=replace(profile.budget, wall_seconds=420))
                    rt, rd = await phase(
                        folder / "reuse", reuse_profile, "reuse", {"inputs.json": inputs}, rc, spec("reuse")["prompt"]
                    )
                    rr = await execute(reuse_profile, rt, runtimes, credential=key)
                    ra = audit_events(rt.output, "reuse", rr, rd["required_tools"], rd["recoverable_tools"])
                    rv.verify(rid)
                    entry["reuse"] = {"run": rr, "scientific_audit": ra}
                    entry["status"] = "reuse_failed"
                    if rr["status"] == "completed" and ra["passed"] and (rt.workspace / "predictions.json").exists():
                        entry["reuse_event"] = lifecycle.record_reuse(
                            mid,
                            session_id=rr["session_id"],
                            mode=profile.mode,
                            inputs=inputs,
                            predictions=json.loads((rt.workspace / "predictions.json").read_text()),
                            scope=scope,
                        )
                        entry["status"] = "completed"
                        future_queue.append((entry, lifecycle, mid, future, scope))
        except Exception as exc:
            entry.update(status="controller_error", error_type=type(exc).__name__)
        write_json(folder / "report.json", entry)
        write_json(args.output / "report.json", report)
        print(json.dumps({"id": entry["id"], "status": entry["status"]}), flush=True)
    for entry, lifecycle, mid, future, scope in future_queue:
        entry["future_metrics"] = get_model_plugin("generated_python.v1").evaluate(lifecycle._model(mid), future, scope)
        write_json(args.output / ("arm_{:03d}".format(entry["id"])) / "report.json", entry)
    report["status"] = "campaign_completed"
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--runtimes", type=Path)
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    parser.add_argument("--execute", action="store_true")
    asyncio.run(main(parser.parse_args()))
