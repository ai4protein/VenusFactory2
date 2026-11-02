"""Matched bounded Agent/Scientist model generation on frozen REAP features.

Both components share data, scientific tools and inference ceilings. This first
integration matrix is not a persistent-goal superiority or cross-family study.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

from run_generated_lifecycle import SYSTEM, spec
from run_harness_probe import default_runtimes
from run_paired_generated import make_bundle
from run_protein_lifecycle import audit_events

from agent.harness import Profile, Task, builtin_registry
from agent.harness.catalog import resolve_credential
from agent.harness.contracts import digest, file_hash
from agent.harness.host import execute, write_json
from agent.rsi.experience import LIMITATION, ModelExperienceStore
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.reap_experience import load_library, project_for_target, transfer_prompt
from agent.rsi.reap_model_case import prepare_reap_model_case

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = "generated_features.v1"
FAMILIES = ("P450_BM3_FL62", "SaSrtA")


def policy(scope, training, acceptance):
    plugin = get_model_plugin(PLUGIN)
    return {
        "schema": 1,
        "scope": scope,
        "plugin_id": PLUGIN,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(training),
        "validation_sha256": digest(acceptance),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }


def development_prompt(scope):
    return (
        f"Develop a supervised predictor of measured {scope['target']} for {scope['protein']}. "
        "Full protein sequences have already been encoded by a frozen ESM2-650M model. "
        "Your numeric ABI is fit(features, targets, seed) and predict(model, features), where features is a list of fixed 1280-dimensional vectors. "
        "Do not treat these numeric vectors as sequence strings. Inspect the task, discover models and read the permitted experience. "
        "Write your own complete deterministic fitting and prediction source; infer dimension and choose preprocessing and an estimator. "
        "No fitted parameters, reference implementation or donor model is supplied. All preprocessing must be learned from training rows. "
        "Return JSON numeric state, not a pickled estimator. Predict must return one finite real activity per row. "
        "Evaluate with the training-only tool, then improve or repair within two source attempts and submit one actually tested hash. "
        "Use the measured targets, not sequence length. Primary acceptance is MSE skill above the training-mean predictor; ranking correlation is also reported. "
        "No external acceptance or future labels are accessible. The worker allows numpy/sklearn, 90 CPU seconds, 120 wall seconds and 4 GiB per operation. "
        "Keep source concise, preferably below 6000 characters; generate parameters inside fit. "
        "Send source directly to evaluate_model_code, use rationale under 120 words, and submit before your final evidence summary. "
        "Do not repeat unchanged inspection calls or claim algorithmic novelty."
    )


async def main(args):
    profile = Profile.from_dict(json.loads(args.profile.read_text()))
    if profile.harness != "langgraph":
        raise ValueError("This component comparison requires a common LangGraph engine")
    profiles = {mode: replace(profile, mode=mode) for mode in ["science_agent", "science_expert"]}
    runtimes = default_runtimes()
    admissions = {mode: builtin_registry().preflight(p, runtimes) for mode, p in profiles.items()}
    if not all(p["ready"] for p in admissions.values()):
        raise ValueError("Both runtime components must pass admission")
    cases = {
        family: prepare_reap_model_case(args.admission, args.features, args.baseline, family, args.seed)
        for family in FAMILIES
    }
    library = getattr(args, "experience_library", None)
    condition = getattr(args, "condition", "none")
    if condition not in {"none", "summary", "source"} or (condition != "none" and library is None):
        raise ValueError("Nonempty experience requires a frozen donor library")
    library_manifest = load_library(library)[0] if library else None
    empty = {"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION}
    projections = {
        f: project_for_target(library, case[0], condition) if library else empty for f, case in cases.items()
    }
    prompts = {
        f: transfer_prompt(case[0], development_prompt(case[0])) if library else development_prompt(case[0])
        for f, case in cases.items()
    }
    allocations = []
    for i, family in enumerate(FAMILIES):
        order = ["science_agent", "science_expert"] if i == 0 else ["science_expert", "science_agent"]
        for mode in order:
            allocations.append(
                {"id": len(allocations), "family": family, "mode": mode, "condition": condition, "status": "allocated"}
            )
    args.output.mkdir(parents=True, exist_ok=False)
    if library:
        shutil.copytree(library, args.output / "donor_library")
    paths = [
        Path(__file__),
        *(
            ROOT / "scripts/rsi" / n
            for n in [
                "run_generated_lifecycle.py",
                "run_harness_probe.py",
                "run_paired_generated.py",
                "run_protein_lifecycle.py",
            ]
        ),
        *sorted((ROOT / "src/agent/rsi").glob("*.py")),
        *sorted((ROOT / "src/agent/harness").glob("*.py")),
        *sorted((ROOT / "src/agent/graph").rglob("*.py")),
        *sorted((ROOT / "src/agent/prompts").glob("*.md")),
        ROOT / "src/agent/kimi_client.py",
    ]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for relative in hashes:
        dest = args.output / "implementation" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    case_records = {}
    for family, (scope, train, val, future, source, split) in cases.items():
        folder = args.output / "data" / family
        folder.mkdir(parents=True)
        for name, rows in [("training", train), ("acceptance", val), ("future", future)]:
            write_json(folder / (name + ".json"), rows)
        case_records[family] = {
            "scope": scope,
            "source": source,
            "split": split,
            "data_hashes": {n: digest(v) for n, v in [("training", train), ("acceptance", val), ("future", future)]},
            "prompt": prompts[family],
            "experience_projection": projections[family],
        }
    protocol = {
        "schema": 1,
        "experience_library": library_manifest,
        "defer_future_scoring": getattr(args, "defer_future_scoring", False),
        "profiles": {m: p.public() for m, p in profiles.items()},
        "cases": case_records,
        "plugin": PLUGIN,
        "plugin_sha256": get_model_plugin(PLUGIN).source_sha256,
        "implementation": hashes,
        "allocations": allocations,
        "system_sha256": digest(SYSTEM),
        "training_feedback": "Two sequence-hash 80/20 internal training splits; one submitted candidate; external acceptance labels unavailable",
        "acceptance": "Independent training replay and MSE skill strictly above training mean on the frozen external single-mutant acceptance subset",
        "future_policy": "After all arms, score every independently refitted submitted candidate, including rejected candidates, on higher-order variants; no score fed back to agents",
        "registration": "Only accepted models are published; source-only experiences are stored separately per arm and not fed to other allocations",
        "resources": "Common profile per phase; reuse wall ceiling 420 seconds; code worker 90 CPU seconds/120 wall seconds/4 GiB",
        "scope": "Four bounded workflow-component allocations on public retrospective data; not full persistent-goal versus conversational behavior, mode superiority, baseline-compute equality or cross-family RSI efficacy",
    }
    write_json(args.output / "protocol.json", protocol)
    report = {"status": "preflight", "protocol_sha256": digest(protocol), "arms": [dict(x) for x in allocations]}
    write_json(args.output / "report.json", report)
    if not args.execute:
        return
    key = resolve_credential(profile.model, args.credential_toml)
    report["status"] = "running"
    write_json(args.output / "report.json", report)
    started = time.monotonic()
    queue = []
    for arm in report["arms"]:
        if any(file_hash(ROOT / n) != h for n, h in hashes.items()):
            raise ValueError("Frozen implementation changed")
        scope, train, val, future, _, _ = cases[arm["family"]]
        folder = args.output / f"arm_{arm['id']:03d}"
        folder.mkdir()
        arm["status"] = "running"
        write_json(args.output / "report.json", report)
        print(json.dumps({k: arm[k] for k in ["id", "family", "mode", "status"]}), flush=True)
        then = time.monotonic()
        p = profiles[arm["mode"]]
        experience = projections[arm["family"]]
        try:
            lc = ModelLifecycle(folder / "controller", policy(scope, train, val))
            versions, version, cap = make_bundle(
                folder / "development_bundle",
                "develop",
                {"training.json": train},
                scope,
                p.public()["model"],
                experience=experience,
                plugin_id=PLUGIN,
            )
            work = folder / "develop/workspace"
            work.mkdir(parents=True)
            write_json(work / "training.json", train)
            task = Task(
                work,
                cap,
                folder / "develop/execution",
                prompts[arm["family"]],
                SYSTEM,
                frozenset(spec("develop")["tools"] | {"read_model_experience"}),
                phase="develop",
            )
            run = await execute(p, task, runtimes, credential=key)
            audit = audit_events(
                task.output,
                "develop",
                run,
                {"inspect_model_code_task", "read_model_experience", "submit_model_code"},
                {"evaluate_model_code"},
            )
            versions.verify(version)
            arm.update(status="development_failed", development={"run": run, "scientific_audit": audit})
            if (
                run["status"] == "completed"
                and run["audit"]["passed"]
                and audit["passed"]
                and (work / "candidate.json").exists()
            ):
                mid = lc.submit(work / "candidate.json", session_id=run["session_id"], mode=p.mode)
                assessment = lc.assess(mid, training=train, validation=val)
                arm.update(
                    status="candidate_rejected",
                    model_id=mid,
                    assessment=assessment,
                    source_sha256=lc._model(mid)["source_sha256"],
                    exposed_experience_ids=[e["experience_id"] for e in experience["entries"]],
                    matches_exposed_source=[
                        e["experience_id"]
                        for e in experience["entries"]
                        if e["source_sha256"] == lc._model(mid)["source_sha256"]
                    ],
                )
                queue.append((arm, lc, mid, future, scope))
                if assessment["passed"]:
                    if library:
                        shutil.copytree(args.output / "donor_library/experience", folder / "experience")
                    store = ModelExperienceStore(folder / "experience")
                    arm["registered_experience_id"] = store.publish(
                        lc,
                        mid,
                        summary="GLM-generated numeric protein-feature model; independently replayed and accepted on the declared single-mutant subset; no future efficacy implied.",
                        parents=arm["exposed_experience_ids"],
                    )
                    inputs = [{k: r[k] for k in ["id", "sequence", "features"]} for r in future]
                    rv, rid, rc = make_bundle(
                        folder / "reuse_bundle",
                        "reuse",
                        {"inputs.json": inputs},
                        scope,
                        p.public()["model"],
                        lc,
                        mid,
                        plugin_id=PLUGIN,
                    )
                    rw = folder / "reuse/workspace"
                    rw.mkdir(parents=True)
                    write_json(rw / "inputs.json", inputs)
                    rt = Task(
                        rw,
                        rc,
                        folder / "reuse/execution",
                        spec("reuse")["prompt"],
                        SYSTEM,
                        frozenset(spec("reuse")["tools"]),
                        phase="reuse",
                    )
                    rr = await execute(
                        replace(p, budget=replace(p.budget, wall_seconds=420)), rt, runtimes, credential=key
                    )
                    ra = audit_events(
                        rt.output, "reuse", rr, {"discover_generated_models", "predict_with_generated_model"}
                    )
                    rv.verify(rid)
                    arm.update(status="reuse_failed", reuse={"run": rr, "scientific_audit": ra})
                    if rr["status"] == "completed" and rr["audit"]["passed"] and ra["passed"]:
                        arm["reuse_event"] = lc.record_reuse(
                            mid,
                            session_id=rr["session_id"],
                            mode=p.mode,
                            inputs=inputs,
                            predictions=json.loads((rw / "predictions.json").read_text()),
                            scope=scope,
                        )
                        arm["status"] = "completed"
        except Exception as exc:
            import traceback

            arm.update(status="controller_error", error_type=type(exc).__name__)
            (folder / "controller_error.log").write_text(traceback.format_exc().replace(key, "[redacted]"))
        arm["wall_seconds"] = time.monotonic() - then
        write_json(folder / "report.json", arm)
        write_json(args.output / "report.json", report)
        print(json.dumps({"id": arm["id"], "status": arm["status"]}), flush=True)
    if getattr(args, "defer_future_scoring", False):
        report.update(
            status="development_completed",
            wall_seconds=time.monotonic() - started,
            source_intact=all(file_hash(ROOT / n) == h for n, h in hashes.items()),
        )
        write_json(args.output / "report.json", report)
        return
    for arm, lc, mid, future, scope in queue:
        arm["future_metrics"] = get_model_plugin(PLUGIN).evaluate(lc._model(mid), future, scope)
        arm["future_metric_role"] = (
            "Prespecified post-allocation scoring, irrespective of acceptance/reuse; not selection feedback"
        )
        write_json(args.output / f"arm_{arm['id']:03d}" / "report.json", arm)
    report.update(
        status="campaign_completed",
        wall_seconds=time.monotonic() - started,
        source_intact=all(file_hash(ROOT / n) == h for n, h in hashes.items()),
    )
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["profile", "admission", "features", "baseline", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--credential-toml", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experience-library", type=Path)
    p.add_argument("--condition", choices=["none", "summary", "source"], default="none")
    p.add_argument("--defer-future-scoring", action="store_true")
    p.add_argument("--execute", action="store_true")
    asyncio.run(main(p.parse_args()))
