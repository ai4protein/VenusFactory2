"""Public-data GFP generated-code lifecycle pilot with all-allocation scoring.

Both modes run one bounded development/reuse component. This does not estimate
persistent-scientist-goal versus conversational-agent effects. No prior internal
history, donor source, author model implementation or fitted state is exposed.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


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
from agent.rsi.cases.htfunclib_model_case import prepare_htfunclib_model_case
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = "generated_classifier.large.v1"
TERMINAL = {"development_failed", "candidate_rejected", "reuse_failed", "completed", "controller_error"}
PROMPT = (
    "Develop a GFP four-class probability model from all 116,116 permitted public htFuncLib training rows. "
    "Each input is the author's aligned string of 22 variable-position amino acids in one fixed GFP scaffold, "
    "not a full protein sequence. Labels are the author's NGS/FACS assignments in the task's ordered classes. "
    "Inspect the task and discover compatible models. Author complete deterministic fit(sequences, targets, seed) "
    "and predict(model, sequences) Python. Fit must return JSON numerical parameters; predict must return one "
    "ordered four-probability vector per sequence with row sum one. Infer length and class count from training. "
    "Choose the representation, estimator and optimization yourself; no model implementation is supplied. "
    "Use only training data for preprocessing and selection; handle every training row, without downsampling. "
    "The primary metric is Brier skill relative to the training-class prior. Class balancing can affect probability "
    "calibration; use the actual training-only feedback. Inspect/discover once, then send concise complete source "
    "directly to evaluate_model_code. Repair or improve within two source attempts, then submit one tested hash. "
    "No acceptance or future labels are accessible through these tools. No prior private history or donor model "
    "is available. Keep state and probabilities efficient within the declared worker limits. "
    "Report the implementation, evidence and limitations without claiming algorithmic novelty."
)


def allocations():
    return [
        {"id": 0, "mode": "science_agent", "status": "allocated"},
        {"id": 1, "mode": "science_expert", "status": "allocated"},
    ]


def model_policy(scope, train, validation):
    plugin = get_model_plugin(PLUGIN)
    return {
        "schema": 1,
        "scope": scope,
        "plugin_id": PLUGIN,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(train),
        "validation_sha256": digest(validation),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }


def seal_barrier(output, arms):
    if [x["id"] for x in arms] != [0, 1] or any(x["status"] not in TERMINAL for x in arms):
        raise ValueError("Every preallocated workflow must terminate before future scoring")
    sealed = []
    for arm in arms:
        folder = output / ("arm_{:03d}".format(arm["id"]))
        candidate = folder / "develop/workspace/candidate.json"
        terminal = folder / "terminal.json"
        if json.loads(terminal.read_text()) != arm:
            raise ValueError("Terminal allocation record changed")
        sealed.append(
            {
                "id": arm["id"],
                "status": arm["status"],
                "terminal_sha256": file_hash(terminal),
                "candidate_sha256": file_hash(candidate)
                if candidate.is_file() and not candidate.is_symlink()
                else None,
            }
        )
    value = {
        "all_allocations_terminal": True,
        "allocations": sealed,
        "future_labels_delivered_to_agent": False,
        "time_unix": time.time(),
    }
    write_json(output / "future_barrier.json", value)
    return value


def score_all(output, arms, barrier, train, future, scope):
    if not barrier.get("all_allocations_terminal") or len(barrier["allocations"]) != len(arms):
        raise ValueError("Missing complete future barrier")
    plugin = get_model_plugin(PLUGIN)
    for arm, sealed in zip(arms, barrier["allocations"], strict=True):
        folder = output / ("arm_{:03d}".format(arm["id"]))
        if arm["id"] != sealed["id"] or file_hash(folder / "terminal.json") != sealed["terminal_sha256"]:
            raise ValueError("Frozen allocation changed after barrier")
        score = {
            "all_allocation_brier_skill": 0.0,
            "raw_metrics": None,
            "status": "missing_submitted_candidate",
            "failure_convention": "training-prior fallback has zero Brier skill",
        }
        candidate = folder / "develop/workspace/candidate.json"
        if sealed["candidate_sha256"] is not None:
            try:
                if candidate.is_symlink() or file_hash(candidate) != sealed["candidate_sha256"]:
                    raise ValueError("Candidate changed after barrier")
                model = json.loads(candidate.read_text())
                rebuilt = plugin.fit(train, scope, model["recipe"])
                if digest(model) != digest(rebuilt):
                    raise ValueError("Submitted artifact differs from independent full-training replay")
                metrics = plugin.evaluate(rebuilt, future, scope)
                score.update(
                    status="scored_submitted_model",
                    raw_metrics=metrics,
                    all_allocation_brier_skill=metrics[plugin.acceptance_metric],
                    source_sha256=rebuilt["source_sha256"],
                    independent_training_replay=True,
                )
            except Exception as exc:
                score.update(
                    status="unverifiable_or_unscorable_candidate", error_type=type(exc).__name__, error=str(exc)[:3000]
                )
        arm["future_scoring"] = score
        write_json(folder / "future_scoring.json", score)
    return arms


async def main(args):
    profile = Profile.from_dict(json.loads(args.profile.read_text()))
    profiles = {mode: replace(profile, mode=mode) for mode in ["science_agent", "science_expert"]}
    runtimes = default_runtimes()
    preflight = {m: builtin_registry().preflight(p, runtimes) for m, p in profiles.items()}
    if not all(x["ready"] for x in preflight.values()):
        raise ValueError("Both workflow modes must pass the selected harness admission")
    scope, train, validation, future, source, split = prepare_htfunclib_model_case(args.assets)
    args.output.mkdir(parents=True, exist_ok=False)
    paths = sorted(
        {
            Path(__file__).resolve(),
            *list((ROOT / "src/agent").rglob("*.py")),
            *list((ROOT / "src/agent/prompts").glob("*.md")),
            *list((ROOT / "scripts/rsi").glob("*.py")),
        }
    )
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for name in hashes:
        dest = args.output / "implementation" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    for name, rows in [("training", train), ("acceptance", validation), ("future", future)]:
        write_json(args.output / (name + ".json"), rows)
    protocol = {
        "schema": 1,
        "case": "htfunclib_generated_code_public_only_v1",
        "source": source,
        "split": split,
        "profiles": {m: p.public() for m, p in profiles.items()},
        "preflight": preflight,
        "scope": scope,
        "plugin_id": PLUGIN,
        "plugin_sha256": get_model_plugin(PLUGIN).source_sha256,
        "model_policy": model_policy(scope, train, validation),
        "allocations": allocations(),
        "implementation": hashes,
        "development_prompt": PROMPT,
        "development_instructions": SYSTEM,
        "experience": "none; no history tool, private donor source or fitted author model",
        "future_policy": "all allocations terminate and artifacts are sealed before host scoring; rejected and failed allocations remain",
        "failure_policy": "score every independently replayable submitted model, including rejected or unreused models; missing/invalid predictions receive training-prior Brier skill zero",
        "comparison_scope": "one bounded workflow component per mode, one allocation each; not a persistent-goal comparison or an efficacy estimate",
        "representation_scope": "22 variable-position amino acids; distinct from the author's 74 engineered classifier features",
        "data_scope": "full published partitions, retrospective and already examined by investigators",
        "resource_limits": "common profile development budget; reuse 420 seconds, at most 12 model calls and 8 tool calls; workers 90 CPU seconds, 120 wall seconds, 4 GiB",
        "publication_rule": "requires complete request/tool audit and independent positive acceptance Brier skill",
    }
    write_json(args.output / "protocol.json", protocol)
    report = {"status": "prepared", "protocol_sha256": digest(protocol), "arms": allocations()}
    write_json(args.output / "report.json", report)
    if not args.execute:
        return report
    key = resolve_credential(profile.model, args.credential_toml)
    started = time.monotonic()
    report["status"] = "running"
    write_json(args.output / "report.json", report)
    for arm in report["arms"]:
        if any(file_hash(ROOT / n) != h for n, h in hashes.items()):
            raise ValueError("Frozen implementation changed")
        folder = args.output / ("arm_{:03d}".format(arm["id"]))
        folder.mkdir()
        p = profiles[arm["mode"]]
        arm["status"] = "running"
        write_json(args.output / "report.json", report)
        then = time.monotonic()
        try:
            lc = ModelLifecycle(folder / "controller", model_policy(scope, train, validation))
            versions, vid, cap = make_bundle(
                folder / "development_bundle",
                "develop",
                {"training.json": train},
                scope,
                p.public()["model"],
                plugin_id=PLUGIN,
            )
            work = folder / "develop/workspace"
            work.mkdir(parents=True)
            write_json(work / "training.json", train)
            task = Task(
                work,
                cap,
                folder / "develop/execution",
                PROMPT,
                SYSTEM,
                frozenset(spec("develop")["tools"]),
                phase="develop",
            )
            run = await execute(p, task, runtimes, credential=key)
            checked = audit_events(
                task.output, "develop", run, {"inspect_model_code_task", "submit_model_code"}, {"evaluate_model_code"}
            )
            versions.verify(vid)
            arm.update(status="development_failed", development={"run": run, "scientific_audit": checked})
            if (
                run["status"] == "completed"
                and run["audit"]["passed"]
                and checked["passed"]
                and (work / "candidate.json").is_file()
            ):
                mid = lc.submit(work / "candidate.json", session_id=run["session_id"], mode=p.mode)
                assessment = lc.assess(mid, training=train, validation=validation)
                arm.update(
                    status="candidate_rejected",
                    model_id=mid,
                    assessment=assessment,
                    source_sha256=lc._model(mid)["source_sha256"],
                )
                if assessment["passed"]:
                    inputs = [{"id": row["id"], "sequence": row["sequence"]} for row in future]
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
                    rp = replace(
                        p,
                        budget=replace(
                            p.budget,
                            wall_seconds=420,
                            model_calls=min(p.budget.model_calls, 12),
                            tool_calls=min(p.budget.tool_calls, 8),
                        ),
                    )
                    rt = Task(
                        rw,
                        rc,
                        folder / "reuse/execution",
                        spec("reuse")["prompt"],
                        SYSTEM,
                        frozenset(spec("reuse")["tools"]),
                        phase="reuse",
                    )
                    rr = await execute(rp, rt, runtimes, credential=key)
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
        write_json(folder / "terminal.json", arm)
        write_json(args.output / "report.json", report)
        print(json.dumps({"id": arm["id"], "mode": arm["mode"], "status": arm["status"]}), flush=True)
    barrier = seal_barrier(args.output, report["arms"])
    score_all(args.output, report["arms"], barrier, train, future, scope)
    report.update(
        status="campaign_completed",
        wall_seconds=time.monotonic() - started,
        source_intact=all(file_hash(ROOT / n) == h for n, h in hashes.items()),
        all_allocations_scored=True,
        future_barrier_sha256=file_hash(args.output / "future_barrier.json"),
    )
    write_json(args.output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    asyncio.run(main(parser.parse_args()))
