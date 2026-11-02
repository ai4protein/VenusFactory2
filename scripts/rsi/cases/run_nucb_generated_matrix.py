"""Full-sequence NucB model code generation, independent acceptance and reuse.

A frozen exploratory workflow-component/source-experience matrix. It does not
reproduce author CNN training or estimate persistent-versus-conversational effects.
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
from agent.rsi.engine.experience import ModelExperienceStore
from agent.rsi.engine.model_lifecycle import ModelLifecycle
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.cases.nucb_case import prepare_nucb_case

ROOT = Path(__file__).resolve().parents[3]
PLUGIN = "generated_classifier.v1"


def policy(scope, train, val):
    plugin = get_model_plugin(PLUGIN)
    return {
        "schema": 1,
        "scope": scope,
        "plugin_id": PLUGIN,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(train),
        "validation_sha256": digest(val),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }


def development_prompt(guidance="standard"):
    if guidance not in {"standard", "compact_probability"}:
        raise ValueError("Unknown development guidance")
    prompt = (
        "Develop a full-sequence NucB binary activity classifier from all 22,980 permitted G1/G2 sequences. "
        "The label is the author's generation-specific FDR activity-above-wildtype call, not continuous activity. "
        "Inputs are full 142-residue sequences. Inspect the task, discover compatible models and read the permitted model experience. "
        "The donor, if any, concerns a different protein and possibly a different prediction ABI; it is optional implementation precedent, "
        "not a validated target model. Independently refit and adapt to this task. Do not reuse donor fitted weights. "
        "Author complete deterministic fit(sequences, targets, seed) and predict(model, sequences) Python. "
        "Fit returns JSON parameters; predict returns two probabilities per sequence in the declared class order. "
        "Each row must sum to one. Classes are imbalanced; use only training feedback for model selection. "
        "Choose an efficient numerical fitting method under the code worker resource limits. "
        "Evaluate and repair or improve within two source attempts, then submit one actually tested hash. "
        "No G3/G4 labels are accessible through the tools. Independent acceptance and future scoring occur later. "
        "Report actual evidence and limitations without claiming algorithmic novelty."
    )
    if guidance == "compact_probability":
        prompt += (
            " Execution guidance: inspect/discover/read experience once, then act on those results without repeating unchanged lookups. "
            "Write concise complete fit/predict source, preferably below 6,000 characters, using shared encoding helpers rather than repeated blocks. "
            "Generate numerical parameters inside fit; do not emit fitted arrays or training rows as source literals. "
            "Send the implementation directly to evaluate_model_code; keep rationale below 120 words and omit long docstrings or design essays. "
            "The primary objective is probability accuracy measured by Brier skill, not accuracy, balanced recall or ranking AUC. "
            "Class reweighting may change probability calibration; any calibration must be learned from permitted training data only. "
            "Inspect the actual training-only Brier feedback before the second attempt, and submit one tested source hash before reporting. "
            "You choose the representation, estimator and optimization; no fitted model or source implementation is supplied."
        )
    return prompt


def allocate(conditions):
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or not set(conditions) <= {"none", "summary", "source"}
    ):
        raise ValueError("Conditions must be distinct registered projections")
    rows = []
    for ci, condition in enumerate(conditions):
        modes = ["science_agent", "science_expert"] if ci % 2 == 0 else ["science_expert", "science_agent"]
        for mode in modes:
            rows.append({"id": len(rows), "condition": condition, "mode": mode, "status": "allocated"})
    return rows


async def main(args):
    profile = Profile.from_dict(json.loads(args.profile.read_text()))
    if profile.harness != "langgraph":
        raise ValueError("This matched workflow-component matrix requires LangGraph")
    allocations = allocate(args.conditions)
    prompt = development_prompt(args.development_guidance)
    scope, train, val, future, source, split = prepare_nucb_case(args.assets, args.rebuilt)
    original = ModelExperienceStore(args.donor_store)
    selected = original._entry(args.donor_id)
    if selected["donor_scope"]["protein"] == scope["protein"]:
        raise ValueError("Cross-protein donor required")
    projections = {c: original.project(c, [args.donor_id]) for c in args.conditions}
    args.output.mkdir(parents=True, exist_ok=False)
    runtimes = default_runtimes()
    registry = builtin_registry()
    profiles = {mode: replace(profile, mode=mode) for mode in ("science_agent", "science_expert")}
    admissions = {mode: registry.preflight(p, runtimes) for mode, p in profiles.items()}
    if not all(x["ready"] for x in admissions.values()):
        raise ValueError("Both modes must pass admission")
    paths = [
        Path(__file__),
        ROOT / "scripts/rsi/run_harness_probe.py",
        ROOT / "scripts/rsi/run_generated_lifecycle.py",
        ROOT / "scripts/rsi/run_paired_generated.py",
        ROOT / "scripts/rsi/run_protein_lifecycle.py",
        *sorted((ROOT / "src/agent/rsi").rglob("*.py")),
        *sorted((ROOT / "src/agent/harness").glob("*.py")),
        *sorted((ROOT / "src/agent/graph").rglob("*.py")),
        *sorted((ROOT / "src/agent/prompts").glob("*.md")),
        ROOT / "src/agent/kimi_client.py",
    ]
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
    for name in hashes:
        dest = args.output / "implementation" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    donor_dir = args.output / "donor_evidence"
    donor_dir.mkdir()
    write_json(donor_dir / "entry.json", selected)
    write_json(args.output / "data_source.json", source)
    for name, rows in (("training", train), ("acceptance", val), ("future", future)):
        write_json(args.output / (name + ".json"), rows)
    for condition, projection in projections.items():
        write_json(donor_dir / (condition + ".json"), projection)
    protocol = {
        "schema": 1,
        "profiles": {m: p.public() for m, p in profiles.items()},
        "scope": scope,
        "source": source,
        "split": split,
        "data_hashes": {"training": digest(train), "acceptance": digest(val), "future": digest(future)},
        "plugin": PLUGIN,
        "plugin_sha256": get_model_plugin(PLUGIN).source_sha256,
        "allocations": allocations,
        "implementation": hashes,
        "donor_id": args.donor_id,
        "donor_selection": "first verified McbA generation from earlier browser goal; fixed before NucB model evaluation",
        "donor_scope": selected["donor_scope"],
        "experience_projections": {c: digest(p) for c, p in projections.items()},
        "primary_acceptance": "Brier skill strictly above training-prior baseline on G3; one submitted candidate",
        "future_policy": "G4 labels assessed only after all development/reuse allocations finish; no future feedback",
        "training_feedback": "two source attempts; deterministic sequence-hash 80/20 split of G1/G2 training",
        "comparison": "two bounded workflow components x declared donor-experience projections; one allocation each",
        "conditions": args.conditions,
        "development_guidance": args.development_guidance,
        "development_prompt": prompt,
        "development_prompt_sha256": digest(prompt),
        "development_instructions_sha256": digest(SYSTEM),
        "guidance_scope": "human-authored execution and scoring guidance; not autonomous improver evolution",
        "limits": [
            "not a full persistent-goal versus conversational comparison",
            "public retrospective data and shared LLM network; not blind",
            "single donor and target family; no generalized transfer-effect inference",
            "author CNN training and MBO are not reproduced",
        ],
        "resources": "common profile limits per phase; reuse wall limit 420 seconds; code worker 90 CPU s / 4 GiB / 120 wall s per call",
    }
    write_json(args.output / "protocol.json", protocol)
    report = {"status": "preflight", "protocol_sha256": digest(protocol), "arms": [dict(x) for x in allocations]}
    write_json(args.output / "report.json", report)
    if not args.execute:
        return
    key = resolve_credential(profile.model, args.credential_toml)
    report["status"] = "running"
    write_json(args.output / "report.json", report)
    queue = []
    started = time.monotonic()
    for entry in report["arms"]:
        if any(file_hash(ROOT / n) != h for n, h in hashes.items()):
            raise ValueError("Frozen implementation changed")
        p = profiles[entry["mode"]]
        folder = args.output / ("arm_{:03d}".format(entry["id"]))
        folder.mkdir()
        entry["status"] = "running"
        write_json(args.output / "report.json", report)
        print(
            json.dumps({"id": entry["id"], "mode": entry["mode"], "condition": entry["condition"], "phase": "develop"}),
            flush=True,
        )
        then = time.monotonic()
        try:
            lc = ModelLifecycle(folder / "controller", policy(scope, train, val))
            versions, version, cap = make_bundle(
                folder / "development_bundle",
                "develop",
                {"training.json": train},
                scope,
                p.public()["model"],
                experience=projections[entry["condition"]],
                plugin_id=PLUGIN,
            )
            prompt = development_prompt(args.development_guidance)
            work = folder / "develop/workspace"
            work.mkdir(parents=True)
            write_json(work / "training.json", train)
            tools = spec("develop")["tools"] | {"read_model_experience"}
            task = Task(work, cap, folder / "develop/execution", prompt, SYSTEM, frozenset(tools), phase="develop")
            run = await execute(p, task, runtimes, credential=key)
            audit = audit_events(
                task.output,
                "develop",
                run,
                {"inspect_model_code_task", "read_model_experience", "submit_model_code"},
                {"evaluate_model_code"},
            )
            versions.verify(version)
            entry.update(development={"run": run, "scientific_audit": audit}, status="development_failed")
            if (
                run["status"] == "completed"
                and run["audit"]["passed"]
                and audit["passed"]
                and (work / "candidate.json").exists()
            ):
                mid = lc.submit(work / "candidate.json", session_id=run["session_id"], mode=p.mode)
                assessment = lc.assess(mid, training=train, validation=val)
                entry.update(
                    model_id=mid,
                    assessment=assessment,
                    source_sha256=lc._model(mid)["source_sha256"],
                    status="candidate_rejected",
                )
                if assessment["passed"]:
                    store = ModelExperienceStore(folder / "experience")
                    parents = []
                    if entry["condition"] != "none":
                        if selected["parents"]:
                            raise ValueError("This protocol requires a root donor experience")
                        write_json(store.root / "entries" / (args.donor_id + ".json"), selected)
                        write_json(store.root / "catalog.json", {"schema": 1, "entries": [args.donor_id]})
                        parents = [args.donor_id]
                    entry["registered_experience_id"] = store.publish(
                        lc,
                        mid,
                        summary="GLM-generated full-sequence NucB classifier; independently replayed training and G3 Brier acceptance. Source donor condition: "
                        + entry["condition"],
                        parents=parents,
                    )
                    inputs = [{"id": r["id"], "sequence": r["sequence"]} for r in future]
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
                    rp = replace(p, budget=replace(p.budget, wall_seconds=420))
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
                    entry.update(reuse={"run": rr, "scientific_audit": ra}, status="reuse_failed")
                    if rr["status"] == "completed" and rr["audit"]["passed"] and ra["passed"]:
                        entry["reuse_event"] = lc.record_reuse(
                            mid,
                            session_id=rr["session_id"],
                            mode=p.mode,
                            inputs=inputs,
                            predictions=json.loads((rw / "predictions.json").read_text()),
                            scope=scope,
                        )
                        entry["status"] = "completed"
                        queue.append((entry, lc, mid))
        except Exception as exc:
            entry.update(status="controller_error", error_type=type(exc).__name__)
            import traceback

            (folder / "controller_error.log").write_text(traceback.format_exc().replace(key, "[redacted]"))
            # Provider error text is not emitted to the user-facing report.
        entry["wall_seconds"] = time.monotonic() - then
        write_json(folder / "report.json", entry)
        write_json(args.output / "report.json", report)
        print(json.dumps({"id": entry["id"], "status": entry["status"]}), flush=True)
    for entry, lc, mid in queue:
        entry["future_metrics"] = get_model_plugin(PLUGIN).evaluate(lc._model(mid), future, scope)
        write_json(args.output / ("arm_{:03d}".format(entry["id"])) / "report.json", entry)
    report.update(
        status="campaign_completed",
        wall_seconds=time.monotonic() - started,
        source_intact=all(file_hash(ROOT / n) == h for n, h in hashes.items()),
    )
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--credential-toml", type=Path)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--rebuilt", type=Path, required=True)
    p.add_argument("--donor-store", type=Path, required=True)
    p.add_argument("--donor-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--conditions", nargs="+", choices=["none", "summary", "source"], default=["none", "summary", "source"]
    )
    p.add_argument("--development-guidance", choices=["standard", "compact_probability"], default="standard")
    p.add_argument("--execute", action="store_true")
    asyncio.run(main(p.parse_args()))
