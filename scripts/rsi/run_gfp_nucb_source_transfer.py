"""Fixed GFP-generated source transferred to all NucB G1/G2 training sequences.

No LLM requests or target source edits. This tests source portability and target
performance, not autonomous target redesign or improvement of an improver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi.classifier_transfer import LOGISTIC_REFERENCE, refit_recipe, registered_source  # noqa: E402
from agent.rsi.model_lifecycle import ModelLifecycle  # noqa: E402
from agent.rsi.model_plugins import get_model_plugin  # noqa: E402
from agent.rsi.nucb_case import prepare_nucb_case  # noqa: E402
from agent.rsi.protein_model import digest  # noqa: E402
from agent.rsi.storage import atomic_json, sha256_file  # noqa: E402

PLUGIN = "generated_classifier.large.v1"


def run(args):
    source, donor, donor_model = registered_source(args.donor_run, args.donor_audit)
    scope, train, validation, future, provenance, split = prepare_nucb_case(args.assets, args.rebuilt)
    if scope["protein"] == donor["donor_scope"]["protein"]:
        raise ValueError("A different protein is required")
    plugin = get_model_plugin(PLUGIN)
    for rows in [train, validation, future]:
        plugin.validate_rows(rows, scope, labels=True)
    inputs = [{"id": r["id"], "sequence": r["sequence"]} for r in future]
    try:
        plugin.predict(donor_model, inputs, scope)
    except ValueError:
        scope_guard = True
    else:
        raise ValueError("Donor fitted model incorrectly accepted a different protein scope")
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output
    arms = [
        {
            "id": "generated_source_transfer",
            "source_origin": "GLM-generated on GFP; unchanged source",
            "source_code": source,
        },
        {
            "id": "fixed_logistic_reference",
            "source_origin": "operator-fixed C=1 one-hot logistic reference",
            "source_code": LOGISTIC_REFERENCE,
        },
    ]
    recipes = {a["id"]: refit_recipe(a.pop("source_code")) for a in arms}
    paths = [
        Path(__file__).resolve(),
        ROOT / "src/agent/rsi/classifier_transfer.py",
        ROOT / "src/agent/rsi/nucb_case.py",
        ROOT / "src/agent/rsi/generated_classifier_large.py",
        ROOT / "src/agent/rsi/generated_classifier.py",
        ROOT / "src/agent/rsi/code_sandbox.py",
        ROOT / "src/agent/rsi/code_worker.py",
        ROOT / "src/agent/rsi/numeric_runtime.py",
        ROOT / "src/agent/rsi/model_lifecycle.py",
        ROOT / "src/agent/rsi/model_plugins.py",
        ROOT / "src/agent/rsi/protein_model.py",
    ]
    binding = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    for name in binding:
        p = root / "implementation" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    for name, value in [("training", train), ("acceptance", validation), ("future", future)]:
        atomic_json(root / (name + ".json"), value)
    protocol = {
        "schema": 1,
        "execution_origin": "operator_driven_frozen_source_transfer",
        "external_model_calls": 0,
        "donor": donor,
        "scope": scope,
        "source": provenance,
        "split": split,
        "data_hashes": {"training": digest(train), "acceptance": digest(validation), "future": digest(future)},
        "allocations": arms,
        "recipes": recipes,
        "plugin_id": PLUGIN,
        "plugin_sha256": plugin.source_sha256,
        "implementation": binding,
        "donor_fitted_scope_guard_passed": scope_guard,
        "selection": "One fixed source per arm; no search, replacement or target-driven source repair",
        "calibration": "Any calibration in the unchanged generated algorithm is refitted using target training data only",
        "acceptance": "G3 Brier skill strictly above G1/G2 class-prior baseline; one candidate per arm",
        "future_policy": "Every G3 decision is final before any G4 scoring; score rejected models too; failed predictions receive prior fallback skill zero",
        "limitations": [
            "No autonomous target redesign, author CNN retraining or general RSI efficacy claim",
            "Retrospective public generations already examined by investigators",
            "One source-target pair, no biological replicates; common lineage and generation-dependent selection remain",
            "Different source and target assays; GFP has 22 variable-position residues and four classes, NucB has 142 residues and two classes",
        ],
    }
    atomic_json(root / "protocol.json", protocol)
    report = {
        "status": "running",
        "protocol_sha256": digest(protocol),
        "arms": [{**a, "status": "allocated"} for a in arms],
    }
    atomic_json(root / "report.json", report)
    start = time.monotonic()
    for arm in report["arms"]:
        folder = root / arm["id"]
        folder.mkdir()
        policy = {
            "schema": 1,
            "scope": scope,
            "plugin_id": PLUGIN,
            "plugin_sha256": plugin.source_sha256,
            "training_sha256": digest(train),
            "validation_sha256": digest(validation),
            "minimum_skill": 0.0,
            "max_candidates": 1,
        }
        lc = ModelLifecycle(folder / "controller", policy)
        then = time.monotonic()
        try:
            model = plugin.fit(train, scope, recipes[arm["id"]])
            assert model["source_sha256"] == hashlib.sha256(recipes[arm["id"]]["source_code"].encode()).hexdigest()
            candidate = folder / "candidate.json"
            atomic_json(candidate, model)
            mid = lc.submit(candidate, session_id="operator-source-transfer-" + arm["id"], mode="science_agent")
            assessed = lc.assess(mid, training=train, validation=validation)
            arm.update(
                status="registered" if assessed["passed"] else "rejected",
                model_id=mid,
                assessment=assessed,
                candidate_sha256=sha256_file(candidate),
                source_sha256=model["source_sha256"],
            )
            if assessed["passed"]:
                lc.export(mid, folder / "registered")
        except Exception as exc:
            arm.update(status="execution_failed", error_type=type(exc).__name__, error=str(exc)[:3000])
        arm["wall_seconds"] = time.monotonic() - then
        atomic_json(folder / "terminal.json", arm)
        atomic_json(root / "report.json", report)
    barrier = {"all_acceptance_decisions_final": True, "time_unix": time.time(), "arms": []}
    for arm in report["arms"]:
        folder = root / arm["id"]
        candidate = folder / "candidate.json"
        barrier["arms"].append(
            {
                "id": arm["id"],
                "terminal_sha256": sha256_file(folder / "terminal.json"),
                "candidate_sha256": sha256_file(candidate) if candidate.is_file() else None,
            }
        )
    atomic_json(root / "future_barrier.json", barrier)
    for arm, sealed in zip(report["arms"], barrier["arms"], strict=True):
        folder = root / arm["id"]
        assert sha256_file(folder / "terminal.json") == sealed["terminal_sha256"]
        scored = {"status": "missing_model", "raw_metrics": None, "all_allocation_brier_skill": 0.0}
        if sealed["candidate_sha256"]:
            try:
                candidate = folder / "candidate.json"
                assert sha256_file(candidate) == sealed["candidate_sha256"]
                model = json.loads(candidate.read_text())
                rebuilt = plugin.fit(train, scope, recipes[arm["id"]])
                if digest(rebuilt) != digest(model):
                    raise ValueError("Candidate differs from independent full-training replay")
                metrics = plugin.evaluate(rebuilt, future, scope)
                scored.update(
                    status="scored", raw_metrics=metrics, all_allocation_brier_skill=metrics[plugin.acceptance_metric]
                )
            except Exception as exc:
                scored.update(status="unscorable_model", error_type=type(exc).__name__, error=str(exc)[:3000])
        arm["future_scoring"] = scored
        atomic_json(folder / "future_scoring.json", scored)
    if any(sha256_file(ROOT / n) != h for n, h in binding.items()):
        raise ValueError("Study implementation changed")
    report.update(
        status="completed",
        wall_seconds=time.monotonic() - start,
        future_barrier_sha256=sha256_file(root / "future_barrier.json"),
        external_model_calls=0,
        autonomous_target_redesign=False,
        source_intact=True,
    )
    atomic_json(root / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["donor-run", "donor-audit", "assets", "rebuilt", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    value = run(parser.parse_args())
    print(json.dumps({"status": value["status"], "arms": value["arms"]}))
