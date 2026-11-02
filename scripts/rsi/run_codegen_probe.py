"""Bounded code-delivery feasibility checks, separate from scientific efficacy.

Use synthetic inputs only. A passing run proves that the configured mode can
submit executable fit/predict code under its declared budget; it establishes
neither autonomous model improvement nor generalization on a protein task.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from run_harness_probe import credential_from_toml, default_runtimes

from agent.harness import Budget, ModelBinding, Profile, Task, builtin_registry
from agent.harness.contracts import digest, file_hash
from agent.harness.host import execute, write_json
from agent.rsi.code_sandbox import guard_source, run_code

PROMPT = (
    "This is a synthetic code-delivery compatibility check, not a biological discovery task. "
    "Inspect the model task and discover available models and experience, then write and evaluate a "
    "complete lightweight binary classifier through evaluate_model_code and submit its tested source "
    "hash through submit_model_code. Use amino-acid composition or position features. "
    "Provide executable fit(sequences, targets, seed) and predict(model, sequences). "
    "fit must return JSON-serializable numeric state; predict must return one [p0,p1] vector per sequence. "
    "The host will independently fit the source twice with the same seed and check deterministic state, "
    "finite normalized nonconstant predictions, and execution in a fresh isolated process. "
    "Keep the implementation concise; finish the actual code submission within the declared budget. "
    "All labels are a synthetic function of residue counts, not experimental biology."
)
TOOLS = frozenset(
    {
        "inspect_model_code_task",
        "discover_generated_models",
        "read_model_experience",
        "evaluate_model_code",
        "submit_model_code",
    }
)


def synthetic_case():
    rows = [
        {"id": f"synthetic_{i:03d}", "sequence": "".join(seq), "target": int(seq.count("C") >= 3)}
        for i, seq in enumerate(itertools.product("AC", repeat=6))
    ]
    scope = {
        "protein": "synthetic code-delivery fixture",
        "assay": "deterministic residue-count labels; not measured",
        "target": "at_least_three_cysteines",
        "unit": "class probability",
        "sequence_length": 6,
        "class_labels": ["below_three", "at_least_three"],
    }
    return rows, scope


def prepare_task(folder):
    rows, scope = synthetic_case()
    work, capability = folder / "workspace", folder / "capability"
    work.mkdir(parents=True)
    capability.mkdir()
    source_root = Path(__file__).resolve().parents[2] / "src/agent/rsi"
    for source in sorted(source_root.glob("*.py")):
        shutil.copy2(source, capability / source.name)
    write_json(work / "training.json", rows)
    write_json(
        capability / "task.json",
        {
            "phase": "develop",
            "plugin_id": "generated_classifier.v1",
            "scope": scope,
            "inputs": {"training.json": digest(rows)},
        },
    )
    return Task(
        work,
        capability,
        folder / "execution",
        PROMPT,
        "Deliver executable code through the registered tools; host audit, not self-report, determines success.",
        TOOLS,
        phase="develop",
    )


def validate_source(source):
    guard = guard_source(source)
    rows, _ = synthetic_case()
    sequences = [r["sequence"] for r in rows]
    targets = [r["target"] for r in rows]
    request = {"operation": "fit", "sequences": sequences, "targets": targets, "seed": 17}
    first, second = run_code(source, request)["model"], run_code(source, request)["model"]
    if digest(first) != digest(second):
        raise ValueError("Fit differs under the same seed")
    queries = ["AAAAAA", "CCCCCC", "ACACAC", "CACACA"]
    prediction = run_code(source, {"operation": "predict", "model": first, "sequences": queries, "seed": 17})[
        "predictions"
    ]
    values = np.asarray(prediction, dtype=float)
    if (
        values.shape != (len(queries), 2)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-8)
        or np.ptp(values[:, 1]) <= 1e-8
    ):
        raise ValueError("Invalid or constant class-probability predictions")
    return {
        **guard,
        "same_seed_state_equal": True,
        "fresh_process_predictions_valid": True,
        "prediction_rows": len(queries),
        "predictions_sha256": digest(prediction),
        "scientific_generalization_test": False,
    }


async def main(args):
    model = ModelBinding(**json.loads(args.model.read_text()))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtimes = default_runtimes()
    if args.runtimes:
        runtimes.update(json.loads(args.runtimes.read_text()))
    budget = Budget(
        wall_seconds=600,
        request_seconds=300,
        model_calls=24,
        tool_calls=12,
        output_tokens=12288,
        thinking=args.thinking,
    )
    base = Profile("science_agent", "langgraph", model, budget)
    profiles = [
        replace(base, mode=mode)
        for repeat in range(args.repeats)
        for mode in (["science_agent", "science_expert"] if repeat % 2 == 0 else ["science_expert", "science_agent"])
    ]
    protocol = {
        "kind": "synthetic_model_code_delivery_preflight",
        "scientific_efficacy_comparison": False,
        "profiles": [p.public() for p in profiles],
        "prompt_sha256": digest(PROMPT),
        "tool_interface": "generated_classifier.v1 development episode",
        "synthetic_data_sha256": digest(synthetic_case()),
        "entry_sha256": file_hash(Path(__file__)),
        "synthetic_inputs_only": True,
        "all_allocations_predeclared": True,
        "model_call_limit": sum(p.budget.model_calls for p in profiles),
    }
    write_json(output / "protocol.json", protocol)
    registry = builtin_registry()
    preflight = [registry.preflight(p, runtimes) for p in profiles]
    report = {
        "status": "prepared",
        "protocol_sha256": digest(protocol),
        "preflight": preflight,
        "runs": [],
    }
    write_json(output / "report.json", report)
    if not args.execute:
        return report
    if not all(p["ready"] for p in preflight):
        raise ValueError("Runtime preflight failed; no external calls made")
    key = credential_from_toml(args.credential_toml, model)
    report["status"] = "running"
    write_json(output / "report.json", report)
    for index, profile in enumerate(profiles):
        folder = output / f"arm_{index:03d}_{profile.mode}"
        task = prepare_task(folder)
        row = {"index": index, "mode": profile.mode, "thinking": profile.budget.thinking, "passed": False}
        try:
            result = await execute(profile, task, runtimes, credential=key)
            row.update(
                status=result["status"],
                audit_passed=result["audit"]["passed"],
                physical_model_calls=result["audit"]["requests"]["model_calls"],
                tool_calls=result["audit"]["events"]["tool_calls"],
                wall_seconds=result.get("wall_seconds"),
            )
            candidate = task.workspace / "candidate.json"
            if candidate.is_file() and not candidate.is_symlink():
                source = json.loads(candidate.read_text())["recipe"]["source_code"]
                row["source_check"] = await asyncio.to_thread(validate_source, source)
            from run_protein_lifecycle import audit_events

            row["scientific_tool_chain"] = audit_events(
                task.output,
                "develop",
                result,
                {"inspect_model_code_task", "evaluate_model_code", "submit_model_code"},
            )
            row["passed"] = (
                row["status"] == "completed"
                and row["audit_passed"]
                and "source_check" in row
                and row["scientific_tool_chain"]["passed"]
            )
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:500])
        report["runs"].append(row)
        write_json(folder / "assessment.json", row)
        write_json(output / "report.json", report)
        print(json.dumps(row), flush=True)
    report.update(status="completed", all_passed=all(r["passed"] for r in report["runs"]))
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--credential-toml", type=Path, required=True)
    parser.add_argument("--runtimes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], required=True)
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--execute", action="store_true")
    asyncio.run(main(parser.parse_args()))
