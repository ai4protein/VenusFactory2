"""GLM-authored Python models and frozen-source transfer to another protein."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from run_protein_lifecycle import ROOT, phase_run, prepare_case

from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.model_policy import read_model_configuration
from agent.rsi.protein_model import digest
from agent.rsi.protein_model import evaluate as fixed_evaluate
from agent.rsi.protein_model import fit_candidate as fixed_fit
from agent.rsi.storage import atomic_json, sha256_file

FILES = [
    "generated_tools.py",
    "generated_model.py",
    "generated_sequence.py",
    "generated_classifier.py",
    "generated_classifier_large.py",
    "generated_features.py",
    "generated_context.py",
    "generated_context_v2.py",
    "context_interfaces.py",
    "code_sandbox.py",
    "code_worker.py",
    "numeric_runtime.py",
    "protein_references.py",
    "protein_model.py",
    "experience.py",
    "library_source.py",
]
SYSTEM = "You are the VenusFactory scientific coding agent. Author executable modelling Python, not merely a built-in estimator configuration. Use MCP task inspection and training-feedback tools. Model code is a fresh fit/predict implementation chosen by you. Do not ask for human input. No shell or outside network tools are available. Work from permitted training measurements only. Keep state JSON-serializable and predictions deterministic. Independent acceptance labels are unavailable. Choose a defensible simple approach; describe the implementation without claiming algorithmic novelty."
PROMPTS = {
    "develop": "Inspect this modelling task and discover any existing validated models. If none is suitable, write the full fit/predict Python source yourself. Evaluate it with evaluate_model_code, use training-only feedback to repair or improve within the two-attempt budget, then submit one tested source hash using submit_model_code. Implement encoding, fitting and prediction in your source; infer length rather than hardcoding this protein. Inspect the tool ABI. Independent acceptance occurs after this session.",
    "reuse": "Inspect the new task, discover a compatible validated generated-code model, and use it to save every prediction. Do not train or modify the model. Report the artifact and row count.",
}


def spec(phase):
    tools = {"inspect_model_code_task", "discover_generated_models"}
    tools.update(
        {"evaluate_model_code", "submit_model_code"} if phase == "develop" else {"predict_with_generated_model"}
    )
    required = (
        {"submit_model_code"} if phase == "develop" else {"discover_generated_models", "predict_with_generated_model"}
    )
    return {
        "files": FILES,
        "mcp_entry": "generated_tools.py",
        "tools": tools,
        "required_tools": required,
        "recoverable_tools": {"evaluate_model_code"} if phase == "develop" else set(),
        "retryable_read_tools": {"inspect_model_code_task", "discover_generated_models"},
        "system_prompt": SYSTEM,
        "prompt": PROMPTS[phase],
        "timeout_s": 420,
        "max_tool_calls": 12,
    }


def policy_for(scope, train, val):
    plugin = get_model_plugin("generated_python.v1")
    return {
        "schema": 1,
        "scope": scope,
        "plugin_id": plugin.id,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(train),
        "validation_sha256": digest(val),
        "minimum_skill": 0.0,
        "max_candidates": 1,
    }


async def assess_and_reuse(folder, lc, artifact, dev_session, scope, train, val, future, config, identity):
    mid = lc.submit(artifact, session_id=dev_session, mode="science_agent")
    assessment = lc.assess(mid, training=train, validation=val)
    out = {"model_id": mid, "assessment": assessment, "status": "candidate_rejected"}
    fixed = fixed_fit(train, scope, {"encoding": "onehot", "alphas": [0.1, 1.0, 10.0, 100.0]})
    out["fixed_recipe_baseline"] = {
        "validation": fixed_evaluate(fixed, val, scope),
        "future": fixed_evaluate(fixed, future, scope),
    }
    if assessment["passed"]:
        inputs = [{"id": r["id"], "sequence": r["sequence"]} for r in future]
        run, audit, workspace = await phase_run(
            folder, "reuse", {"inputs.json": inputs}, scope, config, identity, lc, mid, spec("reuse")
        )
        out["reuse"] = {
            "status": run["status"],
            "session_id": run["session_id"],
            "audit": audit,
            "tool_calls": run["tool_calls"],
        }
        if audit["passed"] and (workspace / "predictions.json").exists():
            predictions = json.loads((workspace / "predictions.json").read_text())
            out["reuse"]["event"] = lc.record_reuse(
                mid,
                session_id=run["session_id"],
                mode="science_agent",
                inputs=inputs,
                predictions=predictions,
                scope=scope,
            )
            out["reuse"]["future_metrics"] = get_model_plugin("generated_python.v1").evaluate(
                lc._model(mid), future, scope
            )
            out["status"] = "completed"
        else:
            out["status"] = "reuse_failed"
    return out


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    _, identity = read_model_configuration(Path(config["kimi_config"]), required=config["required_model"])
    scope, train, val, future, source, split = prepare_case(args.case, args.source_cache)
    lc = ModelLifecycle(args.output / "controller", policy_for(scope, train, val))
    for name, rows in [("training", train), ("validation", val), ("future", future)]:
        atomic_json(args.output / "controller" / f"{name}.json", rows)
    report = {
        "schema": 1,
        "status": "running",
        "case": args.case,
        "model": identity,
        "source": source,
        "split": split,
        "rows": {"training": len(train), "validation": len(val), "future": len(future)},
        "policy_sha256": digest(lc.policy),
        "future_sha256": digest(future),
        "transfer_target_predeclared": args.transfer_to,
        "development_type": "agent_generated_model_code",
        "claim_scope": "retrospective code construction and validated reuse; algorithmic novelty and generalization efficacy unproven",
        "code_worker_network_isolated": True,
        "llm_runtime_network_isolated": False,
    }
    atomic_json(args.output / "report.json", report)
    implementation = args.output / "implementation"
    implementation.mkdir()
    paths = [
        Path(__file__),
        ROOT / "scripts/rsi/run_protein_lifecycle.py",
        *[ROOT / "src/agent/rsi" / name for name in FILES],
        ROOT / "src/agent/rsi/model_lifecycle.py",
        ROOT / "src/agent/rsi/model_plugins.py",
        ROOT / "src/agent/rsi/runtime.py",
        ROOT / "src/agent/rsi/sandbox.py",
    ]
    for path in paths:
        dest = implementation / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    atomic_json(implementation / "manifest.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in paths})
    print(json.dumps({"case": args.case, "phase": "code_development"}), flush=True)
    development_spec = spec("develop")
    development_spec["prompt"] = (
        "Build a numerical predictor of " + scope["target"] + " for " + scope["protein"] + ". "
        "Assay: " + scope["assay"] + "; units: " + scope["unit"] + ". "
        "The inputs are amino-acid strings, and the labels are measured activity/yield values. "
        "Your target is NOT sequence length. First call inspect_model_code_task. "
        "Then author complete Python functions fit(sequences, targets, seed) and predict(model, sequences). "
        "fit returns JSON model parameters; predict returns numeric predictions in input order. "
        "Implement a real supervised predictor with numerical fitting. No pass, TODO, empty functions or invented hashes. "
        "Call evaluate_model_code with your full source, read its returned hash and feedback, "
        "then submit that exact returned hash. You may repair the source once using training-only feedback. "
        "Never submit a hash before a successful code test."
    )
    dev, audit, workspace = await phase_run(
        args.output, "develop", {"training.json": train}, scope, config, identity, plugin_spec=development_spec
    )
    report["development"] = {
        "status": dev["status"],
        "session_id": dev["session_id"],
        "audit": audit,
        "tool_calls": dev["tool_calls"],
    }
    if not audit["passed"] or not (workspace / "candidate.json").exists():
        report["status"] = "development_failed"
        atomic_json(args.output / "report.json", report)
        print(json.dumps(report["development"]), flush=True)
        return 1
    candidate = json.loads((workspace / "candidate.json").read_text())
    report["source_sha256"] = candidate["source_sha256"]
    report["source_provenance"] = json.loads((workspace / "source_provenance.json").read_text())
    report.update(
        await assess_and_reuse(
            args.output,
            lc,
            workspace / "candidate.json",
            dev["session_id"],
            scope,
            train,
            val,
            future,
            config,
            identity,
        )
    )
    atomic_json(args.output / "report.json", report)
    print(
        json.dumps({"case": args.case, "status": report["status"], "assessment": report["assessment"]["metrics"]}),
        flush=True,
    )
    if args.transfer_to and report["status"] == "completed":
        target = args.output / "transfer"
        target.mkdir()
        scope2, train2, val2, future2, source2, split2 = prepare_case(args.transfer_to, args.source_cache)
        lc2 = ModelLifecycle(target / "controller", policy_for(scope2, train2, val2))
        for name, rows in [("training", train2), ("validation", val2), ("future", future2)]:
            atomic_json(target / "controller" / f"{name}.json", rows)
        transfer = {
            "source_case": args.case,
            "target_case": args.transfer_to,
            "source_sha256": candidate["source_sha256"],
            "target_source": source2,
            "split": split2,
            "operation": "frozen generated source refitted on target training data; no source editing or incompatible-weight reuse",
            "target_labels_returned_to_proposer": False,
            "policy_sha256": digest(lc2.policy),
            "future_sha256": digest(future2),
        }
        atomic_json(target / "report.json", transfer)
        frozen = get_model_plugin("generated_python.v1").fit(train2, scope2, candidate["recipe"])
        atomic_json(target / "candidate.json", frozen)
        transfer.update(
            await assess_and_reuse(
                target,
                lc2,
                target / "candidate.json",
                "controller_frozen_source_refit",
                scope2,
                train2,
                val2,
                future2,
                config,
                identity,
            )
        )
        atomic_json(target / "report.json", transfer)
        report["transfer"] = transfer
        atomic_json(args.output / "report.json", report)
        print(
            json.dumps(
                {
                    "transfer_to": args.transfer_to,
                    "status": transfer["status"],
                    "assessment": transfer["assessment"]["metrics"],
                }
            ),
            flush=True,
        )
    return 0 if report["status"] in {"completed", "candidate_rejected"} else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["alde_parpgb", "mcba_itopride"], required=True)
    parser.add_argument("--transfer-to", choices=["alde_parpgb", "mcba_itopride"])
    parser.add_argument("--source-cache", type=Path, default=ROOT / "research_runs/rsi/20261011_paper_case_audit")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rsi/engineering_smoke_glm.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.case == args.transfer_to:
        parser.error("transfer must use another protein")
    try:
        result = asyncio.run(main(args))
    except Exception as exc:
        path = args.output / "report.json"
        if path.exists():
            report = json.loads(path.read_text())
            report.update(status="controller_error", error_type=type(exc).__name__)
            atomic_json(path, report)
        raise
    raise SystemExit(result)
