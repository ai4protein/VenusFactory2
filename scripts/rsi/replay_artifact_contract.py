#!/usr/bin/env python3
"""Offline projection of old synthetic tool results: no LLM calls or live RSI gain."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from agent.rsi.artifact_projection import project, successful_results
from agent.rsi.diagnostic_audit import audit
from agent.rsi.storage import atomic_json, sha256_file


def public_contract(kind, calls):
    # No expected answers are passed into this public contract builder.
    def find(name, **args):
        found = [
            c
            for c in calls
            if c["tool_name"].split("__")[-1] == name and all(c.get("args", {}).get(k) == v for k, v in args.items())
        ]
        if not found:
            raise ValueError("required successful tool call absent")
        return found[-1]["tool_call_id"]

    def ref(cid, *path):
        return {"op": "ref", "call_id": cid, "path": list(path)}

    if kind == "sequence":
        cid = find("sequence_properties")
        return {
            "op": "object",
            "fields": {
                k: ref(cid, "properties", k)
                for k in ["sequence_id", "sequence_length", "aromaticity", "molecular_weight"]
            },
        }
    if kind in ("invalid", "temperature", "ids"):
        return ref(
            find(
                {
                    "invalid": "sequence_properties",
                    "temperature": "convert_temperature",
                    "ids": "audit_prediction_table",
                }[kind]
            )
        )
    if kind == "mutation":
        return {
            "op": "object",
            "fields": {
                "valid_mutation": ref(find("check_mutation", mutation="C2A")),
                "invalid_mutation": ref(find("check_mutation", mutation="A2V")),
            },
        }
    if kind != "reuse":
        raise ValueError("unsupported contract")
    prediction = [
        c
        for c in calls
        if c["tool_name"].endswith("__predict_linear_fixture")
        and c.get("args", {}).get("task", "synthetic_linear_fixture") == "synthetic_linear_fixture"
    ][-1]["tool_call_id"]
    rejected = find("predict_linear_fixture", task="unrelated_protein_property")
    return {
        "op": "object",
        "fields": {
            "predictions": ref(prediction, "predictions"),
            "registered": ref(find("validate_linear_fixture"), "registered"),
            "incompatible_reuse_rejected": {
                "op": "all",
                "values": [
                    {"op": "equals", "value": ref(rejected, "valid"), "expected_literal": False},
                    {
                        "op": "equals",
                        "value": ref(rejected, "error"),
                        "expected_literal": "incompatible_or_unvalidated_model",
                    },
                ],
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to((REPO / "research_runs/rsi").resolve())
    output.mkdir(parents=True, exist_ok=False)
    source = REPO / "research_runs/rsi/20261010_glm_autonomous_pilot30_v1"
    results = []
    for task in json.loads((source / "tasks_private.json").read_text()):
        original = source / "runs" / task["task_id"]
        path = original / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        payloads = successful_results(events)
        calls = [e for e in events if e.get("kind") == "tool_call_start" and e.get("tool_call_id") in payloads]
        contract = public_contract(task["kind"], calls)
        value, used = project(contract, payloads)
        derived = output / task["task_id"]
        work = derived / "workspace"
        work.mkdir(parents=True)
        atomic_json(work / "output.json", value)
        if (original / "workspace/registry").exists():
            shutil.copytree(original / "workspace/registry", work / "registry")
        atomic_json(derived / "contract.json", contract)
        atomic_json(
            derived / "receipt.json",
            {
                "source_events_sha256": sha256_file(path),
                "tool_call_ids": used,
                "artifact_sha256": sha256_file(work / "output.json"),
                "llm_calls": 0,
                "origin": "offline human-authored public contract",
            },
        )
        run = json.loads((original / "run.json").read_text())
        score = audit(task, work, run, path)
        score.update(
            task_id=task["task_id"],
            projected_evidence_and_artifact_valid=score["tool_chain_correct"] and score["artifact_values_correct"],
            original_runtime_status=run["status"],
        )
        atomic_json(derived / "audit.json", score)
        results.append(score)
    summary = {
        "tasks": len(results),
        "projected_evidence_and_artifact_valid": sum(x["projected_evidence_and_artifact_valid"] for x in results),
        "original_runtime_completed": sum(x["runtime_completed"] for x in results),
        "llm_calls": 0,
        "contracts": "human-authored from public task requests",
        "claim_scope": "Post-hoc reconstructability only; not live GLM success, service recovery or RSI gain.",
        "results": results,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
