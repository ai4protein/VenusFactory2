"""Measure fixed DeepLoc controls through the native isolated Scientist API.

This operator-run reference study does not call an LLM or evaluate final labels.
It is not an autonomous RSI experiment and never publishes a model.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_root), str(_root / "src")]

from agent.rsi.datasets.deeploc_protocol import SCHEMA
from agent.rsi.datasets.deeploc_source import regular_file, require, sha256


def write_report(output, report):
    path = output / "results.json"
    temporary = output / "results.pending"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


async def measure(protocol_path, expected_sha256, output):
    protocol_path = Path(protocol_path).absolute()
    raw = regular_file(protocol_path, 1024**2)
    require(sha256(raw) == expected_sha256, "Protocol checksum changed")
    protocol = json.loads(raw)
    require(protocol.get("schema") == SCHEMA, "Use a frozen native DeepLoc protocol")
    methods = protocol.get("reference_methods")
    require(methods == ["training_prevalence", "aac_logistic", "terminal_logistic"], "Reference protocol changed")
    assets = protocol["artifacts"]
    dataset = None
    for name in ("dataset.csv", "membership.jsonl"):
        content = regular_file(protocol_path.parent / name, 20 * 1024**2)
        require(
            len(content) == assets[name]["bytes"] and sha256(content) == assets[name]["sha256"],
            "Protocol input checksum or size changed",
        )
        if name == "dataset.csv":
            dataset = content
    output = Path(output).absolute()
    require(
        not output.exists() and not any(p.is_symlink() for p in [output, *output.parents]),
        "Use a new run directory without symlinks",
    )
    output.mkdir(parents=True, mode=0o700)
    os.environ.update(
        PYTHON_DOTENV_DISABLED="1",
        WEBUI_V2_MODE="local",
        VENUS_SESSION_SANDBOX="bwrap",
        VENUS_JOB_QUEUE_ROOT=str(output / "queue"),
        VENUS_WORKER_AUTOSTART="0",
        VENUS_MODEL_CONFIG=str(_root / "model-config.yaml.example"),
    )
    from agent.harness.research_data_store import ResearchDataStore
    from agent.harness.research_workspace import ResearchWorkspace

    work = output / "workspace"
    store = ResearchDataStore(output / "private_data", work)
    service = ResearchWorkspace(
        work,
        output / "control",
        {"steps": [], "deliverable": "model"},
        development_root=output / "development",
        data_store=store,
    )
    store.ingest(protocol["native_prepare"]["data_file"], dataset, assets["dataset.csv"]["sha256"])
    prepared = await service.prepare_research_dataset(**protocol["native_prepare"])
    (output / "prepared.json").write_text(json.dumps(prepared, indent=2) + "\n")
    report = {
        "schema": "deeploc_fixed_references.v1",
        "status": "running",
        "started": time.time(),
        "kind": "operator_fixed_reference_study",
        "protocol_sha256": expected_sha256,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_root, text=True).strip(),
        "source_sha256": protocol["source_sha256"],
        "cohorts": protocol["cohorts"],
        "references": [],
        "external_model_requests": 0,
        "autonomous_models": 0,
        "registered_models": 0,
        "test_and_hpa_scored": False,
        "model_selection_scope": "validation only",
    }
    write_report(output, report)
    for method in methods:
        print("Starting " + method, flush=True)
        start = time.monotonic()
        item = await service.get_research_baseline(method, output_file="src/" + method + ".py", task="multilabel")
        row = {"reference": method, "source_sha256": item["source_sha256"], "upstream_sha256": item["upstream_sha256"]}
        try:
            result = await service.train_research_model(
                method,
                item["source_file"],
                prepared["evaluation_file"],
                "Predeclared fixed sequence-only localization control; matched partitions, no held-out fitting.",
                variant="baseline",
                seed=0,
            )
            experiment, training = result["experiment"], result["training"]
            require(
                not experiment["test_used_for_feedback"] and not training["validation_and_test_mounted"],
                "Held-out isolation invariant failed",
            )
            row.update(
                status="completed",
                experiment_id=experiment["id"],
                training_id=training["id"],
                validation=experiment["validation"],
                fresh_process_replay=experiment["fresh_process_replay"],
                training_rows=training["training_rows"],
                dependency_versions=training["dependency_versions"],
            )
        except Exception as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1600])
        row["elapsed_seconds"] = time.monotonic() - start
        report["references"].append(row)
        write_report(output, report)
        print(json.dumps({key: row[key] for key in ("reference", "status", "elapsed_seconds")}), flush=True)
    report.update(
        status="completed" if all(row["status"] == "completed" for row in report["references"]) else "failed",
        finished=time.time(),
    )
    write_report(output, report)
    lines = [
        "# DeepLoc fixed reference controls",
        "",
        "One joint ten-output localization task; operator-run reference controls, not autonomous RSI results.",
        "",
        "Training: original folds 0/1/2 in full. Validation: 512 fixed hash-selected proteins from fold 3.",
        "Internal test and HPA were not scored. No models were registered.",
        "",
        "| Control | Status | Validation macro Brier | Seconds |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in report["references"]:
        score = row.get("validation", {}).get("score")
        lines.append(
            "| "
            + row["reference"]
            + " | "
            + row["status"]
            + " | "
            + (f"{score:.6f}" if score is not None else "unavailable")
            + f" | {row['elapsed_seconds']:.1f} |"
        )
    lines += [
        "",
        "Per-label counts, probability metrics, hashes and dependency versions are in results.json.",
        "These bounded validation measurements are not published five-fold DeepLoc benchmark scores.",
        "Full-fold/external assessment and independent scientific acceptance remain separate steps.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = asyncio.run(measure(args.protocol, args.sha256, args.output))
    print(json.dumps({"status": report["status"], "references": len(report["references"])}))
    raise SystemExit(0 if report["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
