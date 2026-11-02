"""Predeclared transfer of one agent-generated source to GB1 and TrpB."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from run_generated_lifecycle import ROOT, assess_and_reuse, policy_for

from agent.rsi.landscape_cases import prepare_landscape
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.model_policy import read_model_configuration
from agent.rsi.protein_model import digest
from agent.rsi.storage import atomic_json, sha256_file


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    donor = json.loads(args.donor.read_text())
    if donor.get("development_type") != "agent_generated_model_code":
        raise ValueError("donor is not generated code")
    config = json.loads(args.config.read_text())
    _, identity = read_model_configuration(Path(config["kimi_config"]), required=config["required_model"])
    protocol = {
        "source_sha256": donor["source_sha256"],
        "donor_artifact_sha256": sha256_file(args.donor),
        "cases": ["GB1", "TrpB"],
        "model": identity,
        "source_update_allowed": False,
        "target_feedback_to_proposer": False,
        "interpretation": "Frozen implementation refitted on each protein training partition; no shared fitted weights and no proof of improved RSI strategy.",
    }
    atomic_json(args.output / "protocol.json", protocol)
    (args.output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "generated_model.py").write_text(donor["recipe"]["source_code"])
    reports = []
    for protein in protocol["cases"]:
        folder = args.output / protein
        folder.mkdir()
        scope, train, val, future, source, split = prepare_landscape(protein, args.source)
        lc = ModelLifecycle(folder / "controller", policy_for(scope, train, val))
        for name, rows in [("training", train), ("validation", val), ("future", future)]:
            atomic_json(folder / "controller" / (name + ".json"), rows)
        report = {
            "case": protein,
            "status": "running",
            "source": source,
            "split": split,
            "protocol_sha256": digest(protocol),
        }
        atomic_json(folder / "report.json", report)
        candidate = get_model_plugin("generated_python.v1").fit(train, scope, donor["recipe"])
        atomic_json(folder / "candidate.json", candidate)
        report.update(
            await assess_and_reuse(
                folder,
                lc,
                folder / "candidate.json",
                "controller_frozen_source_refit",
                scope,
                train,
                val,
                future,
                config,
                identity,
            )
        )
        atomic_json(folder / "report.json", report)
        reports.append(report)
        print(
            json.dumps(
                {
                    "case": protein,
                    "status": report["status"],
                    "assessment": report["assessment"]["metrics"],
                    "reuse": report.get("reuse", {}).get("future_metrics"),
                }
            ),
            flush=True,
        )
    atomic_json(args.output / "report.json", {"protocol": protocol, "results": reports})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rsi/generated_models_glm51.json")
    p.add_argument("--source", type=Path, default=ROOT / "research_runs/rsi/case_assets/alde_landscapes")
    args = p.parse_args()
    try:
        asyncio.run(main(args))
    except Exception as exc:
        if args.output.exists():
            atomic_json(
                args.output / "failure.json",
                {"status": "controller_error", "error_type": type(exc).__name__, "message": str(exc)[:300]},
            )
        raise
