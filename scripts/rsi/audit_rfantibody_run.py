"""Read-only integrity and structural-filter replay for a completed VHH case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.rsi.case_workflow import audit_quiver
from agent.rsi.protein_model import digest
from agent.rsi.rfantibody_audit import export_audit
from agent.rsi.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def main(args):
    root = args.run.resolve()
    report, workflow = read(root / "report.json"), read(root / "workflow.json")
    if report["status"] != "completed" or report["settings"] != workflow:
        raise ValueError("Run is incomplete or workflow differs")
    args.output.mkdir(parents=True, exist_ok=False)
    counts = [
        workflow["scale"]["backbones"],
        workflow["scale"]["backbones"] * workflow["scale"]["sequences_per_backbone"],
    ]
    checked = []
    for index, name in enumerate(["backbone", "sequence", "refinement"]):
        stage = read(root / (name + ".json"))
        if stage != report["stages"][index] or stage["status"] != "completed":
            raise ValueError("Reported stage differs from original receipt")
        binding = stage["binding"]
        if digest(binding) != stage["recipe_sha256"]:
            raise ValueError("Stage recipe changed")
        for value, sha in binding["input_hashes"].items():
            if sha256_file(Path(value)) != sha:
                raise ValueError("Stage input changed")
        output = Path(binding["output"])
        output.resolve().relative_to(root)
        if sha256_file(output) != stage["output_sha256"]:
            raise ValueError("Stage output changed")
        actual = audit_quiver(output, counts[0 if index == 0 else 1])
        if actual != stage["artifact_audit"]:
            raise ValueError("Structural completeness audit changed")
        checked.append(
            {
                "stage": name,
                "recipe_sha256": stage["recipe_sha256"],
                "output_sha256": stage["output_sha256"],
                "designs": actual["designs"],
                "wall_seconds": stage["wall_seconds"],
            }
        )
    source = Path(report["stages"][0]["binding"]["cwd"])
    for relative, sha in workflow["inputs"].items():
        if sha256_file(source / relative) != sha:
            raise ValueError("Workflow source asset changed")
    quality = export_audit(root / "2_sequences.qv", root / "3_refined.qv", args.output / "quality_replay", counts[1])
    if quality != read(root / "quality_audit/audit.json"):
        raise ValueError("Independent structural/sequence replay differs")
    for name, expected in report["quality_audit"].items():
        if quality[name] != expected:
            raise ValueError("Reported filter count differs")
    continuation = read(root / "continuation.json") if (root / "continuation.json").exists() else None
    result = {
        "status": "audit_completed",
        "run_report_sha256": sha256_file(root / "report.json"),
        "auditor_sha256": sha256_file(Path(__file__)),
        "stages": checked,
        "quality": report["quality_audit"],
        "continuation": continuation,
        "sequence_preserving_replay": True,
        "new_generation": False,
        "agent_orchestrated": workflow["agent_orchestrated"],
        "experimental_binding_claim": False,
        "scope": "Frozen input/output and structural-filter replay; no new GPU replicate or direct experimental binding measurement",
        "historical_runtime_limit": "Stage receipts bind input weights and workflow assets; audit does not reconstruct every historical dependency or GPU RNG state",
    }
    atomic_json(args.output / "report.json", result)
    print(json.dumps({"status": result["status"], "quality": result["quality"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
