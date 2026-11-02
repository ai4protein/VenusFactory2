"""Independently reanalyse an existing VHH design run against publication criteria."""

import argparse
import json
from pathlib import Path

from agent.rsi.rfantibody_evidence import validate_target_report, verify_target_artifacts
from agent.rsi.rfantibody_publication import summarize_quivers
from agent.rsi.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]


def run(source, output):
    source = source.resolve()
    output = output.resolve()
    source.relative_to(ROOT / "research_runs/rsi")
    output.relative_to(ROOT / "research_runs/rsi")
    if output.is_relative_to(source) or output == ROOT / "research_runs/rsi":
        raise ValueError("Audit output must be a new directory outside the historical run")
    original = source / "report.json"
    original_sha = sha256_file(original)
    report = json.loads(original.read_text())
    facts = validate_target_report(report)
    binding = verify_target_artifacts(report, source)
    stages = report["stages"]
    argv = stages[2]["binding"]["argv"]
    if argv.count("--hotspot-show-prop") != 1:
        raise ValueError("Explicit RF2 hotspot configuration required")
    proportion = float(argv[argv.index("--hotspot-show-prop") + 1])
    result = summarize_quivers(
        *[s["binding"]["output"] for s in stages],
        target=facts["target_antigen"],
        backbones=facts["backbones"],
        sequences_per_backbone=facts["sequences_per_backbone"],
        hotspot_proportion=proportion,
        source_commit=facts["source_commit"],
    )
    if sha256_file(original) != original_sha:
        raise ValueError("Historical report changed")
    result.update(
        source_run=str(source),
        source_report_sha256=original_sha,
        legacy_artifact_verification=binding,
        historical_artifacts_unchanged=True,
        new_model_calls=0,
        new_gpu_inference=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    implementation = [
        Path(__file__),
        ROOT / "src/agent/rsi/rfantibody_publication.py",
        ROOT / "src/agent/rsi/rfantibody_evidence.py",
        ROOT / "src/agent/rsi/rfantibody_audit.py",
    ]
    (output / "implementation").mkdir()
    for p in implementation:
        (output / "implementation" / p.name).write_bytes(p.read_bytes())
    result["implementation"] = {p.name: sha256_file(p) for p in implementation}
    atomic_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.run, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ["rows", "backbone_records", "inputs"]}, indent=2))
