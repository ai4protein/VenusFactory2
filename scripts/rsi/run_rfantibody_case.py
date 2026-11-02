"""Run all three author RFantibody stages with explicit scale and artifact checks."""

import argparse
import json
from pathlib import Path

from agent.rsi.case_workflow import CaseStage, CaseWorkflow, audit_quiver
from agent.rsi.rfantibody_audit import export_audit
from agent.rsi.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "3e5a5bb1cb98befde9fc049ae3ff58cabb491f67"


def main(args):
    source = ROOT / "research_runs/rsi/case_assets/RFantibody" / COMMIT / "source"
    environment = ROOT / "research_runs/rsi/envs/rfantibody/bin"
    out = args.output.resolve()
    target = source / "scripts/examples/example_inputs/flu_HA.pdb"
    framework = source / "scripts/examples/example_inputs/h-NbBCII10.pdb"
    settings = {
        "case_id": "rfantibody_ha_nanobody",
        "repository": "RosettaCommons/RFantibody",
        "commit": COMMIT,
        "scale": {
            "backbones": args.backbones,
            "sequences_per_backbone": args.sequences,
            "rf2_recycles": 10,
            "diffusion_steps": 50,
        },
        "inputs": {str(x.relative_to(source)): sha256_file(x) for x in [target, framework, source / "uv.lock"]},
        "scope": "complete three-stage computational pipeline at declared scale; not full publication statistics or experimental binding reproduction",
        "agent_orchestrated": False,
        "paper_default_backbones": 1000,
    }
    if args.deterministic_sequences:
        settings["sequence_sampling_deterministic"] = True
    workflow = CaseWorkflow(out, settings)
    weights = source / "weights"
    files = [out / "1_backbones.qv", out / "2_sequences.qv", out / "3_refined.qv"]
    stages = [
        CaseStage(
            "backbone",
            (
                str(environment / "rfdiffusion"),
                "--target",
                str(target),
                "--framework",
                str(framework),
                "--output-quiver",
                str(files[0]),
                "--num-designs",
                str(args.backbones),
                "--design-loops",
                "H1:7,H2:6,H3:5-13",
                "--hotspots",
                "B146,B170,B177",
                "--diffuser-t",
                "50",
                "--deterministic",
                "--no-trajectory",
                "--weights",
                str(weights / "RFdiffusion_Ab.pt"),
            ),
            source,
            (target, framework, weights / "RFdiffusion_Ab.pt"),
            files[0],
            3600,
            lambda path: audit_quiver(path, args.backbones),
        ),
        CaseStage(
            "sequence",
            (
                str(environment / "proteinmpnn"),
                *(("--deterministic",) if args.deterministic_sequences else ()),
                "--input-quiver",
                str(files[0]),
                "--output-quiver",
                str(files[1]),
                "--seqs-per-struct",
                str(args.sequences),
                "--temperature",
                "0.2",
                "--weights",
                str(weights / "ProteinMPNN_v48_noise_0.2.pt"),
            ),
            source,
            (files[0], weights / "ProteinMPNN_v48_noise_0.2.pt"),
            files[1],
            1800,
            lambda path: audit_quiver(path, args.backbones * args.sequences),
        ),
        CaseStage(
            "refinement",
            (
                str(environment / "rf2"),
                "--input-quiver",
                str(files[1]),
                "--output-quiver",
                str(files[2]),
                "--hotspot-show-prop",
                "0.0",
                "--num-recycles",
                "10",
                "--seed",
                "20261011",
                "--weights",
                str(weights / "RF2_ab.pt"),
            ),
            source,
            (files[1], weights / "RF2_ab.pt"),
            files[2],
            7200,
            lambda path: audit_quiver(path, args.backbones * args.sequences),
        ),
    ]
    report = {"status": "running", "settings": settings, "stages": []}
    atomic_json(out / "report.json", report)
    for stage in stages:
        print(json.dumps({"stage": stage.id, "status": "starting"}), flush=True)
        try:
            result = workflow.run_stage(stage)
        except Exception:
            report["status"] = "failed"
            report["failed_stage"] = stage.id
            atomic_json(out / "report.json", report)
            raise
        report["stages"].append(result)
        atomic_json(out / "report.json", report)
        print(
            json.dumps({"stage": stage.id, "status": result["status"], "wall_seconds": result["wall_seconds"]}),
            flush=True,
        )
    audit_dir = out / "quality_audit"
    if audit_dir.exists():
        audit = json.loads((audit_dir / "audit.json").read_text())
        if audit["inputs"] != {str(p): sha256_file(p) for p in files[1:]}:
            raise ValueError("quality audit source changed")
    else:
        audit = export_audit(files[1], files[2], audit_dir, args.backbones * args.sequences)
    report["quality_audit"] = {
        k: audit[k]
        for k in ["designs", "sequence_preservation_verified", "author_minimum_passes", "additional_interface_passes"]
    }
    report["status"] = "completed"
    report["computational_chain_completed"] = True
    report["full_paper_reproduced"] = False
    atomic_json(out / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbones", type=int, default=2)
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--deterministic-sequences", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.backbones <= 1000 or not 1 <= args.sequences <= 20:
        parser.error("invalid campaign size")
    main(args)
