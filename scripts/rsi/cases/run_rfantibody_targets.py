"""Author-parameter nanobody pipelines over explicit antigen presets.

Extends the single-HA case to the supplied RSV site-3 target. Every stage uses
pinned author code/weights and records inputs. This is computational reproduction,
not a binding experiment or a new independently sampled HA replicate.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import json
from pathlib import Path

from agent.rsi.campaigns.case_workflow import CaseStage, CaseWorkflow, audit_quiver
from agent.rsi.cases.rfantibody_audit import export_audit
from agent.rsi.engine.storage import atomic_json, case_gpu_lock, sha256_file

ROOT = Path(__file__).resolve().parents[3]
from agent.rsi.cases.rfantibody_evidence import COMMIT, PRESETS


def main(args):
    preset = PRESETS[args.target]
    source = ROOT / "research_runs/rsi/case_assets/RFantibody" / COMMIT / "source"
    environment = ROOT / "research_runs/rsi/envs/rfantibody/bin"
    inputs = source / "scripts/examples/example_inputs"
    target, framework = inputs / preset["target"], inputs / "h-NbBCII10.pdb"
    out = args.output.resolve()
    out.relative_to((ROOT / "research_runs/rsi").resolve())
    if out == (ROOT / "research_runs/rsi").resolve():
        raise ValueError("dedicated run directory required")
    weights = source / "weights"
    identity = {
        "case_id": "rfantibody_" + args.target + "_nanobody",
        "repository": "RosettaCommons/RFantibody",
        "commit": COMMIT,
        "target_preset": preset,
        "scale": {
            "backbones": args.backbones,
            "sequences_per_backbone": args.sequences,
            "diffusion_steps": 50,
            "rf2_recycles": 10,
        },
        "inputs": {
            str(path.relative_to(source)): sha256_file(path)
            for path in [target, framework, source / preset["author_recipe"], source / "uv.lock"]
        },
        "sampling": {"backbone_and_mpnn": "author deterministic mode", "rf2_seed": 20261011},
        "filter_policy": "global PAE <10 and target-aligned antibody RMSD <2 A; interface PAE <10 reported separately; no binding claim",
        "scope": "complete three-stage computational pipeline at declared scale, not full publication statistics",
        "agent_orchestrated": False,
    }
    workflow = CaseWorkflow(out, identity)
    (out / "implementation.py").write_bytes(Path(__file__).read_bytes())
    files = [out / "1_backbones.qv", out / "2_sequences.qv", out / "3_refined.qv"]
    specifications = [
        (
            "backbone",
            "rfdiffusion",
            [
                "--target",
                str(target),
                "--framework",
                str(framework),
                "--output-quiver",
                str(files[0]),
                "--num-designs",
                str(args.backbones),
                "--design-loops",
                preset["loops"],
                "--hotspots",
                preset["hotspots"],
                "--diffuser-t",
                "50",
                "--deterministic",
                "--no-trajectory",
                "--weights",
                str(weights / "RFdiffusion_Ab.pt"),
            ],
            (target, framework, weights / "RFdiffusion_Ab.pt"),
            files[0],
            args.backbones,
            7200,
        ),
        (
            "sequence",
            "proteinmpnn",
            [
                "--input-quiver",
                str(files[0]),
                "--output-quiver",
                str(files[1]),
                "--seqs-per-struct",
                str(args.sequences),
                "--temperature",
                "0.2",
                "--deterministic",
                "--weights",
                str(weights / "ProteinMPNN_v48_noise_0.2.pt"),
            ],
            (files[0], weights / "ProteinMPNN_v48_noise_0.2.pt"),
            files[1],
            args.backbones * args.sequences,
            3600,
        ),
        (
            "refinement",
            "rf2",
            [
                "--input-quiver",
                str(files[1]),
                "--output-quiver",
                str(files[2]),
                "--num-recycles",
                "10",
                "--hotspot-show-prop",
                "0.0",
                "--seed",
                "20261011",
                "--weights",
                str(weights / "RF2_ab.pt"),
            ],
            (files[1], weights / "RF2_ab.pt"),
            files[2],
            args.backbones * args.sequences,
            10800,
        ),
    ]
    report = {"status": "running", "settings": identity, "stages": []}
    atomic_json(out / "report.json", report)
    with case_gpu_lock(ROOT / "research_runs/rsi/.case_gpu.lock"):
        for name, command, params, data, destination, expected, timeout in specifications:
            stage = CaseStage(
                name,
                (str(environment / command), *params),
                source,
                data,
                destination,
                timeout,
                lambda path, count=expected: audit_quiver(path, count),
            )
            print(json.dumps({"target": args.target, "stage": name, "status": "starting"}), flush=True)
            try:
                result = workflow.run_stage(stage)
            except Exception:
                report.update(status="failed", failed_stage=name)
                atomic_json(out / "report.json", report)
                raise
            report["stages"].append(result)
            atomic_json(out / "report.json", report)
            print(
                json.dumps(
                    {
                        "target": args.target,
                        "stage": name,
                        "status": result["status"],
                        "wall_seconds": result["wall_seconds"],
                    }
                ),
                flush=True,
            )
    audit_path = out / "quality_audit/audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit["inputs"] != {str(path): sha256_file(path) for path in files[1:]}:
            raise ValueError("audit inputs changed")
    else:
        audit = export_audit(files[1], files[2], out / "quality_audit", args.backbones * args.sequences)
    report.update(
        status="completed",
        computational_chain_completed=True,
        full_paper_reproduced=False,
        quality_audit={
            key: audit[key]
            for key in [
                "designs",
                "sequence_preservation_verified",
                "author_minimum_passes",
                "additional_interface_passes",
            ]
        },
    )
    atomic_json(out / "report.json", report)
    print(json.dumps(report["quality_audit"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(PRESETS), required=True)
    parser.add_argument("--backbones", type=int, default=8)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.backbones <= 1000 or not 1 <= args.sequences <= 20:
        parser.error("invalid design count")
    main(args)
