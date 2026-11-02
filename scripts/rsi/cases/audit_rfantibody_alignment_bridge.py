"""Replay admitted original USalign functions on existing VHH structures, CPU only."""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import contextlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

from agent.rsi.cases.rfantibody_alignment import compare_alignment_rows
from agent.rsi.cases.rfantibody_evidence import validate_target_report, verify_target_artifacts
from agent.rsi.cases.rfantibody_publication import summarize_quivers
from agent.rsi.engine.storage import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]
LEGACY_COMMIT = "8d9d402a99fd63052ad538b3d670e467b800123c"


def quiver_pdb_lines(path):
    records = {}
    current = None
    with path.open() as stream:
        for line in stream:
            if line.startswith("QV_TAG "):
                current = line.split()[1]
                if current in records:
                    raise ValueError("Duplicate structure identity")
                records[current] = []
            elif line.startswith(("ATOM ", "REMARK")):
                if current is None:
                    raise ValueError("Structure before identity")
                records[current].append(line)
    return records


def run(args):
    base = ROOT / "research_runs/rsi"
    source = args.legacy_source.resolve()
    output = args.output.resolve()
    for path in [
        source,
        output,
        args.admission.resolve(),
        args.build_receipt.resolve(),
        *[p.resolve() for p in args.run],
    ]:
        path.relative_to(base)
    if output == base or output.exists():
        raise ValueError("A fresh output directory is required")
    for path in [source, *[p.resolve() for p in args.run]]:
        if output.is_relative_to(path):
            raise ValueError("Outputs must not alter an admitted source or historical run")
    admission = json.loads(args.admission.read_text())
    build = json.loads(args.build_receipt.read_text())
    if admission["commit"] != LEGACY_COMMIT or build["source_commit"] != LEGACY_COMMIT:
        raise ValueError("Unexpected public source snapshot")
    if build["status"] != "built" or build["admission_sha256"] != sha256_file(args.admission):
        raise ValueError("Unbound alignment build")
    binary = Path(build["command"][-1]).resolve()
    binary.relative_to(base)
    if sha256_file(binary) != build["binary_sha256"]:
        raise ValueError("Alignment executable changed")
    bindings = {source / name: digest for name, digest in admission["source_sha256"].items()}
    bindings.update(
        {
            binary: build["binary_sha256"],
            args.admission: sha256_file(args.admission),
            args.build_receipt: sha256_file(args.build_receipt),
        }
    )
    for path, digest in bindings.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Admitted source changed: {path}")
    sys.path.insert(0, str(source / "src"))
    pu = importlib.import_module("rfantibody.rf2.modules.pose_util")
    rmsd = importlib.import_module("rfantibody.rf2.modules.rmsd")
    for module in [pu, rmsd]:
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise ValueError("Wrong RFantibody implementation imported")
    # The sole upstream override relocates the verified compiled executable.
    rmsd.usalign = str(binary)
    output.mkdir(parents=True)
    implementation = [
        Path(__file__),
        ROOT / "src/agent/rsi/cases/rfantibody_alignment.py",
        ROOT / "src/agent/rsi/cases/rfantibody_publication.py",
        ROOT / "src/agent/rsi/cases/rfantibody_evidence.py",
        ROOT / "src/agent/rsi/cases/rfantibody_audit.py",
    ]
    for path in implementation:
        destination = output / "implementation" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        bindings[path] = sha256_file(path)
    proc = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    report = {
        "status": "running",
        "started_unix": time.time(),
        "legacy_public_commit": LEGACY_COMMIT,
        "source_code_override": "rmsd.usalign executable path only",
        "binary_sha256": sha256_file(binary),
        "original_files_verified": len(admission["source_sha256"]),
        "process": {
            "pid": os.getpid(),
            "start_ticks": proc[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        },
        "cases": [],
        "new_model_calls": 0,
        "new_gpu_inference": False,
    }
    atomic_json(output / "report.json", report)
    previous_cwd = Path.cwd()
    try:
        for run_dir in args.run:
            run_dir = run_dir.resolve()
            original = run_dir / "report.json"
            bindings[original] = sha256_file(original)
            historical = json.loads(original.read_text())
            facts = validate_target_report(historical)
            verify_target_artifacts(historical, run_dir)
            stages = historical["stages"]
            argv = stages[2]["binding"]["argv"]
            if argv.count("--hotspot-show-prop") != 1:
                raise ValueError("Explicit hotspot proportion required")
            paths = [Path(stage["binding"]["output"]) for stage in stages]
            primary = summarize_quivers(
                *paths,
                target=facts["target_antigen"],
                backbones=facts["backbones"],
                sequences_per_backbone=facts["sequences_per_backbone"],
                hotspot_proportion=float(argv[argv.index("--hotspot-show-prop") + 1]),
                source_commit=facts["source_commit"],
            )
            bindings.update({Path(name): digest for name, digest in primary["inputs"].items()})
            case_dir = output / facts["target_antigen"]
            case_dir.mkdir()
            work = case_dir / "alignment_work"
            work.mkdir()
            original_lines, refined_lines = [quiver_pdb_lines(path) for path in paths[1:]]
            legacy = {}
            os.chdir(work)
            with (case_dir / "upstream_alignment.log").open("w") as log:
                for index, row in enumerate(primary["rows"]):
                    tag = row["id"]
                    with contextlib.redirect_stdout(log):
                        print(tag, flush=True)
                        predicted = pu.pose_from_pdblines(refined_lines[tag])
                        reference = pu.pose_from_pdblines(original_lines[tag.removesuffix("_best")])
                        rmsd.tmalign_to_subset(predicted, reference, subset="target")
                        legacy[tag] = rmsd.calc_prealigned_rmsd(predicted, reference, predicted.antibody_mask)
                    if (index + 1) % 16 == 0:
                        atomic_json(case_dir / "partial_rmsds.json", legacy)
                        report["inflight"] = {
                            "target": facts["target_antigen"],
                            "completed_structures": index + 1,
                            "allocated_structures": len(primary["rows"]),
                        }
                        atomic_json(output / "report.json", report)
                        print(json.dumps(report["inflight"]), flush=True)
            os.chdir(previous_cwd)
            result = compare_alignment_rows(primary["rows"], legacy)
            result.update(
                target=facts["target_antigen"],
                source_run=str(run_dir),
                source_report_sha256=bindings[original],
                inputs=primary["inputs"],
                legacy_public_commit=LEGACY_COMMIT,
                current_source_commit=facts["source_commit"],
            )
            atomic_json(case_dir / "audit.json", result)
            report["cases"].append({k: v for k, v in result.items() if k not in ["rows", "backbone_records", "inputs"]})
            report["inflight"] = None
            atomic_json(output / "report.json", report)
        if any(sha256_file(path) != digest for path, digest in bindings.items()):
            raise ValueError("A bound source changed during replay")
        report.update(
            status="completed",
            finished_unix=time.time(),
            sources_unchanged=True,
            source_bindings={str(p): h for p, h in bindings.items()},
        )
    except Exception as error:
        report.update(status="failed", finished_unix=time.time(), failure_type=type(error).__name__, failure=str(error))
        atomic_json(output / "report.json", report)
        raise
    finally:
        os.chdir(previous_cwd)
    atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != "source_bindings"}, indent=2))
