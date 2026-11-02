"""Run a source-bound, local BLASTP cohort audit without modifying input data."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi import protein_cohort_audit as audit  # noqa: E402


def run(args):
    root, output, source = args.workspace_root.resolve(), args.output.resolve(), args.plan.resolve()
    research = root / "research_runs/rsi"
    for path in (output, source):
        path.relative_to(research)
        if path == research:
            raise ValueError("Dedicated research paths required")
    if output.exists():
        raise FileExistsError("Never overwrite or resume an audit directory")
    raw = source.read_bytes()
    if audit.digest(raw.decode()) != args.plan_sha256:
        raise ValueError("Plan hash differs from the frozen input")
    records = audit.select_records(json.loads(raw), args.train_cohort, args.evaluation_cohorts)
    mapping = audit.token_map(records, args.train_cohort)
    binaries = {name: (args.blast_bin / name).resolve(strict=True) for name in ("blastp", "makeblastdb")}
    versions = {k: subprocess.check_output([str(p), "-version"], text=True, timeout=15) for k, p in binaries.items()}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "running", "stage": "prepare", "external_model_calls": 0}
    audit.save_new(output / "started.json", status)
    try:
        (output / "input_plan.json").write_bytes(raw)
        audit.save_new(output / "records.json", mapping)
        audit.write_fastas(output, mapping)
        audit.save_new(output / "cohort_summary.json", audit.cohort_summary(records, args.train_cohort))
        sources = {}
        for path in (Path(__file__), Path(audit.__file__)):
            shutil.copy2(path, output / path.name)
            sources[path.name] = audit.sha(path)
        db = [
            str(binaries["makeblastdb"]),
            "-in",
            "train.fasta",
            "-dbtype",
            "prot",
            "-parse_seqids",
            "-out",
            "train_db",
        ]
        search = audit.search_arguments(binaries["blastp"], sum(k.startswith("t") for k in mapping))
        protocol = {
            "schema": 1,
            "plan_sha256": args.plan_sha256,
            "train_cohort": args.train_cohort,
            "evaluation_cohorts": args.evaluation_cohorts,
            "sources": sources,
            "binaries": {
                k: {"path": str(p), "sha256": audit.sha(p), "version": versions[k]} for k, p in binaries.items()
            },
            "commands": [db, search],
            "search_timeout_seconds": 7200,
            "cpu_threads": 2,
            "coverage_percent_both_sequences": audit.COVERAGE_PERCENT,
            "identity_thresholds_percent": audit.THRESHOLDS,
            "normalization": "BJOUXZ to X for search only; original sequences used for exact overlap; X-X is never counted as identical",
            "hsp_policy": "All reported HSPs; no max_hsps cap; no aggregation of separate HSPs; retain every query including no-hit queries",
            "boundary": audit.BOUNDARY,
            "prior_performance_exposure": args.prior_performance_exposure,
            "allocation_policy": "Post hoc descriptive input audit; no tuning, row exclusion or confirmatory performance subgroup claims",
        }
        audit.save_new(output / "protocol.json", protocol)
        for stage, command, timeout in [("database", db, 300), ("search", search, 7200)]:
            status["stage"] = stage
            with (output / (stage + ".log")).open("x") as log:
                result = subprocess.run(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            if result.returncode:
                raise RuntimeError(f"{stage} exited {result.returncode}; log retained")
        status["stage"] = "validate_alignments"
        result = audit.summarize_alignments(output / "alignments.tsv", mapping, args.train_cohort)
        audit.save_new(output / "sequence_audit.json", result)
        # Recheck immutable source, data, and executables after computation.
        for path in (Path(__file__), Path(audit.__file__)):
            if audit.sha(path) != sources[path.name]:
                raise ValueError("Audit source changed during execution")
        if audit.sha(source) != args.plan_sha256:
            raise ValueError("Original input plan changed during execution")
        for name, path in binaries.items():
            if audit.sha(path) != protocol["binaries"][name]["sha256"]:
                raise ValueError("BLAST binary changed during execution")
        status.update(
            status="completed", stage="audited", rows=len(result["rows"]), validated_hsps=result["validated_hsps"]
        )
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - started
        status["artifact_sha256"] = {p.name: audit.sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        audit.save_new(output / "report.json", status)
    print(json.dumps({k: v for k, v in status.items() if k != "artifact_sha256"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blast-bin", type=Path, required=True)
    parser.add_argument("--train-cohort", default="train")
    parser.add_argument("--evaluation-cohorts", nargs="+", required=True)
    parser.add_argument("--prior-performance-exposure", choices=("yes", "no", "unknown"), default="unknown")
    os.umask(0o077)
    run(parser.parse_args())
