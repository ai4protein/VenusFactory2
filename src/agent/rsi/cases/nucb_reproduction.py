"""Pinned NucB raw-count reproduction and native artifact verification.

Registration covers this computational workflow, not new model invention.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

from agent.rsi.engine.storage import sha256_file

COMMIT = "702284847c4558567d8288169a86dc147cfaab32"
SOURCE_MANIFEST_SHA256 = "236a5426bcd54f6e8ddfa9596df7fa9230dac61331f24e8ece92afd83257537d"
ASSET_RECEIPT = "research_runs/rsi/20261011_nucb_raw_assets_v1/report.json"
ASSET_RECEIPT_SHA256 = "fa4235c64c00803b2ec570a5ba1e96448ab421ea4ffb4b073ca6a0aad19b0227"
ROWS = {"g4": 15404, "g3": 18618, "g2": 14678, "g1": 9441}
FIDUCIALS = {
    "g1": ("neg_control", "wt"),
    "g2": ("neg_control", "wt"),
    "g3": ("neg_control", "wt", "a73r"),
    "g4": ("neg_control", "wt", "a73r", "a73r_d74s", "a63p_a73r_d74h_i84y"),
}
LANDSCAPE = "processed_data/landscape.csv"
SCRIPTS = (
    "run_nucb_case.py",
    "run_nucb_author_reproduction.py",
    "run_nucb_analysis_reproduction.py",
    "nucb_reproduction_checks.py",
    "prepare_nucb_assets.py",
)
ANALYSIS_FILES = (
    "hit_rates.csv",
    "author_hit_rate_stdout.txt",
    "purified_initial_velocity.csv",
    "purified_time_series.csv",
    "purified_slope_ratios.csv",
    *(f"hit_rate_precision_{f}.csv" for f in ("neg_control", "wt", "A73R", "a73r_d74s", "a63p_a73r_d74h_i84y")),
)
LIMITS = [
    "Released sequencing counts and plate measurements; no new biological experiments",
    "Original CNN/VAE training and sequence design are outside this scope",
    "Diversity analysis, including the known G1 reference discrepancy, is outside this scope",
    "Unseeded WT-denominator visualization is not reproduced; fitted-slope ratios are reported",
    "Workflow registration and repeat execution do not establish model novelty or causal RSI benefit",
]


def read(path):
    return json.loads(Path(path).read_text())


def safe_file(base, name):
    """Only ordinary files below an explicit evidence root, including all parents."""
    base = Path(base).absolute()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Invalid NucB evidence path")
    target = base / relative
    for path in (base, *base.parents, target, *target.parents):
        if path.is_symlink():
            raise ValueError("Symlinked NucB evidence is not admitted")
    target.resolve().relative_to(base.resolve())
    if not target.is_file():
        raise ValueError("Missing NucB evidence file")
    return target


def checked(base, name, sha):
    path = safe_file(base, name)
    if sha256_file(path) != sha:
        raise ValueError(f"NucB evidence changed: {name}")
    return path


def assets_at(root):
    # Frozen runs intentionally share the admitted, hash-pinned public cache.
    return (Path(root) / "research_runs/rsi/case_assets/nuclease_design" / COMMIT).resolve()


def admit(root):
    """Hash pinned author source and every public input, before any computation."""
    assets = assets_at(root)
    manifest = read(checked(assets, "download_manifest.json", SOURCE_MANIFEST_SHA256))
    receipt = (Path(root) / ASSET_RECEIPT).resolve()
    cache = read(checked(receipt.parent, receipt.name, ASSET_RECEIPT_SHA256))
    if manifest["commit"] != COMMIT or cache["status"] != "verified":
        raise ValueError("NucB source admission failed")
    for row in manifest["files"]:
        checked(assets / "source", row["path"], row["sha256"])
    hashes = {}
    for row in cache["files"]:
        p = checked(assets / "gcs", row["path"], row["sha256"])
        if p.stat().st_size != row["bytes"] or row["path"] in hashes:
            raise ValueError("NucB public input size or identity changed")
        hashes[row["path"]] = row["sha256"]
    return manifest, hashes


def implementation(root):
    return {name: sha256_file(Path(root) / "scripts/rsi" / name) for name in SCRIPTS} | {
        "nucb_reproduction.py": sha256_file(Path(root) / "src/agent/rsi/cases/nucb_reproduction.py")
    }


def tables(generation):
    return [
        f"processed_data/{generation}.csv",
        *(f"processed_fiducial_data/{generation}_{f}.csv" for f in FIDUCIALS[generation]),
    ]


def clean_environment(root):
    python = Path(root) / "research_runs/rsi/envs/nuclease/bin/python"
    return {
        "PATH": str(python.parent) + ":/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(Path(root) / "src"),
        "CUDA_VISIBLE_DEVICES": "",
        **dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"),
    }


def _compare_tables(root, actual, reference):
    # Only the already fingerprinted repository comparator is imported.
    spec = importlib.util.spec_from_file_location(
        "nucb_table_audit", Path(root) / "scripts/rsi/cases/nucb_reproduction_checks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_tables(actual, reference)


def _replay_analysis(root, output):
    """Recompute every analysis output from this run's reconstructed tables.

    The helper also independently reconstructs all 28 fits using NumPy OLS,
    distinct from the author's scipy.linregress implementation.
    """
    with tempfile.TemporaryDirectory(prefix="venus-nucb-audit-") as folder:
        fresh = Path(folder) / "analysis"
        argv = [
            str(Path(root) / "research_runs/rsi/envs/nuclease/bin/python"),
            str(Path(root) / "scripts/rsi/cases/run_nucb_analysis_reproduction.py"),
            "--assets",
            str(assets_at(root)),
            "--asset-report",
            str(Path(root) / ASSET_RECEIPT),
            "--data-root",
            str(output / "raw/data"),
            "--output",
            str(fresh),
        ]
        result = subprocess.run(argv, env=clean_environment(root), capture_output=True, timeout=180)
        if result.returncode:
            raise ValueError("NucB independent analysis replay failed")
        replay = read(fresh / "report.json")
        if replay["status"] != "replayed" or replay["source_unchanged"] is not True:
            raise ValueError("NucB independent analysis incomplete")
        expected_order = read(safe_file(output / "analysis", "protocol.json")).get("bootstrap_row_order")
        replay_order = read(fresh / "protocol.json").get("bootstrap_row_order")
        if not expected_order or expected_order != replay_order:
            raise ValueError("NucB bootstrap sampling frame differs from independent replay")
        for name in ANALYSIS_FILES:
            if sha256_file(safe_file(output / "analysis", name)) != sha256_file(fresh / name):
                raise ValueError(f"NucB analysis differs from numerical replay: {name}")
        return replay


def canonical_nucb(report):
    if (
        report.get("status") != "completed"
        or report.get("source_commit") != COMMIT
        or report.get("execution_origin") != "new_raw_count_and_analysis_computation"
        or report.get("limits") != LIMITS
    ):
        raise ValueError("Invalid NucB reproduction identity or scope")
    expected = {
        "generation_rows": ROWS,
        "processed_and_fiducial_tables": 16,
        "unique_landscape_sequences": 55760,
        "hit_rate_rows": 44,
        "purified_timepoints": 8428,
        "initial_rate_fits": 28,
        "reference_tables_passed": 17,
        "independent_analysis_replay_passed": True,
    }
    if report.get("evidence") != expected:
        raise ValueError("NucB computational evidence incomplete")
    return {
        "computational_scope": "nucb_raw_counts_to_landscape_and_library_activity_analysis",
        "execution_origin": report["execution_origin"],
        "source_commit": COMMIT,
        **expected,
        "limits": list(LIMITS),
        "new_model_developed": False,
    }


def verify_artifacts(report, output, root):
    """Native checks run again before review, registration and reuse acceptance."""
    output, root = Path(output), Path(root)
    manifest, inputs = admit(root)
    protocol = read(checked(output, "protocol.json", report.get("protocol_sha256")))
    if (
        protocol.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or protocol.get("asset_receipt_sha256") != ASSET_RECEIPT_SHA256
        or protocol.get("implementation") != implementation(root)
        or protocol.get("limits") != LIMITS
    ):
        raise ValueError("NucB protocol or implementation changed")
    for name, sha in protocol["implementation"].items():
        checked(output / "implementation", name, sha)
    raw = read(checked(output, "raw/report.json", report.get("raw_report_sha256")))
    analysis = read(checked(output, "analysis/report.json", report.get("analysis_report_sha256")))
    raw_protocol = read(checked(output / "raw", "protocol.json", raw.get("protocol_sha256")))
    if (
        raw.get("status") != "replayed"
        or raw.get("source_unchanged") is not True
        or raw_protocol.get("source") != manifest
        or raw_protocol.get("input_hashes") != inputs
        or raw_protocol.get("execution_order") != list(ROWS)
    ):
        raise ValueError("NucB raw computation incomplete or unbound")
    stages = raw.get("stages", [])
    if [s.get("generation") for s in stages] != list(ROWS):
        raise ValueError("Missing or duplicate NucB generation")
    for stage in stages:
        g = stage["generation"]
        if (
            stage.get("status") != "replayed"
            or stage.get("rows") != ROWS[g]
            or set(stage.get("output_hashes", {})) != set(tables(g))
        ):
            raise ValueError("Incomplete NucB raw allocation")
        checks = {}
        for name in tables(g):
            p = checked(output / "raw/data", name, stage["output_hashes"][name])
            checks[name] = _compare_tables(root, p, assets_at(root) / "gcs" / name)
        if (
            not all(c["passed"] for c in checks.values())
            or stage.get("comparisons") != checks
            or stage.get("matches_author_tables") is not True
        ):
            raise ValueError("NucB raw tables differ from independent reference comparison")
    landscape = raw.get("landscape", {})
    p = checked(output / "raw/data", LANDSCAPE, landscape.get("sha256"))
    comparison = _compare_tables(root, p, assets_at(root) / "gcs" / LANDSCAPE)
    if (
        not comparison["passed"]
        or comparison["actual_rows"] != 55760
        or landscape.get("gcs_reference") != comparison
        or landscape.get("rows") != 55760
        or landscape.get("unique_sequences") != 55760
    ):
        raise ValueError("NucB landscape differs")
    for name, sha in analysis.get("outputs", {}).items():
        checked(output / "analysis", name, sha)
    if not set(ANALYSIS_FILES) <= set(analysis.get("outputs", {})):
        raise ValueError("Missing NucB analysis outputs")
    ap = read(safe_file(output / "analysis", "protocol.json"))
    if analysis.get("status") != "replayed":
        raise ValueError("NucB analysis has a reference discrepancy or incomplete computation")
    if (
        analysis.get("source_unchanged") is not True
        or ap.get("source_manifest") != manifest
        or ap.get("execution_origin") != "provided_reconstructed_processed_tables"
        or ap.get("data_root") != str(output / "raw/data")
        or ap.get("analysis_inputs")
        != {
            f"processed_data/{s['generation']}.csv": s["output_hashes"][f"processed_data/{s['generation']}.csv"]
            for s in stages
        }
    ):
        raise ValueError("NucB analysis is not bound to this run's reconstructed data")
    replay = _replay_analysis(root, output)
    if (
        analysis.get("hit_rates") != replay["hit_rates"]
        or analysis.get("purified_protein") != replay["purified_protein"]
    ):
        raise ValueError("NucB analysis report differs from native replay")
    evidence = {
        "generation_rows": {s["generation"]: s["rows"] for s in stages},
        "processed_and_fiducial_tables": sum(len(s["output_hashes"]) for s in stages),
        "unique_landscape_sequences": comparison["actual_rows"],
        "hit_rate_rows": replay["hit_rates"]["sublibraries"],
        "purified_timepoints": replay["purified_protein"]["time_series_points"],
        "initial_rate_fits": replay["purified_protein"]["fits"],
        "reference_tables_passed": 17,
        "independent_analysis_replay_passed": True,
    }
    if report.get("evidence", evidence) != evidence:
        raise ValueError("NucB evidence projection changed")
    canonical_nucb({**report, "evidence": evidence})
    return evidence
