"""Evidence boundary for newly executed, bounded PHL7 WT HREX validation."""

import json
from pathlib import Path

from .phl7_case import ARCHIVE_SHA256
from .storage import sha256_file

PLUGIN_ID = "phl7.hrex_validation.v1"
STAGES = ("preparation", "energy_validation", "dynamics", "native_audit")


def runtime_configuration(root, backend):
    root = Path(root).resolve()
    if backend not in {"cpu", "cuda"}:
        raise ValueError("Unknown HREX backend")
    path = root / "research_runs/rsi/runtime_configs/phl7_hrex" / (backend + ".json")
    config = json.loads(path.read_text())
    expected = {
        "schema",
        "backend",
        "build",
        "build_report_sha256",
        "plumed",
        "plumed_sha256",
        "gmx_double",
        "gmx_double_sha256",
    }
    if set(config) != expected or config["schema"] != 1 or config["backend"] != backend:
        raise ValueError("HREX runtime manifest contract differs")
    resolved = {}
    for name in ["build", "plumed", "gmx_double"]:
        relative = Path(config[name])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("HREX runtime path must remain within the research directory")
        location = (root / relative).resolve()
        location.relative_to((root / "research_runs/rsi").resolve())
        resolved[name] = location
        target = location / "build_result.json" if name == "build" else location
        field = "build_report_sha256" if name == "build" else name + "_sha256"
        if sha256_file(target) != config[field]:
            raise ValueError("HREX configured runtime changed")
    build = json.loads((resolved["build"] / "build_result.json").read_text())
    if build["status"] != "completed" or (backend == "cuda" and build.get("gpu_backend") != "CUDA"):
        raise ValueError("Requested HREX runtime has not passed verification")
    return config, resolved


def canonical_hrex(report):
    if (
        report.get("status") != "completed"
        or report.get("execution_origin") != "new_computation"
        or report.get("archive_sha256") != ARCHIVE_SHA256
        or report.get("compute_backend") not in {"cpu", "cuda"}
        or report.get("new_molecular_dynamics_performed") is not True
        or report.get("full_paper_reproduced") is not False
        or [row.get("id") for row in report.get("stages", [])] != list(STAGES)
        or any(row.get("status") != "completed" for row in report["stages"])
    ):
        raise ValueError("Incomplete or unsupported PHL7 HREX evidence")
    summary = report.get("summary", {})
    integer_fields = [
        "replicas",
        "ps_per_replica",
        "frames_per_replica",
        "exchange_schedule_events",
        "pair_attempts",
        "accepted_swaps",
        "energy_identities",
        "atoms_per_replica",
    ]
    if any(type(summary.get(name)) is not int for name in integer_fields):
        raise ValueError("HREX counts must be integers")
    if (
        summary.get("replicas") != 8
        or summary.get("ps_per_replica") != 4
        or summary.get("frames_per_replica") != 21
        or summary.get("exchange_schedule_events") != 19
        or summary.get("pair_attempts") != 67
        or type(summary.get("accepted_swaps")) is not int
        or not 0 <= summary["accepted_swaps"] <= 67
        or summary.get("energy_identities") != 24
        or summary.get("atoms_per_replica") != 40557
    ):
        raise ValueError("Bounded HREX allocation or observables changed")
    return {
        "computational_scope": "phl7_wt_topologies_energy_checks_and_four_ps_hrex",
        "execution_origin": "new_computation",
        "compute_backend": report["compute_backend"],
        **summary,
        "new_molecular_dynamics_performed": True,
        "full_paper_reproduced": False,
        "convergence_established": False,
        "production_stability_established": False,
        "new_algorithm_or_agent_superiority_established": False,
        "known_fractional_charge_pme_warnings": 7,
        "limits": [
            "WT only; mutant initial systems incomplete",
            "Four ps per replica; published production is 500 ns per replica",
            "Production exchange interval and main/supplement burn-in discrepancy unresolved",
        ],
    }


def verify_artifacts(report, output):
    output = Path(output).resolve()
    facts = canonical_hrex(report)
    protocol_path = output / "protocol.json"
    if sha256_file(protocol_path) != report.get("protocol_sha256"):
        raise ValueError("HREX pipeline protocol changed")
    protocol = json.loads(protocol_path.read_text())
    if sha256_file(output / "runtime_manifest.json") != protocol.get("runtime_manifest_sha256"):
        raise ValueError("HREX runtime manifest changed")
    if (
        protocol["plugin_id"] != PLUGIN_ID
        or protocol["compute_backend"] != report["compute_backend"]
        or protocol["archive_sha256"] != ARCHIVE_SHA256
    ):
        raise ValueError("HREX pipeline request changed")
    for name, value in protocol["implementation"].items():
        if Path(name).name != name or sha256_file(output / "implementation" / name) != value:
            raise ValueError("HREX pipeline implementation changed")
    stage_reports = {}
    for row in report["stages"]:
        path = output / row["id"] / "report.json"
        if path.is_symlink() or sha256_file(path) != row["report_sha256"]:
            raise ValueError("HREX stage report changed")
        stage_reports[row["id"]] = json.loads(path.read_text())
    prepared, energy, dynamics, audit = [stage_reports[key] for key in STAGES]
    if (
        prepared["status"] != "completed"
        or energy["status"] != "completed"
        or dynamics["status"] != "completed"
        or audit["status"] != "passed"
        or not energy["explicit_pair_aware_scaling_validated"]
        or energy["generic_uniform_half_scaling_validated"]
        or len(energy["energy_comparisons"]) != 24
    ):
        raise ValueError("HREX stage validation incomplete")
    ep = json.loads((output / "energy_validation/protocol.json").read_text())
    dp = json.loads((output / "dynamics/protocol.json").read_text())
    if (
        ep["preparation_report_sha256"] != report["stages"][0]["report_sha256"]
        or dp["energy_validation_report_sha256"] != report["stages"][1]["report_sha256"]
        or dp["compute_backend"] != report["compute_backend"]
        or audit["source_report_sha256"] != report["stages"][2]["report_sha256"]
        or Path(audit["source_run"]).resolve() != output / "dynamics"
        or audit["auditor_sha256"] != protocol["implementation"]["audit_phl7_hrex.py"]
        or audit["historical_files_unchanged"] is not True
    ):
        raise ValueError("HREX stage provenance binding changed")
    for key in ["preparation", "energy_validation", "dynamics"]:
        if sha256_file(output / key / "protocol.json") != stage_reports[key]["protocol_sha256"]:
            raise ValueError("HREX stage protocol changed")
    if len(dynamics["replicas"]) != 8 or len(audit["replicas"]) != 8:
        raise ValueError("HREX replica evidence missing")
    for i, (replica, replay) in enumerate(zip(dynamics["replicas"], audit["replicas"], strict=True)):
        if set(replica["artifacts"]) != {"topol.tpr", "md.gro", "md.edr", "md.log", "md.cpt", "observables.xvg"}:
            raise ValueError("HREX trajectory artifact coverage changed")
        if replica["index"] != i or replay["replica"] != i or replay["frames"] != 21 or replay["ps"] != 4:
            raise ValueError("HREX trajectory extent changed")
        for name, value in replica["artifacts"].items():
            if Path(name).name != name or sha256_file(output / "dynamics" / f"replica_{i}" / name) != value:
                raise ValueError("HREX trajectory artifact changed")
        for name in ["edr", "xtc"]:
            if sha256_file(output / "dynamics" / f"replica_{i}" / ("md." + name)) != replay[name + "_sha256"]:
                raise ValueError("HREX native trajectory binding changed")
    if (
        audit["global_pair_attempts"] != 67
        or audit["global_exchange_schedule_events"] != 19
        or audit["global_accepted_swaps"] != report["summary"]["accepted_swaps"]
    ):
        raise ValueError("HREX exchange count changed")
    return facts
