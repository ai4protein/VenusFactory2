"""Fixed-scale, paired RF2 hotspot benchmark over admitted antigen structures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from agent.rsi.campaigns.case_workflow import CaseStage, audit_quiver
from agent.rsi.engine.protein_model import digest
from .rfantibody_evidence import COMMIT
from .rfantibody_publication import summarize_quivers
from .rfantibody_targets import TARGETS, validate_target_input
from agent.rsi.engine.storage import atomic_json, sha256_file

CATALOG = "configs/rsi/rfantibody_benchmark_targets.json"
WEIGHT_HASHES = {
    "ProteinMPNN_v48_noise_0.2.pt": "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
    "RF2_ab.pt": "cb38916458f5a9ecad0cf33182d816a836204b58896c59bd5a4ef475f63950e7",
    "RFdiffusion_Ab.pt": "736c18ae8e867ed505d19853fa5380514ff439675d002ff3a73c7287acf07e3c",
}

SCALE = {"backbones": 200, "sequences_per_backbone": 8, "diffusion_steps": 50, "rf2_recycles": 10}
CONDITIONS = ("1.0", "0.1", "0.0")
STAGES = ("backbone", "sequence", "refinement_1p0", "refinement_0p1", "refinement_0p0")
LAUNCHER = r"""import os, sys
from pathlib import Path
root = Path(__file__).resolve().parent / 'author_source'
paths = [str(root / 'src'), str(root / 'include/SE3Transformer')]
os.environ['PYTHONPATH'] = os.pathsep.join(paths)
os.environ['RFANTIBODY_ROOT'] = str(root)
os.environ['RFANTIBODY_SCRIPTS'] = str(root / 'scripts')
sys.path[:0] = paths
from rfantibody.cli.inference import rfdiffusion, proteinmpnn, rf2
command = sys.argv.pop(1)
{'rfdiffusion': rfdiffusion, 'proteinmpnn': proteinmpnn, 'rf2': rf2}[command]()
"""


def verify_weights(root):
    folder = root / "research_runs/rsi/case_assets/RFantibody" / COMMIT / "source/weights"
    if any(sha256_file(folder / name) != expected for name, expected in WEIGHT_HASHES.items()):
        raise ValueError("Admitted model weights changed")


def load_catalog(root, target):
    if target not in TARGETS:
        raise ValueError("Registered benchmark target required")
    catalog = json.loads((root / CATALOG).read_text())
    if catalog.get("source_commit") != COMMIT or set(catalog["targets"]) != set(TARGETS):
        raise ValueError("Benchmark source or target allocation changed")
    record = catalog["targets"][target]
    validate_target_input(root, record)
    framework = (root / catalog["framework"]["path"]).resolve()
    framework.relative_to((root / "research_runs/rsi").resolve())
    if sha256_file(framework) != catalog["framework"]["sha256"]:
        raise ValueError("Framework changed")
    return catalog, record


def freeze_author_source(root, out):
    """Copy hash-admitted author files and import the copy in child processes."""
    asset = root / "research_runs/rsi/case_assets/RFantibody" / COMMIT
    admission = json.loads((asset / "asset_manifest.json").read_text())
    chosen = {}
    for row in admission["files"]:
        relative = Path(row["path"])
        relative.relative_to("source")
        local = relative.relative_to("source")
        if local.parts[0] not in {"src", "scripts", "include"} and str(local) not in {
            "pyproject.toml",
            "uv.lock",
            "LICENSE",
        }:
            continue
        path = asset / relative
        if path.is_symlink() or sha256_file(path) != row["sha256"]:
            raise ValueError("Author file differs from admitted asset: " + str(relative))
        chosen[str(local)] = row["sha256"]
    required = {
        "src/rfantibody/cli/inference.py",
        "scripts/rfdiffusion_inference.py",
        "scripts/proteinmpnn_interface_design.py",
        "scripts/rf2_predict.py",
        "uv.lock",
    }
    if not required <= set(chosen):
        raise ValueError("Incomplete admitted author source")
    destination = out / "author_source"
    destination.mkdir(exist_ok=False)
    for name, expected in chosen.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(asset / "source" / name, path)
        if sha256_file(path) != expected:
            raise ValueError("Author snapshot changed during copying")
    (out / "author_entry.py").write_text(LAUNCHER)
    frozen = {"author_source/" + name: value for name, value in chosen.items()}
    frozen["author_entry.py"] = sha256_file(out / "author_entry.py")
    atomic_json(out / "author_manifest.json", frozen)
    return frozen


def verify_author_source(out, expected_manifest_sha256):
    manifest = out / "author_manifest.json"
    if sha256_file(manifest) != expected_manifest_sha256:
        raise ValueError("Author manifest changed")
    for name, expected in json.loads(manifest.read_text()).items():
        path = out / name
        path.resolve().relative_to(out.resolve())
        if path.is_symlink() or sha256_file(path) != expected:
            raise ValueError("Frozen author implementation changed")


def build_stages(root, out, record, framework):
    """All three RF2 conditions consume exactly the same 1,600 sequence records."""
    resources = (root / "research_runs/rsi").resolve()
    source = resources / "case_assets/RFantibody" / COMMIT / "source"
    weights = source / "weights"
    python = resources / "envs/rfantibody/bin/python"
    command = (str(python), "-u", str(out / "author_entry.py"))
    target = (root / record["path"]).resolve()
    framework = (root / framework["path"]).resolve()
    backbones, sequences = out / "1_backbones.qv", out / "2_sequences.qv"
    common = (out / "author_manifest.json", out / "author_entry.py", out / "input_catalog.json")
    specs = [
        (
            "backbone",
            "rfdiffusion",
            [
                "--target",
                str(target),
                "--framework",
                str(framework),
                "--output-quiver",
                str(backbones),
                "--num-designs",
                "200",
                "--design-loops",
                record["loops"],
                "--hotspots",
                ",".join(h["chain"] + str(h["number"]) for h in record["hotspots"]),
                "--diffuser-t",
                "50",
                "--deterministic",
                "--no-trajectory",
                "--weights",
                str(weights / "RFdiffusion_Ab.pt"),
            ],
            (target, framework, weights / "RFdiffusion_Ab.pt"),
            backbones,
            200,
            86400,
        ),
        (
            "sequence",
            "proteinmpnn",
            [
                "--input-quiver",
                str(backbones),
                "--output-quiver",
                str(sequences),
                "--seqs-per-struct",
                "8",
                "--temperature",
                "0.2",
                "--deterministic",
                "--weights",
                str(weights / "ProteinMPNN_v48_noise_0.2.pt"),
            ],
            (backbones, weights / "ProteinMPNN_v48_noise_0.2.pt"),
            sequences,
            1600,
            21600,
        ),
    ]
    for condition in CONDITIONS:
        output = out / ("3_refined_" + condition.replace(".", "p") + ".qv")
        specs.append(
            (
                "refinement_" + condition.replace(".", "p"),
                "rf2",
                [
                    "--input-quiver",
                    str(sequences),
                    "--output-quiver",
                    str(output),
                    "--num-recycles",
                    "10",
                    "--hotspot-show-prop",
                    condition,
                    "--seed",
                    "20261018",
                    "--weights",
                    str(weights / "RF2_ab.pt"),
                ],
                (sequences, weights / "RF2_ab.pt"),
                output,
                1600,
                172800,
            )
        )
    return [
        CaseStage(
            name,
            (*command, cli, *args),
            out / "author_source",
            (*inputs, *common),
            output,
            timeout,
            lambda p, count=count: audit_quiver(p, count),
        )
        for name, cli, args, inputs, output, count, timeout in specs
    ]


def summarize_condition(out, target, condition):
    return summarize_quivers(
        out / "1_backbones.qv",
        out / "2_sequences.qv",
        out / ("3_refined_" + condition.replace(".", "p") + ".qv"),
        target=target,
        backbones=200,
        sequences_per_backbone=8,
        hotspot_proportion=float(condition),
        source_commit=COMMIT,
    )


def compact_summary(audit):
    return {
        k: audit[k]
        for k in [
            "allocation",
            "sequence_passes",
            "selected_backbone_passes",
            "any_passing_sequence_backbones",
            "selected_backbone_pass_percent",
            "any_passing_sequence_backbone_percent",
            "independent_rmsd_count",
            "max_rmsd_absolute_difference",
        ]
    }


def canonical_benchmark(report, parameters=None):
    settings = report.get("settings", {})
    target = settings.get("target")
    conditions = report.get("conditions", {})
    if (
        report.get("status") != "completed"
        or report.get("computational_chain_completed") is not True
        or settings.get("scale") != SCALE
        or target not in TARGETS
        or settings.get("source_commit") != COMMIT
    ):
        raise ValueError("Incomplete benchmark allocation")
    if parameters is not None and parameters != {"target": target}:
        raise ValueError("Benchmark request changed")
    if set(conditions) != set(CONDITIONS) or [s.get("binding", {}).get("id") for s in report.get("stages", [])] != list(
        STAGES
    ):
        raise ValueError("All three paired conditions and five stages are required")
    if any(s.get("status") != "completed" for s in report["stages"]):
        raise ValueError("All stages must be completed")
    for condition, row in conditions.items():
        if (
            row.get("allocation")
            != {
                "backbones": 200,
                "sequences_per_backbone": 8,
                "total_sequences": 1600,
                "hotspot_proportion": float(condition),
            }
            or row.get("independent_rmsd_count") != 1600
        ):
            raise ValueError("Incomplete condition denominator")
        for key, limit in [
            ("sequence_passes", 1600),
            ("selected_backbone_passes", 200),
            ("any_passing_sequence_backbones", 200),
        ]:
            if type(row.get(key)) is not int or not 0 <= row[key] <= limit:
                raise ValueError("Invalid pass count")
        if (
            row["selected_backbone_passes"] > row["any_passing_sequence_backbones"]
            or row["any_passing_sequence_backbones"] > row["sequence_passes"]
        ):
            raise ValueError("Inconsistent structure and backbone counts")
        if (
            row["selected_backbone_pass_percent"] != row["selected_backbone_passes"] / 2
            or row["any_passing_sequence_backbone_percent"] != row["any_passing_sequence_backbones"] / 2
        ):
            raise ValueError("Invalid backbone percentages")
    return {
        "computational_scope": "rfantibody_paired_full_scale_target",
        "target_antigen": target,
        "source_commit": COMMIT,
        "backbones": 200,
        "sequences_per_backbone": 8,
        "sequence_design_records": 1600,
        "refinement_assessments": 4800,
        "paired_hotspot_conditions": conditions,
        "sequence_preservation_verified": True,
        "publication_crop_equivalence_established": False,
        "experimental_binding_verified": False,
    }


def verify_benchmark_artifacts(root, out, report, parameters):
    """Recompute each condition from raw coordinates before accepting evidence."""
    claims = canonical_benchmark(report, parameters)
    verify_weights(root)
    settings = report["settings"]
    if sha256_file(out / "input_catalog.json") != settings["catalog_sha256"]:
        raise ValueError("Frozen input catalog changed")
    catalog = json.loads((out / "input_catalog.json").read_text())
    target = settings["target"]
    record = catalog["targets"][target]
    validate_target_input(root, record)
    if sha256_file(root / catalog["framework"]["path"]) != catalog["framework"]["sha256"]:
        raise ValueError("Framework changed")
    verify_author_source(out, settings["author_manifest_sha256"])
    for stage, row in zip(build_stages(root, out, record, catalog["framework"]), report["stages"], strict=True):
        binding = {
            "id": stage.id,
            "argv": list(stage.argv),
            "cwd": str(stage.cwd),
            "input_hashes": {str(p): sha256_file(p) for p in stage.inputs},
            "output": str(stage.output),
            "timeout_s": stage.timeout_s,
        }
        if (
            row.get("status") != "completed"
            or row.get("binding") != binding
            or row.get("recipe_sha256") != digest(binding)
            or row.get("output_sha256") != sha256_file(stage.output)
        ):
            raise ValueError("Stage recipe or artifact changed")
    for condition in CONDITIONS:
        path = out / "audits" / (condition + ".json")
        if sha256_file(path) != report["audit_hashes"][condition]:
            raise ValueError("Condition audit changed")
        stored = json.loads(path.read_text())
        fresh = summarize_condition(out, target, condition)
        if stored != fresh or compact_summary(fresh) != report["conditions"][condition]:
            raise ValueError("Independent condition replay disagrees")
    return claims
