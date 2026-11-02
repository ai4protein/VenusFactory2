"""Target- and scale-bound evidence for author nanobody pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protein_model import digest

COMMIT = "3e5a5bb1cb98befde9fc049ae3ff58cabb491f67"
PRESETS = {
    "ha": {
        "target": "flu_HA.pdb",
        "hotspots": "B146,B170,B177",
        "loops": "H1:7,H2:6,H3:5-13",
        "author_recipe": "scripts/examples/nanobody_full_pipeline.sh",
    },
    "rsv_site3": {
        "target": "rsv_site3.pdb",
        "hotspots": "T305,T456",
        "loops": "L1:8-13,L2:7,L3:9-11,H1:7,H2:6,H3:5-13",
        "author_recipe": "scripts/examples/rfdiffusion/nanobody_qvdesign.sh",
    },
}


def validate_target_report(report, parameters=None):
    settings = report.get("settings", {})
    target = next((name for name, preset in PRESETS.items() if settings.get("target_preset") == preset), None)
    scale = settings.get("scale", {})
    backbones, sequences = scale.get("backbones"), scale.get("sequences_per_backbone")
    if (
        target is None
        or settings.get("commit") != COMMIT
        or settings.get("repository") != "RosettaCommons/RFantibody"
        or settings.get("case_id") != "rfantibody_" + target + "_nanobody"
        or type(backbones) is not int
        or not 1 <= backbones <= 1000
        or type(sequences) is not int
        or not 1 <= sequences <= 20
        or scale.get("diffusion_steps") != 50
        or scale.get("rf2_recycles") != 10
    ):
        raise ValueError("Antibody target, author version or declared scale differs")
    if parameters is not None and {"target": target, "backbones": backbones, "sequences": sequences} != {
        "target": parameters.get("target"),
        "backbones": parameters.get("backbones", 2),
        "sequences": parameters.get("sequences", 2),
    }:
        raise ValueError("Antibody result differs from requested target or scale")
    quality = report.get("quality_audit", {})
    designs = backbones * sequences
    if (
        report.get("status") != "completed"
        or report.get("computational_chain_completed") is not True
        or quality.get("sequence_preservation_verified") is not True
        or type(quality.get("designs")) is not int
        or quality["designs"] != designs
    ):
        raise ValueError("Incomplete antibody chain, sequence preservation or design count")
    for name in ["author_minimum_passes", "additional_interface_passes"]:
        if type(quality.get(name)) is not int or not 0 <= quality[name] <= designs:
            raise ValueError("Invalid antibody structural-filter count")
    if quality["additional_interface_passes"] > quality["author_minimum_passes"]:
        raise ValueError("Additional interface filter cannot increase passing designs")
    stages = report.get("stages", [])
    if [s.get("binding", {}).get("id") for s in stages] != ["backbone", "sequence", "refinement"]:
        raise ValueError("Incomplete ordered antibody stages")
    for stage, expected in zip(stages, [backbones, designs, designs], strict=True):
        if stage.get("status") != "completed" or stage.get("artifact_audit", {}).get("designs") != expected:
            raise ValueError("Antibody stage count differs from declared allocation")
        if digest(stage["binding"]) != stage.get("recipe_sha256"):
            raise ValueError("Antibody stage recipe changed")
    return {
        "computational_scope": "antibody_backbone_sequence_structure",
        "target_antigen": target,
        "source_commit": COMMIT,
        "backbones": backbones,
        "sequences_per_backbone": sequences,
        "designs": designs,
        "sequence_preservation_verified": True,
        "author_minimum_passes": quality["author_minimum_passes"],
        "structural_filter_passes": quality["additional_interface_passes"],
        "experimental_binding_verified": False,
        "repeat_computation_is_independent_efficacy_evidence": False,
    }


def verify_target_artifacts(report, root):
    """Rehash stage inputs/outputs and the sequence-preserving quality evidence."""
    validate_target_report(report)
    root = Path(root).resolve()
    cache = {}

    def hashed(path):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("Antibody evidence file is a symlink")
        name = str(path.resolve())
        if name not in cache:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1048576), b""):
                    h.update(block)
            cache[name] = h.hexdigest()
        return cache[name]

    outputs = []
    for stage in report["stages"]:
        path = Path(stage["binding"]["output"])
        path.resolve().relative_to(root)
        if hashed(path) != stage["output_sha256"]:
            raise ValueError("Antibody stage output changed")
        for name, expected in stage["binding"]["input_hashes"].items():
            if hashed(name) != expected:
                raise ValueError("Antibody stage input changed")
        outputs.append(path)
    audit_path = root / "quality_audit/audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["inputs"] != {str(path): hashed(path) for path in outputs[1:]}:
        raise ValueError("Antibody quality audit references different outputs")
    if {key: audit[key] for key in report["quality_audit"]} != report["quality_audit"]:
        raise ValueError("Antibody quality projection differs from original audit")
    return {"stage_outputs_verified": len(outputs), "quality_audit_sha256": hashed(audit_path)}
