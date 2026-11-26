import hashlib
import json
from copy import deepcopy

import pytest

from agent.rsi.engine.case_evidence import canonical_claims, validate_review
from agent.rsi.engine.protein_model import digest
from agent.rsi.cases.rfantibody_evidence import COMMIT, PRESETS, validate_target_report, verify_target_artifacts


def fixture(root):
    stages = []
    inputs = []
    for name, count in [("backbone", 2), ("sequence", 8), ("refinement", 8)]:
        output = root / (name + ".qv")
        output.write_text(name)
        binding = {
            "id": name,
            "output": str(output),
            "input_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        }
        stages.append(
            {
                "status": "completed",
                "binding": binding,
                "recipe_sha256": digest(binding),
                "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "artifact_audit": {"designs": count},
            }
        )
        inputs = [output]
    quality = {
        "designs": 8,
        "sequence_preservation_verified": True,
        "author_minimum_passes": 3,
        "additional_interface_passes": 1,
    }
    audit = {**quality, "inputs": {s["binding"]["output"]: s["output_sha256"] for s in stages[1:]}}
    (root / "quality_audit").mkdir()
    (root / "quality_audit/audit.json").write_text(json.dumps(audit))
    return {
        "status": "completed",
        "computational_chain_completed": True,
        "quality_audit": quality,
        "settings": {
            "case_id": "rfantibody_ha_nanobody",
            "commit": COMMIT,
            "repository": "RosettaCommons/RFantibody",
            "target_preset": PRESETS["ha"],
            "scale": {"backbones": 2, "sequences_per_backbone": 4, "diffusion_steps": 50, "rf2_recycles": 10},
        },
        "stages": stages,
    }


def test_target_scope_and_actual_artifact_bindings(tmp_path):
    report = fixture(tmp_path)
    claims = canonical_claims("rfantibody.targets.v1", report)
    assert claims["target_antigen"] == "ha" and claims["designs"] == 8
    assert claims["structural_filter_passes"] == 1 and not claims["experimental_binding_verified"]
    assert not claims["repeat_computation_is_independent_efficacy_evidence"]
    assert verify_target_artifacts(report, tmp_path)["stage_outputs_verified"] == 3
    with pytest.raises(ValueError, match="requested target"):
        validate_target_report(report, {"target": "rsv_site3", "backbones": 2, "sequences": 4})
    with pytest.raises(ValueError, match="disagree"):
        validate_review({**claims, "experimental_binding_verified": True}, claims)
    (tmp_path / "sequence.qv").write_text("different sequence")
    with pytest.raises(ValueError, match="output changed"):
        verify_target_artifacts(report, tmp_path)


@pytest.mark.parametrize("change", ["count", "stage", "commit", "quality"])
def test_completed_label_does_not_override_incomplete_domain_evidence(tmp_path, change):
    report = deepcopy(fixture(tmp_path))
    if change == "count":
        report["quality_audit"]["designs"] = 7
    if change == "stage":
        report["stages"].pop()
    if change == "commit":
        report["settings"]["commit"] = "different"
    if change == "quality":
        report["quality_audit"]["additional_interface_passes"] = 4
    with pytest.raises(ValueError):
        canonical_claims("rfantibody.targets.v1", report)
