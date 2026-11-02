"""No-network checks for family-isolated projections and all-arm analysis."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_generated_features import CODE, SCOPE

from agent.rsi.engine.experience import LIMITATION
from agent.rsi.engine.protein_model import digest
from agent.rsi.cases.reap_experience import design, load_library, project_for_target, transfer_prompt
from agent.rsi.engine.storage import sha256_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases.run_reap_generated_matrix import development_prompt
from scripts.rsi.cases.run_reap_transfer_campaign import analysis_rows, freeze_development_barrier, paired_contrasts


def library(tmp_path):
    root = tmp_path / "library"
    (root / "experience/entries").mkdir(parents=True)
    donors, ids = [], []
    for family in ["P450_BM3_FL62", "SaSrtA"]:
        scope = {**deepcopy(SCOPE), "protein": family}
        entry = {
            "schema": 1,
            "source_code": CODE,
            "source_sha256": __import__("hashlib").sha256(CODE.encode()).hexdigest(),
            "donor_scope": scope,
            "method_summary": "Fitted numerical linear model",
            "acceptance": {"passed": True, "training_replayed": True},
            "parents": [],
            "model_sha256": "b" * 64,
            "assessment_sha256": "c" * 64,
            "model_id": "d" * 64,
        }
        eid = digest(entry)
        (root / "experience/entries" / (eid + ".json")).write_text(json.dumps(entry))
        ids.append(eid)
        donors.append({"protein": family, "experience_id": eid, "source_sha256": entry["source_sha256"]})
    (root / "experience/catalog.json").write_text(json.dumps({"schema": 1, "entries": ids}))
    manifest = {
        "schema": 1,
        "donors": donors,
        "files": {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def test_projection_is_other_family_and_excludes_fitted_state(tmp_path):
    root = library(tmp_path)
    scope = {**deepcopy(SCOPE), "protein": "P450_BM3_FL62"}
    none = project_for_target(root, scope, "none")
    summary = project_for_target(root, scope, "summary")
    source = project_for_target(root, scope, "source")
    assert none == {"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION}
    assert summary["entries"][0]["donor_scope"]["protein"] == "SaSrtA"
    assert {k: v for k, v in source["entries"][0].items() if k != "source_code"} == summary["entries"][0]
    assert source["entries"][0]["source_code"] == CODE
    assert (
        not {"model_state", "model_sha256", "assessment_sha256", "model_id", "future_labels"}
        & source["entries"][0].keys()
    )


def test_encoder_mismatch_and_unknown_family_rejected(tmp_path):
    root = library(tmp_path)
    changed = {**deepcopy(SCOPE), "protein": "SaSrtA"}
    changed["representation"]["encoder_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="encoder"):
        project_for_target(root, changed, "source")
    with pytest.raises(ValueError, match="other-protein"):
        project_for_target(root, SCOPE, "source")


def test_tampered_library_rejected(tmp_path):
    root = library(tmp_path)
    manifest, _ = load_library(root)
    p = root / "experience/entries" / (manifest["donors"][0]["experience_id"] + ".json")
    value = json.loads(p.read_text())
    value["source_code"] += "\n# changed\n"
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed"):
        load_library(root)


def test_balanced_design_and_common_prompt():
    groups = design()
    assert len(groups) == 9
    for seed in [1, 2, 3]:
        assert {x["condition"] for x in groups if x["seed"] == seed} == {"none", "summary", "source"}
    for index in range(3):
        assert {groups[index + 3 * block]["condition"] for block in range(3)} == {"none", "summary", "source"}
    with pytest.raises(ValueError):
        design((0, 1, 2))
    with pytest.raises(ValueError):
        design((1, 2))
    prompt = transfer_prompt(SCOPE, development_prompt(SCOPE))
    assert "reference implementation or donor model is supplied" not in prompt
    assert "reuse, revise or ignore" in prompt


def test_failed_and_rejected_allocations_remain_in_paired_utility():
    reports = []
    for index, condition in enumerate(["none", "summary", "source"]):
        arm = {"id": 0, "family": "SaSrtA", "mode": "science_agent", "status": "development_failed"}
        if condition == "source":
            arm.update(
                status="candidate_rejected",
                assessment={"passed": False},
                future_metrics={"spearman": 0.4, "mse_skill_vs_training_mean": -0.2},
            )
        reports.append(({"id": index, "seed": 1, "condition": condition}, {"arms": [arm]}))
    rows = analysis_rows(reports)
    assert len(rows) == 3 and rows[0]["spearman"] is None and rows[0]["ranking_utility"] == -1
    assert rows[2]["ranking_utility"] == 0.4 and rows[2]["accepted"] is False
    assert len(paired_contrasts(rows)) == 2
    assert all(x["ranking_utility_difference"] == 1.4 for x in paired_contrasts(rows))


def test_global_future_barrier_requires_all_terminal_unscored_allocations(tmp_path):
    groups = design()
    for group in groups:
        folder = tmp_path / f"group_{group['id']:02d}"
        folder.mkdir()
        arms = [
            {
                "id": i,
                "family": "P450" if i < 2 else "SaSrtA",
                "mode": "science_agent" if i % 2 else "science_expert",
                "condition": group["condition"],
                "status": "development_failed",
            }
            for i in range(4)
        ]
        (folder / "report.json").write_text(json.dumps({"status": "development_completed", "arms": arms}))
    identity = freeze_development_barrier(tmp_path, groups)
    assert identity == sha256_file(tmp_path / "development_barrier.json")
    barrier = json.loads((tmp_path / "development_barrier.json").read_text())
    assert barrier["allocations"] == 36 and barrier["future_metrics_present"] is False
    for child in barrier["groups"]:
        assert sha256_file(tmp_path / f"group_{child['id']:02d}/development_terminal.json") == child["terminal_sha256"]


@pytest.mark.parametrize("bad", [{"status": "running"}, {"future_metrics": {"spearman": 0.9}}])
def test_global_future_barrier_rejects_nonterminal_or_prescored_arm(tmp_path, bad):
    folder = tmp_path / "group_00"
    folder.mkdir()
    arms = [
        {"id": i, "family": "P450", "mode": "science_agent", "condition": "none", "status": "completed"}
        for i in range(4)
    ]
    arms[0].update(bad)
    (folder / "report.json").write_text(json.dumps({"status": "development_completed", "arms": arms}))
    with pytest.raises(ValueError, match="Nonterminal or prematurely"):
        freeze_development_barrier(tmp_path, design())
    assert not (tmp_path / "development_barrier.json").exists()
