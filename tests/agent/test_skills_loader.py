"""Smoke tests for skill discovery, read paths, and skill-first picker."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.graph.helpers.plan_helpers import _pick_skill_for_code_step
from agent.skills import (
    get_skill_content,
    get_skill_root,
    get_skills_metadata,
    invalidate_skills_cache,
    list_skill_ids,
    resolve_skill_path,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    invalidate_skills_cache()
    yield
    invalidate_skills_cache()


def test_discovers_skills_and_skips_shared():
    ids = list_skill_ids()
    assert "uniprot_database" in ids
    assert "zero_shot_mutation_workflow" in ids
    assert "nature_figure" in ids
    assert "hpa_expression_context" in ids
    assert "venus_finetune_workflow" in ids
    assert "structure_file_prep" in ids
    assert "protein_engineering_hypothesis" in ids
    assert len(ids) >= 36
    assert not any(i.startswith("_") for i in ids)
    assert "_shared_nature" not in ids


def test_name_matches_dir_for_critical_skills():
    by_id = {m["skill_id"]: m for m in get_skills_metadata()}
    for sid in ("fda", "nature_figure", "nature_writing", "nature_polishing"):
        assert by_id[sid]["name_matches_dir"] is True, sid


def test_read_skill_relative_path_and_traversal():
    root = get_skill_root("nature_figure")
    assert root and Path(root).is_dir()
    manifest = get_skill_content("nature_figure", "manifest.yaml")
    assert manifest and "backend" in manifest
    assert resolve_skill_path("nature_figure", "../secrets") is None
    assert resolve_skill_path("nature_figure", "..") is None
    assert get_skill_content("no_such_skill") is None
    # Whitelisted shared-nature alias (nature writing/polishing always_load)
    ethics = resolve_skill_path("nature_writing", "_shared_nature/core/ethics.md")
    assert ethics is not None and ethics.is_file()
    ethics_legacy = resolve_skill_path("nature_writing", "../_shared/core/ethics.md")
    assert ethics_legacy is not None and ethics_legacy.is_file()
    # Arbitrary sibling escape still blocked
    assert resolve_skill_path("nature_writing", "../alphafold_database/SKILL.md") is None


def test_read_skill_tool_json():
    from tools.skill.tools_agent import read_skill_tool

    raw = read_skill_tool.invoke({"skill_id": "nature_figure", "relative_path": "manifest.yaml"})
    data = json.loads(raw)
    assert data["success"] is True
    assert data["skill_root"]
    assert data["relative_path"] == "manifest.yaml"


def test_pick_skill_domain_rules():
    ids = list_skill_ids()
    assert _pick_skill_for_code_step("绘制论文配图 heatmap", ids) == "nature_figure"
    assert _pick_skill_for_code_step("可视化出图", ids) == "nature_figure"
    assert _pick_skill_for_code_step("zero-shot mutation prediction", ids) == "zero_shot_mutation_workflow"
    assert _pick_skill_for_code_step("foldseek structural similar", ids) == "foldseek_structural_similarity"
    assert _pick_skill_for_code_step("HPA tissue expression TP53", ids) == "hpa_expression_context"
    assert _pick_skill_for_code_step("finetune custom protein model", ids) == "venus_finetune_workflow"
    # Domain must win over generic "visual"/"plot"
    assert _pick_skill_for_code_step("STRING interaction network visualization", ids) == "string_database"
    assert _pick_skill_for_code_step("plot mutation scores zero-shot", ids) == "zero_shot_mutation_workflow"
    assert _pick_skill_for_code_step("rcsb experimental structure download", ids) == "rcsb_database"
    assert _pick_skill_for_code_step("unrelated random task xyz", ids) is None


def test_thick_skills_are_slim():
    """Progressive disclosure: main SKILL.md should stay agent-short."""
    thick = [
        "alphafold_database",
        "rdkit",
        "seaborn",
        "brenda_database",
        "fda",
        "biopython",
        "chembl_database",
        "kegg_database",
        "string_database",
        "matplotlib",
    ]
    root = Path(__file__).resolve().parents[2] / "src" / "agent" / "skills"
    for sid in thick:
        n = len((root / sid / "SKILL.md").read_text(encoding="utf-8").splitlines())
        assert n <= 180, f"{sid} still too thick: {n} lines"
        legacy = root / sid / "references" / "legacy_guide.md"
        assert legacy.is_file(), f"{sid} missing archived legacy_guide.md"
