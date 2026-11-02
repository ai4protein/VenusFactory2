import hashlib

import numpy as np
import pytest

from agent.rsi.rfantibody_publication import read_metadata, summarize_quivers, target_aligned_ca_rmsd

COMMIT = "3e5a5bb1cb98befde9fc049ae3ff58cabb491f67"
BASE = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def record(tag, *, shift=0.0, pae=1.0, interface=1.0, reported=None, rotation=None, translation=None, scores=False):
    rows = ["QV_TAG " + tag]
    if scores:
        rows.append(
            "QV_SCORE "
            + tag
            + f" pae={pae}|interaction_pae={interface}|target_aligned_antibody_rmsd={shift if reported is None else reported}|pred_lddt=0.9"
        )
    serial = 0
    for chain, xyz in [("H", BASE + np.array([3.0 + shift, 2.0, 1.0])), ("T", BASE)]:
        if rotation is not None:
            xyz = xyz @ rotation
        if translation is not None:
            xyz = xyz + translation
        for index, (x, y, z) in enumerate(xyz, 1):
            serial += 1
            rows.append(
                f"ATOM  {serial:5d}  CA  ALA {chain}{index:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
    return "\n".join(rows) + "\n"


def fixture(tmp_path):
    paths = [tmp_path / n for n in ["backbones.qv", "sequences.qv", "refined.qv"]]
    paths[0].write_text("".join(record("b" + str(i)) for i in range(2)))
    paths[1].write_text("".join(record(f"b{i}_dldesign_{j}") for i in range(2) for j in range(2)))
    values = [(0.8, 1, 15), (1.1, 20, 8), (3.0, 2, 2), (1.0, 5, 5)]
    paths[2].write_text(
        "".join(
            record(f"b{i // 2}_dldesign_{i % 2}_best", shift=a, pae=b, interface=c, scores=True)
            for i, (a, b, c) in enumerate(values)
        )
    )
    return paths


def audit(paths, **kwargs):
    options = {
        "target": "ha",
        "backbones": 2,
        "sequences_per_backbone": 2,
        "hotspot_proportion": 0.0,
        "source_commit": COMMIT,
    }
    options.update(kwargs)
    return summarize_quivers(*paths, **options)


def test_interface_rule_and_backbone_denominator_are_independent_of_legacy_global_filter(tmp_path):
    paths = fixture(tmp_path)
    before = [p.read_bytes() for p in paths]
    result = audit(paths)
    assert result["sequence_passes"] == 2
    assert result["selected_backbone_passes"] == 1
    assert result["any_passing_sequence_backbones"] == 2
    assert result["selected_backbone_pass_percent"] == 50
    assert result["any_passing_sequence_backbone_percent"] == 100
    assert result["independent_rmsd_count"] == 4
    assert result["max_rmsd_absolute_difference"] < 1e-12
    assert result["rows"][0]["legacy_global_rule_pass"] and not result["rows"][0]["interface_rule_pass"]
    assert result["rows"][1]["interface_rule_pass"] and not result["rows"][1]["legacy_global_rule_pass"]
    assert not result["matching_published_scale"] and not result["exact_paper_replication_established"]
    assert not result["experimental_binding_verified"]
    assert [p.read_bytes() for p in paths] == before
    assert result["inputs"] == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_native_replay_removes_global_rotation_and_translation(tmp_path):
    a, b = tmp_path / "a.qv", tmp_path / "b.qv"
    a.write_text(record("x"))
    b.write_text(
        record(
            "x",
            shift=1.5,
            rotation=np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
            translation=np.array([10, 20, -3.0]),
        )
    )
    assert target_aligned_ca_rmsd(read_metadata(a)["x"], read_metadata(b)["x"]) == pytest.approx(1.5, abs=1e-10)


@pytest.mark.parametrize(
    "problem",
    [
        "wrong_rmsd",
        "changed_sequence",
        "missing_refinement",
        "unknown_parent",
        "duplicate_score",
        "missing_ca",
        "nonfinite_score",
        "duplicate_tag",
        "missing_sequence_index",
    ],
)
def test_invalid_or_incomplete_artifacts_are_rejected(tmp_path, problem):
    paths = fixture(tmp_path)
    if problem == "wrong_rmsd":
        paths[2].write_text(paths[2].read_text().replace("rmsd=0.8", "rmsd=0.1"))
    elif problem == "changed_sequence":
        paths[2].write_text(paths[2].read_text().replace("ALA H", "CYS H", 1))
    elif problem == "missing_refinement":
        paths[2].write_text(paths[2].read_text().split("QV_TAG b1_dldesign_1_best")[0])
    elif problem == "unknown_parent":
        paths[0].write_text(paths[0].read_text().replace("QV_TAG b0", "QV_TAG other"))
    elif problem == "duplicate_score":
        paths[2].write_text(paths[2].read_text().replace("|pred_lddt=0.9", "|pae=2|pred_lddt=0.9", 1))
    elif problem == "missing_ca":
        paths[2].write_text(paths[2].read_text().replace(" CA ", " CB ", 1))
    elif problem == "nonfinite_score":
        paths[2].write_text(paths[2].read_text().replace("interaction_pae=15", "interaction_pae=nan", 1))
    elif problem == "duplicate_tag":
        paths[2].write_text(paths[2].read_text() + record("b0_dldesign_0_best", scores=True))
    else:
        for p in paths[1:]:
            p.write_text(p.read_text().replace("b1_dldesign_1", "b1_dldesign_3"))
    with pytest.raises(ValueError):
        audit(paths)


def test_rounded_boundary_is_flagged_without_changing_the_strict_rule(tmp_path):
    paths = [tmp_path / n for n in ["b.qv", "s.qv", "r.qv"]]
    paths[0].write_text(record("b"))
    paths[1].write_text(record("b_dldesign_0"))
    paths[2].write_text(record("b_dldesign_0_best", shift=2.0, interface=9.0, scores=True))
    result = audit(paths, backbones=1, sequences_per_backbone=1)
    assert result["sequence_passes"] == 0 and result["near_rounded_threshold_sequences"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hotspot_proportion": 0.3},
        {"backbones": True},
        {"sequences_per_backbone": 0},
        {"source_commit": "not-pinned"},
        {"target": "unknown"},
    ],
)
def test_unbound_allocations_are_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        audit(fixture(tmp_path), **kwargs)
