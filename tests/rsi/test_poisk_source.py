import math
from pathlib import Path

import pytest

from agent.rsi import poisk_source as source
from agent.rsi import poisk_training as training
from agent.rsi.storage import atomic_json, sha256_file


@pytest.mark.parametrize(
    "raw,endpoint,molar",
    [
        ("KD = 2.4 x 10^-9 M", "KD", 2.4e-9),
        ("Kd: 2.4E-9 M", "KD", 2.4e-9),
        ("0.092 nM", "reported_molar_concentration", 9.2e-11),
        ("IC50 = 0.5 µM", "IC50", 5e-7),
        ("EC50: 2 pM", "EC50", 2e-12),
        ("KD=1×10^−9 M", "KD", 1e-9),
    ],
)
def test_quantity(raw, endpoint, molar):
    result = source.parse_quantity(raw)
    assert result["status"] == "point" and result["endpoint"] == endpoint
    assert result["molar"] == pytest.approx(molar)
    assert result["negative_log10_molar"] == pytest.approx(-math.log10(molar))


@pytest.mark.parametrize("raw", ["<0.01 nM", "KD >= 5 nM", "≤ 2 nM", "> 0.1 uM"])
def test_censoring_not_point_or_negative_label(raw):
    assert source.parse_quantity(raw)["status"] == "censored"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "NA/NA",
        "1-3 nM",
        "5 ug/ml",
        "kd: 3E-2 1/s",
        "1.8e8 M^-1",
        "~4-fold lower",
        "5 nM at 7.4",
        "0 M",
        "1e999 M",
        "1 nM; 2 nM",
        "10-9 M",
        "−1 nM",
    ],
)
def test_ambiguous_or_invalid_quantity_rejected(raw):
    assert source.parse_quantity(raw)["status"] not in {"point", "censored"}


def raw_row():
    return {
        "unique_record_id": "one",
        "patent_id": "123",
        "target.uniprot_id": "P02768",
        "target.organism": "Human",
        "audit.verdict": "CONFIRMED",
        "audit.flags": "",
        "antibody_type": "Conventional",
        "vh.sequence_from_patent": "ACDE" * 25,
        "vl.sequence_from_patent": "FGHI" * 25,
        "vh.is_full_heavy_chain": "False",
        "vl.is_full_light_chain": "False",
        "target.sequence": "KLMN" * 10,
        "target.binding_affinity.method": "SPR (Biacore)",
        "target.binding_affinity.affinity_value": "KD=5 nM",
        "target.binding_conditions": "",
    }


def test_source_admission_and_pair_preservation():
    row = raw_row()
    refs = {"P02768": {"gene": "ALB", "sequence": "KLMN" * 10}}
    normalized, reason = source.normalize_row(row, refs)
    assert reason is None and normalized["heavy"] != normalized["light"]
    for key, value in [
        ("target.organism", "Mouse"),
        ("audit.verdict", "REVIEW"),
        ("audit.flags", "target.binding_affinity"),
        ("antibody_type", "Bispecific"),
        ("vl.sequence_from_patent", ""),
        ("target.sequence", "ACDE" * 10),
        ("target.binding_affinity.method", "SPR and ELISA"),
    ]:
        altered = dict(row)
        altered[key] = value
        assert source.normalize_row(altered, refs)[1] is not None


def test_deduplication_conflicts_and_patent_provenance():
    row, _ = source.normalize_row(raw_row(), {"P02768": {"gene": "ALB", "sequence": "KLMN" * 10}})
    other = dict(row, source_record="two", patent="456")
    kept, conflict = source.deduplicate([row, other])
    assert not conflict and kept[0]["patents"] == ["123", "456"]
    other["molar"] *= 2
    assert source.deduplicate([row, other])[0] == []


def test_grouping_shared_chains_patents_and_near_clones():
    rows = [
        {"id": "a", "antibody": "a", "heavy": "A" * 100, "light": "C" * 100, "patents": ["1"]},
        {"id": "b", "antibody": "b", "heavy": "D" * 100, "light": "E" * 100, "patents": ["1"]},
        {"id": "c", "antibody": "c", "heavy": "A" * 100, "light": "F" * 100, "patents": ["2"]},
        {"id": "d", "antibody": "d", "heavy": "G" * 100, "light": "F" * 100, "patents": ["3"]},
        {"id": "e", "antibody": "e", "heavy": "G" * 99 + "A", "light": "F" * 99 + "C", "patents": ["4"]},
        {"id": "f", "antibody": "f", "heavy": "H" * 100, "light": "I" * 100, "patents": ["5"]},
    ]
    groups, _ = training.components(rows)
    assert len({groups[k] for k in "abcde"}) == 1 and groups["f"] != groups["a"]
    reversed_groups, _ = training.components(list(reversed(rows)))
    assert reversed_groups == groups


def test_encoding_is_pair_order_sensitive_and_target_free():
    row = {"heavy": "ACDE" * 25, "light": "FGHI" * 25, "target_accession": "P02768"}
    encoded = training.encode(row)
    assert len(encoded) == 118
    assert encoded != training.encode(dict(row, heavy=row["light"], light=row["heavy"]))
    assert encoded == training.encode(dict(row, target=-999))


def test_split_uses_counts_and_keeps_components_whole():
    rows = [{"id": f"{g}_{i}", "group": str(g), "target": i} for g in range(10) for i in range(4)]
    first = training.choose_split(rows)
    assert first is not None and len(first) == 10
    for r in rows:
        r["target"] = 99999
    assert training.choose_split(rows) == first


def test_controls_do_not_read_outer_labels(tmp_path, monkeypatch):
    folder = tmp_path / "tasks"
    folder.mkdir()
    rows = [
        {"id": f"{g}_{i}", "group": str(g), "values": [g / 10, i / 4], "target": g + i / 10}
        for g in range(6)
        for i in range(4)
    ]
    atomic_json(folder / "training.json", rows)
    task = {
        "id": "fixture",
        "scope": {"endpoint": "reported_molar_concentration"},
        "data": {
            "training": {"path": str(folder / "training.json"), "sha256": sha256_file(folder / "training.json")},
            "validation": {"path": "FORBIDDEN"},
            "future": {"path": "FORBIDDEN"},
        },
    }
    atomic_json(
        folder / "manifest.json", {"encoder_source_sha256": sha256_file(Path(training.__file__)), "tasks": [task]}
    )
    original = Path.read_text

    def checked(path, *args, **kwargs):
        assert path.name not in {"validation.json", "future.json", "FORBIDDEN"}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checked)
    report = training.train_controls(folder, tmp_path / "controls")
    assert report["final_fitted_control_models"] == 4 and report["internal_cv_fits"] == 12
    assert report["new_autonomous_models"] == 0 and not report["outer_labels_read"]
