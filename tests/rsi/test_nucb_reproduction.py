"""NucB comparisons must detect changed scientific labels and row identities."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from scripts.rsi.cases.nucb_reproduction_checks import compare_tables


def tables(tmp_path):
    frame = pd.DataFrame(
        {
            "mutations": ["[]", '[["A",73,"R"]]'],
            "sublibrary_names": ['["g1","g2"]', '["g2"]'],
            "ef": [1.0, 2.0],
            "active": [False, True],
        }
    )
    a, b = tmp_path / "actual.csv", tmp_path / "reference.csv"
    frame.to_csv(a, index=False)
    frame.to_csv(b, index=False)
    return frame, a, b


def test_permutations_do_not_hide_scientific_changes(tmp_path):
    frame, a, b = tables(tmp_path)
    frame.loc[0, "sublibrary_names"] = '["g2","g1"]'
    frame.iloc[::-1].to_csv(a, index=False)
    assert compare_tables(a, b)["passed"]
    frame.loc[0, "active"] = True
    frame.to_csv(a, index=False)
    result = compare_tables(a, b)
    assert not result["passed"]
    assert result["differences"]["active"]["mismatched_rows"] == 1


def test_row_replacement_and_duplicate_keys_fail(tmp_path):
    frame, a, b = tables(tmp_path)
    frame.loc[0, "mutations"] = '[["A",63,"P"]]'
    frame.to_csv(a, index=False)
    result = compare_tables(a, b)
    assert result["actual_only_keys"] == result["reference_only_keys"] == 1
    assert not result["passed"]
    frame.loc[0, "mutations"] = frame.loc[1, "mutations"]
    frame.to_csv(a, index=False)
    assert compare_tables(a, b)["problem"] == "duplicate_key"


def test_numeric_tolerance_is_fixed_and_nested_multiplicity_preserved(tmp_path):
    frame, a, b = tables(tmp_path)
    frame.loc[0, "ef"] += 1e-10
    frame.to_csv(a, index=False)
    assert compare_tables(a, b)["passed"]
    frame.loc[0, "ef"] += 1e-5
    frame.to_csv(a, index=False)
    assert not compare_tables(a, b)["passed"]
    frame.loc[0, "ef"] = 1.0
    frame.loc[0, "sublibrary_names"] = '["g1","g2","g2"]'
    frame.to_csv(a, index=False)
    assert not compare_tables(a, b)["passed"]


def test_author_dna_tuple_serialization_matches_json_without_relaxing_identity(tmp_path):
    a, b = tmp_path / "dna.csv", tmp_path / "json.csv"
    pd.DataFrame(
        {"nuc_mutations": ["()", "(('A', 9, 'G'),)"], "mutations": ["[]", '[["A",3,"G"]]'], "count": [100, 90]}
    ).to_csv(a, index=False)
    pd.DataFrame(
        {"nuc_mutations": ["[]", '[["A",9,"G"]]'], "mutations": ["[]", '[["A",3,"G"]]'], "count": [100, 90]}
    ).to_csv(b, index=False)
    assert compare_tables(a, b)["passed"]
    frame = pd.read_csv(a)
    frame.loc[1, "nuc_mutations"] = "(('A', 10, 'G'),)"
    frame.to_csv(a, index=False)
    assert not compare_tables(a, b)["passed"]


def test_published_label_aliases_preserve_every_evaluation_row():
    from scripts.rsi.cases.run_nucb_cnn_reproduction import normalize_activity_labels

    frame = pd.DataFrame({"activity_level": ["non-functional", "activity > 0", "activity > WT", "activity > A73R"]})
    normalized = normalize_activity_labels(frame)
    assert normalized["activity_level"].tolist() == [
        "non-functional",
        "activity_greater_than_0",
        "activity_greater_than_WT",
        "activity_greater_than_A73R",
    ]
    assert normalize_activity_labels(normalized).equals(normalized)
    assert frame.loc[1, "activity_level"] == "activity > 0"


def test_unknown_or_missing_activity_cannot_be_silently_dropped():
    import pytest
    from scripts.rsi.cases.run_nucb_cnn_reproduction import normalize_activity_labels

    for value in ["unknown", None]:
        with pytest.raises(ValueError, match="Unknown or missing"):
            normalize_activity_labels(pd.DataFrame({"activity_level": [value]}))


def test_bootstrap_order_uses_identity_only_and_preserves_reconstructed_values():
    from scripts.rsi.cases.run_nucb_analysis_reproduction import align_bootstrap_row_order

    data = pd.DataFrame({"generation": ["g1", "g1", "g2"], "sequence": ["B", "A", "A"], "value": [2.0, 1.0, 3.0]})
    reference = pd.DataFrame(
        {"generation": ["g1", "g1", "g2"], "sequence": ["A", "B", "A"], "value": [99.0, 98.0, 97.0]}
    )
    before = data.copy(deep=True)
    ordered, receipt = align_bootstrap_row_order(data, reference)
    assert ordered["value"].tolist() == [1.0, 2.0, 3.0]
    assert ordered[["generation", "sequence"]].equals(reference[["generation", "sequence"]])
    assert data.equals(before)
    assert receipt["rows"] == 3 and receipt["reordered_rows"] == 2
    again, repeated = align_bootstrap_row_order(ordered, reference)
    assert again.equals(ordered) and repeated["reordered_rows"] == 0
    assert repeated["identity_order_sha256"] == receipt["identity_order_sha256"]
    changed_values = reference.assign(value=[1000, 2000, 3000])
    same, same_receipt = align_bootstrap_row_order(data, changed_values)
    assert same.equals(ordered) and same_receipt == receipt


def test_bootstrap_order_rejects_missing_replaced_or_duplicate_identities():
    import pytest
    from scripts.rsi.cases.run_nucb_analysis_reproduction import align_bootstrap_row_order

    data = pd.DataFrame({"generation": ["g1", "g2"], "sequence": ["A", "A"], "value": [1.0, 2.0]})
    for reference in [data.iloc[:1], data.assign(sequence=["B", "A"]), pd.concat([data, data.iloc[:1]])]:
        with pytest.raises(ValueError):
            align_bootstrap_row_order(data, reference)
    with pytest.raises(ValueError):
        align_bootstrap_row_order(pd.concat([data, data.iloc[:1]]), data)
