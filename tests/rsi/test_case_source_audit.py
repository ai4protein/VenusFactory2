import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("audit_case_sources", Path(__file__).resolve().parents[2] / "scripts/rsi/audit_case_sources.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_tampered_source_is_rejected(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes(b"id,value\na,2\n")
    asset = {"path": "data.csv", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    path.write_bytes(b"id,value\na,3\n")
    with pytest.raises(ValueError, match="checksum"):
        audit.verify_asset(tmp_path, asset)


def test_outside_source_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        audit.contained_file(tmp_path, "../heldout_labels.csv")


def test_missing_values_are_not_zero_or_dropped(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("id,value\na,0\na,\nb,inf\n")
    result = audit.summarize_csv(path, "id")
    assert result["rows"] == 3
    assert result["duplicate_identifier_rows"] == 1
    assert result["missing_by_column"]["value"] == 1
    assert result["nonfinite_numeric_by_column"]["value"] == 1


def test_misaligned_csv_fails(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("id,value\na,1,2\n")
    with pytest.raises(ValueError, match="length"):
        audit.summarize_csv(path)


def test_zero_denominator_is_explicit(tmp_path):
    path = tmp_path / "alde.csv"
    path.write_text("round,yield1,yield2\ninitial,10,0\ninitial,8,2\n")
    result = audit.alde_trajectory(path)
    assert result["rounds"]["initial"]["undefined_selectivity_rows"] == 1
    assert result["rounds"]["initial"]["best_by_yield_difference_selectivity"] is None
    assert result["full_author_baseline_completed"] is False
    assert result["agent_run"] is False
